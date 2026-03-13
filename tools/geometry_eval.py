import argparse
import json
import math
import os
import warnings

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_feature_file", default=None, type=str)
    parser.add_argument("--target_feature_file", default=None, type=str)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--num_classes", default=None, type=int)
    parser.add_argument("--samples_per_class", default=None, type=int)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--metrics", default="centroid_cosine,centroid_euclidean,cov_frobenius,pca_subspace,pairwise_spearman,pairwise_pearson", type=str)
    parser.add_argument("--class_indices", default=None, type=str, help="Comma-separated ImageNet class indices to evaluate.")
    parser.add_argument("--pca_dim", default=8, type=int)
    parser.add_argument("--self_test_pairwise_corr", action="store_true", default=False)
    return parser.parse_args()


def load_feature_file(path):
    payload = torch.load(path, map_location="cpu")
    required_keys = {"features", "labels", "classnames", "dataset_name", "backbone"}
    missing = sorted(required_keys - set(payload.keys()))
    if missing:
        raise KeyError(f"Feature file '{path}' is missing required keys: {missing}")
    return payload


def parse_metrics(metrics_arg):
    return [metric.strip() for metric in metrics_arg.split(",") if metric.strip()]


def parse_class_indices(class_indices_arg):
    if not class_indices_arg:
        return None
    return [int(value.strip()) for value in class_indices_arg.split(",") if value.strip()]


def build_class_index(labels):
    class_to_indices = {}
    for index, label in enumerate(labels.tolist()):
        class_to_indices.setdefault(label, []).append(index)
    return class_to_indices


def get_class_name(payload, class_index):
    classnames = payload.get("classnames", [])
    if class_index < len(classnames):
        return classnames[class_index]
    return str(class_index)


def sample_class_features(features, class_to_indices, class_index, samples_per_class, rng):
    candidate_indices = class_to_indices[class_index]
    # We compare class geometry using the same number of examples per class
    # so the covariance/PCA/pairwise metrics are on equal footing.
    sampled_indices = rng.choice(candidate_indices, size=samples_per_class, replace=False)
    sampled_indices = np.sort(sampled_indices)
    return features[sampled_indices]


def compute_covariance(matrix):
    # Basic sample covariance of one class cloud in CLIP feature space.
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    denom = max(matrix.shape[0] - 1, 1)
    return centered.T @ centered / denom


def compute_centroid_cosine(source, target):
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    denom = np.linalg.norm(source_centroid) * np.linalg.norm(target_centroid)
    if denom == 0:
        return float("nan")
    return float(np.dot(source_centroid, target_centroid) / denom)


def compute_centroid_euclidean(source, target):
    return float(np.linalg.norm(source.mean(axis=0) - target.mean(axis=0)))


def compute_cov_frobenius(source, target):
    return float(np.linalg.norm(compute_covariance(source) - compute_covariance(target), ord="fro"))


def compute_pca_subspace_similarity(source, target, pca_dim):
    # Compare the top principal directions of two class clouds.
    # Higher means the dominant geometric directions are more aligned.
    source_centered = source - source.mean(axis=0, keepdims=True)
    target_centered = target - target.mean(axis=0, keepdims=True)
    max_rank = min(source_centered.shape[0] - 1, target_centered.shape[0] - 1, source_centered.shape[1], pca_dim)
    if max_rank <= 0:
        return float("nan")

    source_basis = np.linalg.svd(source_centered, full_matrices=False)[2][:max_rank].T
    target_basis = np.linalg.svd(target_centered, full_matrices=False)[2][:max_rank].T
    singular_values = np.linalg.svd(source_basis.T @ target_basis, compute_uv=False)
    singular_values = np.clip(singular_values, -1.0, 1.0)
    return float(np.mean(singular_values))


def compute_pairwise_distance_vector(features):
    num_samples = features.shape[0]
    if num_samples < 3:
        warnings.warn(
            f"Pairwise correlation needs at least 3 samples to be stable, got {num_samples}. Returning NaN.",
            RuntimeWarning,
        )
        return None

    # Build the within-class distance matrix, then keep only the unique
    # pair distances (upper triangle without the diagonal).
    diffs = features[:, None, :] - features[None, :, :]
    distance_matrix = np.linalg.norm(diffs, axis=-1)
    upper_triangular = np.triu_indices(num_samples, k=1)
    return distance_matrix[upper_triangular]


def compute_pairwise_correlation(source, target, method):
    if source.shape[0] != target.shape[0]:
        num_samples = min(source.shape[0], target.shape[0])
        warnings.warn(
            f"Source/target sample counts differ ({source.shape[0]} vs {target.shape[0]}). "
            f"Using the first {num_samples} samples from each.",
            RuntimeWarning,
        )
        source = source[:num_samples]
        target = target[:num_samples]

    source_distances = compute_pairwise_distance_vector(source)
    target_distances = compute_pairwise_distance_vector(target)

    if source_distances is None or target_distances is None:
        return float("nan")

    if source_distances.shape != target_distances.shape:
        warnings.warn(
            f"Pairwise vector sizes do not match ({source_distances.shape} vs {target_distances.shape}). Returning NaN.",
            RuntimeWarning,
        )
        return float("nan")

    if np.array_equal(source_distances, target_distances):
        return 1.0

    if np.var(source_distances) == 0 or np.var(target_distances) == 0:
        warnings.warn("Zero variance detected in pairwise distance vector. Returning NaN.", RuntimeWarning)
        return float("nan")

    if method == "pairwise_spearman":
        return float(spearmanr(source_distances, target_distances).statistic)
    if method == "pairwise_pearson":
        return float(pearsonr(source_distances, target_distances)[0])
    raise ValueError(f"Unsupported pairwise correlation metric: {method}")


def run_pairwise_corr_self_test(seed):
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(8, 16))
    identical_features = features.copy()

    spearman_identical = compute_pairwise_correlation(features, identical_features, "pairwise_spearman")
    pearson_identical = compute_pairwise_correlation(features, identical_features, "pairwise_pearson")

    if spearman_identical != 1.0 or pearson_identical != 1.0:
        raise AssertionError("Identical pairwise vectors must yield correlation 1.0.")

    pairwise_vector = compute_pairwise_distance_vector(features)
    permuted_vector = rng.permutation(pairwise_vector)

    spearman_permuted = float(spearmanr(pairwise_vector, permuted_vector).statistic)
    pearson_permuted = float(pearsonr(pairwise_vector, permuted_vector)[0])

    if spearman_permuted >= spearman_identical:
        raise AssertionError("Spearman correlation should decrease after permutation.")
    if pearson_permuted >= pearson_identical:
        raise AssertionError("Pearson correlation should decrease after permutation.")

    print("pairwise_spearman identical:", spearman_identical)
    print("pairwise_pearson identical:", pearson_identical)
    print("pairwise_spearman permuted:", spearman_permuted)
    print("pairwise_pearson permuted:", pearson_permuted)
    print("Pairwise correlation self-test passed.")


def compute_metrics(source, target, metrics, pca_dim):
    results = {}
    for metric in metrics:
        if metric == "centroid_cosine":
            results[metric] = compute_centroid_cosine(source, target)
        elif metric == "centroid_euclidean":
            results[metric] = compute_centroid_euclidean(source, target)
        elif metric == "cov_frobenius":
            results[metric] = compute_cov_frobenius(source, target)
        elif metric == "pca_subspace":
            results[metric] = compute_pca_subspace_similarity(source, target, pca_dim)
        elif metric in {"pairwise_spearman", "pairwise_pearson"}:
            results[metric] = compute_pairwise_correlation(source, target, metric)
        else:
            raise ValueError(f"Unsupported metric: {metric}")
    return results


def get_retrieval_modes():
    return ["centroid_cosine", "centroid_euclidean", "pca_subspace"]


def compute_retrieval_score(source, target, mode, pca_dim):
    if mode == "centroid_cosine":
        return compute_centroid_cosine(source, target)
    if mode == "centroid_euclidean":
        return -compute_centroid_euclidean(source, target)
    if mode == "pca_subspace":
        return compute_pca_subspace_similarity(source, target, pca_dim)
    raise ValueError(f"Unsupported retrieval mode: {mode}")


def summarize_top_mistakes(retrieval_results, top_k=5):
    mistake_counts = {}
    for result in retrieval_results:
        if result["correct"]:
            continue
        key = (result["source_class_index"], result["predicted_target_class_index"])
        mistake_counts[key] = mistake_counts.get(key, 0) + 1

    sorted_mistakes = sorted(mistake_counts.items(), key=lambda item: (-item[1], item[0]))
    top_mistakes = []
    for (source_class_index, predicted_target_class_index), count in sorted_mistakes[:top_k]:
        top_mistakes.append({
            "source_class_index": source_class_index,
            "predicted_target_class_index": predicted_target_class_index,
            "count": count,
        })
    return top_mistakes


def compute_retrieval_results(selected_classes, source_class_samples, target_class_samples, class_names, pca_dim):
    retrieval = {}
    for mode in get_retrieval_modes():
        per_class = []
        num_correct = 0

        for class_index in selected_classes:
            # Retrieval asks: "which target class geometry looks most like
            # this source class geometry?"
            scored_targets = []
            for target_class_index in selected_classes:
                score = compute_retrieval_score(
                    source_class_samples[class_index],
                    target_class_samples[target_class_index],
                    mode,
                    pca_dim,
                )
                scored_targets.append((target_class_index, score))

            scored_targets.sort(key=lambda item: item[1], reverse=True)
            best_class_index, best_score = scored_targets[0]
            second_best_score = scored_targets[1][1] if len(scored_targets) > 1 else float("nan")
            correct = best_class_index == class_index
            num_correct += int(correct)

            per_class.append({
                "source_class_index": class_index,
                "source_class_name": class_names[class_index],
                "predicted_target_class_index": best_class_index,
                "predicted_target_class_name": class_names[best_class_index],
                "correct": correct,
                "best_score": float(best_score),
                "score_margin": float(best_score - second_best_score) if not math.isnan(second_best_score) else float("nan"),
            })

        retrieval[mode] = {
            "accuracy": float(num_correct / len(selected_classes)) if selected_classes else float("nan"),
            "per_class": per_class,
            "top_mistakes": summarize_top_mistakes(per_class),
        }

    return retrieval


def mean_dict(metric_dicts, metrics):
    summary = {}
    for metric in metrics:
        values = [entry[metric] for entry in metric_dicts if not math.isnan(entry[metric])]
        summary[metric] = float(np.mean(values)) if values else float("nan")
    return summary


def main():
    args = get_arguments()
    if args.self_test_pairwise_corr:
        run_pairwise_corr_self_test(args.seed)
        return

    required_args = ["source_feature_file", "target_feature_file", "output", "num_classes", "samples_per_class"]
    missing_args = [name for name in required_args if getattr(args, name) is None]
    if missing_args:
        raise ValueError(f"Missing required arguments: {missing_args}")

    rng = np.random.default_rng(args.seed)
    metrics = parse_metrics(args.metrics)
    class_subset = parse_class_indices(args.class_indices)

    source_payload = load_feature_file(args.source_feature_file)
    target_payload = load_feature_file(args.target_feature_file)

    source_features = source_payload["features"].float().cpu().numpy()
    target_features = target_payload["features"].float().cpu().numpy()
    source_labels = source_payload["labels"].long().cpu()
    target_labels = target_payload["labels"].long().cpu()

    if source_payload["backbone"] != target_payload["backbone"]:
        raise ValueError("Source and target feature files must use the same backbone.")
    if source_features.shape[1] != target_features.shape[1]:
        raise ValueError("Feature dimensions do not match between source and target files.")

    source_class_to_indices = build_class_index(source_labels)
    target_class_to_indices = build_class_index(target_labels)
    eligible_classes = sorted(
        set(source_class_to_indices).intersection(target_class_to_indices)
    )
    eligible_classes = [
        class_index for class_index in eligible_classes
        if len(source_class_to_indices[class_index]) >= args.samples_per_class
        and len(target_class_to_indices[class_index]) >= args.samples_per_class
    ]

    if class_subset is not None:
        eligible_classes = [class_index for class_index in class_subset if class_index in eligible_classes]

    if len(eligible_classes) < args.num_classes:
        raise ValueError(
            f"Requested {args.num_classes} classes, but only {len(eligible_classes)} classes have at least "
            f"{args.samples_per_class} samples in both datasets."
        )

    selected_classes = sorted(rng.choice(eligible_classes, size=args.num_classes, replace=False).tolist())
    per_class_results = []
    same_class_metrics = []
    different_class_metrics = []
    source_class_samples = {}
    target_class_samples = {}
    class_names = {}

    # Sample each selected class once and reuse those same examples everywhere
    # below. This keeps the metric comparisons and retrieval experiment consistent.
    for class_index in selected_classes:
        source_class_name = get_class_name(source_payload, class_index)
        target_class_name = get_class_name(target_payload, class_index)
        if source_class_name != target_class_name:
            raise ValueError(
                f"Class name mismatch at ImageNet index {class_index}: "
                f"source='{source_class_name}', target='{target_class_name}'."
            )
        source_class_samples[class_index] = sample_class_features(
            source_features, source_class_to_indices, class_index, args.samples_per_class, rng
        )
        target_class_samples[class_index] = sample_class_features(
            target_features, target_class_to_indices, class_index, args.samples_per_class, rng
        )
        class_names[class_index] = source_class_name

    for class_index in selected_classes:
        source_class_name = class_names[class_index]
        source_samples = source_class_samples[class_index]
        target_samples = target_class_samples[class_index]
        same_metrics = compute_metrics(source_samples, target_samples, metrics, args.pca_dim)
        same_class_metrics.append(same_metrics)

        control_metrics_list = []
        for target_class_index in selected_classes:
            if target_class_index == class_index:
                continue
            # Control: compare the source class against all *other* target classes.
            target_control_samples = target_class_samples[target_class_index]
            control_metrics_list.append(compute_metrics(source_samples, target_control_samples, metrics, args.pca_dim))

        averaged_control_metrics = mean_dict(control_metrics_list, metrics)
        different_class_metrics.append(averaged_control_metrics)

        per_class_results.append({
            "class_index": class_index,
            "class_name": source_class_name,
            "same_class": same_metrics,
            "different_class_control": averaged_control_metrics,
        })

    retrieval_results = compute_retrieval_results(
        selected_classes=selected_classes,
        source_class_samples=source_class_samples,
        target_class_samples=target_class_samples,
        class_names=class_names,
        pca_dim=args.pca_dim,
    )

    results = {
        "source_dataset": source_payload["dataset_name"],
        "target_dataset": target_payload["dataset_name"],
        "backbone": source_payload["backbone"],
        "metrics": metrics,
        "num_classes": args.num_classes,
        "samples_per_class": args.samples_per_class,
        "seed": args.seed,
        "selected_classes": selected_classes,
        "averages": {
            "same_class": mean_dict(same_class_metrics, metrics),
            "different_class_control": mean_dict(different_class_metrics, metrics),
        },
        "retrieval": retrieval_results,
        "per_class": per_class_results,
    }

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)

    print(f"Source dataset: {results['source_dataset']}")
    print(f"Target dataset: {results['target_dataset']}")
    print(f"Backbone: {results['backbone']}")
    print(f"Selected classes: {len(selected_classes)}")
    for metric in metrics:
        same_value = results["averages"]["same_class"][metric]
        diff_value = results["averages"]["different_class_control"][metric]
        print(f"{metric}: same={same_value:.6f} different={diff_value:.6f}")
    for mode, retrieval_result in results["retrieval"].items():
        print(f"retrieval_acc_{mode}: {retrieval_result['accuracy']:.6f}")
    print(f"Saved JSON to: {args.output}")


if __name__ == "__main__":
    main()
