import argparse
import json
import math
import os
import time
import warnings

import numpy as np
import torch
from scipy.spatial.distance import jensenshannon
from scipy.stats import pearsonr, spearmanr, wasserstein_distance
from sklearn.manifold import trustworthiness


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
    parser.add_argument("--same_dataset_upper_bound", action="store_true", default=False)
    parser.add_argument("--min_samples_per_class_for_split", default=None, type=int)
    parser.add_argument("--upper_bound_num_repeats", default=1, type=int)
    parser.add_argument("--knn_k", nargs="+", type=int, default=[5, 10, 20])
    parser.add_argument("--compute_knn_distribution", action="store_true", default=False)
    parser.add_argument("--compute_distance_histogram", action="store_true", default=False)
    parser.add_argument("--compute_graph_stats", action="store_true", default=False)
    parser.add_argument("--distance_hist_bins", default=20, type=int)
    parser.add_argument("--device", default=None, type=str, help="Device for geometry cache computation, e.g. cpu, cuda, cuda:0.")
    parser.add_argument("--inter_class_geometry", action="store_true", default=False)
    parser.add_argument("--sign_epsilon", default=0.05, type=float)
    parser.add_argument("--save_inter_class_plots", action="store_true", default=False)
    parser.add_argument("--save_umap", action="store_true", default=False)
    parser.add_argument("--viz_dim", default=2, type=int, choices=[2, 3])
    parser.add_argument("--viz_classes", nargs="+", default=None, help="Class names to visualize, or 'all'.")
    parser.add_argument("--viz_class_ids", nargs="+", type=int, default=None)
    parser.add_argument("--viz_max_points_per_class", default=100, type=int)
    parser.add_argument("--interactive", action="store_true", default=False)
    return parser.parse_args()


def load_feature_file(path):
    payload = torch.load(path, map_location="cpu")
    required_keys = {"features", "labels", "classnames", "dataset_name", "backbone"}
    missing = sorted(required_keys - set(payload.keys()))
    if missing:
        raise KeyError(f"Feature file '{path}' is missing required keys: {missing}")
    return payload


def resolve_device(device_arg):
    if device_arg:
        return torch.device(device_arg)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def log_timing(label, start_time):
    elapsed = time.perf_counter() - start_time
    print(f"[timing] {label}: {elapsed:.2f}s")


def parse_metrics(metrics_arg):
    return [metric.strip() for metric in metrics_arg.split(",") if metric.strip()]


def parse_class_indices(class_indices_arg):
    if not class_indices_arg:
        return None
    return [int(value.strip()) for value in class_indices_arg.split(",") if value.strip()]


def get_output_prefix(output_path):
    directory = os.path.dirname(output_path) or "."
    stem = os.path.splitext(os.path.basename(output_path))[0]
    return os.path.join(directory, stem)


def select_visualization_class_ids(selected_classes, class_names, args):
    if args.viz_class_ids:
        return [class_index for class_index in args.viz_class_ids if class_index in selected_classes]
    if args.viz_classes:
        if len(args.viz_classes) == 1 and args.viz_classes[0].lower() == "all":
            return list(selected_classes)
        requested_names = set(args.viz_classes)
        return [class_index for class_index in selected_classes if class_names[class_index] in requested_names]
    return list(selected_classes)


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


def compute_pca_basis(matrix, pca_dim):
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    max_rank = min(centered.shape[0] - 1, centered.shape[1], pca_dim)
    if max_rank <= 0:
        return None
    return np.linalg.svd(centered, full_matrices=False)[2][:max_rank].T


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


def compute_pca_subspace_similarity_from_basis(source_basis, target_basis):
    if source_basis is None or target_basis is None:
        return float("nan")
    max_rank = min(source_basis.shape[1], target_basis.shape[1])
    if max_rank <= 0:
        return float("nan")
    singular_values = np.linalg.svd(source_basis[:, :max_rank].T @ target_basis[:, :max_rank], compute_uv=False)
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


def compute_distance_matrix(features):
    diffs = features[:, None, :] - features[None, :, :]
    return np.linalg.norm(diffs, axis=-1)


def sanitize_distance_vector(distances, context):
    if distances is None or distances.size == 0:
        warnings.warn(f"{context}: empty distance vector. Skipping metric.", RuntimeWarning)
        return None
    distances = distances[np.isfinite(distances)]
    if distances.size == 0:
        warnings.warn(f"{context}: all distances are NaN/Inf. Skipping metric.", RuntimeWarning)
        return None
    return distances


def compute_knn_distance_values(features, k):
    if features.shape[0] <= k:
        warnings.warn(
            f"kNN distance distribution requires more than k={k} samples, got {features.shape[0]}. Skipping.",
            RuntimeWarning,
        )
        return None

    distance_matrix = compute_distance_matrix(features)
    np.fill_diagonal(distance_matrix, np.inf)
    neighbor_distances = np.sort(distance_matrix, axis=1)[:, :k].reshape(-1)
    return sanitize_distance_vector(neighbor_distances, f"kNN(k={k})")


def compute_knn_distribution_metrics(source, target, k_values):
    metrics = {
        "knn_wasserstein": {},
        "knn_mean_diff": {},
        "knn_std_diff": {},
    }
    for k in k_values:
        source_knn = compute_knn_distance_values(source, k)
        target_knn = compute_knn_distance_values(target, k)
        key = f"k={k}"
        if source_knn is None or target_knn is None:
            metrics["knn_wasserstein"][key] = float("nan")
            metrics["knn_mean_diff"][key] = float("nan")
            metrics["knn_std_diff"][key] = float("nan")
            continue

        metrics["knn_wasserstein"][key] = float(wasserstein_distance(source_knn, target_knn))
        metrics["knn_mean_diff"][key] = float(abs(np.mean(source_knn) - np.mean(target_knn)))
        metrics["knn_std_diff"][key] = float(abs(np.std(source_knn) - np.std(target_knn)))
    return metrics


def compute_knn_distribution_metrics_from_cache(source_cache, target_cache, k_values):
    # Compare local neighborhood geometry without requiring image-to-image
    # correspondence between the two datasets.
    metrics = {
        "knn_wasserstein": {},
        "knn_mean_diff": {},
        "knn_std_diff": {},
    }
    for k in k_values:
        key = f"k={k}"
        source_knn = source_cache["knn_values"].get(key)
        target_knn = target_cache["knn_values"].get(key)
        if source_knn is None or target_knn is None:
            metrics["knn_wasserstein"][key] = float("nan")
            metrics["knn_mean_diff"][key] = float("nan")
            metrics["knn_std_diff"][key] = float("nan")
            continue
        metrics["knn_wasserstein"][key] = float(wasserstein_distance(source_knn, target_knn))
        metrics["knn_mean_diff"][key] = float(abs(np.mean(source_knn) - np.mean(target_knn)))
        metrics["knn_std_diff"][key] = float(abs(np.std(source_knn) - np.std(target_knn)))
    return metrics


def compute_distance_histogram_metrics(source, target, num_bins):
    # Compare the full within-class distance distribution. This is a coarse
    # "map similarity" score for the class cloud.
    source_distances = sanitize_distance_vector(compute_pairwise_distance_vector(source), "distance histogram source")
    target_distances = sanitize_distance_vector(compute_pairwise_distance_vector(target), "distance histogram target")
    if source_distances is None or target_distances is None:
        return {
            "distance_histogram_js": float("nan"),
            "distance_histogram_wasserstein": float("nan"),
        }

    combined = np.concatenate([source_distances, target_distances])
    min_value = float(np.min(combined))
    max_value = float(np.max(combined))
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        warnings.warn("Distance histogram: non-finite bin range. Skipping metric.", RuntimeWarning)
        return {
            "distance_histogram_js": float("nan"),
            "distance_histogram_wasserstein": float("nan"),
        }
    if min_value == max_value:
        bins = np.linspace(min_value, min_value + 1e-6, num_bins + 1)
    else:
        bins = np.linspace(min_value, max_value, num_bins + 1)

    source_hist, _ = np.histogram(source_distances, bins=bins, density=False)
    target_hist, _ = np.histogram(target_distances, bins=bins, density=False)
    source_hist = source_hist.astype(np.float64)
    target_hist = target_hist.astype(np.float64)
    source_hist /= max(source_hist.sum(), 1.0)
    target_hist /= max(target_hist.sum(), 1.0)

    return {
        "distance_histogram_js": float(jensenshannon(source_hist + 1e-12, target_hist + 1e-12) ** 2),
        "distance_histogram_wasserstein": float(wasserstein_distance(source_distances, target_distances)),
    }


def compute_distance_histogram_metrics_from_cache(source_cache, target_cache, num_bins):
    source_distances = source_cache["pairwise_vector"]
    target_distances = target_cache["pairwise_vector"]
    if source_distances is None or target_distances is None:
        return {
            "distance_histogram_js": float("nan"),
            "distance_histogram_wasserstein": float("nan"),
        }

    combined = np.concatenate([source_distances, target_distances])
    min_value = float(np.min(combined))
    max_value = float(np.max(combined))
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        warnings.warn("Distance histogram: non-finite bin range. Skipping metric.", RuntimeWarning)
        return {
            "distance_histogram_js": float("nan"),
            "distance_histogram_wasserstein": float("nan"),
        }
    bins = np.linspace(min_value, min_value + 1e-6, num_bins + 1) if min_value == max_value else np.linspace(min_value, max_value, num_bins + 1)
    source_hist, _ = np.histogram(source_distances, bins=bins, density=False)
    target_hist, _ = np.histogram(target_distances, bins=bins, density=False)
    source_hist = source_hist.astype(np.float64)
    target_hist = target_hist.astype(np.float64)
    source_hist /= max(source_hist.sum(), 1.0)
    target_hist /= max(target_hist.sum(), 1.0)
    return {
        "distance_histogram_js": float(jensenshannon(source_hist + 1e-12, target_hist + 1e-12) ** 2),
        "distance_histogram_wasserstein": float(wasserstein_distance(source_distances, target_distances)),
    }


def compute_undirected_knn_adjacency(features, k):
    if features.shape[0] <= k:
        warnings.warn(
            f"Neighbor graph stats require more than k={k} samples, got {features.shape[0]}. Skipping.",
            RuntimeWarning,
        )
        return None, None
    distance_matrix = compute_distance_matrix(features)
    np.fill_diagonal(distance_matrix, np.inf)
    knn_indices = np.argsort(distance_matrix, axis=1)[:, :k]
    adjacency = np.zeros((features.shape[0], features.shape[0]), dtype=bool)
    rows = np.arange(features.shape[0])[:, None]
    adjacency[rows, knn_indices] = True
    adjacency = np.logical_or(adjacency, adjacency.T)
    np.fill_diagonal(adjacency, False)
    return adjacency, distance_matrix


def summarize_graph_stats(features, k_values):
    summary = {}
    for k in k_values:
        adjacency, distance_matrix = compute_undirected_knn_adjacency(features, k)
        key = f"k={k}"
        if adjacency is None:
            summary[key] = {
                "mean_knn_distance": float("nan"),
                "std_knn_distance": float("nan"),
                "average_node_degree": float("nan"),
            }
            continue

        finite_distances = np.sort(distance_matrix, axis=1)[:, :k].reshape(-1)
        finite_distances = sanitize_distance_vector(finite_distances, f"graph stats k={k}")
        degrees = adjacency.sum(axis=1).astype(np.float64)
        summary[key] = {
            # Average local spacing between points.
            "mean_knn_distance": float(np.mean(finite_distances)) if finite_distances is not None else float("nan"),
            # Whether local density is uniform or very uneven.
            "std_knn_distance": float(np.std(finite_distances)) if finite_distances is not None else float("nan"),
            # Symmetrized kNN graph degree; useful for spotting topology shifts.
            "average_node_degree": float(np.mean(degrees)),
        }
    return summary


def prepare_class_geometry_cache(features, k_values, pca_dim, device):
    # Precompute the geometry summary of one class cloud once, then reuse it
    # for same-class comparison, different-class controls, retrieval, and
    # upper-bound repeats. This avoids repeating expensive distance/PCA work.
    with torch.no_grad():
        feature_tensor = torch.as_tensor(features, dtype=torch.float32, device=device)
        centroid = feature_tensor.mean(dim=0)
        centered = feature_tensor - centroid.unsqueeze(0)
        denom = max(feature_tensor.shape[0] - 1, 1)
        covariance = (centered.T @ centered) / denom

        max_rank = min(feature_tensor.shape[0] - 1, feature_tensor.shape[1], pca_dim)
        if max_rank <= 0:
            pca_basis = None
        else:
            pca_basis = torch.linalg.svd(centered, full_matrices=False)[2][:max_rank].T.contiguous()

        distance_matrix = torch.cdist(feature_tensor, feature_tensor).cpu().numpy()

    distance_matrix_no_diag = distance_matrix.copy()
    np.fill_diagonal(distance_matrix_no_diag, np.inf)
    pairwise_vector = sanitize_distance_vector(
        distance_matrix[np.triu_indices(distance_matrix.shape[0], k=1)] if distance_matrix.shape[0] >= 2 else None,
        "pairwise cache",
    )
    knn_values = {}
    graph_summary = {}
    num_samples = distance_matrix.shape[0]
    for k in k_values:
        if num_samples <= k:
            knn_values[f"k={k}"] = None
            graph_summary[f"k={k}"] = {
                "mean_knn_distance": float("nan"),
                "std_knn_distance": float("nan"),
                "average_node_degree": float("nan"),
            }
            continue
        sorted_distances = np.sort(distance_matrix_no_diag, axis=1)[:, :k]
        flattened_knn = sanitize_distance_vector(sorted_distances.reshape(-1), f"kNN cache k={k}")
        knn_values[f"k={k}"] = flattened_knn

        knn_indices = np.argsort(distance_matrix_no_diag, axis=1)[:, :k]
        adjacency = np.zeros((num_samples, num_samples), dtype=bool)
        rows = np.arange(num_samples)[:, None]
        adjacency[rows, knn_indices] = True
        adjacency = np.logical_or(adjacency, adjacency.T)
        np.fill_diagonal(adjacency, False)
        degrees = adjacency.sum(axis=1).astype(np.float64)
        graph_summary[f"k={k}"] = {
            "mean_knn_distance": float(np.mean(flattened_knn)) if flattened_knn is not None else float("nan"),
            "std_knn_distance": float(np.std(flattened_knn)) if flattened_knn is not None else float("nan"),
            "average_node_degree": float(np.mean(degrees)),
        }

    return {
        "features": features,
        "centroid": centroid.cpu().numpy(),
        "covariance": covariance.cpu().numpy(),
        "pca_basis": None if pca_basis is None else pca_basis.cpu().numpy(),
        "pairwise_vector": pairwise_vector,
        "knn_values": knn_values,
        "graph_summary": graph_summary,
    }


def compute_graph_stats_metrics(source, target, k_values):
    source_summary = summarize_graph_stats(source, k_values)
    target_summary = summarize_graph_stats(target, k_values)
    return {
        "neighbor_graph_stats": {
            f"k={k}": {
                "source": source_summary[f"k={k}"],
                "target": target_summary[f"k={k}"],
                "abs_diff": {
                    stat_name: float(abs(source_summary[f"k={k}"][stat_name] - target_summary[f"k={k}"][stat_name]))
                    if not (math.isnan(source_summary[f"k={k}"][stat_name]) or math.isnan(target_summary[f"k={k}"][stat_name]))
                    else float("nan")
                    for stat_name in ["mean_knn_distance", "std_knn_distance", "average_node_degree"]
                },
            }
            for k in k_values
        }
    }


def compute_graph_stats_metrics_from_cache(source_cache, target_cache, k_values):
    return {
        "neighbor_graph_stats": {
            f"k={k}": {
                "source": source_cache["graph_summary"][f"k={k}"],
                "target": target_cache["graph_summary"][f"k={k}"],
                "abs_diff": {
                    stat_name: float(
                        abs(source_cache["graph_summary"][f"k={k}"][stat_name] - target_cache["graph_summary"][f"k={k}"][stat_name])
                    )
                    if not (
                        math.isnan(source_cache["graph_summary"][f"k={k}"][stat_name]) or
                        math.isnan(target_cache["graph_summary"][f"k={k}"][stat_name])
                    )
                    else float("nan")
                    for stat_name in ["mean_knn_distance", "std_knn_distance", "average_node_degree"]
                },
            }
            for k in k_values
        }
    }


def aggregate_nested_metric_dict(metric_dicts, outer_keys):
    aggregated = {}
    for outer_key in outer_keys:
        nested_keys = sorted({nested_key for metric_dict in metric_dicts for nested_key in metric_dict[outer_key].keys()})
        aggregated[outer_key] = {}
        for nested_key in nested_keys:
            values = [
                metric_dict[outer_key].get(nested_key, float("nan"))
                for metric_dict in metric_dicts
                if not math.isnan(metric_dict[outer_key].get(nested_key, float("nan")))
            ]
            aggregated[outer_key][nested_key] = float(np.mean(values)) if values else float("nan")
    return aggregated


def aggregate_graph_stat_dicts(graph_metric_dicts):
    aggregated = {"neighbor_graph_stats": {}}
    k_keys = sorted({k for metric_dict in graph_metric_dicts for k in metric_dict["neighbor_graph_stats"].keys()})
    for k_key in k_keys:
        aggregated["neighbor_graph_stats"][k_key] = {}
        for stat_name in ["mean_knn_distance", "std_knn_distance", "average_node_degree"]:
            source_values = [
                metric_dict["neighbor_graph_stats"][k_key]["source"][stat_name]
                for metric_dict in graph_metric_dicts
                if k_key in metric_dict["neighbor_graph_stats"]
                and not math.isnan(metric_dict["neighbor_graph_stats"][k_key]["source"][stat_name])
            ]
            target_values = [
                metric_dict["neighbor_graph_stats"][k_key]["target"][stat_name]
                for metric_dict in graph_metric_dicts
                if k_key in metric_dict["neighbor_graph_stats"]
                and not math.isnan(metric_dict["neighbor_graph_stats"][k_key]["target"][stat_name])
            ]
            diff_values = [
                metric_dict["neighbor_graph_stats"][k_key]["abs_diff"][stat_name]
                for metric_dict in graph_metric_dicts
                if k_key in metric_dict["neighbor_graph_stats"]
                and not math.isnan(metric_dict["neighbor_graph_stats"][k_key]["abs_diff"][stat_name])
            ]
            aggregated["neighbor_graph_stats"][k_key][stat_name] = {
                "source_mean": float(np.mean(source_values)) if source_values else float("nan"),
                "target_mean": float(np.mean(target_values)) if target_values else float("nan"),
                "abs_diff_mean": float(np.mean(diff_values)) if diff_values else float("nan"),
            }
    return aggregated


def aggregate_graph_stat_averages(graph_average_dicts):
    aggregated = {"neighbor_graph_stats": {}}
    k_keys = sorted({k for graph_dict in graph_average_dicts for k in graph_dict["neighbor_graph_stats"].keys()})
    for k_key in k_keys:
        aggregated["neighbor_graph_stats"][k_key] = {}
        for stat_name in ["mean_knn_distance", "std_knn_distance", "average_node_degree"]:
            source_values = [
                graph_dict["neighbor_graph_stats"][k_key][stat_name]["source_mean"]
                for graph_dict in graph_average_dicts
                if k_key in graph_dict["neighbor_graph_stats"]
                and not math.isnan(graph_dict["neighbor_graph_stats"][k_key][stat_name]["source_mean"])
            ]
            target_values = [
                graph_dict["neighbor_graph_stats"][k_key][stat_name]["target_mean"]
                for graph_dict in graph_average_dicts
                if k_key in graph_dict["neighbor_graph_stats"]
                and not math.isnan(graph_dict["neighbor_graph_stats"][k_key][stat_name]["target_mean"])
            ]
            diff_values = [
                graph_dict["neighbor_graph_stats"][k_key][stat_name]["abs_diff_mean"]
                for graph_dict in graph_average_dicts
                if k_key in graph_dict["neighbor_graph_stats"]
                and not math.isnan(graph_dict["neighbor_graph_stats"][k_key][stat_name]["abs_diff_mean"])
            ]
            aggregated["neighbor_graph_stats"][k_key][stat_name] = {
                "source_mean": float(np.mean(source_values)) if source_values else float("nan"),
                "target_mean": float(np.mean(target_values)) if target_values else float("nan"),
                "abs_diff_mean": float(np.mean(diff_values)) if diff_values else float("nan"),
            }
    return aggregated


def compute_cosine_similarity_matrix(centered_prototypes):
    # `centered_prototypes`의 shape은 [클래스 수, feature 차원]입니다.
    # 각 행은 "클래스 prototype - 전역 prototype" 입니다.
    # 여기서는 모든 클래스 쌍에 대해 cosine similarity를 한 번에 계산합니다.
    norms = np.linalg.norm(centered_prototypes, axis=1)
    # `norms[:, None]`는 1차원 벡터를 열벡터처럼 바꿉니다.
    # 열벡터 x 행벡터를 하면 cosine 분모에 필요한 모든 조합이 만들어집니다.
    denom = norms[:, None] * norms[None, :]
    similarity = np.full((centered_prototypes.shape[0], centered_prototypes.shape[0]), np.nan, dtype=np.float64)
    valid = denom > 0
    if np.any(valid):
        # NumPy에서 `@`는 행렬곱입니다.
        # 이 한 줄로 모든 클래스 쌍의 내적(dot product)을 동시에 계산합니다.
        similarity[valid] = (centered_prototypes @ centered_prototypes.T)[valid] / denom[valid]
    # 대각선은 자기 자신과의 비교이므로 정의상 항상 1입니다.
    np.fill_diagonal(similarity, 1.0)
    return np.clip(similarity, -1.0, 1.0), norms


def build_inter_class_structure(class_samples, selected_classes):
    class_ids = list(selected_classes)
    # 각 클래스에서 뽑힌 이미지 feature들을 평균내서
    # "클래스 대표 벡터(prototype)"를 만듭니다.
    prototypes = np.stack([class_samples[class_index].mean(axis=0) for class_index in class_ids], axis=0)
    # 선택된 모든 클래스/샘플을 합쳐 전체 평균을 구합니다.
    # 이 벡터가 전체 클래스 구조의 중심점 역할을 합니다.
    pooled_samples = np.concatenate([class_samples[class_index] for class_index in class_ids], axis=0)
    global_prototype = pooled_samples.mean(axis=0)
    # 각 클래스 prototype에서 전역 중심을 빼서 centered prototype을 만듭니다.
    # 이렇게 하면 "절대 위치"보다 "전체 중심 대비 상대 관계"를 보기 쉬워집니다.
    centered_prototypes = prototypes - global_prototype[None, :]
    similarity_matrix, radii = compute_cosine_similarity_matrix(centered_prototypes)
    return {
        "class_ids": class_ids,
        "prototypes": prototypes,
        "global_prototype": global_prototype,
        "centered_prototypes": centered_prototypes,
        "similarity_matrix": similarity_matrix,
        "radii": radii,
    }


def safe_stat_correlation(values_a, values_b, method, context):
    # Pearson: 실제 값의 선형적 유사성
    # Spearman: 값의 크기 순서(rank)가 얼마나 비슷한지
    # 빈 벡터이거나 분산이 0이면 상관계수가 불안정하므로 NaN을 반환합니다.
    if values_a.size == 0 or values_b.size == 0:
        warnings.warn(f"{context}: empty vectors. Returning NaN.", RuntimeWarning)
        return float("nan")
    if np.var(values_a) == 0 or np.var(values_b) == 0:
        warnings.warn(f"{context}: zero variance detected. Returning NaN.", RuntimeWarning)
        return float("nan")
    if method == "pearson":
        return float(pearsonr(values_a, values_b)[0])
    if method == "spearman":
        return float(spearmanr(values_a, values_b).statistic)
    raise ValueError(f"Unsupported correlation method: {method}")


def sign_label(value):
    # 부호를 JSON에 저장하기 쉽도록 -1 / 0 / +1로 바꿉니다.
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def build_inter_class_artifact_prefix(output_prefix, artifact_suffix):
    return f"{output_prefix}{artifact_suffix}_inter_class"


def maybe_save_inter_class_plots(source_matrix, target_matrix, upper_source, upper_target, abs_diff_vector, artifact_prefix, title_prefix):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Inter-class plots require 'matplotlib'. Please install requirements.") from exc

    diff_matrix = np.abs(source_matrix - target_matrix)
    files = {}

    for suffix, matrix, title in [
        ("source_heatmap", source_matrix, f"{title_prefix} source centered prototype similarity"),
        ("target_heatmap", target_matrix, f"{title_prefix} target centered prototype similarity"),
        ("abs_diff_heatmap", diff_matrix, f"{title_prefix} absolute difference"),
    ]:
        figure, axis = plt.subplots(figsize=(7, 6))
        image = axis.imshow(matrix, cmap="coolwarm" if suffix != "abs_diff_heatmap" else "magma", vmin=-1.0 if suffix != "abs_diff_heatmap" else None, vmax=1.0 if suffix != "abs_diff_heatmap" else None)
        axis.set_title(title)
        axis.set_xlabel("Class index")
        axis.set_ylabel("Class index")
        figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
        path = f"{artifact_prefix}_{suffix}.png"
        figure.tight_layout()
        figure.savefig(path, dpi=200)
        plt.close(figure)
        files[suffix] = path

    # scatter plot은 source의 클래스쌍 관계와 target의 클래스쌍 관계가
    # 얼마나 비슷한지 한눈에 보는 용도입니다.
    # 점들이 대각선 주변에 모일수록 관계 구조가 잘 보존된 것입니다.
    pearson_value = safe_stat_correlation(upper_source, upper_target, "pearson", "inter-class scatter pearson")
    spearman_value = safe_stat_correlation(upper_source, upper_target, "spearman", "inter-class scatter spearman")
    figure, axis = plt.subplots(figsize=(7, 6))
    axis.scatter(upper_source, upper_target, alpha=0.6, s=18)
    axis.set_xlabel("Source upper-triangular similarities")
    axis.set_ylabel("Target upper-triangular similarities")
    axis.set_title(f"{title_prefix} scatter\npearson={pearson_value:.4f}, spearman={spearman_value:.4f}")
    scatter_path = f"{artifact_prefix}_scatter.png"
    figure.tight_layout()
    figure.savefig(scatter_path, dpi=200)
    plt.close(figure)
    files["scatter"] = scatter_path

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(abs_diff_vector, bins=min(40, max(10, abs_diff_vector.shape[0] // 5)))
    axis.set_xlabel("|source - target|")
    axis.set_ylabel("Count")
    axis.set_title(f"{title_prefix} absolute-difference histogram")
    histogram_path = f"{artifact_prefix}_abs_diff_hist.png"
    figure.tight_layout()
    figure.savefig(histogram_path, dpi=200)
    plt.close(figure)
    files["abs_diff_histogram"] = histogram_path

    return files


def compute_inter_class_geometry_results(
    selected_classes,
    class_names,
    source_class_samples,
    target_class_samples,
    output_prefix,
    artifact_suffix,
    source_dataset_name,
    target_dataset_name,
    sign_epsilon,
    save_plots,
):
    if len(selected_classes) < 2:
        raise ValueError("Inter-class relational geometry requires at least 2 classes.")

    # 이 함수가 inter-class 분석의 핵심입니다.
    # 흐름은 아래와 같습니다.
    # 1) source/target 각각에서 class prototype을 만듭니다.
    # 2) 전체 평균(global prototype)을 빼서 centered prototype을 만듭니다.
    # 3) centered prototype 사이 cosine similarity matrix S를 만듭니다.
    # 4) source S와 target S를 비교해서 관계 구조가 얼마나 유지됐는지 수치화합니다.

    # source / target 각각에 대해
    # "클래스 prototype -> 전역 중심 제거 -> 클래스 간 유사도 행렬"
    # 구조를 만듭니다.
    source_structure = build_inter_class_structure(source_class_samples, selected_classes)
    target_structure = build_inter_class_structure(target_class_samples, selected_classes)

    # 두 데이터셋에서 만든 클래스-클래스 유사도 행렬 S를 꺼냅니다.
    # shape은 [클래스 수, 클래스 수] 입니다.
    source_matrix = source_structure["similarity_matrix"]
    target_matrix = target_structure["similarity_matrix"]
    # 대각선은 자기 자신과의 비교라 항상 1이고,
    # 아래 삼각형은 위 삼각형과 중복이므로 upper triangle만 사용합니다.
    upper_indices = np.triu_indices(len(selected_classes), k=1)
    upper_source = source_matrix[upper_indices]
    upper_target = target_matrix[upper_indices]
    # `np.triu_indices(..., k=1)`는 대각선을 제외한 위쪽 삼각형 인덱스만 뽑습니다.
    # 이렇게 하면 (a,b)와 (b,a) 중복 없이 클래스쌍을 한 번만 비교할 수 있습니다.

    # NaN / Inf 같은 비정상 값은 비교에서 제외합니다.
    # 예를 들어 centered prototype의 길이가 0이면 cosine이 정의되지 않을 수 있습니다.
    if not np.all(np.isfinite(upper_source)) or not np.all(np.isfinite(upper_target)):
        warnings.warn("Inter-class relational geometry contains non-finite similarity values.", RuntimeWarning)
    valid_mask = np.isfinite(upper_source) & np.isfinite(upper_target)
    valid_source = upper_source[valid_mask]
    valid_target = upper_target[valid_mask]
    # 각 클래스쌍 관계가 source와 target에서 얼마나 달라졌는지 절댓값 차이로 봅니다.
    abs_diff_vector = np.abs(valid_source - valid_target)
    # 여기서 `valid_source`, `valid_target`의 shape은 [유효한 클래스쌍 수]인 1차원 벡터입니다.
    # 즉 행렬 비교를 위해 결국 상삼각 행렬을 "펼친 벡터" 형태로 다루고 있습니다.

    # pair_records에는 클래스쌍별 상세 정보를 저장합니다.
    # 나중에 어떤 클래스쌍에서 관계 변화가 컸는지 추적할 때 사용합니다.
    pair_records = []
    num_pairs_near_zero = 0
    num_sign_comparisons = 0
    num_sign_matches = 0
    top_abs_diff_pairs = []
    for pair_offset, (a_pos, b_pos) in enumerate(zip(*upper_indices)):
        # upper triangle에서 꺼낸 1차원 값 하나가
        # 실제로는 (class a, class b) 쌍 하나에 해당합니다.
        source_value = float(upper_source[pair_offset])
        target_value = float(upper_target[pair_offset])
        # 두 값이 모두 0 근처면 부호 비교가 매우 불안정하므로
        # normal sign 비교에서 제외하고 near_zero로만 표시합니다.
        near_zero = abs(source_value) < sign_epsilon and abs(target_value) < sign_epsilon
        source_sign = sign_label(source_value)
        target_sign = sign_label(target_value)
        sign_match = source_sign == target_sign
        if near_zero:
            num_pairs_near_zero += 1
        else:
            num_sign_comparisons += 1
            num_sign_matches += int(sign_match)
        pair_records.append({
            "class_a_id": selected_classes[a_pos],
            "class_b_id": selected_classes[b_pos],
            "class_a_name": class_names[selected_classes[a_pos]],
            "class_b_name": class_names[selected_classes[b_pos]],
            "source_similarity": source_value,
            "target_similarity": target_value,
            "delta_similarity": float(target_value - source_value),
            "source_sign": source_sign,
            "target_sign": target_sign,
            "sign_match": bool(sign_match),
            "near_zero": bool(near_zero),
            "abs_diff": float(abs(source_value - target_value)),
        })

    # 관계 차이가 큰 클래스쌍 상위 10개를 따로 저장합니다.
    # 해석 단계에서 "어떤 클래스쌍이 가장 많이 변했는가?"를 보기 쉽습니다.
    top_abs_diff_pairs = sorted(pair_records, key=lambda item: item["abs_diff"], reverse=True)[:10]
    top_source_similar_pairs = sorted(pair_records, key=lambda item: item["source_similarity"], reverse=True)[:10]
    top_target_similar_pairs = sorted(pair_records, key=lambda item: item["target_similarity"], reverse=True)[:10]
    largest_similarity_increase_pairs = sorted(pair_records, key=lambda item: item["delta_similarity"], reverse=True)[:10]
    largest_similarity_decrease_pairs = sorted(pair_records, key=lambda item: item["delta_similarity"])[:10]

    # radius는 각 클래스 centered prototype의 길이입니다.
    # 즉, "전체 중심에서 이 클래스가 얼마나 멀리 떨어져 있는가?"를 뜻합니다.
    radius_records = []
    for class_index, source_radius, target_radius in zip(selected_classes, source_structure["radii"], target_structure["radii"]):
        radius_records.append({
            "class_id": class_index,
            "class_name": class_names[class_index],
            "source_radius": float(source_radius),
            "target_radius": float(target_radius),
            "abs_diff": float(abs(source_radius - target_radius)),
        })
    # `zip(selected_classes, source_radii, target_radii)`는 같은 위치의 값들을 묶어 줍니다.
    # 그래서 class id 하나와 source/target radius 하나를 같이 처리하기 좋습니다.

    # 상관계수 계산을 위해 radius를 다시 숫자 벡터로 모읍니다.
    source_radius_values = np.asarray([record["source_radius"] for record in radius_records], dtype=np.float64)
    target_radius_values = np.asarray([record["target_radius"] for record in radius_records], dtype=np.float64)

    artifact_prefix = build_inter_class_artifact_prefix(output_prefix, artifact_suffix)
    os.makedirs(os.path.dirname(output_prefix) or ".", exist_ok=True)

    # 행렬이 커질 수 있으므로 main JSON에 직접 다 넣지 않고
    # 별도 .npy / .json 파일로 저장합니다.
    source_matrix_path = f"{artifact_prefix}_source.npy"
    target_matrix_path = f"{artifact_prefix}_target.npy"
    # `.npy`는 NumPy 배열을 그대로 저장하는 포맷입니다.
    # JSON보다 빠르고, 2차원 행렬을 손실 없이 보관하기 좋습니다.
    np.save(source_matrix_path, source_matrix)
    np.save(target_matrix_path, target_matrix)

    # 클래스쌍별 sign 정보와 클래스별 radius 정보는 사람이 읽기 쉽도록 JSON으로 저장합니다.
    per_pair_path = f"{artifact_prefix}_per_pair_sign_info.json"
    with open(per_pair_path, "w", encoding="utf-8") as handle:
        json.dump(pair_records, handle, indent=2)

    per_radius_path = f"{artifact_prefix}_per_class_radius.json"
    with open(per_radius_path, "w", encoding="utf-8") as handle:
        json.dump(radius_records, handle, indent=2)

    plot_files = None
    if save_plots:
        # heatmap / scatter / histogram은 정량 지표를 보조하는 시각적 해석 도구입니다.
        plot_files = maybe_save_inter_class_plots(
            source_matrix=source_matrix,
            target_matrix=target_matrix,
            upper_source=valid_source,
            upper_target=valid_target,
            abs_diff_vector=abs_diff_vector,
            artifact_prefix=artifact_prefix,
            title_prefix=f"{source_dataset_name} vs {target_dataset_name}",
        )

    return {
        "class_ids": selected_classes,
        "class_names": [class_names[class_index] for class_index in selected_classes],
        "matrix_shape": list(source_matrix.shape),
        "matrix_file_source": source_matrix_path,
        "matrix_file_target": target_matrix_path,
        # upper triangle만 펼친 벡터끼리 비교한 상관계수입니다.
        # 높을수록 "클래스-클래스 관계 지도"가 비슷하다는 뜻입니다.
        "pearson_corr": safe_stat_correlation(valid_source, valid_target, "pearson", "inter-class matrix pearson"),
        "spearman_corr": safe_stat_correlation(valid_source, valid_target, "spearman", "inter-class matrix spearman"),
        # 클래스쌍 관계값이 평균적으로 얼마나 달라졌는지 봅니다.
        "mean_abs_diff": float(np.mean(abs_diff_vector)) if abs_diff_vector.size > 0 else float("nan"),
        # near_zero가 아닌 클래스쌍들 중에서 부호가 얼마나 일치하는지 봅니다.
        "sign_consistency_rate": float(num_sign_matches / num_sign_comparisons) if num_sign_comparisons > 0 else float("nan"),
        "sign_epsilon": float(sign_epsilon),
        "num_pairs_total": int(upper_source.shape[0]),
        "num_pairs_near_zero": int(num_pairs_near_zero),
        # radius는 "각 클래스 prototype이 전체 중심에서 얼마나 멀리 있는가"입니다.
        # 이 값이 비슷하면 전체 클래스 배치의 바깥/안쪽 구조가 유지된 것으로 볼 수 있습니다.
        "radius_corr": safe_stat_correlation(source_radius_values, target_radius_values, "pearson", "inter-class radius correlation"),
        "radius_mean_abs_diff": float(np.mean(np.abs(source_radius_values - target_radius_values))),
        # centered prototype norm 자체도 저장해 두면 나중에 클래스별 해석이 쉬워집니다.
        "source_centered_prototype_norms": [float(value) for value in source_structure["radii"]],
        "target_centered_prototype_norms": [float(value) for value in target_structure["radii"]],
        "per_pair_sign_info_file": per_pair_path,
        "per_class_radius_file": per_radius_path,
        "plot_files": plot_files,
        "top_abs_diff_pairs": top_abs_diff_pairs,
        "top_source_similar_pairs": top_source_similar_pairs,
        "top_target_similar_pairs": top_target_similar_pairs,
        "largest_similarity_increase_pairs": largest_similarity_increase_pairs,
        "largest_similarity_decrease_pairs": largest_similarity_decrease_pairs,
    }


def summarize_inter_class_geometry(inter_class_runs):
    # split-half upper bound에서는 랜덤하게 나눈 결과가 여러 번 생기므로
    # 반복 실험들의 평균과 표준편차를 함께 요약합니다.
    keys = ["pearson_corr", "spearman_corr", "mean_abs_diff", "sign_consistency_rate", "radius_corr", "radius_mean_abs_diff"]
    summary = {}
    for key in keys:
        values = [run[key] for run in inter_class_runs if key in run and not math.isnan(run[key])]
        summary[key] = {
            "mean": float(np.mean(values)) if values else float("nan"),
            "std": float(np.std(values)) if values else float("nan"),
        }
    near_zero_values = [run["num_pairs_near_zero"] for run in inter_class_runs]
    total_values = [run["num_pairs_total"] for run in inter_class_runs]
    summary["num_pairs_near_zero"] = {
        "mean": float(np.mean(near_zero_values)) if near_zero_values else float("nan"),
        "std": float(np.std(near_zero_values)) if near_zero_values else float("nan"),
    }
    summary["num_pairs_total"] = int(total_values[0]) if total_values else 0
    summary["sign_epsilon"] = inter_class_runs[0]["sign_epsilon"] if inter_class_runs else float("nan")
    return summary


def compute_continuity_score(original_features, embedded_features, n_neighbors=5):
    num_samples = original_features.shape[0]
    if num_samples <= n_neighbors + 1:
        warnings.warn(
            f"Continuity requires more than {n_neighbors + 1} samples, got {num_samples}. Returning NaN.",
            RuntimeWarning,
        )
        return float("nan")

    original_dist = compute_distance_matrix(original_features)
    embedded_dist = compute_distance_matrix(embedded_features)
    original_order = np.argsort(original_dist, axis=1)
    embedded_order = np.argsort(embedded_dist, axis=1)
    original_neighbor_sets = [set(order[1:n_neighbors + 1]) for order in original_order]

    penalty = 0.0
    for i in range(num_samples):
        embedded_neighbors = embedded_order[i][1:n_neighbors + 1]
        rank_positions = np.empty(num_samples, dtype=np.int64)
        rank_positions[original_order[i]] = np.arange(num_samples)
        for neighbor in embedded_neighbors:
            if neighbor not in original_neighbor_sets[i]:
                penalty += rank_positions[neighbor] - n_neighbors

    normalizer = num_samples * n_neighbors * (2 * num_samples - 3 * n_neighbors - 1)
    if normalizer <= 0:
        return float("nan")
    return float(1.0 - (2.0 / normalizer) * penalty)


def build_visualization_arrays(selected_classes, source_class_samples, target_class_samples, class_names, args):
    viz_class_ids = select_visualization_class_ids(selected_classes, class_names, args)
    if not viz_class_ids:
        warnings.warn("No classes selected for visualization. Skipping UMAP export.", RuntimeWarning)
        return None

    features = []
    labels = []
    dataset_tags = []
    centroids = []

    for class_index in viz_class_ids:
        source_samples = source_class_samples[class_index][:args.viz_max_points_per_class]
        target_samples = target_class_samples[class_index][:args.viz_max_points_per_class]

        if source_samples.shape[0] == 0 or target_samples.shape[0] == 0:
            warnings.warn(f"Class {class_index} has no samples for visualization. Skipping.", RuntimeWarning)
            continue

        features.append(source_samples)
        features.append(target_samples)
        labels.extend([class_names[class_index]] * source_samples.shape[0])
        labels.extend([class_names[class_index]] * target_samples.shape[0])
        dataset_tags.extend(["source"] * source_samples.shape[0])
        dataset_tags.extend(["target"] * target_samples.shape[0])
        centroids.append({
            "class_index": class_index,
            "class_name": class_names[class_index],
            "source_centroid": source_samples.mean(axis=0),
            "target_centroid": target_samples.mean(axis=0),
        })

    if not features:
        return None

    return {
        "features": np.concatenate(features, axis=0),
        "labels": labels,
        "dataset_tags": dataset_tags,
        "centroids": centroids,
        "viz_class_ids": viz_class_ids,
    }


def save_umap_outputs(viz_bundle, output_prefix, source_dataset_name, target_dataset_name, args):
    try:
        import matplotlib.pyplot as plt
        import umap
    except ImportError as exc:
        raise ImportError(
            "UMAP visualization requires 'matplotlib' and 'umap-learn'. Please install requirements."
        ) from exc

    reducer = umap.UMAP(
        n_components=args.viz_dim,
        random_state=args.seed,
        transform_seed=args.seed,
    )
    embedding = reducer.fit_transform(viz_bundle["features"])
    centroid_stack = np.stack(
        [item["source_centroid"] for item in viz_bundle["centroids"]] +
        [item["target_centroid"] for item in viz_bundle["centroids"]],
        axis=0,
    )
    centroid_embedding = reducer.transform(centroid_stack)
    split_point = len(viz_bundle["centroids"])

    dataset_to_marker = {"source": "o", "target": "^"}
    class_to_color_id = {class_name: index for index, class_name in enumerate(sorted(set(viz_bundle["labels"])))}
    color_values = np.array([class_to_color_id[label] for label in viz_bundle["labels"]], dtype=np.int64)

    figure = plt.figure(figsize=(10, 8))
    if args.viz_dim == 3:
        axis = figure.add_subplot(111, projection="3d")
    else:
        axis = figure.add_subplot(111)

    for dataset_tag in ["source", "target"]:
        indices = [i for i, tag in enumerate(viz_bundle["dataset_tags"]) if tag == dataset_tag]
        if not indices:
            continue
        points = embedding[indices]
        scatter_kwargs = {
            "c": color_values[indices],
            "marker": dataset_to_marker[dataset_tag],
            "alpha": 0.7,
            "cmap": "tab20",
            "label": dataset_tag,
        }
        if args.viz_dim == 3:
            axis.scatter(points[:, 0], points[:, 1], points[:, 2], **scatter_kwargs)
        else:
            axis.scatter(points[:, 0], points[:, 1], **scatter_kwargs)

    for centroid_index, centroid_info in enumerate(viz_bundle["centroids"]):
        source_point = centroid_embedding[centroid_index]
        target_point = centroid_embedding[split_point + centroid_index]
        delta = target_point - source_point
        if args.viz_dim == 3:
            axis.quiver(source_point[0], source_point[1], source_point[2], delta[0], delta[1], delta[2], color="black", alpha=0.5)
        else:
            axis.annotate("", xy=target_point[:2], xytext=source_point[:2], arrowprops={"arrowstyle": "->", "alpha": 0.5, "color": "black"})

    axis.set_title(f"UMAP: {source_dataset_name} vs {target_dataset_name}")
    axis.legend()
    png_path = f"{output_prefix}_umap_{args.viz_dim}d.png"
    figure.tight_layout()
    figure.savefig(png_path, dpi=200)
    plt.close(figure)

    html_path = None
    if args.interactive:
        try:
            import plotly.express as px
            import plotly.graph_objects as go
        except ImportError as exc:
            raise ImportError("Interactive visualization requires 'plotly'. Please install requirements.") from exc

        data_frame = {
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "class": viz_bundle["labels"],
            "dataset": viz_bundle["dataset_tags"],
        }
        if args.viz_dim == 3:
            data_frame["z"] = embedding[:, 2]
            figure_html = px.scatter_3d(data_frame, x="x", y="y", z="z", color="class", symbol="dataset", title=f"UMAP: {source_dataset_name} vs {target_dataset_name}")
            for centroid_index, centroid_info in enumerate(viz_bundle["centroids"]):
                source_point = centroid_embedding[centroid_index]
                target_point = centroid_embedding[split_point + centroid_index]
                figure_html.add_trace(go.Scatter3d(x=[source_point[0], target_point[0]], y=[source_point[1], target_point[1]], z=[source_point[2], target_point[2]], mode="lines", line={"color": "black"}, showlegend=False))
        else:
            figure_html = px.scatter(data_frame, x="x", y="y", color="class", symbol="dataset", title=f"UMAP: {source_dataset_name} vs {target_dataset_name}")
            for centroid_index, centroid_info in enumerate(viz_bundle["centroids"]):
                source_point = centroid_embedding[centroid_index]
                target_point = centroid_embedding[split_point + centroid_index]
                figure_html.add_scatter(x=[source_point[0], target_point[0]], y=[source_point[1], target_point[1]], mode="lines", line={"color": "black"}, showlegend=False)

        html_path = f"{output_prefix}_umap_{args.viz_dim}d.html"
        figure_html.write_html(html_path)

    trust_k = min(10, max(2, viz_bundle["features"].shape[0] - 1))
    viz_metrics = {
        "source_dataset": source_dataset_name,
        "target_dataset": target_dataset_name,
        "viz_dim": args.viz_dim,
        "num_points": int(viz_bundle["features"].shape[0]),
        "num_classes": len(viz_bundle["viz_class_ids"]),
        "trustworthiness": float(trustworthiness(viz_bundle["features"], embedding, n_neighbors=trust_k)),
        "continuity": compute_continuity_score(viz_bundle["features"], embedding, n_neighbors=trust_k),
        "png_path": png_path,
        "html_path": html_path,
    }
    viz_metrics_path = f"{output_prefix}_viz_metrics.json"
    with open(viz_metrics_path, "w", encoding="utf-8") as handle:
        json.dump(viz_metrics, handle, indent=2)

    return {
        "png_path": png_path,
        "html_path": html_path,
        "viz_metrics_path": viz_metrics_path,
        "metrics": viz_metrics,
    }


def compute_pairwise_correlation(source, target, method):
    # Deprecated for cross-dataset comparison: the i-th pair in one dataset
    # is not the same semantic image pair as the i-th pair in the other.
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


def compute_pairwise_correlation_from_cache(source_cache, target_cache, method):
    source_distances = source_cache["pairwise_vector"]
    target_distances = target_cache["pairwise_vector"]

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
    raise ValueError(f"Unsupported metric: {method}")


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


def compute_metrics_from_cache(source_cache, target_cache, metrics):
    results = {}
    source_centroid = source_cache["centroid"]
    target_centroid = target_cache["centroid"]
    centroid_delta = source_centroid - target_centroid
    centroid_denom = np.linalg.norm(source_centroid) * np.linalg.norm(target_centroid)

    for metric in metrics:
        if metric == "centroid_cosine":
            results[metric] = float(np.dot(source_centroid, target_centroid) / centroid_denom) if centroid_denom != 0 else float("nan")
        elif metric == "centroid_euclidean":
            results[metric] = float(np.linalg.norm(centroid_delta))
        elif metric == "cov_frobenius":
            results[metric] = float(np.linalg.norm(source_cache["covariance"] - target_cache["covariance"], ord="fro"))
        elif metric == "pca_subspace":
            results[metric] = compute_pca_subspace_similarity_from_basis(source_cache["pca_basis"], target_cache["pca_basis"])
        elif metric in {"pairwise_spearman", "pairwise_pearson"}:
            results[metric] = compute_pairwise_correlation_from_cache(source_cache, target_cache, metric)
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


def compute_retrieval_score_from_cache(source_cache, target_cache, mode):
    if mode == "centroid_cosine":
        source_centroid = source_cache["centroid"]
        target_centroid = target_cache["centroid"]
        denom = np.linalg.norm(source_centroid) * np.linalg.norm(target_centroid)
        return float(np.dot(source_centroid, target_centroid) / denom) if denom != 0 else float("nan")
    if mode == "centroid_euclidean":
        return -float(np.linalg.norm(source_cache["centroid"] - target_cache["centroid"]))
    if mode == "pca_subspace":
        return compute_pca_subspace_similarity_from_basis(source_cache["pca_basis"], target_cache["pca_basis"])
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


def compute_retrieval_results(selected_classes, source_class_caches, target_class_caches, class_names):
    retrieval = {}
    for mode in get_retrieval_modes():
        per_class = []
        num_correct = 0

        for class_index in selected_classes:
            # Retrieval asks: "which target class geometry looks most like
            # this source class geometry?" If the answer is the same class id,
            # class identity is preserved well enough for nearest-class matching.
            scored_targets = []
            for target_class_index in selected_classes:
                score = compute_retrieval_score_from_cache(
                    source_class_caches[class_index],
                    target_class_caches[target_class_index],
                    mode,
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


def std_dict(metric_dicts, metrics):
    summary = {}
    for metric in metrics:
        values = [entry[metric] for entry in metric_dicts if not math.isnan(entry[metric])]
        summary[metric] = float(np.std(values)) if values else float("nan")
    return summary


def compute_retrieval_summary_stats(retrieval_runs):
    summary = {}
    for mode in get_retrieval_modes():
        accuracies = [run[mode]["accuracy"] for run in retrieval_runs if mode in run and not math.isnan(run[mode]["accuracy"])]
        summary[mode] = {
            "mean_accuracy": float(np.mean(accuracies)) if accuracies else float("nan"),
            "std_accuracy": float(np.std(accuracies)) if accuracies else float("nan"),
        }
    return summary


def build_result(
    selected_classes,
    source_payload,
    target_payload,
    metrics,
    pca_dim,
    source_class_samples,
    target_class_samples,
    class_names,
    samples_per_class,
    mode,
    args,
    output_prefix=None,
    save_visualization=False,
    artifact_suffix="",
):
    device = resolve_device(args.device)
    per_class_results = []
    same_class_metrics = []
    different_class_metrics = []
    knn_metric_results = []
    histogram_metric_results = []
    graph_metric_results = []
    cache_start = time.perf_counter()
    source_class_caches = {
        class_index: prepare_class_geometry_cache(source_class_samples[class_index], args.knn_k, pca_dim, device)
        for class_index in selected_classes
    }
    target_class_caches = {
        class_index: prepare_class_geometry_cache(target_class_samples[class_index], args.knn_k, pca_dim, device)
        for class_index in selected_classes
    }
    log_timing("prepare_class_geometry_cache", cache_start)

    if "pairwise_spearman" in metrics or "pairwise_pearson" in metrics:
        warnings.warn(
            "pairwise distance correlation is not reliable for cross-dataset comparison without sample correspondence",
            RuntimeWarning,
        )

    metric_start = time.perf_counter()
    for class_index in selected_classes:
        source_class_name = class_names[class_index]
        source_cache = source_class_caches[class_index]
        target_cache = target_class_caches[class_index]
        # Main question: does class y in source still look like class y in target?
        same_metrics = compute_metrics_from_cache(source_cache, target_cache, metrics)
        same_class_metrics.append(same_metrics)

        control_metrics_list = []
        for target_class_index in selected_classes:
            if target_class_index == class_index:
                continue
            target_control_cache = target_class_caches[target_class_index]
            # Control question: are we getting a genuinely class-specific match,
            # or would many unrelated target classes score similarly?
            control_metrics_list.append(compute_metrics_from_cache(source_cache, target_control_cache, metrics))

        averaged_control_metrics = mean_dict(control_metrics_list, metrics)
        different_class_metrics.append(averaged_control_metrics)

        per_class_results.append({
            "class_index": class_index,
            "class_name": source_class_name,
            "same_class": same_metrics,
            "different_class_control": averaged_control_metrics,
        })

        if args.compute_knn_distribution:
            knn_metrics = compute_knn_distribution_metrics_from_cache(source_cache, target_cache, args.knn_k)
            per_class_results[-1].update(knn_metrics)
            knn_metric_results.append(knn_metrics)

        if args.compute_distance_histogram:
            histogram_metrics = compute_distance_histogram_metrics_from_cache(source_cache, target_cache, args.distance_hist_bins)
            per_class_results[-1].update(histogram_metrics)
            histogram_metric_results.append(histogram_metrics)

        if args.compute_graph_stats:
            graph_metrics = compute_graph_stats_metrics_from_cache(source_cache, target_cache, args.knn_k)
            per_class_results[-1].update(graph_metrics)
            graph_metric_results.append(graph_metrics)
    log_timing("per-class metrics", metric_start)

    retrieval_start = time.perf_counter()
    retrieval_results = compute_retrieval_results(
        selected_classes=selected_classes,
        source_class_caches=source_class_caches,
        target_class_caches=target_class_caches,
        class_names=class_names,
    )
    log_timing("retrieval", retrieval_start)

    results = {
        "mode": mode,
        "source_dataset": source_payload["dataset_name"],
        "target_dataset": target_payload["dataset_name"],
        "backbone": source_payload["backbone"],
        "metrics": metrics,
        "num_classes": len(selected_classes),
        "samples_per_class": samples_per_class,
        "selected_classes": selected_classes,
        "averages": {
            "same_class": mean_dict(same_class_metrics, metrics),
            "different_class_control": mean_dict(different_class_metrics, metrics),
        },
        "retrieval": retrieval_results,
        "per_class": per_class_results,
    }

    if args.compute_knn_distribution:
        results["averages"].update(aggregate_nested_metric_dict(
            knn_metric_results,
            ["knn_wasserstein", "knn_mean_diff", "knn_std_diff"],
        ))
    if args.compute_distance_histogram:
        js_values = [metric_dict["distance_histogram_js"] for metric_dict in histogram_metric_results if not math.isnan(metric_dict["distance_histogram_js"])]
        wass_values = [metric_dict["distance_histogram_wasserstein"] for metric_dict in histogram_metric_results if not math.isnan(metric_dict["distance_histogram_wasserstein"])]
        results["averages"]["distance_histogram_js"] = float(np.mean(js_values)) if js_values else float("nan")
        results["averages"]["distance_histogram_wasserstein"] = float(np.mean(wass_values)) if wass_values else float("nan")
    if args.compute_graph_stats:
        results["averages"].update(aggregate_graph_stat_dicts(graph_metric_results))
    if "pairwise_spearman" in metrics or "pairwise_pearson" in metrics:
        results["pairwise_deprecated_warning"] = (
            "pairwise distance correlation is not reliable for cross-dataset comparison without sample correspondence"
        )

    if args.inter_class_geometry:
        inter_class_start = time.perf_counter()
        results["inter_class_geometry"] = compute_inter_class_geometry_results(
            selected_classes=selected_classes,
            class_names=class_names,
            source_class_samples=source_class_samples,
            target_class_samples=target_class_samples,
            output_prefix=output_prefix,
            artifact_suffix=artifact_suffix,
            source_dataset_name=source_payload["dataset_name"],
            target_dataset_name=target_payload["dataset_name"],
            sign_epsilon=args.sign_epsilon,
            save_plots=save_visualization and args.save_inter_class_plots,
        )
        log_timing("inter-class relational geometry", inter_class_start)

    if save_visualization and args.save_umap:
        viz_bundle = build_visualization_arrays(
            selected_classes=selected_classes,
            source_class_samples=source_class_samples,
            target_class_samples=target_class_samples,
            class_names=class_names,
            args=args,
        )
        if viz_bundle is not None:
            viz_outputs = save_umap_outputs(
                viz_bundle=viz_bundle,
                output_prefix=output_prefix,
                source_dataset_name=source_payload["dataset_name"],
                target_dataset_name=target_payload["dataset_name"],
                args=args,
            )
            results["visualization"] = viz_outputs

    return results


def run_same_dataset_upper_bound(args, payload, metrics):
    features = payload["features"].float().cpu().numpy()
    labels = payload["labels"].long().cpu()
    class_subset = parse_class_indices(args.class_indices)
    class_to_indices = build_class_index(labels)
    min_samples = args.min_samples_per_class_for_split or (2 * args.samples_per_class)

    eligible_classes = sorted(class_to_indices)
    if class_subset is not None:
        eligible_classes = [class_index for class_index in class_subset if class_index in class_to_indices]
    eligible_classes = [class_index for class_index in eligible_classes if len(class_to_indices[class_index]) >= min_samples]

    for class_index in sorted(class_to_indices):
        if class_subset is not None and class_index not in class_subset:
            continue
        if len(class_to_indices[class_index]) < min_samples:
            warnings.warn(
                f"Skipping class {class_index} for same-dataset upper bound: "
                f"needs at least {min_samples} samples, found {len(class_to_indices[class_index])}.",
                RuntimeWarning,
            )

    if len(eligible_classes) < args.num_classes:
        raise ValueError(
            f"Requested {args.num_classes} classes, but only {len(eligible_classes)} classes have at least "
            f"{min_samples} samples for same-dataset split-half evaluation."
        )

    selection_rng = np.random.default_rng(args.seed)
    selected_classes = sorted(selection_rng.choice(eligible_classes, size=args.num_classes, replace=False).tolist())
    repeat_results = []

    for repeat_index in range(args.upper_bound_num_repeats):
        repeat_rng = np.random.default_rng(args.seed + repeat_index)
        source_class_samples = {}
        target_class_samples = {}
        class_names = {}

        for class_index in selected_classes:
            candidate_indices = np.array(class_to_indices[class_index])
            # Split-half reliability: two disjoint subsets from the same dataset
            # act as a noise ceiling for how high the metric can go in practice.
            shuffled_indices = repeat_rng.permutation(candidate_indices)
            split_size = min(args.samples_per_class, shuffled_indices.shape[0] // 2)
            if split_size < args.samples_per_class:
                warnings.warn(
                    f"Skipping class {class_index} in repeat {repeat_index}: "
                    f"not enough disjoint samples for two halves of size {args.samples_per_class}.",
                    RuntimeWarning,
                )
                continue

            source_indices = np.sort(shuffled_indices[:split_size])
            target_indices = np.sort(shuffled_indices[split_size:2 * split_size])
            source_class_samples[class_index] = features[source_indices]
            target_class_samples[class_index] = features[target_indices]
            class_names[class_index] = get_class_name(payload, class_index)

        repeat_selected_classes = sorted(source_class_samples)
        if not repeat_selected_classes:
            raise ValueError("No classes remained after same-dataset split-half sampling.")

        repeat_result = build_result(
            selected_classes=repeat_selected_classes,
            source_payload={"dataset_name": f"{payload['dataset_name']}_split_A", "backbone": payload["backbone"]},
            target_payload={"dataset_name": f"{payload['dataset_name']}_split_B", "backbone": payload["backbone"]},
            metrics=metrics,
            pca_dim=args.pca_dim,
            source_class_samples=source_class_samples,
            target_class_samples=target_class_samples,
            class_names=class_names,
            samples_per_class=args.samples_per_class,
            mode="same_dataset_upper_bound",
            args=args,
            output_prefix=get_output_prefix(args.output),
            save_visualization=(repeat_index == 0),
            artifact_suffix=f"_repeat_{repeat_index}",
        )
        repeat_result["repeat_index"] = repeat_index
        repeat_results.append(repeat_result)

    same_runs = [run["averages"]["same_class"] for run in repeat_results]
    different_runs = [run["averages"]["different_class_control"] for run in repeat_results]
    retrieval_runs = [run["retrieval"] for run in repeat_results]
    summary = {
        "same_class_mean": mean_dict(same_runs, metrics),
        "same_class_std": std_dict(same_runs, metrics),
        "different_class_control_mean": mean_dict(different_runs, metrics),
        "different_class_control_std": std_dict(different_runs, metrics),
        "retrieval": compute_retrieval_summary_stats(retrieval_runs),
    }

    if args.compute_knn_distribution:
        knn_runs = [
            {
                "knn_wasserstein": run["averages"]["knn_wasserstein"],
                "knn_mean_diff": run["averages"]["knn_mean_diff"],
                "knn_std_diff": run["averages"]["knn_std_diff"],
            }
            for run in repeat_results
        ]
        summary.update(aggregate_nested_metric_dict(
            knn_runs,
            ["knn_wasserstein", "knn_mean_diff", "knn_std_diff"],
        ))
    if args.compute_distance_histogram:
        js_values = [run["averages"]["distance_histogram_js"] for run in repeat_results if not math.isnan(run["averages"]["distance_histogram_js"])]
        wass_values = [run["averages"]["distance_histogram_wasserstein"] for run in repeat_results if not math.isnan(run["averages"]["distance_histogram_wasserstein"])]
        summary["distance_histogram_js"] = float(np.mean(js_values)) if js_values else float("nan")
        summary["distance_histogram_wasserstein"] = float(np.mean(wass_values)) if wass_values else float("nan")
    if args.compute_graph_stats:
        graph_runs = [{"neighbor_graph_stats": run["averages"]["neighbor_graph_stats"]} for run in repeat_results]
        summary.update(aggregate_graph_stat_averages(graph_runs))
    if args.inter_class_geometry:
        summary["inter_class_geometry"] = summarize_inter_class_geometry(
            [run["inter_class_geometry"] for run in repeat_results if "inter_class_geometry" in run]
        )

    return {
        "mode": "same_dataset_upper_bound",
        "source_dataset": f"{payload['dataset_name']}_split_A",
        "target_dataset": f"{payload['dataset_name']}_split_B",
        "backbone": payload["backbone"],
        "metrics": metrics,
        "num_classes": args.num_classes,
        "samples_per_class": args.samples_per_class,
        "seed": args.seed,
        "selected_classes": selected_classes,
        "min_samples_per_class_for_split": min_samples,
        "upper_bound_num_repeats": args.upper_bound_num_repeats,
        "summary": summary,
        "repeats": repeat_results,
    }


def main():
    total_start = time.perf_counter()
    args = get_arguments()
    if args.self_test_pairwise_corr:
        run_pairwise_corr_self_test(args.seed)
        return

    if args.same_dataset_upper_bound:
        required_args = ["source_feature_file", "output", "num_classes", "samples_per_class"]
    else:
        required_args = ["source_feature_file", "target_feature_file", "output", "num_classes", "samples_per_class"]
    missing_args = [name for name in required_args if getattr(args, name) is None]
    if missing_args:
        raise ValueError(f"Missing required arguments: {missing_args}")

    metrics = parse_metrics(args.metrics)
    device = resolve_device(args.device)
    print(f"[timing] geometry device: {device}")
    load_start = time.perf_counter()
    source_payload = load_feature_file(args.source_feature_file)
    if args.same_dataset_upper_bound:
        log_timing("load feature file", load_start)
    if args.same_dataset_upper_bound:
        results = run_same_dataset_upper_bound(args, source_payload, metrics)
    else:
        rng = np.random.default_rng(args.seed)
        class_subset = parse_class_indices(args.class_indices)
        target_payload = load_feature_file(args.target_feature_file)
        log_timing("load feature files", load_start)

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
        eligible_classes = sorted(set(source_class_to_indices).intersection(target_class_to_indices))
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

        results = build_result(
            selected_classes=selected_classes,
            source_payload=source_payload,
            target_payload=target_payload,
            metrics=metrics,
            pca_dim=args.pca_dim,
            source_class_samples=source_class_samples,
            target_class_samples=target_class_samples,
            class_names=class_names,
            samples_per_class=args.samples_per_class,
            mode="cross_dataset",
            args=args,
            output_prefix=get_output_prefix(args.output),
            save_visualization=True,
            artifact_suffix="",
        )
        results["seed"] = args.seed

    save_start = time.perf_counter()
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    log_timing("save json", save_start)

    if results["mode"] == "same_dataset_upper_bound":
        print("Same-dataset upper bound / split-half reliability estimate")
    else:
        print("Cross-dataset geometry evaluation")
    if "pairwise_spearman" in metrics or "pairwise_pearson" in metrics:
        print("pairwise distance correlation is not reliable for cross-dataset comparison without sample correspondence")
    print(f"Source dataset: {results['source_dataset']}")
    print(f"Target dataset: {results['target_dataset']}")
    print(f"Backbone: {results['backbone']}")
    print(f"Selected classes: {len(results['selected_classes'])}")
    if results["mode"] == "same_dataset_upper_bound":
        for metric in metrics:
            same_mean = results["summary"]["same_class_mean"][metric]
            same_std = results["summary"]["same_class_std"][metric]
            diff_mean = results["summary"]["different_class_control_mean"][metric]
            diff_std = results["summary"]["different_class_control_std"][metric]
            print(f"{metric}: same={same_mean:.6f}+/-{same_std:.6f} different={diff_mean:.6f}+/-{diff_std:.6f}")
        for mode, retrieval_result in results["summary"]["retrieval"].items():
            print(f"retrieval_acc_{mode}: {retrieval_result['mean_accuracy']:.6f}+/-{retrieval_result['std_accuracy']:.6f}")
    else:
        for metric in metrics:
            same_value = results["averages"]["same_class"][metric]
            diff_value = results["averages"]["different_class_control"][metric]
            print(f"{metric}: same={same_value:.6f} different={diff_value:.6f}")
        for mode, retrieval_result in results["retrieval"].items():
            print(f"retrieval_acc_{mode}: {retrieval_result['accuracy']:.6f}")

    if args.compute_knn_distribution or args.compute_distance_histogram or args.compute_graph_stats:
        print("Correspondence-free topology metrics")
        if args.compute_knn_distribution:
            knn_wasserstein_summary = results["summary"]["knn_wasserstein"] if results["mode"] == "same_dataset_upper_bound" else results["averages"]["knn_wasserstein"]
            for k_key, value in knn_wasserstein_summary.items():
                print(f"kNN distance Wasserstein ({k_key}): {value:.6f}")
            knn_mean_diff_summary = results["summary"]["knn_mean_diff"] if results["mode"] == "same_dataset_upper_bound" else results["averages"]["knn_mean_diff"]
            for k_key, value in knn_mean_diff_summary.items():
                print(f"kNN mean distance diff ({k_key}): {value:.6f}")
        if args.compute_distance_histogram:
            histogram_js = results["summary"]["distance_histogram_js"] if results["mode"] == "same_dataset_upper_bound" else results["averages"]["distance_histogram_js"]
            histogram_wass = results["summary"]["distance_histogram_wasserstein"] if results["mode"] == "same_dataset_upper_bound" else results["averages"]["distance_histogram_wasserstein"]
            print(f"distance histogram JS: {histogram_js:.6f}")
            print(f"distance histogram Wasserstein: {histogram_wass:.6f}")
        if args.compute_graph_stats:
            graph_stats_summary = results["summary"]["neighbor_graph_stats"] if results["mode"] == "same_dataset_upper_bound" else results["averages"]["neighbor_graph_stats"]
            for k_key, stat_summary in graph_stats_summary.items():
                print(f"neighbor graph mean kNN distance diff ({k_key}): {stat_summary['mean_knn_distance']['abs_diff_mean']:.6f}")

    if args.inter_class_geometry:
        print("Inter-class relational geometry")
        if results["mode"] == "same_dataset_upper_bound":
            inter_class_summary = results["summary"]["inter_class_geometry"]
            print(f"- matrix Pearson corr: {inter_class_summary['pearson_corr']['mean']:.6f}+/-{inter_class_summary['pearson_corr']['std']:.6f}")
            print(f"- matrix Spearman corr: {inter_class_summary['spearman_corr']['mean']:.6f}+/-{inter_class_summary['spearman_corr']['std']:.6f}")
            print(f"- mean abs diff: {inter_class_summary['mean_abs_diff']['mean']:.6f}+/-{inter_class_summary['mean_abs_diff']['std']:.6f}")
            print(f"- sign consistency rate: {inter_class_summary['sign_consistency_rate']['mean']:.6f}+/-{inter_class_summary['sign_consistency_rate']['std']:.6f}")
            print(f"- near-zero pairs: {inter_class_summary['num_pairs_near_zero']['mean']:.2f}+/-{inter_class_summary['num_pairs_near_zero']['std']:.2f}")
            print(f"- radius corr: {inter_class_summary['radius_corr']['mean']:.6f}+/-{inter_class_summary['radius_corr']['std']:.6f}")
            print(f"- radius mean abs diff: {inter_class_summary['radius_mean_abs_diff']['mean']:.6f}+/-{inter_class_summary['radius_mean_abs_diff']['std']:.6f}")
        else:
            inter_class = results["inter_class_geometry"]
            print(f"- matrix Pearson corr: {inter_class['pearson_corr']:.6f}")
            print(f"- matrix Spearman corr: {inter_class['spearman_corr']:.6f}")
            print(f"- mean abs diff: {inter_class['mean_abs_diff']:.6f}")
            print(f"- sign consistency rate: {inter_class['sign_consistency_rate']:.6f}")
            print(f"- near-zero pairs: {inter_class['num_pairs_near_zero']}/{inter_class['num_pairs_total']}")
            print(f"- radius corr: {inter_class['radius_corr']:.6f}")
            print(f"- radius mean abs diff: {inter_class['radius_mean_abs_diff']:.6f}")
            print(f"- source matrix: {inter_class['matrix_file_source']}")
            print(f"- target matrix: {inter_class['matrix_file_target']}")
            print(f"- per-pair signs: {inter_class['per_pair_sign_info_file']}")
            print(f"- per-class radii: {inter_class['per_class_radius_file']}")
            if inter_class["top_source_similar_pairs"]:
                pair = inter_class["top_source_similar_pairs"][0]
                print(f"- most similar pair in source: {pair['class_a_name']} / {pair['class_b_name']} ({pair['source_similarity']:.6f})")
            if inter_class["top_target_similar_pairs"]:
                pair = inter_class["top_target_similar_pairs"][0]
                print(f"- most similar pair in target: {pair['class_a_name']} / {pair['class_b_name']} ({pair['target_similarity']:.6f})")
            if inter_class["largest_similarity_increase_pairs"]:
                pair = inter_class["largest_similarity_increase_pairs"][0]
                print(f"- largest similarity increase: {pair['class_a_name']} / {pair['class_b_name']} (delta={pair['delta_similarity']:.6f})")
            if inter_class["largest_similarity_decrease_pairs"]:
                pair = inter_class["largest_similarity_decrease_pairs"][0]
                print(f"- largest similarity decrease: {pair['class_a_name']} / {pair['class_b_name']} (delta={pair['delta_similarity']:.6f})")
            if inter_class["plot_files"] is not None:
                print(f"- inter-class plots: {inter_class['plot_files']}")

    if "visualization" in results:
        print(f"UMAP figure: {results['visualization']['png_path']}")
        if results["visualization"]["html_path"] is not None:
            print(f"UMAP html: {results['visualization']['html_path']}")
        print(f"UMAP metrics: {results['visualization']['viz_metrics_path']}")
        print(f"trustworthiness: {results['visualization']['metrics']['trustworthiness']:.6f}")
        print(f"continuity: {results['visualization']['metrics']['continuity']:.6f}")

    print("small values -> class geometry preserved")
    print("large values -> geometry differs across datasets")
    print(f"Saved JSON to: {args.output}")
    log_timing("total", total_start)


if __name__ == "__main__":
    main()
