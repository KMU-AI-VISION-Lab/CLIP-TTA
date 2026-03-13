import argparse
import json
import os
import sys
import warnings

import numpy as np
import torch

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.geometry_eval import (
    build_class_index,
    compute_continuity_score,
    compute_distance_histogram_metrics,
    compute_knn_distribution_metrics,
    get_class_name,
    get_output_prefix,
    sample_class_features,
)


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_feature_file", required=True, type=str)
    parser.add_argument("--target_feature_file", required=True, type=str)
    parser.add_argument("--eval_json", default=None, type=str, help="Optional geometry_eval JSON for auto-selecting classes.")
    parser.add_argument("--output_prefix", default=None, type=str)
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--samples_per_class", default=10, type=int)
    parser.add_argument("--viz_dim", default=2, type=int, choices=[2, 3])
    parser.add_argument("--viz_max_points_per_class", default=100, type=int)
    parser.add_argument("--interactive", action="store_true", default=False)
    parser.add_argument("--viz_class_ids", nargs="+", type=int, default=None)
    parser.add_argument("--viz_classes", nargs="+", default=None)
    parser.add_argument("--viz_select_mode", default="manual", choices=["manual", "retrieval_error", "centroid_shift", "topology"])
    parser.add_argument("--viz_topk", default=10, type=int)
    parser.add_argument("--retrieval_mode", default="centroid_cosine", choices=["centroid_cosine", "centroid_euclidean", "pca_subspace"])
    parser.add_argument("--topology_metric", default="distance_histogram_js", type=str)
    parser.add_argument("--knn_k", nargs="+", type=int, default=[5, 10])
    return parser.parse_args()


def load_feature_payload(path):
    payload = torch.load(path, map_location="cpu")
    required = {"features", "labels", "classnames", "dataset_name", "backbone"}
    missing = required - set(payload)
    if missing:
        raise KeyError(f"Missing keys in feature file {path}: {sorted(missing)}")
    return payload


def load_eval_json(path):
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def select_manual_classes(selected_classes, class_names, args):
    if args.viz_class_ids:
        return [class_index for class_index in args.viz_class_ids if class_index in selected_classes]
    if args.viz_classes:
        if len(args.viz_classes) == 1 and args.viz_classes[0].lower() == "all":
            return list(selected_classes)
        requested = set(args.viz_classes)
        return [class_index for class_index in selected_classes if class_names[class_index] in requested]
    return list(selected_classes[:args.viz_topk])


def select_from_eval_json(eval_payload, args):
    selected_classes = eval_payload["selected_classes"]
    per_class = eval_payload["per_class"]
    if args.viz_select_mode == "manual":
        class_names = {entry["class_index"]: entry["class_name"] for entry in per_class}
        return select_manual_classes(selected_classes, class_names, args)

    if args.viz_select_mode == "retrieval_error":
        # Focus on the classes that the quantitative retrieval metric found
        # most confusing. These are usually the most informative to inspect.
        retrieval_entries = eval_payload["retrieval"][args.retrieval_mode]["per_class"]
        ranked = [entry["source_class_index"] for entry in retrieval_entries if not entry["correct"]]
        if len(ranked) < args.viz_topk:
            ranked.extend([entry["source_class_index"] for entry in retrieval_entries if entry["correct"]])
        return ranked[:args.viz_topk]

    if args.viz_select_mode == "centroid_shift":
        # Large centroid movement means the class mean shifts a lot across domains.
        ranked_entries = sorted(per_class, key=lambda entry: entry["same_class"]["centroid_euclidean"], reverse=True)
        return [entry["class_index"] for entry in ranked_entries[:args.viz_topk]]

    if args.viz_select_mode == "topology":
        def get_metric_value(entry):
            metric = entry.get(args.topology_metric)
            if isinstance(metric, dict):
                first_key = sorted(metric.keys())[0]
                return metric[first_key]
            if metric is not None:
                return metric
            return entry["same_class"].get(args.topology_metric, float("nan"))

        ranked_entries = sorted(per_class, key=get_metric_value, reverse=True)
        return [entry["class_index"] for entry in ranked_entries[:args.viz_topk]]

    raise ValueError(f"Unsupported viz_select_mode: {args.viz_select_mode}")


def select_classes(source_payload, target_payload, eval_payload, args):
    source_class_to_indices = build_class_index(source_payload["labels"].long().cpu())
    target_class_to_indices = build_class_index(target_payload["labels"].long().cpu())
    available_classes = sorted(set(source_class_to_indices).intersection(target_class_to_indices))
    class_names = {class_index: get_class_name(source_payload, class_index) for class_index in available_classes}

    if eval_payload is not None:
        candidate_classes = [class_index for class_index in select_from_eval_json(eval_payload, args) if class_index in available_classes]
    else:
        candidate_classes = select_manual_classes(available_classes, class_names, args)

    return candidate_classes, source_class_to_indices, target_class_to_indices, class_names


def build_visualization_bundle(source_payload, target_payload, class_ids, source_class_to_indices, target_class_to_indices, class_names, args):
    rng = np.random.default_rng(args.seed)
    features = []
    labels = []
    dataset_tags = []
    centroids = []

    source_features = source_payload["features"].float().cpu().numpy()
    target_features = target_payload["features"].float().cpu().numpy()

    for class_index in class_ids:
        if len(source_class_to_indices[class_index]) < args.samples_per_class or len(target_class_to_indices[class_index]) < args.samples_per_class:
            warnings.warn(f"Skipping class {class_index}: not enough samples for visualization.", RuntimeWarning)
            continue

        # Keep the number of plotted points per class bounded so one class
        # does not visually dominate the whole UMAP.
        source_samples = sample_class_features(source_features, source_class_to_indices, class_index, args.samples_per_class, rng)
        target_samples = sample_class_features(target_features, target_class_to_indices, class_index, args.samples_per_class, rng)
        source_points = source_samples[:args.viz_max_points_per_class]
        target_points = target_samples[:args.viz_max_points_per_class]

        features.append(source_points)
        features.append(target_points)
        labels.extend([class_names[class_index]] * source_points.shape[0])
        labels.extend([class_names[class_index]] * target_points.shape[0])
        dataset_tags.extend(["source"] * source_points.shape[0])
        dataset_tags.extend(["target"] * target_points.shape[0])
        centroids.append({
            "class_index": class_index,
            "class_name": class_names[class_index],
            "source_centroid": source_samples.mean(axis=0),
            "target_centroid": target_samples.mean(axis=0),
        })

    if not features:
        raise ValueError("No valid classes remained for visualization.")

    return {
        "features": np.concatenate(features, axis=0),
        "labels": labels,
        "dataset_tags": dataset_tags,
        "centroids": centroids,
        "class_ids": [item["class_index"] for item in centroids],
    }


def save_umap_visualization(bundle, source_dataset_name, target_dataset_name, output_prefix, args):
    import matplotlib.pyplot as plt
    import umap
    from sklearn.manifold import trustworthiness

    reducer = umap.UMAP(
        n_components=args.viz_dim,
        random_state=args.seed,
        transform_seed=args.seed,
    )
    # Fit one shared reducer on pooled points so source and target are shown
    # in the same low-dimensional coordinate system.
    embedding = reducer.fit_transform(bundle["features"])
    centroid_stack = np.stack(
        [item["source_centroid"] for item in bundle["centroids"]] +
        [item["target_centroid"] for item in bundle["centroids"]],
        axis=0,
    )
    # Centroids are computed in the original CLIP space first, then projected
    # with the fitted UMAP. The arrows show domain shift direction per class.
    centroid_embedding = reducer.transform(centroid_stack)
    centroid_split = len(bundle["centroids"])

    dataset_to_marker = {"source": "o", "target": "^"}
    class_to_color_id = {class_name: index for index, class_name in enumerate(sorted(set(bundle["labels"])))}
    color_values = np.array([class_to_color_id[label] for label in bundle["labels"]], dtype=np.int64)

    figure = plt.figure(figsize=(10, 8))
    axis = figure.add_subplot(111, projection="3d") if args.viz_dim == 3 else figure.add_subplot(111)
    for dataset_tag in ["source", "target"]:
        indices = [i for i, tag in enumerate(bundle["dataset_tags"]) if tag == dataset_tag]
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

    for centroid_index, centroid_info in enumerate(bundle["centroids"]):
        source_point = centroid_embedding[centroid_index]
        target_point = centroid_embedding[centroid_split + centroid_index]
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
        import plotly.express as px
        import plotly.graph_objects as go

        frame = {
            "x": embedding[:, 0],
            "y": embedding[:, 1],
            "class": bundle["labels"],
            "dataset": bundle["dataset_tags"],
        }
        if args.viz_dim == 3:
            frame["z"] = embedding[:, 2]
            figure_html = px.scatter_3d(frame, x="x", y="y", z="z", color="class", symbol="dataset", title=f"UMAP: {source_dataset_name} vs {target_dataset_name}")
            for centroid_index in range(len(bundle["centroids"])):
                source_point = centroid_embedding[centroid_index]
                target_point = centroid_embedding[centroid_split + centroid_index]
                figure_html.add_trace(go.Scatter3d(x=[source_point[0], target_point[0]], y=[source_point[1], target_point[1]], z=[source_point[2], target_point[2]], mode="lines", line={"color": "black"}, showlegend=False))
        else:
            figure_html = px.scatter(frame, x="x", y="y", color="class", symbol="dataset", title=f"UMAP: {source_dataset_name} vs {target_dataset_name}")
            for centroid_index in range(len(bundle["centroids"])):
                source_point = centroid_embedding[centroid_index]
                target_point = centroid_embedding[centroid_split + centroid_index]
                figure_html.add_scatter(x=[source_point[0], target_point[0]], y=[source_point[1], target_point[1]], mode="lines", line={"color": "black"}, showlegend=False)

        html_path = f"{output_prefix}_umap_{args.viz_dim}d.html"
        figure_html.write_html(html_path)

    trust_k = min(10, max(2, bundle["features"].shape[0] - 1))
    viz_metrics = {
        "source_dataset": source_dataset_name,
        "target_dataset": target_dataset_name,
        "viz_dim": args.viz_dim,
        "viz_select_mode": args.viz_select_mode,
        "num_points": int(bundle["features"].shape[0]),
        "num_classes": len(bundle["class_ids"]),
        "selected_class_ids": bundle["class_ids"],
        # These scores help judge whether the UMAP figure preserved local
        # neighborhood structure well enough to trust qualitatively.
        "trustworthiness": float(trustworthiness(bundle["features"], embedding, n_neighbors=trust_k)),
        "continuity": compute_continuity_score(bundle["features"], embedding, n_neighbors=trust_k),
        "png_path": png_path,
        "html_path": html_path,
    }
    metrics_path = f"{output_prefix}_viz_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as handle:
        json.dump(viz_metrics, handle, indent=2)

    return png_path, html_path, metrics_path


def main():
    args = get_arguments()
    source_payload = load_feature_payload(args.source_feature_file)
    target_payload = load_feature_payload(args.target_feature_file)
    eval_payload = load_eval_json(args.eval_json)

    output_prefix = args.output_prefix or get_output_prefix(args.eval_json or args.target_feature_file)
    class_ids, source_class_to_indices, target_class_to_indices, class_names = select_classes(
        source_payload, target_payload, eval_payload, args
    )
    bundle = build_visualization_bundle(
        source_payload,
        target_payload,
        class_ids,
        source_class_to_indices,
        target_class_to_indices,
        class_names,
        args,
    )
    png_path, html_path, metrics_path = save_umap_visualization(
        bundle,
        source_payload["dataset_name"],
        target_payload["dataset_name"],
        output_prefix,
        args,
    )

    print(f"Selected classes: {bundle['class_ids']}")
    print(f"UMAP figure: {png_path}")
    if html_path is not None:
        print(f"UMAP html: {html_path}")
    print(f"UMAP metrics: {metrics_path}")


if __name__ == "__main__":
    main()
