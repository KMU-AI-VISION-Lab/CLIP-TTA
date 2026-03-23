import argparse
import json
import os

import numpy as np


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--radius_file", required=True, type=str)
    parser.add_argument("--low_quantile", default=0.2, type=float, help="Bottom quantile used for the inner bucket.")
    parser.add_argument("--high_quantile", default=0.8, type=float, help="Top quantile used for the outer bucket.")
    parser.add_argument("--topk", default=10, type=int)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--save_plots", action="store_true")
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def mean_or_nan(values):
    if not values:
        return float("nan")
    return float(sum(values) / len(values))


def std_or_nan(values):
    if not values:
        return float("nan")
    return float(np.std(np.asarray(values, dtype=np.float64)))


def attach_delta_radius(radius_records):
    enriched = []
    for entry in radius_records:
        cloned = dict(entry)
        cloned["delta_radius"] = float(cloned["target_radius"] - cloned["source_radius"])
        enriched.append(cloned)
    return enriched


def assign_buckets(radius_records, low_threshold, high_threshold):
    bucketed = []
    for entry in radius_records:
        # radius bucket은 source radius를 기준으로 정합니다.
        # 즉 source에서 원래 중심에 가까운 클래스(inner)와
        # 원래 바깥에 있는 클래스(outer)가 target에서 어떻게 이동하는지 봅니다.
        source_radius = float(entry["source_radius"])
        cloned = dict(entry)
        if source_radius <= low_threshold:
            cloned["bucket"] = "inner"
        elif source_radius >= high_threshold:
            cloned["bucket"] = "outer"
        else:
            cloned["bucket"] = "mid"
        bucketed.append(cloned)
    return bucketed


def format_radius_record(entry):
    return {
        "class_id": entry["class_id"],
        "class_name": entry["class_name"],
        "source_radius": float(entry["source_radius"]),
        "target_radius": float(entry["target_radius"]),
        "delta_radius": float(entry["delta_radius"]),
        "abs_diff": float(entry["abs_diff"]),
        "bucket": entry["bucket"],
    }


def build_bucket_summary(entries, bucket_name, topk):
    if not entries:
        return {"bucket": bucket_name, "num_classes": 0}

    source_values = [float(entry["source_radius"]) for entry in entries]
    target_values = [float(entry["target_radius"]) for entry in entries]
    delta_values = [float(entry["delta_radius"]) for entry in entries]
    abs_diff_values = [float(entry["abs_diff"]) for entry in entries]

    # representative classes:
    # - inner: source radius가 가장 작은 클래스들
    # - outer: source radius가 가장 큰 클래스들
    # - mid: source radius 중앙값과 가장 가까운 클래스들
    if bucket_name == "inner":
        representative = sorted(entries, key=lambda item: item["source_radius"])[:topk]
    elif bucket_name == "outer":
        representative = sorted(entries, key=lambda item: item["source_radius"], reverse=True)[:topk]
    else:
        median_value = float(np.median(np.asarray(source_values, dtype=np.float64)))
        representative = sorted(entries, key=lambda item: abs(item["source_radius"] - median_value))[:topk]

    return {
        "bucket": bucket_name,
        "num_classes": len(entries),
        "mean_source_radius": mean_or_nan(source_values),
        "std_source_radius": std_or_nan(source_values),
        "mean_target_radius": mean_or_nan(target_values),
        "std_target_radius": std_or_nan(target_values),
        "mean_delta_radius": mean_or_nan(delta_values),
        "std_delta_radius": std_or_nan(delta_values),
        "mean_abs_delta_radius": mean_or_nan(abs_diff_values),
        "std_abs_delta_radius": std_or_nan(abs_diff_values),
        "representative_classes": [format_radius_record(entry) for entry in representative],
        "largest_radius_increase_classes": [
            format_radius_record(entry)
            for entry in sorted(entries, key=lambda item: item["delta_radius"], reverse=True)[:topk]
        ],
        "largest_radius_decrease_classes": [
            format_radius_record(entry)
            for entry in sorted(entries, key=lambda item: item["delta_radius"])[:topk]
        ],
    }


def maybe_save_plots(summary, source_values, target_values, artifact_prefix):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Radius-bucket plots require 'matplotlib'.") from exc

    inner_threshold = summary["bucket_thresholds"]["inner_upper"]
    outer_threshold = summary["bucket_thresholds"]["outer_lower"]
    files = {}

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(source_values, bins=min(40, max(10, len(source_values) // 5)), color="#9ecae1", alpha=0.9)
    axis.axvline(inner_threshold, color="#2b8a3e", linestyle="--", label=f"inner <= {inner_threshold:.3f}")
    axis.axvline(outer_threshold, color="#cc4c4c", linestyle="--", label=f"outer >= {outer_threshold:.3f}")
    axis.set_title("Source Radius Distribution")
    axis.set_xlabel("Source radius")
    axis.set_ylabel("Count")
    axis.legend()
    hist_path = f"{artifact_prefix}_source_hist.png"
    figure.tight_layout()
    figure.savefig(hist_path, dpi=200)
    plt.close(figure)
    files["source_histogram"] = hist_path

    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(source_values, target_values, alpha=0.7, s=28)
    max_value = max(float(np.max(source_values)), float(np.max(target_values)))
    axis.plot([0.0, max_value], [0.0, max_value], linestyle="--", color="gray")
    axis.set_title("Source vs Target Radius")
    axis.set_xlabel("Source radius")
    axis.set_ylabel("Target radius")
    scatter_path = f"{artifact_prefix}_scatter.png"
    figure.tight_layout()
    figure.savefig(scatter_path, dpi=200)
    plt.close(figure)
    files["scatter"] = scatter_path

    bucket_names = ["inner", "mid", "outer"]
    delta_means = [summary["buckets"][name]["mean_delta_radius"] for name in bucket_names]
    abs_means = [summary["buckets"][name]["mean_abs_delta_radius"] for name in bucket_names]

    figure, axis = plt.subplots(figsize=(7, 5))
    x = np.arange(len(bucket_names))
    width = 0.35
    axis.bar(x - width / 2, delta_means, width=width, label="mean delta radius")
    axis.bar(x + width / 2, abs_means, width=width, label="mean abs delta radius")
    axis.set_xticks(x)
    axis.set_xticklabels(bucket_names)
    axis.set_title("Radius Bucket Summary Statistics")
    axis.set_ylabel("Value")
    axis.legend()
    bar_path = f"{artifact_prefix}_bucket_stats.png"
    figure.tight_layout()
    figure.savefig(bar_path, dpi=200)
    plt.close(figure)
    files["bucket_stats"] = bar_path

    return files


def main():
    args = get_arguments()
    if not (0.0 < args.low_quantile < args.high_quantile < 1.0):
        raise ValueError("Require 0 < low_quantile < high_quantile < 1.")

    radius_records = attach_delta_radius(load_json(args.radius_file))
    if not radius_records:
        raise ValueError(f"No radius records found in {args.radius_file}")

    source_values = np.asarray([float(entry["source_radius"]) for entry in radius_records], dtype=np.float64)
    target_values = np.asarray([float(entry["target_radius"]) for entry in radius_records], dtype=np.float64)

    inner_threshold = float(np.quantile(source_values, args.low_quantile))
    outer_threshold = float(np.quantile(source_values, args.high_quantile))
    bucketed_records = assign_buckets(radius_records, inner_threshold, outer_threshold)

    buckets = {
        bucket_name: [entry for entry in bucketed_records if entry["bucket"] == bucket_name]
        for bucket_name in ["inner", "mid", "outer"]
    }

    summary = {
        "radius_file": args.radius_file,
        "bucket_definition": "source-radius-quantiles",
        "quantiles": {
            "low_quantile": float(args.low_quantile),
            "high_quantile": float(args.high_quantile),
        },
        "bucket_thresholds": {
            "inner_upper": inner_threshold,
            "mid_lower": inner_threshold,
            "mid_upper": outer_threshold,
            "outer_lower": outer_threshold,
        },
        "num_classes_total": len(radius_records),
        "mean_delta_radius_overall": mean_or_nan([entry["delta_radius"] for entry in radius_records]),
        "mean_abs_delta_radius_overall": mean_or_nan([entry["abs_diff"] for entry in radius_records]),
        "buckets": {
            bucket_name: build_bucket_summary(entries, bucket_name, args.topk)
            for bucket_name, entries in buckets.items()
        },
    }

    output_path = args.output
    if output_path is None:
        base = os.path.splitext(args.radius_file)[0]
        low_label = int(round(args.low_quantile * 100))
        high_label = int(round(args.high_quantile * 100))
        output_path = f"{base}_radius_buckets_q{low_label}_{high_label}.json"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_files = None
    if args.save_plots:
        artifact_prefix = os.path.splitext(output_path)[0]
        plot_files = maybe_save_plots(summary, source_values, target_values, artifact_prefix)
        summary["plot_files"] = plot_files
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    print(f"Saved radius-bucket summary to: {output_path}")
    print(f"Class count: {len(radius_records)}")
    print(f"Bucket thresholds: inner<={inner_threshold:.6f}, outer>={outer_threshold:.6f}")
    for bucket_name in ["inner", "mid", "outer"]:
        bucket = summary["buckets"][bucket_name]
        print(
            f"{bucket_name}: num_classes={bucket['num_classes']}, "
            f"mean_delta_radius={bucket.get('mean_delta_radius', float('nan')):.6f}, "
            f"mean_abs_delta_radius={bucket.get('mean_abs_delta_radius', float('nan')):.6f}"
        )
    if plot_files:
        print(f"Saved plots: {plot_files}")


if __name__ == "__main__":
    main()
