import argparse
import json
import math
import os

import numpy as np


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_info_file", required=True, type=str)
    parser.add_argument("--low_quantile", default=0.2, type=float, help="Bottom quantile used for the far bucket.")
    parser.add_argument("--high_quantile", default=0.8, type=float, help="Top quantile used for the near bucket.")
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


def format_bucket_record(entry):
    return {
        "class_a_id": entry["class_a_id"],
        "class_b_id": entry["class_b_id"],
        "class_a_name": entry["class_a_name"],
        "class_b_name": entry["class_b_name"],
        "source_similarity": float(entry["source_similarity"]),
        "target_similarity": float(entry["target_similarity"]),
        "delta_similarity": float(entry["delta_similarity"]),
        "abs_diff": float(entry["abs_diff"]),
        "bucket": entry["bucket"],
    }


def attach_missing_delta(pair_records):
    enriched = []
    for entry in pair_records:
        cloned = dict(entry)
        cloned.setdefault("delta_similarity", float(cloned["target_similarity"] - cloned["source_similarity"]))
        cloned.setdefault("abs_diff", float(abs(cloned["target_similarity"] - cloned["source_similarity"])))
        enriched.append(cloned)
    return enriched


def assign_buckets(pair_records, low_threshold, high_threshold):
    bucketed = []
    for entry in pair_records:
        # bucket의 기준은 항상 source similarity입니다.
        # 즉 "원래 비슷했던 쌍" / "원래 멀었던 쌍"을 source에서 먼저 정하고,
        # target에서 그 쌍들이 어떻게 움직였는지를 보는 실험입니다.
        source_similarity = float(entry["source_similarity"])
        cloned = dict(entry)
        if source_similarity <= low_threshold:
            cloned["bucket"] = "far"
        elif source_similarity >= high_threshold:
            cloned["bucket"] = "near"
        else:
            cloned["bucket"] = "mid"
        bucketed.append(cloned)
    return bucketed


def build_bucket_summary(entries, bucket_name, topk):
    if not entries:
        return {"bucket": bucket_name, "num_pairs": 0}

    source_values = [float(entry["source_similarity"]) for entry in entries]
    target_values = [float(entry["target_similarity"]) for entry in entries]
    delta_values = [float(entry["delta_similarity"]) for entry in entries]
    abs_diff_values = [float(entry["abs_diff"]) for entry in entries]

    # representative examples:
    # - near: source similarity가 가장 높은 쌍들
    # - far: source similarity가 가장 낮은 쌍들
    # - mid: 중앙값에 가장 가까운 쌍들
    if bucket_name == "near":
        representative = sorted(entries, key=lambda item: item["source_similarity"], reverse=True)[:topk]
    elif bucket_name == "far":
        representative = sorted(entries, key=lambda item: item["source_similarity"])[:topk]
    else:
        median_value = float(np.median(np.asarray(source_values, dtype=np.float64)))
        representative = sorted(entries, key=lambda item: abs(item["source_similarity"] - median_value))[:topk]

    return {
        "bucket": bucket_name,
        "num_pairs": len(entries),
        "mean_source_similarity": mean_or_nan(source_values),
        "std_source_similarity": std_or_nan(source_values),
        "mean_target_similarity": mean_or_nan(target_values),
        "std_target_similarity": std_or_nan(target_values),
        "mean_delta_similarity": mean_or_nan(delta_values),
        "std_delta_similarity": std_or_nan(delta_values),
        "mean_abs_diff": mean_or_nan(abs_diff_values),
        "std_abs_diff": std_or_nan(abs_diff_values),
        "representative_pairs": [format_bucket_record(entry) for entry in representative],
        "largest_similarity_increase_pairs": [
            format_bucket_record(entry)
            for entry in sorted(entries, key=lambda item: item["delta_similarity"], reverse=True)[:topk]
        ],
        "largest_similarity_decrease_pairs": [
            format_bucket_record(entry)
            for entry in sorted(entries, key=lambda item: item["delta_similarity"])[:topk]
        ],
    }


def maybe_save_plots(summary, source_values, artifact_prefix):
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("Pair-bucket plots require 'matplotlib'.") from exc

    low_threshold = summary["bucket_thresholds"]["far_upper"]
    high_threshold = summary["bucket_thresholds"]["near_lower"]
    files = {}

    figure, axis = plt.subplots(figsize=(7, 5))
    axis.hist(source_values, bins=min(50, max(10, len(source_values) // 100)), color="#8aa4d6", alpha=0.9)
    axis.axvline(low_threshold, color="#cc4c4c", linestyle="--", label=f"far <= {low_threshold:.3f}")
    axis.axvline(high_threshold, color="#2b8a3e", linestyle="--", label=f"near >= {high_threshold:.3f}")
    axis.set_title("Source Inter-Class Similarity Distribution")
    axis.set_xlabel("Source similarity")
    axis.set_ylabel("Count")
    axis.legend()
    hist_path = f"{artifact_prefix}_source_hist.png"
    figure.tight_layout()
    figure.savefig(hist_path, dpi=200)
    plt.close(figure)
    files["source_histogram"] = hist_path

    bucket_names = ["far", "mid", "near"]
    delta_means = [summary["buckets"][name]["mean_delta_similarity"] for name in bucket_names]
    abs_diff_means = [summary["buckets"][name]["mean_abs_diff"] for name in bucket_names]

    figure, axis = plt.subplots(figsize=(7, 5))
    x = np.arange(len(bucket_names))
    width = 0.35
    axis.bar(x - width / 2, delta_means, width=width, label="mean delta")
    axis.bar(x + width / 2, abs_diff_means, width=width, label="mean abs diff")
    axis.set_xticks(x)
    axis.set_xticklabels(bucket_names)
    axis.set_title("Bucket Summary Statistics")
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

    pair_records = attach_missing_delta(load_json(args.pair_info_file))
    if not pair_records:
        raise ValueError(f"No pair records found in {args.pair_info_file}")

    source_values = np.asarray([float(entry["source_similarity"]) for entry in pair_records], dtype=np.float64)
    low_threshold = float(np.quantile(source_values, args.low_quantile))
    high_threshold = float(np.quantile(source_values, args.high_quantile))
    bucketed_records = assign_buckets(pair_records, low_threshold, high_threshold)

    buckets = {
        bucket_name: [entry for entry in bucketed_records if entry["bucket"] == bucket_name]
        for bucket_name in ["far", "mid", "near"]
    }

    summary = {
        "pair_info_file": args.pair_info_file,
        "bucket_definition": "source-similarity-quantiles",
        "quantiles": {
            "low_quantile": float(args.low_quantile),
            "high_quantile": float(args.high_quantile),
        },
        "bucket_thresholds": {
            "far_upper": low_threshold,
            "mid_lower": low_threshold,
            "mid_upper": high_threshold,
            "near_lower": high_threshold,
        },
        "num_pairs_total": len(pair_records),
        "buckets": {
            bucket_name: build_bucket_summary(entries, bucket_name, args.topk)
            for bucket_name, entries in buckets.items()
        },
    }

    output_path = args.output
    if output_path is None:
        base = os.path.splitext(args.pair_info_file)[0]
        low_label = int(round(args.low_quantile * 100))
        high_label = int(round(args.high_quantile * 100))
        output_path = f"{base}_pair_buckets_q{low_label}_{high_label}.json"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    plot_files = None
    if args.save_plots:
        artifact_prefix = os.path.splitext(output_path)[0]
        plot_files = maybe_save_plots(summary, source_values, artifact_prefix)
        summary["plot_files"] = plot_files
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)

    print(f"Saved pair-bucket summary to: {output_path}")
    print(f"Pair count: {len(pair_records)}")
    print(f"Bucket thresholds: far<={low_threshold:.6f}, near>={high_threshold:.6f}")
    for bucket_name in ["far", "mid", "near"]:
        bucket = summary["buckets"][bucket_name]
        print(
            f"{bucket_name}: num_pairs={bucket['num_pairs']}, "
            f"mean_delta_similarity={bucket.get('mean_delta_similarity', float('nan')):.6f}, "
            f"mean_abs_diff={bucket.get('mean_abs_diff', float('nan')):.6f}"
        )
    if plot_files:
        print(f"Saved plots: {plot_files}")


if __name__ == "__main__":
    main()
