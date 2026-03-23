import argparse
import glob
import json
import os


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="./outputs/geometry_vit_b16", type=str)
    parser.add_argument("--output", default=None, type=str)
    parser.add_argument("--low_quantile", default=0.2, type=float)
    parser.add_argument("--high_quantile", default=0.8, type=float)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def find_bucket_summaries(output_root, low_quantile, high_quantile):
    low_label = int(round(low_quantile * 100))
    high_label = int(round(high_quantile * 100))
    pattern = os.path.join(output_root, "**", f"*pair_buckets_q{low_label}_{high_label}.json")
    return sorted(glob.glob(pattern, recursive=True))


def format_pair(entry):
    return (
        f"{entry['class_a_name']} ({entry['class_a_id']}) / "
        f"{entry['class_b_name']} ({entry['class_b_id']}) | "
        f"source={entry['source_similarity']:.4f}, "
        f"target={entry['target_similarity']:.4f}, "
        f"delta={entry['delta_similarity']:.4f}"
    )


def parse_dataset_pair_name(path):
    base = os.path.basename(path)
    return base.split("_inter_class_per_pair_sign_info_")[0]


def build_section(summary_path):
    payload = load_json(summary_path)
    lines = []
    lines.append(f"## {parse_dataset_pair_name(summary_path)}")
    lines.append("")
    lines.append(f"- bucket_definition: {payload['bucket_definition']}")
    lines.append(
        "- thresholds: "
        f"far <= {payload['bucket_thresholds']['far_upper']:.4f}, "
        f"near >= {payload['bucket_thresholds']['near_lower']:.4f}"
    )
    lines.append(f"- num_pairs_total: {payload['num_pairs_total']}")
    lines.append("")

    for bucket_name in ["far", "mid", "near"]:
        bucket = payload["buckets"][bucket_name]
        lines.append(
            f"- {bucket_name}: "
            f"pairs={bucket['num_pairs']}, "
            f"mean_source_similarity={bucket.get('mean_source_similarity', float('nan')):.4f}, "
            f"mean_target_similarity={bucket.get('mean_target_similarity', float('nan')):.4f}, "
            f"mean_delta_similarity={bucket.get('mean_delta_similarity', float('nan')):.4f}, "
            f"mean_abs_diff={bucket.get('mean_abs_diff', float('nan')):.4f}"
        )
    lines.append("")

    for bucket_name in ["far", "mid", "near"]:
        bucket = payload["buckets"][bucket_name]
        representative = bucket.get("representative_pairs", [])
        if representative:
            lines.append(f"{bucket_name.title()} representative pairs:")
            for entry in representative[:3]:
                lines.append(f"- {format_pair(entry)}")
            lines.append("")

    for bucket_name in ["far", "mid", "near"]:
        bucket = payload["buckets"][bucket_name]
        increases = bucket.get("largest_similarity_increase_pairs", [])
        decreases = bucket.get("largest_similarity_decrease_pairs", [])
        if increases:
            lines.append(f"{bucket_name.title()} top increase:")
            lines.append(f"- {format_pair(increases[0])}")
        if decreases:
            lines.append(f"{bucket_name.title()} top decrease:")
            lines.append(f"- {format_pair(decreases[0])}")
        if increases or decreases:
            lines.append("")

    return "\n".join(lines)


def main():
    args = get_arguments()
    summary_files = find_bucket_summaries(args.output_root, args.low_quantile, args.high_quantile)
    if not summary_files:
        raise FileNotFoundError("No pair-bucket summary JSON files found. Run the bucket-summary step first.")

    output_path = args.output
    if output_path is None:
        low_label = int(round(args.low_quantile * 100))
        high_label = int(round(args.high_quantile * 100))
        output_path = os.path.join(args.output_root, f"pair_bucket_report_q{low_label}_{high_label}.md")

    lines = [
        "# Pair Bucket Hypothesis Report",
        "",
        "This report tests whether originally similar class pairs become closer and originally dissimilar class pairs become farther apart.",
        "",
        "Bucket rule:",
        f"- far: source similarity <= {int(round(args.low_quantile * 100))}th percentile",
        f"- mid: between the two thresholds",
        f"- near: source similarity >= {int(round(args.high_quantile * 100))}th percentile",
        "",
    ]
    for summary_path in summary_files:
        lines.append(build_section(summary_path))

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).strip() + "\n")

    print(f"Saved pair-bucket report to: {output_path}")


if __name__ == "__main__":
    main()
