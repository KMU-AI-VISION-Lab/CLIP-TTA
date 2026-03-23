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


def find_radius_summaries(output_root, low_quantile, high_quantile):
    low_label = int(round(low_quantile * 100))
    high_label = int(round(high_quantile * 100))
    pattern = os.path.join(output_root, "**", f"*radius_buckets_q{low_label}_{high_label}.json")
    return sorted(glob.glob(pattern, recursive=True))


def parse_dataset_pair_name(path):
    base = os.path.basename(path)
    return base.split("_inter_class_per_class_radius_")[0]


def format_class(entry):
    return (
        f"{entry['class_name']} ({entry['class_id']}) | "
        f"source_radius={entry['source_radius']:.4f}, "
        f"target_radius={entry['target_radius']:.4f}, "
        f"delta_radius={entry['delta_radius']:.4f}"
    )


def build_section(summary_path):
    payload = load_json(summary_path)
    lines = []
    lines.append(f"## {parse_dataset_pair_name(summary_path)}")
    lines.append("")
    lines.append(f"- bucket_definition: {payload['bucket_definition']}")
    lines.append(
        "- thresholds: "
        f"inner <= {payload['bucket_thresholds']['inner_upper']:.4f}, "
        f"outer >= {payload['bucket_thresholds']['outer_lower']:.4f}"
    )
    lines.append(f"- num_classes_total: {payload['num_classes_total']}")
    lines.append(f"- mean_delta_radius_overall: {payload['mean_delta_radius_overall']:.4f}")
    lines.append(f"- mean_abs_delta_radius_overall: {payload['mean_abs_delta_radius_overall']:.4f}")
    lines.append("")

    for bucket_name in ["inner", "mid", "outer"]:
        bucket = payload["buckets"][bucket_name]
        lines.append(
            f"- {bucket_name}: "
            f"classes={bucket['num_classes']}, "
            f"mean_source_radius={bucket.get('mean_source_radius', float('nan')):.4f}, "
            f"mean_target_radius={bucket.get('mean_target_radius', float('nan')):.4f}, "
            f"mean_delta_radius={bucket.get('mean_delta_radius', float('nan')):.4f}, "
            f"mean_abs_delta_radius={bucket.get('mean_abs_delta_radius', float('nan')):.4f}"
        )
    lines.append("")

    for bucket_name in ["inner", "mid", "outer"]:
        bucket = payload["buckets"][bucket_name]
        representatives = bucket.get("representative_classes", [])
        if representatives:
            lines.append(f"{bucket_name.title()} representative classes:")
            for entry in representatives[:3]:
                lines.append(f"- {format_class(entry)}")
            lines.append("")

    for bucket_name in ["inner", "mid", "outer"]:
        bucket = payload["buckets"][bucket_name]
        increases = bucket.get("largest_radius_increase_classes", [])
        decreases = bucket.get("largest_radius_decrease_classes", [])
        if increases:
            lines.append(f"{bucket_name.title()} top radius increase:")
            lines.append(f"- {format_class(increases[0])}")
        if decreases:
            lines.append(f"{bucket_name.title()} top radius decrease:")
            lines.append(f"- {format_class(decreases[0])}")
        if increases or decreases:
            lines.append("")

    return "\n".join(lines)


def main():
    args = get_arguments()
    summary_files = find_radius_summaries(args.output_root, args.low_quantile, args.high_quantile)
    if not summary_files:
        raise FileNotFoundError("No radius-bucket summary JSON files found. Run the radius-summary step first.")

    output_path = args.output
    if output_path is None:
        low_label = int(round(args.low_quantile * 100))
        high_label = int(round(args.high_quantile * 100))
        output_path = os.path.join(args.output_root, f"radius_bucket_report_q{low_label}_{high_label}.md")

    lines = [
        "# Radius Bucket Report",
        "",
        "This report tests whether classes near the global center move further inward and classes far from the center move further outward.",
        "",
        "Bucket rule:",
        f"- inner: source radius <= {int(round(args.low_quantile * 100))}th percentile",
        f"- mid: between the two thresholds",
        f"- outer: source radius >= {int(round(args.high_quantile * 100))}th percentile",
        "",
    ]

    for summary_path in summary_files:
        lines.append(build_section(summary_path))

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).strip() + "\n")

    print(f"Saved radius-bucket report to: {output_path}")


if __name__ == "__main__":
    main()
