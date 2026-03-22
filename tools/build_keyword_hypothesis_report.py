import argparse
import glob
import json
import os


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="./outputs/geometry_vit_b16", type=str)
    parser.add_argument("--keywords", nargs="+", default=["dog", "wolf", "frog"])
    parser.add_argument("--match_mode", default="any", choices=["any", "both"])
    parser.add_argument("--output", default=None, type=str)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def find_summary_files(output_root, keywords, match_mode):
    keyword_suffix = "_".join(keyword.lower() for keyword in keywords)
    pattern = os.path.join(
        output_root,
        "**",
        f"*per_pair_sign_info_{keyword_suffix}_{match_mode}_summary.json",
    )
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
    group_summary = payload["group_relation_summary"]
    lines = []

    title = parse_dataset_pair_name(summary_path)
    lines.append(f"## {title}")
    lines.append("")
    lines.append(f"- keywords: {', '.join(payload['keywords'])}")
    lines.append(f"- match_mode: {payload['match_mode']}")
    lines.append(f"- filtered_pairs: {payload['num_filtered_pairs']}")

    overall_within = group_summary.get("overall_within_group", {})
    overall_cross = group_summary.get("overall_cross_group", {})
    if overall_within.get("num_pairs", 0) > 0:
        lines.append(
            "- overall within-group: "
            f"pairs={overall_within['num_pairs']}, "
            f"mean_delta_similarity={overall_within['mean_delta_similarity']:.4f}, "
            f"mean_abs_diff={overall_within['mean_abs_diff']:.4f}"
        )
    if overall_cross.get("num_pairs", 0) > 0:
        lines.append(
            "- overall cross-group: "
            f"pairs={overall_cross['num_pairs']}, "
            f"mean_delta_similarity={overall_cross['mean_delta_similarity']:.4f}, "
            f"mean_abs_diff={overall_cross['mean_abs_diff']:.4f}"
        )
    lines.append("")

    if payload["largest_similarity_increase_pairs"]:
        lines.append("Top increase:")
        lines.append(f"- {format_pair(payload['largest_similarity_increase_pairs'][0])}")
    if payload["largest_similarity_decrease_pairs"]:
        lines.append("Top decrease:")
        lines.append(f"- {format_pair(payload['largest_similarity_decrease_pairs'][0])}")
    lines.append("")

    within_group = group_summary.get("within_group", {})
    if within_group:
        lines.append("Within-group summary:")
        for key, bucket in sorted(within_group.items()):
            lines.append(
                f"- {key}: pairs={bucket['num_pairs']}, "
                f"mean_delta_similarity={bucket['mean_delta_similarity']:.4f}, "
                f"mean_abs_diff={bucket['mean_abs_diff']:.4f}"
            )
        lines.append("")

    cross_group = group_summary.get("cross_group", {})
    if cross_group:
        lines.append("Cross-group summary:")
        for key, bucket in sorted(cross_group.items()):
            lines.append(
                f"- {key}: pairs={bucket['num_pairs']}, "
                f"mean_delta_similarity={bucket['mean_delta_similarity']:.4f}, "
                f"mean_abs_diff={bucket['mean_abs_diff']:.4f}"
            )
        lines.append("")

    return "\n".join(lines)


def main():
    args = get_arguments()
    summary_files = find_summary_files(args.output_root, args.keywords, args.match_mode)
    if not summary_files:
        raise FileNotFoundError("No keyword summary JSON files found. Run the keyword-summary step first.")

    output_path = args.output
    if output_path is None:
        keyword_suffix = "_".join(keyword.lower() for keyword in args.keywords)
        output_path = os.path.join(args.output_root, f"{keyword_suffix}_{args.match_mode}_hypothesis_report.md")

    lines = [
        "# Keyword Hypothesis Report",
        "",
        "This report focuses on within-group and cross-group relational changes for selected keywords.",
        "",
    ]

    for summary_path in summary_files:
        lines.append(build_section(summary_path))

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).strip() + "\n")

    print(f"Saved keyword hypothesis report to: {output_path}")


if __name__ == "__main__":
    main()
