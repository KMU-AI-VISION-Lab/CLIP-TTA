import argparse
import json
import os
from collections import defaultdict


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair_info_file", required=True, type=str)
    parser.add_argument("--mapping_file", default="./imagenet-class-ids.txt", type=str)
    parser.add_argument("--keywords", nargs="+", required=True, help="Examples: dog wolf frog")
    parser.add_argument("--match_mode", default="any", choices=["any", "both"])
    parser.add_argument("--topk", default=20, type=int)
    parser.add_argument("--output", default=None, type=str)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_mapping(path):
    mapping = {}
    with open(path, "r", encoding="utf-8") as handle:
        header = next(handle, None)
        for line in handle:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t", maxsplit=1)
            if len(parts) != 2:
                continue
            class_id, class_name = parts
            mapping[int(class_id)] = class_name
    return mapping


def normalize_text(text):
    return text.lower().strip()


def build_keyword_to_class_ids(mapping, keywords):
    keyword_to_ids = defaultdict(list)
    normalized_mapping = {class_id: normalize_text(class_name) for class_id, class_name in mapping.items()}
    for keyword in keywords:
        keyword_norm = normalize_text(keyword)
        for class_id, class_name in normalized_mapping.items():
            if keyword_norm in class_name:
                keyword_to_ids[keyword_norm].append(class_id)
    return dict(keyword_to_ids)


def pair_matches(entry, keyword_to_ids, match_mode):
    matched_keywords = set()
    class_a_id = entry["class_a_id"]
    class_b_id = entry["class_b_id"]
    for keyword, class_ids in keyword_to_ids.items():
        if class_a_id in class_ids or class_b_id in class_ids:
            matched_keywords.add(keyword)

    if match_mode == "any":
        return len(matched_keywords) > 0, sorted(matched_keywords)
    if match_mode == "both":
        return len(matched_keywords) >= 2 or (
            len(matched_keywords) == 1 and class_a_id in keyword_to_ids[next(iter(matched_keywords))] and class_b_id in keyword_to_ids[next(iter(matched_keywords))]
        ), sorted(matched_keywords)
    raise ValueError(f"Unsupported match_mode: {match_mode}")


def get_entry_keywords(entry, keyword_to_ids):
    class_a_id = entry["class_a_id"]
    class_b_id = entry["class_b_id"]
    class_a_keywords = sorted([keyword for keyword, class_ids in keyword_to_ids.items() if class_a_id in class_ids])
    class_b_keywords = sorted([keyword for keyword, class_ids in keyword_to_ids.items() if class_b_id in class_ids])
    return class_a_keywords, class_b_keywords


def summarize_pairs(filtered_pairs, topk):
    top_source = sorted(filtered_pairs, key=lambda item: item["source_similarity"], reverse=True)[:topk]
    top_target = sorted(filtered_pairs, key=lambda item: item["target_similarity"], reverse=True)[:topk]
    top_increase = sorted(filtered_pairs, key=lambda item: item["delta_similarity"], reverse=True)[:topk]
    top_decrease = sorted(filtered_pairs, key=lambda item: item["delta_similarity"])[:topk]
    top_abs_diff = sorted(filtered_pairs, key=lambda item: item["abs_diff"], reverse=True)[:topk]
    return {
        "top_source_similar_pairs": top_source,
        "top_target_similar_pairs": top_target,
        "largest_similarity_increase_pairs": top_increase,
        "largest_similarity_decrease_pairs": top_decrease,
        "largest_abs_diff_pairs": top_abs_diff,
    }


def mean_or_nan(values):
    return float(sum(values) / len(values)) if values else float("nan")


def build_group_bucket_summary(pairs, topk):
    return {
        "num_pairs": len(pairs),
        "mean_source_similarity": mean_or_nan([entry["source_similarity"] for entry in pairs]),
        "mean_target_similarity": mean_or_nan([entry["target_similarity"] for entry in pairs]),
        "mean_delta_similarity": mean_or_nan([entry["delta_similarity"] for entry in pairs]),
        "mean_abs_diff": mean_or_nan([entry["abs_diff"] for entry in pairs]),
        "top_source_similar_pairs": sorted(pairs, key=lambda item: item["source_similarity"], reverse=True)[:topk],
        "top_target_similar_pairs": sorted(pairs, key=lambda item: item["target_similarity"], reverse=True)[:topk],
        "largest_similarity_increase_pairs": sorted(pairs, key=lambda item: item["delta_similarity"], reverse=True)[:topk],
        "largest_similarity_decrease_pairs": sorted(pairs, key=lambda item: item["delta_similarity"])[:topk],
    }


def summarize_group_relations(filtered_pairs, keywords, topk):
    within_group = {keyword: [] for keyword in keywords}
    cross_group = {}
    group_pairwise = {}

    for keyword_a in keywords:
        for keyword_b in keywords:
            if keyword_a < keyword_b:
                cross_group[f"{keyword_a}__{keyword_b}"] = []

    for entry in filtered_pairs:
        class_a_keywords = entry["class_a_keywords"]
        class_b_keywords = entry["class_b_keywords"]
        shared_keywords = sorted(set(class_a_keywords).intersection(class_b_keywords))

        # within-group: 두 클래스가 같은 keyword 그룹에 모두 속하는 경우
        for keyword in shared_keywords:
            within_group[keyword].append(entry)

        # cross-group: class A와 class B가 서로 다른 keyword 그룹에 속하는 경우
        unique_cross_pairs = set()
        for keyword_a in class_a_keywords:
            for keyword_b in class_b_keywords:
                if keyword_a == keyword_b:
                    continue
                pair_key = "__".join(sorted([keyword_a, keyword_b]))
                unique_cross_pairs.add(pair_key)
        for pair_key in unique_cross_pairs:
            if pair_key in cross_group:
                cross_group[pair_key].append(entry)

    within_group_summary = {
        keyword: build_group_bucket_summary(pairs, topk)
        for keyword, pairs in within_group.items()
        if pairs
    }
    cross_group_summary = {
        pair_key: build_group_bucket_summary(pairs, topk)
        for pair_key, pairs in cross_group.items()
        if pairs
    }

    all_within_pairs = [entry for pairs in within_group.values() for entry in pairs]
    all_cross_pairs = [entry for pairs in cross_group.values() for entry in pairs]

    return {
        "within_group": within_group_summary,
        "cross_group": cross_group_summary,
        "overall_within_group": build_group_bucket_summary(all_within_pairs, topk) if all_within_pairs else {"num_pairs": 0},
        "overall_cross_group": build_group_bucket_summary(all_cross_pairs, topk) if all_cross_pairs else {"num_pairs": 0},
    }


def print_pair(label, entry, score_key):
    print(
        f"{label}: {entry['class_a_name']} ({entry['class_a_id']}) / "
        f"{entry['class_b_name']} ({entry['class_b_id']}) -> {score_key}={entry[score_key]:.6f}"
    )


def main():
    args = get_arguments()
    pair_info = load_json(args.pair_info_file)
    mapping = load_mapping(args.mapping_file)
    keywords = [normalize_text(keyword) for keyword in args.keywords]
    keyword_to_ids = build_keyword_to_class_ids(mapping, keywords)

    filtered_pairs = []
    for entry in pair_info:
        matches, matched_keywords = pair_matches(entry, keyword_to_ids, args.match_mode)
        if not matches:
            continue
        entry = dict(entry)
        entry["matched_keywords"] = matched_keywords
        class_a_keywords, class_b_keywords = get_entry_keywords(entry, keyword_to_ids)
        entry["class_a_keywords"] = class_a_keywords
        entry["class_b_keywords"] = class_b_keywords
        # Older pair files may not have delta_similarity yet.
        entry.setdefault("delta_similarity", float(entry["target_similarity"] - entry["source_similarity"]))
        filtered_pairs.append(entry)

    summary = {
        "pair_info_file": args.pair_info_file,
        "mapping_file": args.mapping_file,
        "keywords": keywords,
        "match_mode": args.match_mode,
        "keyword_to_class_ids": keyword_to_ids,
        "num_filtered_pairs": len(filtered_pairs),
        "pairs": filtered_pairs,
    }
    summary.update(summarize_pairs(filtered_pairs, args.topk))
    summary["group_relation_summary"] = summarize_group_relations(filtered_pairs, keywords, args.topk)

    output_path = args.output
    if output_path is None:
        base = os.path.splitext(args.pair_info_file)[0]
        keywords_suffix = "_".join(keywords)
        output_path = f"{base}_{keywords_suffix}_{args.match_mode}_summary.json"

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    print(f"Saved filtered summary to: {output_path}")
    print(f"Keywords: {', '.join(keywords)}")
    print(f"Match mode: {args.match_mode}")
    print(f"Filtered pairs: {len(filtered_pairs)}")
    for keyword, class_ids in keyword_to_ids.items():
        print(f"Classes matching '{keyword}': {len(class_ids)}")

    if summary["top_source_similar_pairs"]:
        print_pair("Most similar in source", summary["top_source_similar_pairs"][0], "source_similarity")
    if summary["top_target_similar_pairs"]:
        print_pair("Most similar in target", summary["top_target_similar_pairs"][0], "target_similarity")
    if summary["largest_similarity_increase_pairs"]:
        print_pair("Largest similarity increase", summary["largest_similarity_increase_pairs"][0], "delta_similarity")
    if summary["largest_similarity_decrease_pairs"]:
        print_pair("Largest similarity decrease", summary["largest_similarity_decrease_pairs"][0], "delta_similarity")

    within_summary = summary["group_relation_summary"]["overall_within_group"]
    cross_summary = summary["group_relation_summary"]["overall_cross_group"]
    if within_summary.get("num_pairs", 0) > 0:
        print(
            "Overall within-group: "
            f"pairs={within_summary['num_pairs']}, "
            f"mean_delta_similarity={within_summary['mean_delta_similarity']:.6f}, "
            f"mean_abs_diff={within_summary['mean_abs_diff']:.6f}"
        )
    if cross_summary.get("num_pairs", 0) > 0:
        print(
            "Overall cross-group: "
            f"pairs={cross_summary['num_pairs']}, "
            f"mean_delta_similarity={cross_summary['mean_delta_similarity']:.6f}, "
            f"mean_abs_diff={cross_summary['mean_abs_diff']:.6f}"
        )


if __name__ == "__main__":
    main()
