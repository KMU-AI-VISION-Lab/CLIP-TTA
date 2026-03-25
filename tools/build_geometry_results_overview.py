import argparse
import glob
import json
import math
import os


def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_root", default="./outputs/geometry_vit_b16", type=str)
    parser.add_argument("--output", default=None, type=str)
    return parser.parse_args()


def load_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def find_result_files(output_root):
    # ncloud 정리본은 `main_results/` 아래에 있고,
    # 로컬에서는 flat layout일 수도 있으므로 둘 다 지원합니다.
    candidates = []
    for pattern in [
        os.path.join(output_root, "main_results", "*.json"),
        os.path.join(output_root, "*.json"),
    ]:
        candidates.extend(glob.glob(pattern))

    files = []
    seen = set()
    for path in sorted(candidates):
        base = os.path.basename(path)
        if path in seen:
            continue
        seen.add(path)
        # 요약 대상은 "메인 결과 JSON"만입니다.
        # pair/radius/detail/viz summary JSON은 제외합니다.
        if any(
            token in base
            for token in [
                "_inter_class_per_",
                "_viz_metrics.json",
                "_summary.json",
                "_pair_buckets_",
                "_radius_buckets_",
            ]
        ):
            continue
        files.append(path)
    return files


def fmt(value, digits=4):
    if value is None:
        return "-"
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float)):
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return "NaN"
        return f"{value:.{digits}f}"
    return str(value)


def safe_get(mapping, *keys, default=None):
    current = mapping
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def build_cross_dataset_table(rows):
    lines = [
        "## Cross-Dataset Summary",
        "",
        "| Comparison | nc | spc | Retrieval Cos | Retrieval Euc | Retrieval PCA | Same Centroid Cos | Diff Centroid Cos | Same PCA | Diff PCA | Hist JS | kNN WD (k=5) | Inter-Class Pearson | Inter-Class MAD | Radius Corr |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["comparison"],
                    str(row["num_classes"]),
                    str(row["samples_per_class"]),
                    fmt(row["retrieval_cos"]),
                    fmt(row["retrieval_euc"]),
                    fmt(row["retrieval_pca"]),
                    fmt(row["same_centroid_cos"]),
                    fmt(row["diff_centroid_cos"]),
                    fmt(row["same_pca"]),
                    fmt(row["diff_pca"]),
                    fmt(row["hist_js"]),
                    fmt(row["knn_wasserstein_k5"]),
                    fmt(row["inter_pearson"]),
                    fmt(row["inter_mad"]),
                    fmt(row["radius_corr"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def build_upper_bound_table(rows):
    if not rows:
        return []
    lines = [
        "## Same-Dataset Upper Bound Summary",
        "",
        "| Dataset | nc | spc | repeats | Retrieval Cos (mean±std) | Retrieval Euc (mean±std) | Retrieval PCA (mean±std) | Same Centroid Cos (mean±std) | Hist JS (mean±std) | kNN WD (k=5 mean±std) |",
        "| --- | ---: | ---: | ---: | --- | --- | --- | --- | --- | --- |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["dataset"],
                    str(row["num_classes"]),
                    str(row["samples_per_class"]),
                    str(row["repeats"]),
                    row["retrieval_cos"],
                    row["retrieval_euc"],
                    row["retrieval_pca"],
                    row["same_centroid_cos"],
                    row["hist_js"],
                    row["knn_wasserstein_k5"],
                ]
            )
            + " |"
        )
    lines.append("")
    return lines


def collect_cross_dataset_rows(paths):
    rows = []
    for path in paths:
        payload = load_json(path)
        if payload.get("mode") != "cross_dataset":
            continue
        averages = payload.get("averages", {})
        retrieval = payload.get("retrieval", {})
        inter_class = payload.get("inter_class_geometry", {})
        rows.append(
            {
                "comparison": f"{payload.get('source_dataset', '?')} vs {payload.get('target_dataset', '?')}",
                "num_classes": payload.get("num_classes", "-"),
                "samples_per_class": payload.get("samples_per_class", "-"),
                "retrieval_cos": safe_get(retrieval, "centroid_cosine"),
                "retrieval_euc": safe_get(retrieval, "centroid_euclidean"),
                "retrieval_pca": safe_get(retrieval, "pca_subspace"),
                "same_centroid_cos": safe_get(averages, "same_class", "centroid_cosine"),
                "diff_centroid_cos": safe_get(averages, "different_class_control", "centroid_cosine"),
                "same_pca": safe_get(averages, "same_class", "pca_subspace"),
                "diff_pca": safe_get(averages, "different_class_control", "pca_subspace"),
                "hist_js": safe_get(averages, "distance_histogram_js"),
                "knn_wasserstein_k5": safe_get(averages, "knn_wasserstein", "k=5"),
                "inter_pearson": safe_get(inter_class, "pearson_corr"),
                "inter_mad": safe_get(inter_class, "mean_abs_diff"),
                "radius_corr": safe_get(inter_class, "radius_corr"),
                "path": path,
            }
        )
    rows.sort(key=lambda item: item["comparison"])
    return rows


def mean_std_text(mean_value, std_value):
    return f"{fmt(mean_value)} ± {fmt(std_value)}"


def collect_upper_bound_rows(paths):
    rows = []
    for path in paths:
        payload = load_json(path)
        if payload.get("mode") != "same_dataset_upper_bound":
            continue
        summary = payload.get("summary", {})
        retrieval = summary.get("retrieval", {})
        rows.append(
            {
                "dataset": payload.get("source_dataset", "?"),
                "num_classes": payload.get("num_classes", "-"),
                "samples_per_class": payload.get("samples_per_class", "-"),
                "repeats": payload.get("upper_bound_num_repeats", "-"),
                "retrieval_cos": mean_std_text(
                    safe_get(retrieval, "centroid_cosine", "mean"),
                    safe_get(retrieval, "centroid_cosine", "std"),
                ),
                "retrieval_euc": mean_std_text(
                    safe_get(retrieval, "centroid_euclidean", "mean"),
                    safe_get(retrieval, "centroid_euclidean", "std"),
                ),
                "retrieval_pca": mean_std_text(
                    safe_get(retrieval, "pca_subspace", "mean"),
                    safe_get(retrieval, "pca_subspace", "std"),
                ),
                "same_centroid_cos": mean_std_text(
                    safe_get(summary, "same_class_mean", "centroid_cosine"),
                    safe_get(summary, "same_class_std", "centroid_cosine"),
                ),
                "hist_js": mean_std_text(
                    safe_get(summary, "distance_histogram_js", "mean"),
                    safe_get(summary, "distance_histogram_js", "std"),
                ),
                "knn_wasserstein_k5": mean_std_text(
                    safe_get(summary, "knn_wasserstein", "k=5", "mean"),
                    safe_get(summary, "knn_wasserstein", "k=5", "std"),
                ),
                "path": path,
            }
        )
    rows.sort(key=lambda item: item["dataset"])
    return rows


def build_metric_guide():
    return [
        "## Metric Guide",
        "",
        "- `same_class.centroid_cosine`: 같은 클래스 prototype끼리 중심 방향이 얼마나 비슷한지. 높을수록 좋습니다.",
        "- `different_class_control.centroid_cosine`: 다른 클래스끼리 중심 방향이 얼마나 비슷한지. 낮을수록 클래스 구분이 잘 됩니다.",
        "- `same_class.centroid_euclidean`: 같은 클래스 prototype 중심 간 거리. 낮을수록 좋습니다.",
        "- `different_class_control.centroid_euclidean`: 다른 클래스 prototype 중심 간 거리. 높을수록 클래스가 잘 분리됩니다.",
        "- `cov_frobenius`: 클래스 내부 분포 모양(공분산)이 얼마나 비슷한지 보는 차이값. 낮을수록 shape 보존이 좋습니다.",
        "- `pca_subspace`: 클래스 내부 주성분 방향이 얼마나 겹치는지. 높을수록 orientation 보존이 좋습니다.",
        "- `pairwise_spearman` / `pairwise_pearson`: 샘플 간 거리 관계가 얼마나 유지되는지 보는 보조 지표입니다.",
        "- `retrieval.centroid_cosine` / `retrieval.centroid_euclidean` / `retrieval.pca_subspace`: target 클래스가 source에서 자기 짝을 얼마나 잘 찾아오는지. 높을수록 좋습니다.",
        "- `knn_wasserstein`: 최근접 이웃 거리 분포 차이. 낮을수록 local geometry가 비슷합니다.",
        "- `knn_mean_diff`: kNN 평균 거리 차이. 낮을수록 좋습니다.",
        "- `knn_std_diff`: kNN 거리의 퍼짐 정도 차이. 낮을수록 좋습니다.",
        "- `distance_histogram_js`: 전체 pairwise distance 분포 차이. 낮을수록 global geometry가 비슷합니다.",
        "- `distance_histogram_wasserstein`: 전체 거리 분포를 Wasserstein으로 비교한 값. 낮을수록 좋습니다.",
        "- `neighbor_graph_stats`: kNN graph 기준 local 구조 통계입니다. `abs_diff_mean`이 낮을수록 source/target이 비슷합니다.",
        "- `inter_class_geometry.pearson_corr`: centered class prototype similarity matrix가 얼마나 비슷한지. 높을수록 좋습니다.",
        "- `inter_class_geometry.mean_abs_diff`: inter-class similarity matrix 평균 차이. 낮을수록 좋습니다.",
        "- `inter_class_geometry.radius_corr`: 클래스별 centered prototype norm(반지름) 순서가 얼마나 유지되는지. 높을수록 좋습니다.",
        "- `inter_class_geometry.global_prototype_shift_l2`: source 전체 중심 `m_0`와 target 전체 중심 `m'_0` 사이의 L2 거리입니다. 낮을수록 두 dataset의 전역 중심이 비슷합니다.",
        "- `inter_class_geometry.global_prototype_shift_cosine`: source/target 전체 중심 벡터 방향이 얼마나 비슷한지 보는 cosine입니다. 높을수록 전역 방향이 비슷합니다.",
        "",
    ]


def build_notes(cross_rows):
    if not cross_rows:
        return []
    # 사람이 제일 먼저 볼 만한 포인트를 몇 줄로 요약합니다.
    best_retrieval = max(cross_rows, key=lambda item: (-1 if item["retrieval_cos"] is None else item["retrieval_cos"]))
    worst_hist = max(cross_rows, key=lambda item: (-1 if item["hist_js"] is None else item["hist_js"]))
    best_inter = max(cross_rows, key=lambda item: (-1 if item["inter_pearson"] is None else item["inter_pearson"]))
    lines = [
        "## Quick Takeaways",
        "",
        f"- Highest centroid-cosine retrieval: `{best_retrieval['comparison']}` ({fmt(best_retrieval['retrieval_cos'])})",
        f"- Highest distance-histogram JS divergence: `{worst_hist['comparison']}` ({fmt(worst_hist['hist_js'])})",
        f"- Highest inter-class Pearson correlation: `{best_inter['comparison']}` ({fmt(best_inter['inter_pearson'])})",
        "",
    ]
    return lines


def build_averages_only_sections(result_files):
    lines = [
        "## Averages-Only Tables",
        "",
        "아래 표는 각 결과 JSON의 `averages`와 `inter_class_geometry`에서 핵심 값만 뽑은 것입니다.",
        "",
    ]

    for path in result_files:
        payload = load_json(path)
        base = os.path.basename(path)
        lines.append(f"### {base}")
        lines.append("")
        lines.append(f"- mode: `{payload.get('mode', '-')}`")
        lines.append(f"- source_dataset: `{payload.get('source_dataset', '-')}`")
        lines.append(f"- target_dataset: `{payload.get('target_dataset', '-')}`")
        lines.append(f"- backbone: `{payload.get('backbone', '-')}`")
        lines.append(f"- num_classes: `{payload.get('num_classes', '-')}`")
        lines.append(f"- samples_per_class: `{payload.get('samples_per_class', '-')}`")
        lines.append("")

        if payload.get("mode") == "cross_dataset":
            averages = payload.get("averages", {})
            inter_class = payload.get("inter_class_geometry", {})
            lines.extend(
                [
                    "| Category | Metric | Value |",
                    "| --- | --- | ---: |",
                    f"| same_class | centroid_cosine | {fmt(safe_get(averages, 'same_class', 'centroid_cosine'))} |",
                    f"| same_class | centroid_euclidean | {fmt(safe_get(averages, 'same_class', 'centroid_euclidean'))} |",
                    f"| same_class | cov_frobenius | {fmt(safe_get(averages, 'same_class', 'cov_frobenius'))} |",
                    f"| same_class | pca_subspace | {fmt(safe_get(averages, 'same_class', 'pca_subspace'))} |",
                    f"| same_class | pairwise_spearman | {fmt(safe_get(averages, 'same_class', 'pairwise_spearman'))} |",
                    f"| same_class | pairwise_pearson | {fmt(safe_get(averages, 'same_class', 'pairwise_pearson'))} |",
                    f"| different_class_control | centroid_cosine | {fmt(safe_get(averages, 'different_class_control', 'centroid_cosine'))} |",
                    f"| different_class_control | centroid_euclidean | {fmt(safe_get(averages, 'different_class_control', 'centroid_euclidean'))} |",
                    f"| different_class_control | cov_frobenius | {fmt(safe_get(averages, 'different_class_control', 'cov_frobenius'))} |",
                    f"| different_class_control | pca_subspace | {fmt(safe_get(averages, 'different_class_control', 'pca_subspace'))} |",
                    f"| different_class_control | pairwise_spearman | {fmt(safe_get(averages, 'different_class_control', 'pairwise_spearman'))} |",
                    f"| different_class_control | pairwise_pearson | {fmt(safe_get(averages, 'different_class_control', 'pairwise_pearson'))} |",
                    f"| topology | knn_wasserstein (k=5) | {fmt(safe_get(averages, 'knn_wasserstein', 'k=5'))} |",
                    f"| topology | knn_mean_diff (k=5) | {fmt(safe_get(averages, 'knn_mean_diff', 'k=5'))} |",
                    f"| topology | knn_std_diff (k=5) | {fmt(safe_get(averages, 'knn_std_diff', 'k=5'))} |",
                    f"| topology | distance_histogram_js | {fmt(safe_get(averages, 'distance_histogram_js'))} |",
                    f"| topology | distance_histogram_wasserstein | {fmt(safe_get(averages, 'distance_histogram_wasserstein'))} |",
                    f"| graph (k=5) | mean_knn_distance abs_diff_mean | {fmt(safe_get(averages, 'neighbor_graph_stats', 'k=5', 'mean_knn_distance', 'abs_diff_mean'))} |",
                    f"| graph (k=5) | std_knn_distance abs_diff_mean | {fmt(safe_get(averages, 'neighbor_graph_stats', 'k=5', 'std_knn_distance', 'abs_diff_mean'))} |",
                    f"| graph (k=5) | average_node_degree abs_diff_mean | {fmt(safe_get(averages, 'neighbor_graph_stats', 'k=5', 'average_node_degree', 'abs_diff_mean'))} |",
                    f"| inter_class | pearson_corr | {fmt(safe_get(inter_class, 'pearson_corr'))} |",
                    f"| inter_class | spearman_corr | {fmt(safe_get(inter_class, 'spearman_corr'))} |",
                    f"| inter_class | mean_abs_diff | {fmt(safe_get(inter_class, 'mean_abs_diff'))} |",
                    f"| inter_class | sign_consistency_rate | {fmt(safe_get(inter_class, 'sign_consistency_rate'))} |",
                    f"| inter_class | radius_corr | {fmt(safe_get(inter_class, 'radius_corr'))} |",
                    f"| inter_class | radius_mean_abs_diff | {fmt(safe_get(inter_class, 'radius_mean_abs_diff'))} |",
                    f"| inter_class | global_prototype_shift_l2 | {fmt(safe_get(inter_class, 'global_prototype_shift_l2'))} |",
                    f"| inter_class | global_prototype_shift_cosine | {fmt(safe_get(inter_class, 'global_prototype_shift_cosine'))} |",
                    "",
                ]
            )
        else:
            summary = payload.get("summary", {})
            lines.extend(
                [
                    "| Category | Metric | Mean | Std |",
                    "| --- | --- | ---: | ---: |",
                    f"| same_class | centroid_cosine | {fmt(safe_get(summary, 'same_class_mean', 'centroid_cosine'))} | {fmt(safe_get(summary, 'same_class_std', 'centroid_cosine'))} |",
                    f"| same_class | centroid_euclidean | {fmt(safe_get(summary, 'same_class_mean', 'centroid_euclidean'))} | {fmt(safe_get(summary, 'same_class_std', 'centroid_euclidean'))} |",
                    f"| same_class | cov_frobenius | {fmt(safe_get(summary, 'same_class_mean', 'cov_frobenius'))} | {fmt(safe_get(summary, 'same_class_std', 'cov_frobenius'))} |",
                    f"| same_class | pca_subspace | {fmt(safe_get(summary, 'same_class_mean', 'pca_subspace'))} | {fmt(safe_get(summary, 'same_class_std', 'pca_subspace'))} |",
                    f"| topology | knn_wasserstein (k=5) | {fmt(safe_get(summary, 'knn_wasserstein', 'k=5', 'mean'))} | {fmt(safe_get(summary, 'knn_wasserstein', 'k=5', 'std'))} |",
                    f"| topology | knn_mean_diff (k=5) | {fmt(safe_get(summary, 'knn_mean_diff', 'k=5', 'mean'))} | {fmt(safe_get(summary, 'knn_mean_diff', 'k=5', 'std'))} |",
                    f"| topology | knn_std_diff (k=5) | {fmt(safe_get(summary, 'knn_std_diff', 'k=5', 'mean'))} | {fmt(safe_get(summary, 'knn_std_diff', 'k=5', 'std'))} |",
                    f"| topology | distance_histogram_js | {fmt(safe_get(summary, 'distance_histogram_js', 'mean'))} | {fmt(safe_get(summary, 'distance_histogram_js', 'std'))} |",
                    f"| topology | distance_histogram_wasserstein | {fmt(safe_get(summary, 'distance_histogram_wasserstein', 'mean'))} | {fmt(safe_get(summary, 'distance_histogram_wasserstein', 'std'))} |",
                    "",
                ]
            )
        lines.append("---")
        lines.append("")
    return lines


def main():
    args = get_arguments()
    result_files = find_result_files(args.output_root)
    if not result_files:
        raise FileNotFoundError(f"No result JSON files found under {args.output_root}")

    cross_rows = collect_cross_dataset_rows(result_files)
    upper_rows = collect_upper_bound_rows(result_files)

    output_path = args.output
    if output_path is None:
        output_path = os.path.join(args.output_root, "geometry_results_overview.md")

    lines = [
        "# Geometry Results Overview",
        "",
        "This file summarizes only the core `averages` and `inter_class_geometry` fields from the main geometry-evaluation JSON results.",
        "",
    ]
    lines.extend(build_metric_guide())
    lines.extend(build_averages_only_sections(result_files))

    with open(output_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines).strip() + "\n")

    print(f"Saved geometry results overview to: {output_path}")


if __name__ == "__main__":
    main()
