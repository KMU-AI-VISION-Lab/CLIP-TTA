# Geometry Results Overview

This file summarizes only the core `averages` and `inter_class_geometry` fields from the main geometry-evaluation JSON results.

## Metric Guide

- `same_class.centroid_cosine`: 같은 클래스 prototype끼리 중심 방향이 얼마나 비슷한지. 높을수록 좋습니다.
- `different_class_control.centroid_cosine`: 다른 클래스끼리 중심 방향이 얼마나 비슷한지. 낮을수록 클래스 구분이 잘 됩니다.
- `same_class.centroid_euclidean`: 같은 클래스 prototype 중심 간 거리. 낮을수록 좋습니다.
- `different_class_control.centroid_euclidean`: 다른 클래스 prototype 중심 간 거리. 높을수록 클래스가 잘 분리됩니다.
- `cov_frobenius`: 클래스 내부 분포 모양(공분산)이 얼마나 비슷한지 보는 차이값. 낮을수록 shape 보존이 좋습니다.
- `pca_subspace`: 클래스 내부 주성분 방향이 얼마나 겹치는지. 높을수록 orientation 보존이 좋습니다.
- `pairwise_spearman` / `pairwise_pearson`: 샘플 간 거리 관계가 얼마나 유지되는지 보는 보조 지표입니다.
- `retrieval.centroid_cosine` / `retrieval.centroid_euclidean` / `retrieval.pca_subspace`: target 클래스가 source에서 자기 짝을 얼마나 잘 찾아오는지. 높을수록 좋습니다.
- `knn_wasserstein`: 최근접 이웃 거리 분포 차이. 낮을수록 local geometry가 비슷합니다.
- `knn_mean_diff`: kNN 평균 거리 차이. 낮을수록 좋습니다.
- `knn_std_diff`: kNN 거리의 퍼짐 정도 차이. 낮을수록 좋습니다.
- `distance_histogram_js`: 전체 pairwise distance 분포 차이. 낮을수록 global geometry가 비슷합니다.
- `distance_histogram_wasserstein`: 전체 거리 분포를 Wasserstein으로 비교한 값. 낮을수록 좋습니다.
- `neighbor_graph_stats`: kNN graph 기준 local 구조 통계입니다. `abs_diff_mean`이 낮을수록 source/target이 비슷합니다.
- `inter_class_geometry.pearson_corr`: centered class prototype similarity matrix가 얼마나 비슷한지. 높을수록 좋습니다.
- `inter_class_geometry.mean_abs_diff`: inter-class similarity matrix 평균 차이. 낮을수록 좋습니다.
- `inter_class_geometry.radius_corr`: 클래스별 centered prototype norm(반지름) 순서가 얼마나 유지되는지. 높을수록 좋습니다.
- `inter_class_geometry.global_prototype_shift_l2`: source 전체 중심 `m_0`와 target 전체 중심 `m'_0` 사이의 L2 거리입니다. 낮을수록 두 dataset의 전역 중심이 비슷합니다.
- `inter_class_geometry.global_prototype_shift_cosine`: source/target 전체 중심 벡터 방향이 얼마나 비슷한지 보는 cosine입니다. 높을수록 전역 방향이 비슷합니다.

## Averages-Only Tables

아래 표는 각 결과 JSON의 `averages`와 `inter_class_geometry`에서 핵심 값만 뽑은 것입니다.

### ImageNet_v1_vs_ImageNet_R_nc200_sc10_knn5_10_seed1.json

- mode: `cross_dataset`
- source_dataset: `ImageNet_v1`
- target_dataset: `ImageNet_R`
- backbone: `vit_b16`
- num_classes: `200`
- samples_per_class: `10`

| Category | Metric | Value |
| --- | --- | ---: |
| same_class | centroid_cosine | 0.8567 |
| same_class | centroid_euclidean | 0.4585 |
| same_class | cov_frobenius | 0.1574 |
| same_class | pca_subspace | 0.2085 |
| same_class | pairwise_spearman | 0.0155 |
| same_class | pairwise_pearson | 0.0107 |
| different_class_control | centroid_cosine | 0.6865 |
| different_class_control | centroid_euclidean | 0.6775 |
| different_class_control | cov_frobenius | 0.1600 |
| different_class_control | pca_subspace | 0.1723 |
| different_class_control | pairwise_spearman | 0.0007 |
| different_class_control | pairwise_pearson | 0.0006 |
| topology | knn_wasserstein (k=5) | 0.1462 |
| topology | knn_mean_diff (k=5) | 0.1415 |
| topology | knn_std_diff (k=5) | 0.0299 |
| topology | distance_histogram_js | 0.3053 |
| topology | distance_histogram_wasserstein | 0.1473 |
| graph (k=5) | mean_knn_distance abs_diff_mean | 0.1415 |
| graph (k=5) | std_knn_distance abs_diff_mean | 0.0299 |
| graph (k=5) | average_node_degree abs_diff_mean | 0.2830 |
| inter_class | pearson_corr | 0.8116 |
| inter_class | spearman_corr | 0.7228 |
| inter_class | mean_abs_diff | 0.0903 |
| inter_class | sign_consistency_rate | 0.8138 |
| inter_class | radius_corr | 0.4165 |
| inter_class | radius_mean_abs_diff | 0.1086 |
| inter_class | global_prototype_shift_l2 | 0.2622 |
| inter_class | global_prototype_shift_cosine | 0.9360 |

---

### ImageNet_v1_vs_ImageNet_R_nc200_sc10_knn5_10_seed1_global_prototype_shift.json

- mode: `-`
- source_dataset: `ImageNet_v1`
- target_dataset: `ImageNet_R`
- backbone: `vit_b16`
- num_classes: `200`
- samples_per_class: `10`

| Category | Metric | Mean | Std |
| --- | --- | ---: | ---: |
| same_class | centroid_cosine | - | - |
| same_class | centroid_euclidean | - | - |
| same_class | cov_frobenius | - | - |
| same_class | pca_subspace | - | - |
| topology | knn_wasserstein (k=5) | - | - |
| topology | knn_mean_diff (k=5) | - | - |
| topology | knn_std_diff (k=5) | - | - |
| topology | distance_histogram_js | - | - |
| topology | distance_histogram_wasserstein | - | - |

---

### ImageNet_Sketch_split_half_upper_bound_nc200_sc10_rep3_knn5_10_seed1.json

- mode: `same_dataset_upper_bound`
- source_dataset: `ImageNet_Sketch_split_A`
- target_dataset: `ImageNet_Sketch_split_B`
- backbone: `vit_b16`
- num_classes: `200`
- samples_per_class: `10`

| Category | Metric | Mean | Std |
| --- | --- | ---: | ---: |
| same_class | centroid_cosine | 0.9697 | 0.0008 |
| same_class | centroid_euclidean | 0.2131 | 0.0027 |
| same_class | cov_frobenius | 0.1157 | 0.0003 |
| same_class | pca_subspace | 0.4098 | 0.0033 |
| topology | knn_wasserstein (k=5) | - | - |
| topology | knn_mean_diff (k=5) | - | - |
| topology | knn_std_diff (k=5) | - | - |
| topology | distance_histogram_js | - | - |
| topology | distance_histogram_wasserstein | - | - |

---

### ImageNet_v1_split_half_upper_bound_nc200_sc25_rep3_knn5_10_seed1.json

- mode: `same_dataset_upper_bound`
- source_dataset: `ImageNet_v1_split_A`
- target_dataset: `ImageNet_v1_split_B`
- backbone: `vit_b16`
- num_classes: `200`
- samples_per_class: `25`

| Category | Metric | Mean | Std |
| --- | --- | ---: | ---: |
| same_class | centroid_cosine | 0.9841 | 0.0001 |
| same_class | centroid_euclidean | 0.1469 | 0.0007 |
| same_class | cov_frobenius | 0.0853 | 0.0003 |
| same_class | pca_subspace | 0.4370 | 0.0007 |
| topology | knn_wasserstein (k=5) | - | - |
| topology | knn_mean_diff (k=5) | - | - |
| topology | knn_std_diff (k=5) | - | - |
| topology | distance_histogram_js | - | - |
| topology | distance_histogram_wasserstein | - | - |

---

### ImageNet_v1_vs_ImageNet_A_nc180_sc10_knn5_10_seed1.json

- mode: `cross_dataset`
- source_dataset: `ImageNet_v1`
- target_dataset: `ImageNet_A`
- backbone: `vit_b16`
- num_classes: `180`
- samples_per_class: `10`

| Category | Metric | Value |
| --- | --- | ---: |
| same_class | centroid_cosine | 0.8760 |
| same_class | centroid_euclidean | 0.4208 |
| same_class | cov_frobenius | 0.1661 |
| same_class | pca_subspace | 0.2438 |
| same_class | pairwise_spearman | 0.0351 |
| same_class | pairwise_pearson | 0.0355 |
| different_class_control | centroid_cosine | 0.6984 |
| different_class_control | centroid_euclidean | 0.6541 |
| different_class_control | cov_frobenius | 0.1719 |
| different_class_control | pca_subspace | 0.1811 |
| different_class_control | pairwise_spearman | 0.0022 |
| different_class_control | pairwise_pearson | 0.0018 |
| topology | knn_wasserstein (k=5) | 0.1612 |
| topology | knn_mean_diff (k=5) | 0.1586 |
| topology | knn_std_diff (k=5) | 0.0261 |
| topology | distance_histogram_js | 0.3277 |
| topology | distance_histogram_wasserstein | 0.1585 |
| graph (k=5) | mean_knn_distance abs_diff_mean | 0.1586 |
| graph (k=5) | std_knn_distance abs_diff_mean | 0.0261 |
| graph (k=5) | average_node_degree abs_diff_mean | 0.3044 |
| inter_class | pearson_corr | 0.8521 |
| inter_class | spearman_corr | 0.8484 |
| inter_class | mean_abs_diff | 0.1098 |
| inter_class | sign_consistency_rate | 0.8722 |
| inter_class | radius_corr | 0.4878 |
| inter_class | radius_mean_abs_diff | 0.1033 |
| inter_class | global_prototype_shift_l2 | 0.1457 |
| inter_class | global_prototype_shift_cosine | 0.9791 |

---

### ImageNet_v1_vs_ImageNet_A_nc180_sc10_knn5_10_seed1_global_prototype_shift.json

- mode: `-`
- source_dataset: `ImageNet_v1`
- target_dataset: `ImageNet_A`
- backbone: `vit_b16`
- num_classes: `180`
- samples_per_class: `10`

| Category | Metric | Mean | Std |
| --- | --- | ---: | ---: |
| same_class | centroid_cosine | - | - |
| same_class | centroid_euclidean | - | - |
| same_class | cov_frobenius | - | - |
| same_class | pca_subspace | - | - |
| topology | knn_wasserstein (k=5) | - | - |
| topology | knn_mean_diff (k=5) | - | - |
| topology | knn_std_diff (k=5) | - | - |
| topology | distance_histogram_js | - | - |
| topology | distance_histogram_wasserstein | - | - |

---

### ImageNet_v1_vs_ImageNet_R_nc200_sc10_knn5_10_seed1.json

- mode: `cross_dataset`
- source_dataset: `ImageNet_v1`
- target_dataset: `ImageNet_R`
- backbone: `vit_b16`
- num_classes: `200`
- samples_per_class: `10`

| Category | Metric | Value |
| --- | --- | ---: |
| same_class | centroid_cosine | 0.8567 |
| same_class | centroid_euclidean | 0.4585 |
| same_class | cov_frobenius | 0.1574 |
| same_class | pca_subspace | 0.2085 |
| same_class | pairwise_spearman | 0.0155 |
| same_class | pairwise_pearson | 0.0107 |
| different_class_control | centroid_cosine | 0.6865 |
| different_class_control | centroid_euclidean | 0.6775 |
| different_class_control | cov_frobenius | 0.1600 |
| different_class_control | pca_subspace | 0.1723 |
| different_class_control | pairwise_spearman | 0.0007 |
| different_class_control | pairwise_pearson | 0.0006 |
| topology | knn_wasserstein (k=5) | 0.1462 |
| topology | knn_mean_diff (k=5) | 0.1415 |
| topology | knn_std_diff (k=5) | 0.0299 |
| topology | distance_histogram_js | 0.3053 |
| topology | distance_histogram_wasserstein | 0.1473 |
| graph (k=5) | mean_knn_distance abs_diff_mean | 0.1415 |
| graph (k=5) | std_knn_distance abs_diff_mean | 0.0299 |
| graph (k=5) | average_node_degree abs_diff_mean | 0.2830 |
| inter_class | pearson_corr | 0.8116 |
| inter_class | spearman_corr | 0.7228 |
| inter_class | mean_abs_diff | 0.0903 |
| inter_class | sign_consistency_rate | 0.8138 |
| inter_class | radius_corr | 0.4165 |
| inter_class | radius_mean_abs_diff | 0.1086 |
| inter_class | global_prototype_shift_l2 | 0.2622 |
| inter_class | global_prototype_shift_cosine | 0.9360 |

---

### ImageNet_v1_vs_ImageNet_R_nc200_sc10_knn5_10_seed1_global_prototype_shift.json

- mode: `-`
- source_dataset: `ImageNet_v1`
- target_dataset: `ImageNet_R`
- backbone: `vit_b16`
- num_classes: `200`
- samples_per_class: `10`

| Category | Metric | Mean | Std |
| --- | --- | ---: | ---: |
| same_class | centroid_cosine | - | - |
| same_class | centroid_euclidean | - | - |
| same_class | cov_frobenius | - | - |
| same_class | pca_subspace | - | - |
| topology | knn_wasserstein (k=5) | - | - |
| topology | knn_mean_diff (k=5) | - | - |
| topology | knn_std_diff (k=5) | - | - |
| topology | distance_histogram_js | - | - |
| topology | distance_histogram_wasserstein | - | - |

---

### ImageNet_v1_vs_ImageNet_Sketch_nc200_sc10_knn5_10_seed1.json

- mode: `cross_dataset`
- source_dataset: `ImageNet_v1`
- target_dataset: `ImageNet_Sketch`
- backbone: `vit_b16`
- num_classes: `200`
- samples_per_class: `10`

| Category | Metric | Value |
| --- | --- | ---: |
| same_class | centroid_cosine | 0.8338 |
| same_class | centroid_euclidean | 0.5023 |
| same_class | cov_frobenius | 0.1446 |
| same_class | pca_subspace | 0.2137 |
| same_class | pairwise_spearman | 0.0093 |
| same_class | pairwise_pearson | 0.0144 |
| different_class_control | centroid_cosine | 0.6579 |
| different_class_control | centroid_euclidean | 0.7222 |
| different_class_control | cov_frobenius | 0.1477 |
| different_class_control | pca_subspace | 0.1637 |
| different_class_control | pairwise_spearman | -0.0027 |
| different_class_control | pairwise_pearson | -0.0026 |
| topology | knn_wasserstein (k=5) | 0.1228 |
| topology | knn_mean_diff (k=5) | 0.1128 |
| topology | knn_std_diff (k=5) | 0.0389 |
| topology | distance_histogram_js | 0.2534 |
| topology | distance_histogram_wasserstein | 0.1206 |
| graph (k=5) | mean_knn_distance abs_diff_mean | 0.1128 |
| graph (k=5) | std_knn_distance abs_diff_mean | 0.0389 |
| graph (k=5) | average_node_degree abs_diff_mean | 0.2980 |
| inter_class | pearson_corr | 0.7961 |
| inter_class | spearman_corr | 0.7598 |
| inter_class | mean_abs_diff | 0.0977 |
| inter_class | sign_consistency_rate | 0.8212 |
| inter_class | radius_corr | 0.3215 |
| inter_class | radius_mean_abs_diff | 0.0713 |
| inter_class | global_prototype_shift_l2 | 0.3151 |
| inter_class | global_prototype_shift_cosine | 0.9138 |

---

### ImageNet_v1_vs_ImageNet_Sketch_nc200_sc10_knn5_10_seed1_global_prototype_shift.json

- mode: `-`
- source_dataset: `ImageNet_v1`
- target_dataset: `ImageNet_Sketch`
- backbone: `vit_b16`
- num_classes: `200`
- samples_per_class: `10`

| Category | Metric | Mean | Std |
| --- | --- | ---: | ---: |
| same_class | centroid_cosine | - | - |
| same_class | centroid_euclidean | - | - |
| same_class | cov_frobenius | - | - |
| same_class | pca_subspace | - | - |
| topology | knn_wasserstein (k=5) | - | - |
| topology | knn_mean_diff (k=5) | - | - |
| topology | knn_std_diff (k=5) | - | - |
| topology | distance_histogram_js | - | - |
| topology | distance_histogram_wasserstein | - | - |

---

### ImageNet_v1_vs_ImageNet_v2_nc200_sc10_knn5_10_seed1.json

- mode: `cross_dataset`
- source_dataset: `ImageNet_v1`
- target_dataset: `ImageNet_v2`
- backbone: `vit_b16`
- num_classes: `200`
- samples_per_class: `10`

| Category | Metric | Value |
| --- | --- | ---: |
| same_class | centroid_cosine | 0.9431 |
| same_class | centroid_euclidean | 0.2802 |
| same_class | cov_frobenius | 0.1487 |
| same_class | pca_subspace | 0.2979 |
| same_class | pairwise_spearman | 0.0008 |
| same_class | pairwise_pearson | 0.0006 |
| different_class_control | centroid_cosine | 0.6766 |
| different_class_control | centroid_euclidean | 0.6854 |
| different_class_control | cov_frobenius | 0.1604 |
| different_class_control | pca_subspace | 0.1820 |
| different_class_control | pairwise_spearman | 0.0037 |
| different_class_control | pairwise_pearson | 0.0039 |
| topology | knn_wasserstein (k=5) | 0.0749 |
| topology | knn_mean_diff (k=5) | 0.0672 |
| topology | knn_std_diff (k=5) | 0.0259 |
| topology | distance_histogram_js | 0.1943 |
| topology | distance_histogram_wasserstein | 0.0758 |
| graph (k=5) | mean_knn_distance abs_diff_mean | 0.0672 |
| graph (k=5) | std_knn_distance abs_diff_mean | 0.0259 |
| graph (k=5) | average_node_degree abs_diff_mean | 0.3010 |
| inter_class | pearson_corr | 0.9280 |
| inter_class | spearman_corr | 0.9130 |
| inter_class | mean_abs_diff | 0.0564 |
| inter_class | sign_consistency_rate | 0.9249 |
| inter_class | radius_corr | 0.7900 |
| inter_class | radius_mean_abs_diff | 0.0341 |
| inter_class | global_prototype_shift_l2 | 0.0674 |
| inter_class | global_prototype_shift_cosine | 0.9955 |

---

### ImageNet_v1_vs_ImageNet_v2_nc200_sc10_knn5_10_seed1_global_prototype_shift.json

- mode: `-`
- source_dataset: `ImageNet_v1`
- target_dataset: `ImageNet_v2`
- backbone: `vit_b16`
- num_classes: `200`
- samples_per_class: `10`

| Category | Metric | Mean | Std |
| --- | --- | ---: | ---: |
| same_class | centroid_cosine | - | - |
| same_class | centroid_euclidean | - | - |
| same_class | cov_frobenius | - | - |
| same_class | pca_subspace | - | - |
| topology | knn_wasserstein (k=5) | - | - |
| topology | knn_mean_diff (k=5) | - | - |
| topology | knn_std_diff (k=5) | - | - |
| topology | distance_histogram_js | - | - |
| topology | distance_histogram_wasserstein | - | - |

---

### ImageNet_v2_split_half_upper_bound_nc200_sc5_rep3_knn5_10_seed1.json

- mode: `same_dataset_upper_bound`
- source_dataset: `ImageNet_v2_split_A`
- target_dataset: `ImageNet_v2_split_B`
- backbone: `vit_b16`
- num_classes: `200`
- samples_per_class: `5`

| Category | Metric | Mean | Std |
| --- | --- | ---: | ---: |
| same_class | centroid_cosine | 0.9153 | 0.0011 |
| same_class | centroid_euclidean | 0.3442 | 0.0022 |
| same_class | cov_frobenius | 0.2267 | 0.0004 |
| same_class | pca_subspace | 0.2379 | 0.0008 |
| topology | knn_wasserstein (k=5) | - | - |
| topology | knn_mean_diff (k=5) | - | - |
| topology | knn_std_diff (k=5) | - | - |
| topology | distance_histogram_js | - | - |
| topology | distance_histogram_wasserstein | - | - |

---
