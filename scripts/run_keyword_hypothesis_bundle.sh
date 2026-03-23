#!/usr/bin/env bash

set -euo pipefail

DATASET_LOCATION="${1:-naver}"
BACKBONE="${2:-vit_b16}"
EXPERIMENT_TAG="${EXPERIMENT_TAG:-}"
CACHE_ROOT="${3:-./caches/geometry_${BACKBONE}}"
DEFAULT_OUTPUT_ROOT="./outputs/geometry_${BACKBONE}"
if [[ -n "${EXPERIMENT_TAG}" ]]; then
  DEFAULT_OUTPUT_ROOT="${DEFAULT_OUTPUT_ROOT}_${EXPERIMENT_TAG}"
fi
OUTPUT_ROOT="${4:-${DEFAULT_OUTPUT_ROOT}}"
MAPPING_FILE="${5:-./imagenet-class-ids.txt}"
GROUP_FILE="${GROUP_FILE:-./configs/imagenet_curated_groups.json}"
KEYWORDS=("${@:6}")

if [[ ${#KEYWORDS[@]} -eq 0 ]]; then
  KEYWORDS=("dog" "wolf" "frog")
fi

# 이 번들 스크립트는 hypothesis 검증에 필요한 여러 단계를
# 사람이 일일이 따로 치지 않아도 되도록 순서대로 묶어 둔 entry point입니다.
# 흐름:
# 1) geometry 실험 결과 채우기
# 2) pair-level summary 만들기
# 3) markdown report 만들기
# 4) 선택된 class들의 visualization 만들기

echo "[bundle] dataset location: ${DATASET_LOCATION}"
echo "[bundle] backbone: ${BACKBONE}"
echo "[bundle] cache root: ${CACHE_ROOT}"
echo "[bundle] output root: ${OUTPUT_ROOT}"
if [[ -n "${EXPERIMENT_TAG}" ]]; then
  echo "[bundle] experiment tag: ${EXPERIMENT_TAG}"
fi
echo "[bundle] curated group file: ${GROUP_FILE}"

echo "[step 1/4] Fill missing ImageNet-family outputs, including ImageNet_A when needed"
# 여기서는 실제 feature dump / geometry eval이 돌고,
# 이미 결과가 있으면 run_geometry_imagenet_family.sh 내부에서 skip합니다.
SKIP_EXISTING_EVALS=1 \
RUN_IMAGENET_A="${RUN_IMAGENET_A:-1}" \
RUN_IMAGENET_R="${RUN_IMAGENET_R:-1}" \
PARALLEL_EXECUTION="${PARALLEL_EXECUTION:-1}" \
GPU_IDS="${GPU_IDS:-0,1,2,3}" \
bash scripts/run_geometry_imagenet_family.sh "${DATASET_LOCATION}" "${BACKBONE}" "${CACHE_ROOT}" "${OUTPUT_ROOT}"

echo "[step 2/4] Build keyword summaries"
# inter-class per-pair JSON을 읽어서
# within-group / cross-group summary JSON으로 다시 가공합니다.
GROUP_FILE="${GROUP_FILE}" bash scripts/run_inter_class_keyword_summary.sh "${OUTPUT_ROOT}" "${MAPPING_FILE}" "${KEYWORDS[@]}"

echo "[step 2.5/4] Organize outputs"
bash scripts/organize_geometry_outputs.sh "${OUTPUT_ROOT}"

echo "[step 3/4] Build markdown hypothesis report"
# JSON은 사람이 읽기 불편하므로, report.py가 markdown 문서로 정리합니다.
python tools/build_keyword_hypothesis_report.py \
  --output_root "${OUTPUT_ROOT}" \
  --keywords "${KEYWORDS[@]}" \
  --match_mode "${MATCH_MODE:-any}"

echo "[step 4/4] Generate keyword-focused visualizations"
# 같은 group/class set을 geometry_viz.py에 넘겨서
# 시각화에서도 summary와 동일한 class 집합을 보게 합니다.
BACKBONE="${BACKBONE}" GROUP_FILE="${GROUP_FILE}" bash scripts/run_keyword_visualizations.sh "${OUTPUT_ROOT}" "${CACHE_ROOT}" "${MAPPING_FILE}" "${KEYWORDS[@]}"

echo "[step 4.5/4] Re-organize outputs after report and visualization"
bash scripts/organize_geometry_outputs.sh "${OUTPUT_ROOT}"
