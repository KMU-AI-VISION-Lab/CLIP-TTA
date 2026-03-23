#!/usr/bin/env bash

set -euo pipefail

DATASET_LOCATION="${1:-naver}"
BACKBONE="${2:-vit_b16}"
CACHE_ROOT="${3:-./caches/geometry_${BACKBONE}}"
OUTPUT_ROOT="${4:-./outputs/geometry_${BACKBONE}}"
MAPPING_FILE="${5:-./imagenet-class-ids.txt}"
GROUP_FILE="${GROUP_FILE:-./configs/imagenet_curated_groups.json}"
KEYWORDS=("${@:6}")

if [[ ${#KEYWORDS[@]} -eq 0 ]]; then
  KEYWORDS=("dog" "wolf" "frog")
fi

echo "[step 1/4] Fill missing ImageNet-family outputs, including ImageNet_A when needed"
SKIP_EXISTING_EVALS=1 \
RUN_IMAGENET_A="${RUN_IMAGENET_A:-1}" \
RUN_IMAGENET_R="${RUN_IMAGENET_R:-1}" \
PARALLEL_EXECUTION="${PARALLEL_EXECUTION:-1}" \
GPU_IDS="${GPU_IDS:-0,1,2,3}" \
bash scripts/run_geometry_imagenet_family.sh "${DATASET_LOCATION}" "${BACKBONE}" "${CACHE_ROOT}" "${OUTPUT_ROOT}"

echo "[step 2/4] Build keyword summaries"
GROUP_FILE="${GROUP_FILE}" bash scripts/run_inter_class_keyword_summary.sh "${OUTPUT_ROOT}" "${MAPPING_FILE}" "${KEYWORDS[@]}"

echo "[step 2.5/4] Organize outputs"
bash scripts/organize_geometry_outputs.sh "${OUTPUT_ROOT}"

echo "[step 3/4] Build markdown hypothesis report"
python tools/build_keyword_hypothesis_report.py \
  --output_root "${OUTPUT_ROOT}" \
  --keywords "${KEYWORDS[@]}" \
  --match_mode "${MATCH_MODE:-any}"

echo "[step 4/4] Generate keyword-focused visualizations"
BACKBONE="${BACKBONE}" GROUP_FILE="${GROUP_FILE}" bash scripts/run_keyword_visualizations.sh "${OUTPUT_ROOT}" "${CACHE_ROOT}" "${MAPPING_FILE}" "${KEYWORDS[@]}"

echo "[step 4.5/4] Re-organize outputs after report and visualization"
bash scripts/organize_geometry_outputs.sh "${OUTPUT_ROOT}"
