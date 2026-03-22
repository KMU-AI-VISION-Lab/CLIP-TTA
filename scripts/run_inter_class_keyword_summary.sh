#!/usr/bin/env bash

set -euo pipefail

OUTPUT_ROOT="${1:-./outputs/geometry_vit_b16}"
MAPPING_FILE="${2:-./imagenet-class-ids.txt}"
TOPK="${TOPK:-20}"
MATCH_MODE="${MATCH_MODE:-any}"
KEYWORDS=("${@:3}")

if [[ ${#KEYWORDS[@]} -eq 0 ]]; then
  KEYWORDS=("dog" "wolf" "frog")
fi

PAIR_FILES=(
  "${OUTPUT_ROOT}/ImageNet_v1_vs_ImageNet_v2_inter_class_inter_class_per_pair_sign_info.json"
  "${OUTPUT_ROOT}/ImageNet_v1_vs_ImageNet_Sketch_inter_class_inter_class_per_pair_sign_info.json"
  "${OUTPUT_ROOT}/ImageNet_v1_vs_ImageNet_A_inter_class_inter_class_per_pair_sign_info.json"
  "${OUTPUT_ROOT}/ImageNet_v1_vs_ImageNet_R_inter_class_inter_class_per_pair_sign_info.json"
)

for pair_file in "${PAIR_FILES[@]}"; do
  if [[ ! -f "${pair_file}" ]]; then
    echo "Skipping missing pair file: ${pair_file}"
    continue
  fi

  echo "Summarizing ${pair_file}"
  python tools/filter_inter_class_pairs.py \
    --pair_info_file "${pair_file}" \
    --mapping_file "${MAPPING_FILE}" \
    --keywords "${KEYWORDS[@]}" \
    --match_mode "${MATCH_MODE}" \
    --topk "${TOPK}"
done
