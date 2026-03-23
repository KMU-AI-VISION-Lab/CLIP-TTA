#!/usr/bin/env bash

set -euo pipefail

BACKBONE="${1:-vit_b16}"
CACHE_ROOT="${2:-./caches/geometry_${BACKBONE}}"
OUTPUT_DIR="${3:-./configs/generated}"
GROUP_CANDIDATES_FILE="${GROUP_CANDIDATES_FILE:-./configs/imagenet_supergroup_candidates.json}"
MAX_CLASSES_PER_GROUP="${MAX_CLASSES_PER_GROUP:-}"

mkdir -p "${OUTPUT_DIR}"

run_builder() {
  local config_name="$1"
  shift
  local cmd=(
    python tools/build_balanced_group_configs.py
    --config_name "${config_name}"
    --output_dir "${OUTPUT_DIR}"
    --group_candidates_file "${GROUP_CANDIDATES_FILE}"
    --groups dog bird vehicle
    --feature_files "$@"
  )
  if [[ -n "${MAX_CLASSES_PER_GROUP}" ]]; then
    cmd+=(--max_classes_per_group "${MAX_CLASSES_PER_GROUP}")
  fi
  echo "Building ${config_name}"
  "${cmd[@]}"
}

run_builder "balanced_ImageNet_v1_ImageNet_v2_ImageNet_Sketch" \
  "ImageNet_v1=${CACHE_ROOT}/ImageNet_v1/ImageNet_v1_${BACKBONE}_features.pt" \
  "ImageNet_v2=${CACHE_ROOT}/ImageNet_v2/ImageNet_v2_${BACKBONE}_features.pt" \
  "ImageNet_Sketch=${CACHE_ROOT}/ImageNet_Sketch/ImageNet_Sketch_${BACKBONE}_features.pt"

run_builder "balanced_ImageNet_v1_ImageNet_A" \
  "ImageNet_v1=${CACHE_ROOT}/ImageNet_v1/ImageNet_v1_${BACKBONE}_features.pt" \
  "ImageNet_A=${CACHE_ROOT}/ImageNet_A/ImageNet_A_${BACKBONE}_features.pt"

run_builder "balanced_ImageNet_v1_ImageNet_R" \
  "ImageNet_v1=${CACHE_ROOT}/ImageNet_v1/ImageNet_v1_${BACKBONE}_features.pt" \
  "ImageNet_R=${CACHE_ROOT}/ImageNet_R/ImageNet_R_${BACKBONE}_features.pt"
