#!/usr/bin/env bash

set -euo pipefail

OUTPUT_ROOT="${1:-./outputs/geometry_vit_b16}"
CACHE_ROOT="${2:-./caches/geometry_vit_b16}"
MAPPING_FILE="${3:-./imagenet-class-ids.txt}"
BACKBONE="${BACKBONE:-vit_b16}"
VIZ_DIM="${VIZ_DIM:-3}"
VIZ_MAX_POINTS_PER_CLASS="${VIZ_MAX_POINTS_PER_CLASS:-30}"
VIZ_SAMPLES_PER_CLASS="${VIZ_SAMPLES_PER_CLASS:-10}"
INTERACTIVE="${INTERACTIVE:-1}"
KEYWORDS=("${@:4}")

if [[ ${#KEYWORDS[@]} -eq 0 ]]; then
  KEYWORDS=("dog" "wolf" "frog")
fi

keywords_csv="$(printf '%s,' "${KEYWORDS[@]}")"
keywords_csv="${keywords_csv%,}"

mapfile -t CLASS_IDS < <(
  awk -F '\t' -v keywords="${keywords_csv}" '
    BEGIN {
      split(tolower(keywords), kws, ",")
    }
    NR == 1 { next }
    {
      name = tolower($2)
      for (i in kws) {
        if (index(name, kws[i]) > 0) {
          print $1
          break
        }
      }
    }
  ' "${MAPPING_FILE}"
)

if [[ ${#CLASS_IDS[@]} -eq 0 ]]; then
  echo "[error] No class ids matched keywords: ${KEYWORDS[*]}"
  exit 1
fi

latest_eval_json() {
  local target_name="$1"
  mapfile -t matches < <(find "${OUTPUT_ROOT}" -type f -name "ImageNet_v1_vs_${target_name}*.json" | sort)
  local filtered=()
  local path
  for path in "${matches[@]}"; do
    if [[ ! -f "${path}" ]]; then
      continue
    fi
    if [[ "${path}" == *"_inter_class_"* || "${path}" == *"_viz_metrics.json" ]]; then
      continue
    fi
    filtered+=("${path}")
  done
  if [[ ${#filtered[@]} -eq 0 ]]; then
    return 1
  fi
  ls -1t "${filtered[@]}" | head -n 1
}

run_viz() {
  local target_name="$1"
  local target_feature_file="$2"
  local eval_json
  eval_json="$(latest_eval_json "${target_name}")" || {
    echo "Skipping ${target_name} visualization; no eval JSON found."
    return
  }

  if [[ ! -f "${target_feature_file}" ]]; then
    echo "Skipping ${target_name} visualization; missing feature file ${target_feature_file}."
    return
  fi

  local output_prefix="${OUTPUT_ROOT}/keyword_viz_ImageNet_v1_vs_${target_name}_$(echo "${KEYWORDS[*]}" | tr ' ' '_')"
  local cmd=(
    python tools/geometry_viz.py
    --source_feature_file "${CACHE_ROOT}/ImageNet_v1/ImageNet_v1_${BACKBONE}_features.pt"
    --target_feature_file "${target_feature_file}"
    --eval_json "${eval_json}"
    --output_prefix "${output_prefix}"
    --viz_dim "${VIZ_DIM}"
    --samples_per_class "${VIZ_SAMPLES_PER_CLASS}"
    --viz_max_points_per_class "${VIZ_MAX_POINTS_PER_CLASS}"
    --viz_class_ids "${CLASS_IDS[@]}"
  )
  if [[ "${INTERACTIVE}" == "1" ]]; then
    cmd+=(--interactive)
  fi

  echo "Running keyword visualization for ${target_name}"
  "${cmd[@]}"
}

run_viz "ImageNet_v2" "${CACHE_ROOT}/ImageNet_v2/ImageNet_v2_${BACKBONE}_features.pt"
run_viz "ImageNet_Sketch" "${CACHE_ROOT}/ImageNet_Sketch/ImageNet_Sketch_${BACKBONE}_features.pt"
run_viz "ImageNet_A" "${CACHE_ROOT}/ImageNet_A/ImageNet_A_${BACKBONE}_features.pt"
run_viz "ImageNet_R" "${CACHE_ROOT}/ImageNet_R/ImageNet_R_${BACKBONE}_features.pt"
