#!/usr/bin/env bash

set -euo pipefail

OUTPUT_ROOT="${1:-./outputs/geometry_vit_b16}"
LOW_QUANTILE="${LOW_QUANTILE:-0.2}"
HIGH_QUANTILE="${HIGH_QUANTILE:-0.8}"
TOPK="${TOPK:-10}"
SAVE_PLOTS="${SAVE_PLOTS:-1}"

# 이 스크립트는 기존 per-class radius JSON을 재사용합니다.
# 즉 inter-class geometry를 다시 계산하지 않고도
# source radius 기준 inner / mid / outer bucket 분석을 추가할 수 있습니다.
mapfile -t RADIUS_FILES < <(
  find "${OUTPUT_ROOT}" -type f -name "*per_class_radius.json" | sort | while read -r path; do
    base="$(basename "${path}")"
    lower_base="$(printf '%s' "${base}" | tr '[:upper:]' '[:lower:]')"
    if [[ "${lower_base}" == *"repeat_"* ]]; then
      continue
    fi
    if [[ "${lower_base}" == *"vs"* && "${lower_base}" == *"inter_class"* ]]; then
      printf '%s\n' "${path}"
    fi
  done
)

if [[ ${#RADIUS_FILES[@]} -eq 0 ]]; then
  echo "[error] No inter-class per-class radius files found under ${OUTPUT_ROOT}"
  exit 1
fi

for radius_file in "${RADIUS_FILES[@]}"; do
  echo "Building radius-bucket summary for ${radius_file}"
  cmd=(
    python tools/build_radius_bucket_summary.py
    --radius_file "${radius_file}"
    --low_quantile "${LOW_QUANTILE}"
    --high_quantile "${HIGH_QUANTILE}"
    --topk "${TOPK}"
  )
  if [[ "${SAVE_PLOTS}" == "1" ]]; then
    cmd+=(--save_plots)
  fi
  "${cmd[@]}"
done
