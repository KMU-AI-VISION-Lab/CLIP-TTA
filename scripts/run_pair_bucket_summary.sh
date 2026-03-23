#!/usr/bin/env bash

set -euo pipefail

OUTPUT_ROOT="${1:-./outputs/geometry_vit_b16}"
LOW_QUANTILE="${LOW_QUANTILE:-0.2}"
HIGH_QUANTILE="${HIGH_QUANTILE:-0.8}"
TOPK="${TOPK:-10}"
SAVE_PLOTS="${SAVE_PLOTS:-1}"

# 이 스크립트는 기존 inter-class per-pair JSON을 재사용합니다.
# 즉 geometry_eval을 다시 돌리지 않고도,
# source similarity 분포 기준 near / mid / far bucket 실험을 추가할 수 있습니다.
mapfile -t PAIR_FILES < <(
  find "${OUTPUT_ROOT}" -type f -name "*per_pair_sign_info.json" | sort | while read -r path; do
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

if [[ ${#PAIR_FILES[@]} -eq 0 ]]; then
  echo "[error] No inter-class per-pair files found under ${OUTPUT_ROOT}"
  exit 1
fi

for pair_file in "${PAIR_FILES[@]}"; do
  echo "Building pair-bucket summary for ${pair_file}"
  cmd=(
    python tools/build_pair_bucket_summary.py
    --pair_info_file "${pair_file}"
    --low_quantile "${LOW_QUANTILE}"
    --high_quantile "${HIGH_QUANTILE}"
    --topk "${TOPK}"
  )
  if [[ "${SAVE_PLOTS}" == "1" ]]; then
    cmd+=(--save_plots)
  fi
  "${cmd[@]}"
done
