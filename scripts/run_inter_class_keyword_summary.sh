#!/usr/bin/env bash

set -euo pipefail

OUTPUT_ROOT="${1:-./outputs/geometry_vit_b16}"
MAPPING_FILE="${2:-./imagenet-class-ids.txt}"
TOPK="${TOPK:-20}"
MATCH_MODE="${MATCH_MODE:-any}"
GROUP_FILE="${GROUP_FILE:-}"
KEYWORDS=("${@:3}")

if [[ ${#KEYWORDS[@]} -eq 0 ]]; then
  KEYWORDS=("dog" "wolf" "frog")
fi

# organize 이후 파일이 하위 폴더로 들어갈 수 있으므로
# OUTPUT_ROOT 아래를 재귀적으로 탐색합니다.
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
  if [[ ! -f "${pair_file}" ]]; then
    echo "Skipping missing pair file: ${pair_file}"
    continue
  fi

  echo "Summarizing ${pair_file}"
  if [[ -n "${GROUP_FILE}" ]]; then
    # curated group file이 있으면 substring keyword 대신
    # 명시적인 class id 목록으로 분석합니다.
    python tools/filter_inter_class_pairs.py \
      --pair_info_file "${pair_file}" \
      --mapping_file "${MAPPING_FILE}" \
      --group_file "${GROUP_FILE}" \
      --groups "${KEYWORDS[@]}" \
      --match_mode "${MATCH_MODE}" \
      --topk "${TOPK}"
  else
    # group file이 없을 때만 예전 keyword substring 방식으로 동작합니다.
    python tools/filter_inter_class_pairs.py \
      --pair_info_file "${pair_file}" \
      --mapping_file "${MAPPING_FILE}" \
      --keywords "${KEYWORDS[@]}" \
      --match_mode "${MATCH_MODE}" \
      --topk "${TOPK}"
  fi
done
