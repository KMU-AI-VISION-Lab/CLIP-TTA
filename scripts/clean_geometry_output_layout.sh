#!/usr/bin/env bash

set -euo pipefail

OUTPUT_ROOT="${1:-./outputs/geometry_vit_b16}"
MODE="${2:-copy}"

if [[ ! -d "${OUTPUT_ROOT}" ]]; then
  echo "[error] Output directory not found: ${OUTPUT_ROOT}"
  exit 1
fi

LEGACY_DIR="${OUTPUT_ROOT}/legacy_flat_outputs"
CANONICAL_DIR="${OUTPUT_ROOT}/canonical_flat_outputs"
REPORTS_DIR="${OUTPUT_ROOT}/reports"
MISC_DIR="${OUTPUT_ROOT}/misc_flat_outputs"

mkdir -p "${LEGACY_DIR}" "${CANONICAL_DIR}" "${REPORTS_DIR}" "${MISC_DIR}"

move_or_copy() {
  local src="$1"
  local dst_dir="$2"
  if [[ "${MODE}" == "move" ]]; then
    mv -n "${src}" "${dst_dir}/"
  else
    cp -n "${src}" "${dst_dir}/"
  fi
}

shopt -s nullglob

for path in "${OUTPUT_ROOT}"/*; do
  base="$(basename "${path}")"

  if [[ -d "${path}" ]]; then
    continue
  fi

  case "${base}" in
    dog_wolf_frog_*_hypothesis_report*.md)
      move_or_copy "${path}" "${REPORTS_DIR}"
      ;;
    ImageNet_*)
      move_or_copy "${path}" "${CANONICAL_DIR}"
      ;;
    imagenet_*)
      move_or_copy "${path}" "${LEGACY_DIR}"
      ;;
    output_files.txt|overall.md)
      move_or_copy "${path}" "${MISC_DIR}"
      ;;
    *)
      move_or_copy "${path}" "${MISC_DIR}"
      ;;
  esac
done

cat <<EOF
[done] Split flat geometry outputs under: ${OUTPUT_ROOT}
- canonical flat outputs: ${CANONICAL_DIR}
- legacy flat outputs: ${LEGACY_DIR}
- reports: ${REPORTS_DIR}
- misc flat outputs: ${MISC_DIR}

Mode: ${MODE}
Tip: use a fresh output root for new curated runs:
  EXPERIMENT_TAG=curated MATCH_MODE=both GROUP_FILE=./configs/imagenet_curated_groups.json bash scripts/run_keyword_hypothesis_bundle.sh naver vit_b16
EOF
