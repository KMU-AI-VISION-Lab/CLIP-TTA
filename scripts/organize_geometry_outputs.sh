#!/usr/bin/env bash

set -euo pipefail

OUTPUT_ROOT="${1:-./outputs/geometry_vit_b16}"
MODE="${2:-move}"

if [[ ! -d "${OUTPUT_ROOT}" ]]; then
  echo "[error] Output directory not found: ${OUTPUT_ROOT}"
  exit 1
fi

MAIN_DIR="${OUTPUT_ROOT}/main_results"
INTER_CLASS_DIR="${OUTPUT_ROOT}/inter_class_artifacts"
UPPER_BOUND_DIR="${OUTPUT_ROOT}/upper_bound_repeat_artifacts"
VIS_DIR="${OUTPUT_ROOT}/visualizations"
MISC_DIR="${OUTPUT_ROOT}/misc"

mkdir -p "${MAIN_DIR}" "${INTER_CLASS_DIR}" "${UPPER_BOUND_DIR}" "${VIS_DIR}" "${MISC_DIR}"

move_or_copy() {
  local src="$1"
  local dst_dir="$2"
  if [[ "${MODE}" == "copy" ]]; then
    cp -n "${src}" "${dst_dir}/"
  else
    mv -n "${src}" "${dst_dir}/"
  fi
}

shopt -s nullglob

for path in "${OUTPUT_ROOT}"/*; do
  base="$(basename "${path}")"

  if [[ "${base}" == "logs" || "${base}" == "main_results" || "${base}" == "inter_class_artifacts" || "${base}" == "upper_bound_repeat_artifacts" || "${base}" == "visualizations" || "${base}" == "misc" ]]; then
    continue
  fi

  if [[ -d "${path}" ]]; then
    continue
  fi

  case "${base}" in
    *.json)
      if [[ "${base}" == *"_inter_class_per_pair_sign_info.json" || "${base}" == *"_inter_class_per_class_radius.json" ]]; then
        if [[ "${base}" == *"_repeat_"* ]]; then
          move_or_copy "${path}" "${UPPER_BOUND_DIR}"
        else
          move_or_copy "${path}" "${INTER_CLASS_DIR}"
        fi
      elif [[ "${base}" == *"_viz_metrics.json" ]]; then
        move_or_copy "${path}" "${VIS_DIR}"
      elif [[ "${base}" == "output_files.txt" || "${base}" == "overall.md" ]]; then
        move_or_copy "${path}" "${MISC_DIR}"
      else
        move_or_copy "${path}" "${MAIN_DIR}"
      fi
      ;;
    *.npy)
      if [[ "${base}" == *"_repeat_"* ]]; then
        move_or_copy "${path}" "${UPPER_BOUND_DIR}"
      else
        move_or_copy "${path}" "${INTER_CLASS_DIR}"
      fi
      ;;
    *.png|*.html)
      move_or_copy "${path}" "${VIS_DIR}"
      ;;
    *)
      move_or_copy "${path}" "${MISC_DIR}"
      ;;
  esac
done

cat <<EOF
[done] Organized geometry outputs under: ${OUTPUT_ROOT}
- main results: ${MAIN_DIR}
- inter-class artifacts: ${INTER_CLASS_DIR}
- upper-bound repeat artifacts: ${UPPER_BOUND_DIR}
- visualizations: ${VIS_DIR}
- misc: ${MISC_DIR}

Mode: ${MODE}
EOF
