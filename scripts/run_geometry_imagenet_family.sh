#!/usr/bin/env bash

set -euo pipefail

DATASET_LOCATION="${1:-lab_server}"

if [[ "${DATASET_LOCATION}" == "naver" ]]; then
  ROOT_PATH="/data/tta/ImageNet_Family"
elif [[ "${DATASET_LOCATION}" == "lab_server" ]]; then
  ROOT_PATH="/data2/TTA_dataset"
else
  ROOT_PATH="${DATASET_LOCATION}"
fi

BACKBONE="${2:-vit_b16}"
CACHE_ROOT="${3:-./caches/geometry_${BACKBONE}}"
OUTPUT_ROOT="${4:-./outputs/geometry_${BACKBONE}}"
DEVICE="${DEVICE:-cuda}"
GEOMETRY_DEVICE="${GEOMETRY_DEVICE:-${DEVICE}}"
PARALLEL_EXECUTION="${PARALLEL_EXECUTION:-0}"
GPU_IDS_RAW="${GPU_IDS:-0,1,2,3}"
LOG_ROOT="${LOG_ROOT:-${OUTPUT_ROOT}/logs}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"
NUM_CLASSES="${NUM_CLASSES:-100}"
IMAGENET_SAMPLES_PER_CLASS="${IMAGENET_SAMPLES_PER_CLASS:-30}"
IMAGENET_V2_SAMPLES_PER_CLASS="${IMAGENET_V2_SAMPLES_PER_CLASS:-10}"
IMAGENET_SKETCH_SAMPLES_PER_CLASS="${IMAGENET_SKETCH_SAMPLES_PER_CLASS:-30}"
IMAGENET_A_SAMPLES_PER_CLASS="${IMAGENET_A_SAMPLES_PER_CLASS:-10}"
IMAGENET_R_SAMPLES_PER_CLASS="${IMAGENET_R_SAMPLES_PER_CLASS:-10}"
IMAGENET_UPPER_BOUND_SAMPLES_PER_CLASS="${IMAGENET_UPPER_BOUND_SAMPLES_PER_CLASS:-25}"
IMAGENET_V2_UPPER_BOUND_SAMPLES_PER_CLASS="${IMAGENET_V2_UPPER_BOUND_SAMPLES_PER_CLASS:-5}"
IMAGENET_SKETCH_UPPER_BOUND_SAMPLES_PER_CLASS="${IMAGENET_SKETCH_UPPER_BOUND_SAMPLES_PER_CLASS:-10}"
KNN_K_1="${KNN_K_1:-5}"
KNN_K_2="${KNN_K_2:-10}"
UPPER_BOUND_REPEATS="${UPPER_BOUND_REPEATS:-3}"
RUN_IMAGENET_A="${RUN_IMAGENET_A:-1}"
RUN_IMAGENET_R="${RUN_IMAGENET_R:-1}"
RUN_INTER_CLASS_GEOMETRY="${RUN_INTER_CLASS_GEOMETRY:-1}"
SIGN_EPSILON="${SIGN_EPSILON:-0.05}"
SAVE_INTER_CLASS_PLOTS="${SAVE_INTER_CLASS_PLOTS:-0}"
SKIP_EXISTING_EVALS="${SKIP_EXISTING_EVALS:-0}"

GPU_IDS=()
IFS=', ' read -r -a GPU_IDS <<< "${GPU_IDS_RAW}"

if [[ "${PARALLEL_EXECUTION}" == "1" ]]; then
  PARALLEL_JOBS="${PARALLEL_JOBS:-${#GPU_IDS[@]}}"
else
  PARALLEL_JOBS=1
fi

if (( PARALLEL_JOBS < 1 )); then
  PARALLEL_JOBS=1
fi

IMAGENET_FEATURE_DIR="${CACHE_ROOT}/ImageNet_v1"
IMAGENET_V2_FEATURE_DIR="${CACHE_ROOT}/ImageNet_v2"
IMAGENET_SKETCH_FEATURE_DIR="${CACHE_ROOT}/ImageNet_Sketch"
IMAGENET_A_FEATURE_DIR="${CACHE_ROOT}/ImageNet_A"
IMAGENET_R_FEATURE_DIR="${CACHE_ROOT}/ImageNet_R"

IMAGENET_FEATURE_FILE="${IMAGENET_FEATURE_DIR}/ImageNet_v1_${BACKBONE}_features.pt"
IMAGENET_V2_FEATURE_FILE="${IMAGENET_V2_FEATURE_DIR}/ImageNet_v2_${BACKBONE}_features.pt"
IMAGENET_SKETCH_FEATURE_FILE="${IMAGENET_SKETCH_FEATURE_DIR}/ImageNet_Sketch_${BACKBONE}_features.pt"
IMAGENET_A_FEATURE_FILE="${IMAGENET_A_FEATURE_DIR}/ImageNet_A_${BACKBONE}_features.pt"
IMAGENET_R_FEATURE_FILE="${IMAGENET_R_FEATURE_DIR}/ImageNet_R_${BACKBONE}_features.pt"
IMAGENET_V2_OUTPUT="${OUTPUT_ROOT}/ImageNet_v1_vs_ImageNet_v2_nc${NUM_CLASSES}_sc${IMAGENET_V2_SAMPLES_PER_CLASS}_knn${KNN_K_1}_${KNN_K_2}_seed1.json"
IMAGENET_SKETCH_OUTPUT="${OUTPUT_ROOT}/ImageNet_v1_vs_ImageNet_Sketch_nc${NUM_CLASSES}_sc${IMAGENET_SKETCH_SAMPLES_PER_CLASS}_knn${KNN_K_1}_${KNN_K_2}_seed1.json"
IMAGENET_A_OUTPUT="${OUTPUT_ROOT}/ImageNet_v1_vs_ImageNet_A_nc${NUM_CLASSES}_sc${IMAGENET_A_SAMPLES_PER_CLASS}_knn${KNN_K_1}_${KNN_K_2}_seed1.json"
IMAGENET_R_OUTPUT="${OUTPUT_ROOT}/ImageNet_v1_vs_ImageNet_R_nc${NUM_CLASSES}_sc${IMAGENET_R_SAMPLES_PER_CLASS}_knn${KNN_K_1}_${KNN_K_2}_seed1.json"
IMAGENET_UPPER_BOUND_OUTPUT="${OUTPUT_ROOT}/ImageNet_v1_split_half_upper_bound_nc${NUM_CLASSES}_sc${IMAGENET_UPPER_BOUND_SAMPLES_PER_CLASS}_rep${UPPER_BOUND_REPEATS}_knn${KNN_K_1}_${KNN_K_2}_seed1.json"
IMAGENET_V2_UPPER_BOUND_OUTPUT="${OUTPUT_ROOT}/ImageNet_v2_split_half_upper_bound_nc${NUM_CLASSES}_sc${IMAGENET_V2_UPPER_BOUND_SAMPLES_PER_CLASS}_rep${UPPER_BOUND_REPEATS}_knn${KNN_K_1}_${KNN_K_2}_seed1.json"
IMAGENET_SKETCH_UPPER_BOUND_OUTPUT="${OUTPUT_ROOT}/ImageNet_Sketch_split_half_upper_bound_nc${NUM_CLASSES}_sc${IMAGENET_SKETCH_UPPER_BOUND_SAMPLES_PER_CLASS}_rep${UPPER_BOUND_REPEATS}_knn${KNN_K_1}_${KNN_K_2}_seed1.json"

INTER_CLASS_ARGS=()
if [[ "${RUN_INTER_CLASS_GEOMETRY}" == "1" ]]; then
  INTER_CLASS_ARGS+=(--inter_class_geometry --sign_epsilon "${SIGN_EPSILON}")
  if [[ "${SAVE_INTER_CLASS_PLOTS}" == "1" ]]; then
    INTER_CLASS_ARGS+=(--save_inter_class_plots)
  fi
fi

mkdir -p "${CACHE_ROOT}" "${OUTPUT_ROOT}"
mkdir -p "${LOG_ROOT}"

ACTIVE_PIDS=()
ACTIVE_LABELS=()

job_device_arg() {
  if [[ "$1" == "cpu" ]]; then
    echo "cpu"
  else
    echo "cuda:0"
  fi
}

wait_for_active_jobs() {
  local exit_code=0
  local pid
  local idx
  for idx in "${!ACTIVE_PIDS[@]}"; do
    pid="${ACTIVE_PIDS[$idx]}"
    if ! wait "${pid}"; then
      echo "[error] Job failed: ${ACTIVE_LABELS[$idx]}"
      exit_code=1
    fi
  done
  ACTIVE_PIDS=()
  ACTIVE_LABELS=()
  if (( exit_code != 0 )); then
    exit 1
  fi
}

maybe_wait_for_wave() {
  if (( ${#ACTIVE_PIDS[@]} >= PARALLEL_JOBS )); then
    wait_for_active_jobs
  fi
}

launch_job() {
  local label="$1"
  local gpu_id="$2"
  shift 2
  local log_file="${LOG_ROOT}/${label}.log"
  echo "[launch] ${label} -> ${log_file}"
  if [[ "${gpu_id}" == "cpu" ]]; then
    (
      "$@"
    ) > >(tee "${log_file}") 2>&1 &
  else
    (
      export CUDA_VISIBLE_DEVICES="${gpu_id}"
      "$@"
    ) > >(tee "${log_file}") 2>&1 &
  fi
  ACTIVE_PIDS+=("$!")
  ACTIVE_LABELS+=("${label}")
}

feature_job() {
  local gpu_id="$1"
  local dataset_key="$2"
  local cache_dir="$3"
  local output_file="$4"
  local label="$5"
  if [[ -f "${output_file}" ]]; then
    echo "Skipping ${label} feature dump; found ${output_file}"
    return
  fi
  local device_arg
  device_arg="$(job_device_arg "${gpu_id}")"
  launch_job "feature_${label}" "${gpu_id}" \
    python tools/dump_features.py \
      --dataset "${dataset_key}" \
      --root_path "${ROOT_PATH}" \
      --backbone "${BACKBONE}" \
      --cache_dir "${cache_dir}" \
      --output "${output_file}" \
      --device "${device_arg}" \
      --batch_size "${FEATURE_BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}"
  maybe_wait_for_wave
}

cross_eval_job() {
  local gpu_id="$1"
  local target_feature_file="$2"
  local output_file="$3"
  local samples_per_class="$4"
  local label="$5"
  if [[ "${SKIP_EXISTING_EVALS}" == "1" && -f "${output_file}" ]]; then
    echo "Skipping ${label} eval; found ${output_file}"
    return
  fi
  local device_arg
  device_arg="$(job_device_arg "${gpu_id}")"
  launch_job "eval_${label}" "${gpu_id}" \
    python tools/geometry_eval.py \
      --source_feature_file "${IMAGENET_FEATURE_FILE}" \
      --target_feature_file "${target_feature_file}" \
      --output "${output_file}" \
      --num_classes "${NUM_CLASSES}" \
      --samples_per_class "${samples_per_class}" \
      --seed 1 \
      --compute_knn_distribution \
      --compute_distance_histogram \
      --compute_graph_stats \
      --knn_k "${KNN_K_1}" "${KNN_K_2}" \
      "${INTER_CLASS_ARGS[@]}" \
      --device "${device_arg}"
  maybe_wait_for_wave
}

upper_bound_job() {
  local gpu_id="$1"
  local source_feature_file="$2"
  local output_file="$3"
  local samples_per_class="$4"
  local label="$5"
  if [[ "${SKIP_EXISTING_EVALS}" == "1" && -f "${output_file}" ]]; then
    echo "Skipping ${label} upper bound; found ${output_file}"
    return
  fi
  local device_arg
  device_arg="$(job_device_arg "${gpu_id}")"
  launch_job "upper_${label}" "${gpu_id}" \
    python tools/geometry_eval.py \
      --same_dataset_upper_bound \
      --source_feature_file "${source_feature_file}" \
      --output "${output_file}" \
      --num_classes "${NUM_CLASSES}" \
      --samples_per_class "${samples_per_class}" \
      --min_samples_per_class_for_split "$(( samples_per_class * 2 ))" \
      --upper_bound_num_repeats "${UPPER_BOUND_REPEATS}" \
      --seed 1 \
      --compute_knn_distribution \
      --compute_distance_histogram \
      --compute_graph_stats \
      --knn_k "${KNN_K_1}" "${KNN_K_2}" \
      "${INTER_CLASS_ARGS[@]}" \
      --device "${device_arg}"
  maybe_wait_for_wave
}

gpu_for_job_index() {
  local job_index="$1"
  if [[ "${DEVICE}" == "cpu" && "${GEOMETRY_DEVICE}" == "cpu" ]]; then
    echo "cpu"
    return
  fi
  echo "${GPU_IDS[$(( job_index % ${#GPU_IDS[@]} ))]}"
}

echo "[config] dataset root: ${ROOT_PATH}"
echo "[config] cache root: ${CACHE_ROOT}"
echo "[config] output root: ${OUTPUT_ROOT}"
echo "[config] parallel execution: ${PARALLEL_EXECUTION}"
echo "[config] parallel jobs: ${PARALLEL_JOBS}"
echo "[config] gpu ids: ${GPU_IDS_RAW}"

# Step 1: dump frozen CLIP image features for each dataset once.
# The internal dataset keys stay lowercase (`imagenet`, `imagenet_v2`, ...),
# but displayed dataset names and output file names follow the canonical style:
# ImageNet_v1, ImageNet_v2, ImageNet_Sketch, ImageNet_A, ImageNet_R.
feature_job "$(gpu_for_job_index 0)" imagenet "${IMAGENET_FEATURE_DIR}" "${IMAGENET_FEATURE_FILE}" "ImageNet_v1"
feature_job "$(gpu_for_job_index 1)" imagenet_v2 "${IMAGENET_V2_FEATURE_DIR}" "${IMAGENET_V2_FEATURE_FILE}" "ImageNet_v2"
feature_job "$(gpu_for_job_index 2)" imagenet_sketch "${IMAGENET_SKETCH_FEATURE_DIR}" "${IMAGENET_SKETCH_FEATURE_FILE}" "ImageNet_Sketch"

if [[ "${RUN_IMAGENET_A}" == "1" ]]; then
  feature_job "$(gpu_for_job_index 3)" imagenet_a "${IMAGENET_A_FEATURE_DIR}" "${IMAGENET_A_FEATURE_FILE}" "ImageNet_A"
fi

if [[ "${RUN_IMAGENET_R}" == "1" ]]; then
  feature_job "$(gpu_for_job_index 4)" imagenet_r "${IMAGENET_R_FEATURE_DIR}" "${IMAGENET_R_FEATURE_FILE}" "ImageNet_R"
fi

wait_for_active_jobs

# Step 2: compare class geometry between ImageNet_v1 and each target dataset.
cross_eval_job "$(gpu_for_job_index 0)" "${IMAGENET_V2_FEATURE_FILE}" "${IMAGENET_V2_OUTPUT}" "${IMAGENET_V2_SAMPLES_PER_CLASS}" "ImageNet_v1_vs_ImageNet_v2"
cross_eval_job "$(gpu_for_job_index 1)" "${IMAGENET_SKETCH_FEATURE_FILE}" "${IMAGENET_SKETCH_OUTPUT}" "${IMAGENET_SKETCH_SAMPLES_PER_CLASS}" "ImageNet_v1_vs_ImageNet_Sketch"

if [[ "${RUN_IMAGENET_A}" == "1" ]]; then
  cross_eval_job "$(gpu_for_job_index 2)" "${IMAGENET_A_FEATURE_FILE}" "${IMAGENET_A_OUTPUT}" "${IMAGENET_A_SAMPLES_PER_CLASS}" "ImageNet_v1_vs_ImageNet_A"
fi

if [[ "${RUN_IMAGENET_R}" == "1" ]]; then
  cross_eval_job "$(gpu_for_job_index 3)" "${IMAGENET_R_FEATURE_FILE}" "${IMAGENET_R_OUTPUT}" "${IMAGENET_R_SAMPLES_PER_CLASS}" "ImageNet_v1_vs_ImageNet_R"
fi

wait_for_active_jobs

upper_bound_job "$(gpu_for_job_index 0)" "${IMAGENET_FEATURE_FILE}" "${IMAGENET_UPPER_BOUND_OUTPUT}" "${IMAGENET_UPPER_BOUND_SAMPLES_PER_CLASS}" "ImageNet_v1_split_half_upper_bound"
upper_bound_job "$(gpu_for_job_index 1)" "${IMAGENET_V2_FEATURE_FILE}" "${IMAGENET_V2_UPPER_BOUND_OUTPUT}" "${IMAGENET_V2_UPPER_BOUND_SAMPLES_PER_CLASS}" "ImageNet_v2_split_half_upper_bound"
upper_bound_job "$(gpu_for_job_index 2)" "${IMAGENET_SKETCH_FEATURE_FILE}" "${IMAGENET_SKETCH_UPPER_BOUND_OUTPUT}" "${IMAGENET_SKETCH_UPPER_BOUND_SAMPLES_PER_CLASS}" "ImageNet_Sketch_split_half_upper_bound"

wait_for_active_jobs
