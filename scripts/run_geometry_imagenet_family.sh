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

# Step 1: dump frozen CLIP image features for each dataset once.
# The internal dataset keys stay lowercase (`imagenet`, `imagenet_v2`, ...),
# but displayed dataset names and output file names follow the canonical style:
# ImageNet_v1, ImageNet_v2, ImageNet_Sketch, ImageNet_A, ImageNet_R.
if [[ ! -f "${IMAGENET_FEATURE_FILE}" ]]; then
  python tools/dump_features.py \
    --dataset imagenet \
    --root_path "${ROOT_PATH}" \
    --backbone "${BACKBONE}" \
    --cache_dir "${IMAGENET_FEATURE_DIR}" \
    --output "${IMAGENET_FEATURE_FILE}" \
    --device "${DEVICE}" \
    --batch_size "${FEATURE_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}"
else
  echo "Skipping ImageNet_v1 feature dump; found ${IMAGENET_FEATURE_FILE}"
fi

if [[ ! -f "${IMAGENET_V2_FEATURE_FILE}" ]]; then
  python tools/dump_features.py \
    --dataset imagenet_v2 \
    --root_path "${ROOT_PATH}" \
    --backbone "${BACKBONE}" \
    --cache_dir "${IMAGENET_V2_FEATURE_DIR}" \
    --output "${IMAGENET_V2_FEATURE_FILE}" \
    --device "${DEVICE}" \
    --batch_size "${FEATURE_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}"
else
  echo "Skipping ImageNet_v2 feature dump; found ${IMAGENET_V2_FEATURE_FILE}"
fi

if [[ ! -f "${IMAGENET_SKETCH_FEATURE_FILE}" ]]; then
  python tools/dump_features.py \
    --dataset imagenet_sketch \
    --root_path "${ROOT_PATH}" \
    --backbone "${BACKBONE}" \
    --cache_dir "${IMAGENET_SKETCH_FEATURE_DIR}" \
    --output "${IMAGENET_SKETCH_FEATURE_FILE}" \
    --device "${DEVICE}" \
    --batch_size "${FEATURE_BATCH_SIZE}" \
    --num_workers "${NUM_WORKERS}"
else
  echo "Skipping ImageNet_Sketch feature dump; found ${IMAGENET_SKETCH_FEATURE_FILE}"
fi

if [[ "${RUN_IMAGENET_A}" == "1" ]]; then
  if [[ ! -f "${IMAGENET_A_FEATURE_FILE}" ]]; then
    python tools/dump_features.py \
      --dataset imagenet_a \
      --root_path "${ROOT_PATH}" \
      --backbone "${BACKBONE}" \
      --cache_dir "${IMAGENET_A_FEATURE_DIR}" \
      --output "${IMAGENET_A_FEATURE_FILE}" \
      --device "${DEVICE}" \
      --batch_size "${FEATURE_BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}"
  else
    echo "Skipping ImageNet_A feature dump; found ${IMAGENET_A_FEATURE_FILE}"
  fi
fi

if [[ "${RUN_IMAGENET_R}" == "1" ]]; then
  if [[ ! -f "${IMAGENET_R_FEATURE_FILE}" ]]; then
    python tools/dump_features.py \
      --dataset imagenet_r \
      --root_path "${ROOT_PATH}" \
      --backbone "${BACKBONE}" \
      --cache_dir "${IMAGENET_R_FEATURE_DIR}" \
      --output "${IMAGENET_R_FEATURE_FILE}" \
      --device "${DEVICE}" \
      --batch_size "${FEATURE_BATCH_SIZE}" \
      --num_workers "${NUM_WORKERS}"
  else
    echo "Skipping ImageNet_R feature dump; found ${IMAGENET_R_FEATURE_FILE}"
  fi
fi

# Step 2: compare class geometry between ImageNet_v1 and each target dataset.
python tools/geometry_eval.py \
  --source_feature_file "${IMAGENET_FEATURE_FILE}" \
  --target_feature_file "${IMAGENET_V2_FEATURE_FILE}" \
  --output "${IMAGENET_V2_OUTPUT}" \
  --num_classes "${NUM_CLASSES}" \
  --samples_per_class "${IMAGENET_V2_SAMPLES_PER_CLASS}" \
  --seed 1 \
  --compute_knn_distribution \
  --compute_distance_histogram \
  --compute_graph_stats \
  --knn_k "${KNN_K_1}" "${KNN_K_2}" \
  "${INTER_CLASS_ARGS[@]}" \
  --device "${GEOMETRY_DEVICE}"

python tools/geometry_eval.py \
  --source_feature_file "${IMAGENET_FEATURE_FILE}" \
  --target_feature_file "${IMAGENET_SKETCH_FEATURE_FILE}" \
  --output "${IMAGENET_SKETCH_OUTPUT}" \
  --num_classes "${NUM_CLASSES}" \
  --samples_per_class "${IMAGENET_SKETCH_SAMPLES_PER_CLASS}" \
  --seed 1 \
  --compute_knn_distribution \
  --compute_distance_histogram \
  --compute_graph_stats \
  --knn_k "${KNN_K_1}" "${KNN_K_2}" \
  "${INTER_CLASS_ARGS[@]}" \
  --device "${GEOMETRY_DEVICE}"

if [[ "${RUN_IMAGENET_A}" == "1" ]]; then
  python tools/geometry_eval.py \
    --source_feature_file "${IMAGENET_FEATURE_FILE}" \
    --target_feature_file "${IMAGENET_A_FEATURE_FILE}" \
    --output "${IMAGENET_A_OUTPUT}" \
    --num_classes "${NUM_CLASSES}" \
    --samples_per_class "${IMAGENET_A_SAMPLES_PER_CLASS}" \
    --seed 1 \
    --compute_knn_distribution \
    --compute_distance_histogram \
    --compute_graph_stats \
    --knn_k "${KNN_K_1}" "${KNN_K_2}" \
    "${INTER_CLASS_ARGS[@]}" \
    --device "${GEOMETRY_DEVICE}"
fi

if [[ "${RUN_IMAGENET_R}" == "1" ]]; then
  python tools/geometry_eval.py \
    --source_feature_file "${IMAGENET_FEATURE_FILE}" \
    --target_feature_file "${IMAGENET_R_FEATURE_FILE}" \
    --output "${IMAGENET_R_OUTPUT}" \
    --num_classes "${NUM_CLASSES}" \
    --samples_per_class "${IMAGENET_R_SAMPLES_PER_CLASS}" \
    --seed 1 \
    --compute_knn_distribution \
    --compute_distance_histogram \
    --compute_graph_stats \
    --knn_k "${KNN_K_1}" "${KNN_K_2}" \
    "${INTER_CLASS_ARGS[@]}" \
    --device "${GEOMETRY_DEVICE}"
fi

python tools/geometry_eval.py \
  --same_dataset_upper_bound \
  --source_feature_file "${IMAGENET_FEATURE_FILE}" \
  --output "${IMAGENET_UPPER_BOUND_OUTPUT}" \
  --num_classes "${NUM_CLASSES}" \
  --samples_per_class "${IMAGENET_UPPER_BOUND_SAMPLES_PER_CLASS}" \
  --min_samples_per_class_for_split "$(( IMAGENET_UPPER_BOUND_SAMPLES_PER_CLASS * 2 ))" \
  --upper_bound_num_repeats "${UPPER_BOUND_REPEATS}" \
  --seed 1 \
  --compute_knn_distribution \
  --compute_distance_histogram \
  --compute_graph_stats \
  --knn_k "${KNN_K_1}" "${KNN_K_2}" \
  "${INTER_CLASS_ARGS[@]}" \
  --device "${GEOMETRY_DEVICE}"

python tools/geometry_eval.py \
  --same_dataset_upper_bound \
  --source_feature_file "${IMAGENET_V2_FEATURE_FILE}" \
  --output "${IMAGENET_V2_UPPER_BOUND_OUTPUT}" \
  --num_classes "${NUM_CLASSES}" \
  --samples_per_class "${IMAGENET_V2_UPPER_BOUND_SAMPLES_PER_CLASS}" \
  --min_samples_per_class_for_split "$(( IMAGENET_V2_UPPER_BOUND_SAMPLES_PER_CLASS * 2 ))" \
  --upper_bound_num_repeats "${UPPER_BOUND_REPEATS}" \
  --seed 1 \
  --compute_knn_distribution \
  --compute_distance_histogram \
  --compute_graph_stats \
  --knn_k "${KNN_K_1}" "${KNN_K_2}" \
  "${INTER_CLASS_ARGS[@]}" \
  --device "${GEOMETRY_DEVICE}"

python tools/geometry_eval.py \
  --same_dataset_upper_bound \
  --source_feature_file "${IMAGENET_SKETCH_FEATURE_FILE}" \
  --output "${IMAGENET_SKETCH_UPPER_BOUND_OUTPUT}" \
  --num_classes "${NUM_CLASSES}" \
  --samples_per_class "${IMAGENET_SKETCH_UPPER_BOUND_SAMPLES_PER_CLASS}" \
  --min_samples_per_class_for_split "$(( IMAGENET_SKETCH_UPPER_BOUND_SAMPLES_PER_CLASS * 2 ))" \
  --upper_bound_num_repeats "${UPPER_BOUND_REPEATS}" \
  --seed 1 \
  --compute_knn_distribution \
  --compute_distance_histogram \
  --compute_graph_stats \
  --knn_k "${KNN_K_1}" "${KNN_K_2}" \
  "${INTER_CLASS_ARGS[@]}" \
  --device "${GEOMETRY_DEVICE}"
