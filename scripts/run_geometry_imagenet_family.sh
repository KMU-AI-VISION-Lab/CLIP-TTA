#!/usr/bin/env bash

set -euo pipefail

ROOT_PATH="${1:-/data2/TTA_dataset}"
BACKBONE="${2:-vit_b16}"
CACHE_ROOT="${3:-./caches/geometry_${BACKBONE}}"
OUTPUT_ROOT="${4:-./outputs/geometry_${BACKBONE}}"
DEVICE="${DEVICE:-cuda}"
FEATURE_BATCH_SIZE="${FEATURE_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-8}"

mkdir -p "${CACHE_ROOT}" "${OUTPUT_ROOT}"

# Step 1: dump frozen CLIP image features for each dataset once.
# The later geometry step only reads these saved tensors; it does not touch the images.
python tools/dump_features.py \
  --dataset imagenet \
  --root_path "${ROOT_PATH}" \
  --backbone "${BACKBONE}" \
  --cache_dir "${CACHE_ROOT}/imagenet" \
  --device "${DEVICE}" \
  --batch_size "${FEATURE_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

python tools/dump_features.py \
  --dataset imagenet_v2 \
  --root_path "${ROOT_PATH}" \
  --backbone "${BACKBONE}" \
  --cache_dir "${CACHE_ROOT}/imagenet_v2" \
  --device "${DEVICE}" \
  --batch_size "${FEATURE_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

python tools/dump_features.py \
  --dataset imagenet_sketch \
  --root_path "${ROOT_PATH}" \
  --backbone "${BACKBONE}" \
  --cache_dir "${CACHE_ROOT}/imagenet_sketch" \
  --device "${DEVICE}" \
  --batch_size "${FEATURE_BATCH_SIZE}" \
  --num_workers "${NUM_WORKERS}"

# Step 2: compare class geometry between ImageNet and each target dataset.
python tools/geometry_eval.py \
  --source_feature_file "${CACHE_ROOT}/imagenet/imagenet_${BACKBONE}_features.pt" \
  --target_feature_file "${CACHE_ROOT}/imagenet_v2/imagenet_v2_${BACKBONE}_features.pt" \
  --output "${OUTPUT_ROOT}/imagenet_vs_imagenet_v2.json" \
  --num_classes 200 \
  --samples_per_class 10 \
  --seed 1 \

python tools/geometry_eval.py \
  --source_feature_file "${CACHE_ROOT}/imagenet/imagenet_${BACKBONE}_features.pt" \
  --target_feature_file "${CACHE_ROOT}/imagenet_sketch/imagenet_sketch_${BACKBONE}_features.pt" \
  --output "${OUTPUT_ROOT}/imagenet_vs_imagenet_sketch.json" \
  --num_classes 200 \
  --samples_per_class 10 \
  --seed 1 \
