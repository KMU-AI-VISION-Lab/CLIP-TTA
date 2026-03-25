#!/bin/bash

set -euo pipefail

DATA_ROOT=${1:?Usage: bash scripts/build_imagenet_family_manifest.sh /data/tta/ImageNet_Family [online|batch]}
SETTING=${2:-online}
OUTPUT=${3:-outputs/stata_runs/manifests/imagenet_family_${SETTING}.json}

python tools/build_stata_manifest.py \
  --output "${OUTPUT}" \
  --root_path "${DATA_ROOT}" \
  --setting "${SETTING}" \
  --preset imagenet_family
