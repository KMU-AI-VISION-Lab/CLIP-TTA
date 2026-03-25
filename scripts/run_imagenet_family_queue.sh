#!/bin/bash

set -euo pipefail

MANIFEST=${1:?Usage: bash scripts/run_imagenet_family_queue.sh outputs/stata_runs/manifests/imagenet_family_online.json}
shift

if [ "$#" -eq 0 ]; then
  GPUS=(0 1 2 3)
else
  GPUS=("$@")
fi

python tools/run_manifest_queue.py --manifest "${MANIFEST}" --gpus "${GPUS[@]}"
