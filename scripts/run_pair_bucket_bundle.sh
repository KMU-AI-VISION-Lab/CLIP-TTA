#!/usr/bin/env bash

set -euo pipefail

OUTPUT_ROOT="${1:-./outputs/geometry_vit_b16}"
LOW_QUANTILE="${LOW_QUANTILE:-0.2}"
HIGH_QUANTILE="${HIGH_QUANTILE:-0.8}"

echo "[bundle] output root: ${OUTPUT_ROOT}"
echo "[bundle] pair bucket rule: far <= ${LOW_QUANTILE}, near >= ${HIGH_QUANTILE}"

echo "[step 1/2] Build pair-bucket summaries"
LOW_QUANTILE="${LOW_QUANTILE}" HIGH_QUANTILE="${HIGH_QUANTILE}" bash scripts/run_pair_bucket_summary.sh "${OUTPUT_ROOT}"

echo "[step 2/2] Build pair-bucket markdown report"
python tools/build_pair_bucket_report.py \
  --output_root "${OUTPUT_ROOT}" \
  --low_quantile "${LOW_QUANTILE}" \
  --high_quantile "${HIGH_QUANTILE}"
