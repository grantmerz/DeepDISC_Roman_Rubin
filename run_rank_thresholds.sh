#!/usr/bin/env bash
set -euo pipefail

# Batch runner for rank_thresholds.py
# METRICS_DIR="$HOME/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics"
METRICS_DIR="$HOME/lsst_runs/clip5_30k_4h200_bs64_ep50/metrics"
OUT_DIR="$METRICS_DIR/rank_outs"

mkdir -p "$OUT_DIR"

COMMON_ARGS=(
  --metrics-dir "$METRICS_DIR"
  --csv-files
    gold_buf0/gs_metrics.csv
    gold_buf1/gs_metrics.csv
    gold_buf2/gs_metrics.csv
    pl_buf0/gs_metrics.csv
    pl_buf1/gs_metrics.csv
    pl_buf2/gs_metrics.csv
  --mag-limits
    gold gold gold
    power_law power_law power_law
  --buffers
    0 1 2 0 1 2
)

run_cmd() {
  local out_file="$1"
  shift
  echo "Running -> $out_file"
  python rank_thresholds.py "${COMMON_ARGS[@]}" "$@" > "$out_file" 2>&1
}

# Exact sequence from your terminal history
run_cmd "$OUT_DIR/rank_thresh_ll1.txt"
run_cmd "$OUT_DIR/rank_thresh_ll2.txt" --linking-length 2.0
run_cmd "$OUT_DIR/rank_thresh_ll1_mean.txt" --linking-length 1.0 --strategy mean
run_cmd "$OUT_DIR/rank_thresh_ll2_mean.txt" --linking-length 2.0 --strategy mean
run_cmd "$OUT_DIR/rank_thresh_ll1_min.txt" --linking-length 1.0 --strategy min_value
run_cmd "$OUT_DIR/rank_thresh_ll2_min.txt" --linking-length 2.0 --strategy min_value

echo "Done. Outputs are in: $OUT_DIR"
