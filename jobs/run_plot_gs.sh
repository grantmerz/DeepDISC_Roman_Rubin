#!/usr/bin/env bash
set -euo pipefail

# Batch runner for plot_grid_search.py
# Runs individual heatmaps and comparison grids for all model/mag-limit combos
# Usage:  ./run_plt_gs.sh [gold|power_law|both]   (default: gold)
MAG="${1:-gold}"
CMP_METRICS="completeness purity f1 unrec_blend_frac_total unrec_blend_frac_blended unrec_blend_frac_matched"

# MODELS=(
#     "lsst5_30k_4h200_bs192_ep50"
#     "clip5_30k_4h200_bs64_ep50"
#     "lsst5_all_4h200_bs192_ep20"
# )
MODELS=(
    # "distill_30k_4h200_bs192_ep50"
    "comb_30k_4h200_bs144_ep50"
)
run_mag() {
    local metrics_dir="$1"
    local mag="$2"        # gold or power_law
    local prefix="$3"     # short label for echo

    local csv_pfx
    if [[ "$mag" == "gold" ]]; then csv_pfx="gold"; else csv_pfx="pl"; fi

    # echo "=== $prefix | $mag | Individual ==="
    python plot_grid_search.py \
        --metrics-dir "$metrics_dir" \
        --csv-files   ${csv_pfx}_buf1/gs_metrics.csv ${csv_pfx}_buf2/gs_metrics.csv \
        --output-dir  ${csv_pfx}_buf1/gs_hmaps ${csv_pfx}_buf2/gs_hmaps \
        --mag-limits  "$mag" "$mag" --buffers 1 2 --no-comparison

    # echo "=== $prefix | $mag | Comparison ==="
    python ~/DeepDISC_Roman_Rubin/plot_grid_search.py \
        --metrics-dir "$metrics_dir" \
        --csv-files   ${csv_pfx}_buf1/gs_metrics.csv ${csv_pfx}_buf2/gs_metrics.csv \
        --output-dir  ${csv_pfx}_cmp \
        --mag-limits  "$mag" "$mag" --buffers 1 2 \
        --metrics $CMP_METRICS --no-individual
    
    # just 1 buffer
    # echo "=== $prefix | $mag | Individual ==="
    # python plot_grid_search.py \
    #     --metrics-dir "$metrics_dir" \
    #     --csv-files   ${csv_pfx}_buf1/gs_metrics.csv \
    #     --output-dir  ${csv_pfx}_buf1/gs_hmaps \
    #     --mag-limits  "$mag" --buffers 1 --no-comparison
}

for model in "${MODELS[@]}"; do
    metrics_dir="$HOME/lsst_runs/$model/metrics"
    label="$model"

    if [[ "$MAG" == "both" ]]; then
        run_mag "$metrics_dir" gold      "$label"
        run_mag "$metrics_dir" power_law "$label"
    else
        run_mag "$metrics_dir" "$MAG" "$label"
    fi
done

echo "All done."