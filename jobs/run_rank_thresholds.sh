#!/usr/bin/env bash
set -euo pipefail

# Batch runner for rank_thresholds.py
# Usage:
#   ./run_rank_thresholds.sh [model] [mag] [linking_lengths] [strategies] [buffers]
#
# Arguments (all optional, space-separated lists must be quoted):
#   model            : lsst30k | clip30k | comb30k | lsstall | all  (default: all)
#   mag              : gold | power_law | both             (default: gold)
#   linking_lengths  : e.g. "1.0 2.0"                     (default: "1.0 2.0")
#   strategies       : e.g. "minimax mean min_value"       (default: "minimax mean min_value")
#   buffers          : e.g. "0 1 2"                        (default: "1 2")
#
# Examples:
#   ./run_rank_thresholds.sh
#   ./run_rank_thresholds.sh all gold "1.0" "minimax mean" "0 1"
#   ./run_rank_thresholds.sh lsst30k gold "1.0" "minimax mean" "1"

MODEL="${1:-all}"
MAG="${2:-gold}"
LINKING_LENGTHS=(${3:-1.0 2.0})
STRATEGIES=(${4:-minimax mean min_value})
BUFFERS=(${5:-1 2})

declare -A MODEL_DIRS=(
    [lsst30k]="$HOME/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics"
    [clip30k]="$HOME/lsst_runs/clip5_30k_4h200_bs64_ep50/metrics"
    [comb30k]="$HOME/lsst_runs/comb_30k_4h200_bs144_ep50/metrics"
    [lsstall]="$HOME/lsst_runs/lsst5_all_4h200_bs192_ep20/metrics"
)

if [[ "$MODEL" == "all" ]]; then
    MODELS_TO_RUN=(lsst30k clip30k comb30k lsstall)
elif [[ -n "${MODEL_DIRS[$MODEL]+_}" ]]; then
    MODELS_TO_RUN=("$MODEL")
else
    echo "Unknown model '$MODEL'. Choose: lsst30k | clip30k | comb30k | lsstall | all"; exit 1
fi

build_csv_args() {
    local mag="$1"   # gold | power_law | both
    local csv_files=() mag_limits=() buffers=()
    local mags=()
    if [[ "$mag" == "both" ]]; then
        mags=(gold power_law)
    else
        mags=("$mag")
    fi
    for m in "${mags[@]}"; do
        local pfx; [[ "$m" == "gold" ]] && pfx="gold" || pfx="pl"
        for buf in "${BUFFERS[@]}"; do
            csv_files+=( ${pfx}_buf${buf}/gs_metrics.csv )
            mag_limits+=( "$m" )
            buffers+=( "$buf" )
        done
    done
    echo "--csv-files ${csv_files[*]} --mag-limits ${mag_limits[*]} --buffers ${buffers[*]}"
}

run_model() {
    local model="$1"
    local metrics_dir="${MODEL_DIRS[$model]}"
    local out_dir="$metrics_dir/rank_outs"
    local csv_args
    csv_args=$(build_csv_args "$MAG")
    mkdir -p "$out_dir"
    echo "--- Model: $model ---"
    for ll in "${LINKING_LENGTHS[@]}"; do
        for strategy in "${STRATEGIES[@]}"; do
            local tag="ll${ll}_${strategy}"
            echo "  Running -> rank_${tag}.txt"
            python rank_thresholds.py \
                --metrics-dir "$metrics_dir" \
                $csv_args \
                --linking-length "$ll" --strategy "$strategy" \
                > "$out_dir/rank_${tag}.txt" 2>&1
        done
    done
    echo "  Done. Outputs in: $out_dir"
}

for model in "${MODELS_TO_RUN[@]}"; do
    run_model "$model"
done

echo "All done."
