#!/usr/bin/env bash
set -euo pipefail

# Batch runner for plot_grid_search.py -- individual heatmaps + comparison grids
# across model/mag-limit/buffer combos.
# Usage:
#   ./run_plot_gs.sh [model] [mag] [buffers] [options...]
#
# Positional (all optional):
#   model    : lsst30k | ups30k | comb30k | clip30k | lsstall | all  (default: lsst30k)
#   mag      : gold | power_law | both                       (default: gold, matches
#                                                              plot_grid_search.py's
#                                                              --mag-limits default)
#   buffers  : e.g. "0 1 2"                                  (default: "2", matches
#                                                              plot_grid_search.py's
#                                                              --buffers default)
#
# Options (forwarded to plot_grid_search.py; every default below matches
# plot_grid_search.py's own argparse default, not this wrapper's prior behavior):
#   --pipeline PIPELINE           dd | lsst                             (default: dd)
#   --linking-lengths "LL1 LL2"                                        (default: "0.5 1.0")
#   --metrics "M1 M2 ..."         used for both individual and comparison passes,
#                                 same as a single plot_grid_search.py invocation
#                                 (default: unset -> all 12 metrics)
#   --delta                       also emit DD-LSST delta heatmaps      (default: on)
#   --no-delta                    disable DD-LSST delta heatmaps
#   --pareto                      outline the Pareto front on the scatter (default: on)
#   --no-pareto                   disable Pareto front outline
#   --absolute-paths              treat CSV/output paths as absolute    (default: off)
#   --no-individual               skip the individual-heatmap pass      (default: pass runs)
#   --comparison                  also run the comparison-grid pass     (default: off)
#   --no-annotate                 hide per-cell numbers on heatmaps and delta heatmaps
#                                 (individual pass only; default: numbers shown)
#   --title-template "TPL"        heatmap title, used for both individual and comparison
#                                 passes -- {metric} {pipeline} {ll} placeholders, plus
#                                 {mag_limit} {buffer} in individual mode
#   --delta-title-template "TPL"  delta heatmap title -- {metric} {pipeline} {ll} {lsst},
#                                 plus {mag_limit} {buffer} (individual mode only)
#   --scatter-title-template "TPL" purity-vs-completeness scatter title -- {label}
#                                 (e.g. "gold buf2") {pipeline} {ll} {mag_limit} {buffer}
#
# All three *-title-template flags default to auto-picked-per-model templates -- same
# template text, only the run label inside changes: "Rubin" (default) / "RubinUps"
# (ups30k) / "RubinRoman" (comb30k). See _run_label_for_model and the three
# _*_title_template_for_model functions below; pass the flag explicitly to override
# for every model.
#
# Examples:
#   ./run_plot_gs.sh
#   ./run_plot_gs.sh --no-delta
#   ./run_plot_gs.sh all both "0 1 2" --comparison
#   ./run_plot_gs.sh ups30k gold "1 2" --comparison
#   ./run_plot_gs.sh ups30k gold "2" --no-annotate --title-template '{metric} (LL={ll}")'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Positional args are consumed only while they don't look like a --flag, so
# `run_plot_gs.sh --no-delta` (no model/mag/buffers given) falls straight through
# to the option parser below instead of misreading "--no-delta" as the model name.
MODEL="lsst30k"
[[ $# -gt 0 && "$1" != --* ]] && { MODEL="$1"; shift; }
MAG="gold"
[[ $# -gt 0 && "$1" != --* ]] && { MAG="$1"; shift; }
read -r -a BUFFERS <<< "2"
[[ $# -gt 0 && "$1" != --* ]] && { read -r -a BUFFERS <<< "$1"; shift; }

# plot_grid_search.py passthrough options, defaults matching plot_grid_search.py's own.
PIPELINE="dd"
LINKING_LENGTHS="0.5 1.0"
METRICS=""
DELTA_FLAG="--delta"
PARETO_FLAG="--pareto"
ABSOLUTE_PATHS_FLAG=""
NO_INDIVIDUAL=""
COMPARISON_FLAG=""
NO_ANNOTATE_FLAG=""
# Each *_TEMPLATE var: empty = auto-pick per model below; the matching --*-template
# flag overrides for every model.
TITLE_TEMPLATE=""
DELTA_TITLE_TEMPLATE=""
SCATTER_TITLE_TEMPLATE=""

# Run label swapped into every default template below (only used when the matching
# --*-template flag isn't given explicitly).
_run_label_for_model() {
    case "$1" in
        ups30k) echo "RubinUps" ;;
        comb30k) echo "RubinRoman" ;;
        *) echo "Rubin" ;;
    esac
}

_title_template_for_model() {
    local run_label; run_label="$(_run_label_for_model "$1")"
    printf '%s' '{metric} as a Function of $\theta_{{\mathrm{{ROI}}}}$ and $\theta_{{\mathrm{{NMS}}}}$ (DD '"$run_label"', LL = {ll}", $\Delta$m = {buffer})'
}

_delta_title_template_for_model() {
    local run_label; run_label="$(_run_label_for_model "$1")"
    printf '%s' '{metric} DD '"$run_label"'-LSST (LSST={lsst}, LL = {ll}", $\Delta$m = {buffer})'
}

_scatter_title_template_for_model() {
    local run_label; run_label="$(_run_label_for_model "$1")"
    printf '%s' 'Purity vs Completeness (DD '"$run_label"', LL = {ll}", $\Delta$m = {buffer})'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pipeline) PIPELINE="$2"; shift 2 ;;
        --linking-lengths) LINKING_LENGTHS="$2"; shift 2 ;;
        --metrics) METRICS="$2"; shift 2 ;;
        --delta) DELTA_FLAG="--delta"; shift ;;
        --no-delta) DELTA_FLAG="--no-delta"; shift ;;
        --pareto) PARETO_FLAG="--pareto"; shift ;;
        --no-pareto) PARETO_FLAG="--no-pareto"; shift ;;
        --absolute-paths) ABSOLUTE_PATHS_FLAG="--absolute-paths"; shift ;;
        --no-individual) NO_INDIVIDUAL="1"; shift ;;
        --comparison) COMPARISON_FLAG="1"; shift ;;
        --no-annotate) NO_ANNOTATE_FLAG="--no-annotate"; shift ;;
        --title-template) TITLE_TEMPLATE="$2"; shift 2 ;;
        --delta-title-template) DELTA_TITLE_TEMPLATE="$2"; shift 2 ;;
        --scatter-title-template) SCATTER_TITLE_TEMPLATE="$2"; shift 2 ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done
read -r -a LINKING_LENGTHS_ARR <<< "$LINKING_LENGTHS"
METRICS_ARGS=()
[[ -n "$METRICS" ]] && { read -r -a METRICS_ARR <<< "$METRICS"; METRICS_ARGS=(--metrics "${METRICS_ARR[@]}"); }

declare -A MODEL_DIRS=(
    [lsst30k]="$HOME/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics"
    [ups30k]="$HOME/lsst_runs/lsst5_ups_30k_4h200_bs192_ep50/metrics"
    [comb30k]="$HOME/lsst_runs/comb_30k_4h200_bs144_ep50/metrics"
    [clip30k]="$HOME/lsst_runs/clip5_30k_4h200_bs64_ep50/metrics"
    [lsstall]="$HOME/lsst_runs/lsst5_all_4h200_bs192_ep20/metrics"
)

if [[ "$MODEL" == "all" ]]; then
    MODELS_TO_RUN=(lsst30k ups30k comb30k)
elif [[ -n "${MODEL_DIRS[$MODEL]+_}" ]]; then
    MODELS_TO_RUN=("$MODEL")
else
    echo "Unknown model '$MODEL'. Choose: lsst30k | clip30k | comb30k | ups30k | lsstall | all" >&2
    exit 1
fi

run_mag() {
    local metrics_dir="$1"
    local mag="$2"   # gold or power_law
    local model="$3"
    local pfx; [[ "$mag" == "gold" ]] && pfx="gold" || pfx="pl"

    local title_template="$TITLE_TEMPLATE"
    [[ -z "$title_template" ]] && title_template="$(_title_template_for_model "$model")"
    local title_template_args=(--title-template "$title_template")

    local delta_title_template="$DELTA_TITLE_TEMPLATE"
    [[ -z "$delta_title_template" ]] && delta_title_template="$(_delta_title_template_for_model "$model")"
    local delta_title_template_args=(--delta-title-template "$delta_title_template")

    local scatter_title_template="$SCATTER_TITLE_TEMPLATE"
    [[ -z "$scatter_title_template" ]] && scatter_title_template="$(_scatter_title_template_for_model "$model")"
    local scatter_title_template_args=(--scatter-title-template "$scatter_title_template")

    local csv_files=() ind_out_dirs=() mag_limits=() buffers=()
    for buf in "${BUFFERS[@]}"; do
        csv_files+=( "${pfx}_buf${buf}/gs_metrics.csv" )
        ind_out_dirs+=( "${pfx}_buf${buf}/gs_hmaps" )
        mag_limits+=( "$mag" )
        buffers+=( "$buf" )
    done

    if [[ -z "$NO_INDIVIDUAL" ]]; then
        echo "  [$mag] individual -> ${ind_out_dirs[*]}"
        python plot_grid_search.py \
            --metrics-dir "$metrics_dir" \
            --csv-files "${csv_files[@]}" \
            --output-dir "${ind_out_dirs[@]}" \
            --mag-limits "${mag_limits[@]}" --buffers "${buffers[@]}" \
            --pipeline "$PIPELINE" --linking-lengths "${LINKING_LENGTHS_ARR[@]}" \
            "${METRICS_ARGS[@]}" \
            $DELTA_FLAG $PARETO_FLAG $ABSOLUTE_PATHS_FLAG $NO_ANNOTATE_FLAG \
            "${title_template_args[@]}" "${delta_title_template_args[@]}" \
            "${scatter_title_template_args[@]}" \
            --no-comparison
    fi

    if [[ -n "$COMPARISON_FLAG" ]]; then
        echo "  [$mag] comparison -> ${pfx}_cmp"
        python plot_grid_search.py \
            --metrics-dir "$metrics_dir" \
            --csv-files "${csv_files[@]}" \
            --output-dir "${pfx}_cmp" \
            --mag-limits "${mag_limits[@]}" --buffers "${buffers[@]}" \
            --pipeline "$PIPELINE" --linking-lengths "${LINKING_LENGTHS_ARR[@]}" \
            "${METRICS_ARGS[@]}" \
            "${title_template_args[@]}" \
            $ABSOLUTE_PATHS_FLAG \
            --no-individual --comparison
    fi
}

for model in "${MODELS_TO_RUN[@]}"; do
    metrics_dir="${MODEL_DIRS[$model]}"
    echo "--- Model: $model ---"
    if [[ "$MAG" == "both" ]]; then
        run_mag "$metrics_dir" gold "$model"
        run_mag "$metrics_dir" power_law "$model"
    else
        run_mag "$metrics_dir" "$MAG" "$model"
    fi
done

echo "All done."
