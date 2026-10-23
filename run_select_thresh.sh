#!/usr/bin/env bash
set -euo pipefail

# Batch runner for select_thresh.py
# Usage:
#   ./run_select_thresh.sh [model] [mag] [linking_lengths] [buffers] [options...]
#
# Positional args are consumed only while they don't look like a --flag, so
# `run_select_thresh.sh --pareto` (no model/mag/etc given) falls straight through
# to the option parser below instead of misreading "--pareto" as the model name.
#
# Positional (all optional, space-separated lists must be quoted):
#   model            : lsst30k | ups30k | comb30k | clip30k | lsstall | all  (default: lsst30k)
#   mag              : gold | power_law | both             (default: gold)
#   linking_lengths  : e.g. "0.5 1.0"                      (default: "0.5 1.0")
#   buffers          : e.g. "0 1 2"                        (default: "2")
#
# Options (forwarded to select_thresh.py; defaults match select_thresh.py's own):
#   --run-dir DIR                  override the model-derived metrics dir —
#                                  sets --metrics-dir to DIR/metrics (mirrors gs.sh)
#   --metrics-dir DIR              directly set the metrics directory, bypassing
#                                  the model map and --run-dir entirely
#   --pipeline PIPELINE            dd | lsst                        (default: dd)
#   --compare-pipeline PIPELINE     dd | lsst -- baseline for table delta annotations
#                                    (default: lsst)
#   --metric METRIC                 percentile/stability metric (default: f1)
#   --constraint "METRIC OP VALUE" Constraint like "purity >= 0.85" (repeatable);
#                                    filters combos before ranking/selection/plotting.
#                                    No constraints applied by default.
#   --top N                         rows printed per ranking         (default: 10)
#   --tradeoff-x METRIC             (default: f1)
#   --tradeoff-y METRIC             (default: unrec_blend_frac)
#   --tradeoff-color METRIC         (default: purity)
#   --pareto                        outline the Pareto front on the trade-off plot
#   --optimize-set "M1 M2 ..."       multi-metric operating point set
#                                    (default: "f1 unrec_blend_frac")
#   --pareto-set "M1 M2 ..."        Pareto matrix metric set
#                                    (default: "f1 unrec_blend_frac")
#   --pareto-marks "M1 M2 ..."      combo markers on the Pareto matrix
#                                    (choices: f1 mean min; default: f1 mean)
#   --pareto-matrix / --no-pareto-matrix  write or skip the Pareto matrix plot
#   --table-optimize "M1 M2 ..."     (default: "f1 unrec_blend_frac")
#   --table-cols "M1 M2 ..."         (default: "f1 unrec_blend_frac purity completeness spurious_frac
#                                    shred_frac missed_frac resolved_frac absorbed_frac
#                                    survivor_frac sig_blend_frac")
#   --parcoords                     also emit the parallel-coordinates plot
#   --parcoords-highlight METRIC    (default: f1)
#   --parcoords-cols "M1 M2 ..."    metrics included in the plot
#                                    (default: "f1 unrec_blend_frac purity completeness
#                                     resolved_frac spurious_frac shred_frac")
#   --stability-combo "SCORE NMS"   override the auto-picked stability combo(s)
#   --absolute-paths                treat CSV/output paths as absolute
#   --log-file "NAME"               override the log filename (default: "select.log")
#   --save-csv / --no-save-csv      write or skip CSV outputs (default: write)
#   --tradeoff-title-template "TPL"      trade-off plot title -- {label} {pipeline}
#                                        {x_metric} {y_metric} {ll} {mag_limit} {buffer}
#   --pareto-matrix-title-template "TPL" Pareto matrix title -- {label} {pipeline} {ll}
#                                        {mag_limit} {buffer}
#   --parcoords-title-template "TPL"     parallel coordinates title -- same placeholders
#                                        as the Pareto matrix title
#   All three default to an auto-picked-per-model template matching run_plot_gs.sh's
#   scatter title: same run label ("Rubin" / "RubinUps" / "RubinRoman"); pass the flag
#   explicitly to override for every model.
#
# Examples:
#   ./run_select_thresh.sh
#   ./run_select_thresh.sh all gold "1.0" "0 1"
#   ./run_select_thresh.sh lsst30k gold "1.0" "1" "1.0" --pareto --parcoords
#   ./run_select_thresh.sh lsst30k gold "1.0" "1" "1.0" --tradeoff-x completeness --tradeoff-y spurious_frac
#
#   # ad hoc run not in the model map — use --run-dir directly (mirrors gs.sh):
#   ./run_select_thresh.sh --run-dir ~/lsst_runs/distill_30k_4h200_bs192_ep50 gold "1.0" "2"
#
#   # with --metrics-dir for even more control:
#   ./run_select_thresh.sh --metrics-dir /path/to/metrics gold "1.0" "2"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

MODEL="lsst30k"
[[ $# -gt 0 && "$1" != --* ]] && { MODEL="$1"; shift; }
MAG="gold"
[[ $# -gt 0 && "$1" != --* ]] && { MAG="$1"; shift; }
read -r -a LINKING_LENGTHS <<< "0.5 1.0"
[[ $# -gt 0 && "$1" != --* ]] && { read -r -a LINKING_LENGTHS <<< "$1"; shift; }
read -r -a BUFFERS <<< "2"
[[ $# -gt 0 && "$1" != --* ]] && { read -r -a BUFFERS <<< "$1"; shift; }
HEADLINE_MAG="gold"; [[ "$MAG" == "power_law" ]] && HEADLINE_MAG="power_law"
HEADLINE_BUFFER="${BUFFERS[-1]}"

# select_thresh.py passthrough options, defaults matching select_thresh.py's own.
PIPELINE="dd"
COMPARE_PIPELINE="lsst"
METRIC="f1"
TOP="10"
TRADEOFF_X="f1"
TRADEOFF_Y="unrec_blend_frac"
TRADEOFF_COLOR="purity"
PARETO_FLAG="--pareto"
OPTIMIZE_SET="f1 unrec_blend_frac"
PARETO_SET="f1 unrec_blend_frac"
PARETO_MARKS="f1 mean"
TABLE_OPTIMIZE="f1 unrec_blend_frac"
TABLE_COLS="f1 unrec_blend_frac purity completeness spurious_frac shred_frac missed_frac resolved_frac absorbed_frac survivor_frac sig_blend_frac"
PARCOORDS_FLAG="--parcoords"
PARCOORDS_HIGHLIGHT="f1"
PARCOORDS_COLS="f1 unrec_blend_frac purity completeness resolved_frac spurious_frac shred_frac"
STABILITY_COMBO=""
ABSOLUTE_PATHS_FLAG=""
LOG_FILE="select.log"
RUN_DIR=""
METRICS_DIR_OVERRIDE=""
CONSTRAINTS=()
# Each *_TITLE_TEMPLATE var: empty = auto-pick per model below; the matching
# --*-title-template flag overrides for every model.
TRADEOFF_TITLE_TEMPLATE=""
PARETO_MATRIX_TITLE_TEMPLATE=""
PARCOORDS_TITLE_TEMPLATE=""
EXTRA_ARGS=()

# Run label swapped into every default title template below (only used when the
# matching --*-title-template flag isn't given explicitly) -- matches run_plot_gs.sh's
# _run_label_for_model.
_run_label_for_model() {
    case "$1" in
        ups30k) echo "RubinUps" ;;
        comb30k) echo "RubinRoman" ;;
        _ad_hoc_) echo "Rubin" ;;
        *) echo "Rubin" ;;
    esac
}

_tradeoff_title_template_for_model() {
    local run_label; run_label="$(_run_label_for_model "$1")"
    printf '%s' '{y_metric} vs {x_metric} (DD '"$run_label"', LL = {ll}", $\Delta$m = {buffer})'
}

_pareto_matrix_title_template_for_model() {
    local run_label; run_label="$(_run_label_for_model "$1")"
    printf '%s' 'Pareto Matrix (DD '"$run_label"', LL = {ll}", $\Delta$m = {buffer})'
}

_parcoords_title_template_for_model() {
    local run_label; run_label="$(_run_label_for_model "$1")"
    printf '%s' 'Parallel Coordinates (DD '"$run_label"', LL = {ll}", $\Delta$m = {buffer})'
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --pipeline) PIPELINE="$2"; shift 2 ;;
        --compare-pipeline) COMPARE_PIPELINE="$2"; shift 2 ;;
        --metric) METRIC="$2"; shift 2 ;;
        --constraint) CONSTRAINTS+=("$2"); shift 2 ;;
        --top) TOP="$2"; shift 2 ;;
        --tradeoff-x) TRADEOFF_X="$2"; shift 2 ;;
        --tradeoff-y) TRADEOFF_Y="$2"; shift 2 ;;
        --tradeoff-color) TRADEOFF_COLOR="$2"; shift 2 ;;
        --pareto) PARETO_FLAG="--pareto"; shift ;;
        --optimize-set) OPTIMIZE_SET="$2"; shift 2 ;;
        --pareto-set) PARETO_SET="$2"; shift 2 ;;
        --pareto-marks) PARETO_MARKS="$2"; shift 2 ;;
        --table-optimize) TABLE_OPTIMIZE="$2"; shift 2 ;;
        --table-cols) TABLE_COLS="$2"; shift 2 ;;
        --parcoords) PARCOORDS_FLAG="--parcoords"; shift ;;
        --parcoords-highlight) PARCOORDS_HIGHLIGHT="$2"; shift 2 ;;
        --parcoords-cols) PARCOORDS_COLS="$2"; shift 2 ;;
        --stability-combo) STABILITY_COMBO="$2"; shift 2 ;;
        --absolute-paths) ABSOLUTE_PATHS_FLAG="--absolute-paths"; shift ;;
        --log-file) LOG_FILE="$2"; shift 2 ;;
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --metrics-dir) METRICS_DIR_OVERRIDE="$2"; shift 2 ;;
        --tradeoff-title-template) TRADEOFF_TITLE_TEMPLATE="$2"; shift 2 ;;
        --pareto-matrix-title-template) PARETO_MATRIX_TITLE_TEMPLATE="$2"; shift 2 ;;
        --parcoords-title-template) PARCOORDS_TITLE_TEMPLATE="$2"; shift 2 ;;
        *) EXTRA_ARGS+=("$1"); shift ;;
    esac
done
read -r -a TABLE_OPTIMIZE_ARR <<< "$TABLE_OPTIMIZE"
read -r -a TABLE_COLS_ARR <<< "$TABLE_COLS"
read -r -a OPTIMIZE_SET_ARR <<< "$OPTIMIZE_SET"
read -r -a PARETO_SET_ARR <<< "$PARETO_SET"
read -r -a PARETO_MARKS_ARR <<< "$PARETO_MARKS"
read -r -a PARCOORDS_COLS_ARR <<< "$PARCOORDS_COLS"

declare -A MODEL_DIRS=(
    [lsst30k]="$HOME/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics"
    [ups30k]="$HOME/lsst_runs/lsst5_ups_30k_4h200_bs192_ep50/metrics"
    [comb30k]="$HOME/lsst_runs/comb_30k_4h200_bs144_ep50/metrics"
    [clip30k]="$HOME/lsst_runs/clip5_30k_4h200_bs64_ep50/metrics"
    [lsstall]="$HOME/lsst_runs/lsst5_all_4h200_bs192_ep20/metrics"
)

if [[ -n "$RUN_DIR" || -n "$METRICS_DIR_OVERRIDE" ]]; then
    # --run-dir / --metrics-dir bypass the model map entirely (mirrors gs.sh).
    MODELS_TO_RUN=("_ad_hoc_")
elif [[ "$MODEL" == "all" ]]; then
    MODELS_TO_RUN=(lsst30k ups30k comb30k)
elif [[ -n "${MODEL_DIRS[$MODEL]+_}" ]]; then
    MODELS_TO_RUN=("$MODEL")
else
    echo "Unknown model '$MODEL'. Choose: lsst30k | clip30k | comb30k | ups30k | lsstall | all" >&2
    echo "(or pass --run-dir / --metrics-dir directly for a run not in this map)" >&2
    exit 1
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
    local metrics_dir
    if [[ -n "$METRICS_DIR_OVERRIDE" ]]; then
        metrics_dir="$METRICS_DIR_OVERRIDE"
    elif [[ -n "$RUN_DIR" ]]; then
        metrics_dir="$RUN_DIR/metrics"
    else
        metrics_dir="${MODEL_DIRS[$model]}"
    fi
    local out_dir="$metrics_dir/select_outs"
    local csv_args
    csv_args=$(build_csv_args "$MAG")
    local constraint_args=()
    for c in "${CONSTRAINTS[@]}"; do
        constraint_args+=(--constraint "$c")
    done

    local tradeoff_title_template="$TRADEOFF_TITLE_TEMPLATE"
    [[ -z "$tradeoff_title_template" ]] && tradeoff_title_template="$(_tradeoff_title_template_for_model "$model")"
    local pareto_matrix_title_template="$PARETO_MATRIX_TITLE_TEMPLATE"
    [[ -z "$pareto_matrix_title_template" ]] && pareto_matrix_title_template="$(_pareto_matrix_title_template_for_model "$model")"
    local parcoords_title_template="$PARCOORDS_TITLE_TEMPLATE"
    [[ -z "$parcoords_title_template" ]] && parcoords_title_template="$(_parcoords_title_template_for_model "$model")"

    mkdir -p "$out_dir"
    echo "--- Model: $model ---"
    echo "  Running -> $LOG_FILE"
    python select_thresh.py \
        --metrics-dir "$metrics_dir" \
        $csv_args \
        --output-dir "$out_dir" \
        --linking-lengths "${LINKING_LENGTHS[@]}" \
        --headline-mag "$HEADLINE_MAG" --headline-buffer "$HEADLINE_BUFFER" \
        --pipeline "$PIPELINE" --compare-pipeline "$COMPARE_PIPELINE" \
        --metric "$METRIC" --top "$TOP" \
        --tradeoff-x "$TRADEOFF_X" --tradeoff-y "$TRADEOFF_Y" \
        --tradeoff-color "$TRADEOFF_COLOR" \
        --tradeoff-title-template "$tradeoff_title_template" \
        --pareto-matrix-title-template "$pareto_matrix_title_template" \
        --parcoords-title-template "$parcoords_title_template" \
        --optimize-set "${OPTIMIZE_SET_ARR[@]}" \
        --pareto-set "${PARETO_SET_ARR[@]}" \
        --pareto-marks "${PARETO_MARKS_ARR[@]}" \
        --table-optimize "${TABLE_OPTIMIZE_ARR[@]}" \
        --table-cols "${TABLE_COLS_ARR[@]}" \
        --parcoords-highlight "$PARCOORDS_HIGHLIGHT" \
        --parcoords-cols "${PARCOORDS_COLS_ARR[@]}" \
        "${constraint_args[@]}" \
            $PARETO_FLAG $PARCOORDS_FLAG $ABSOLUTE_PATHS_FLAG "${EXTRA_ARGS[@]}" \
        ${STABILITY_COMBO:+--stability-combo $STABILITY_COMBO} \
        > "$out_dir/$LOG_FILE" 2>&1
    echo "  Done. Outputs in: $out_dir"
}

for model in "${MODELS_TO_RUN[@]}"; do
    run_model "$model"
done

echo "All done."
