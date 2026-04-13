#!/usr/bin/env bash
set -euo pipefail

# ./jobs/run_classify_plot.sh --run-names "lsst5_30k_4h200_bs192_ep50 comb_30k_4h200_bs144_ep50" --score-thresholds "0.4" --nms-thresholds "0.55" --buffers "1 2"

# run_classify_plot.sh — wrapper for run_class.py
# Mirrors submit_plot_fof_binned.sh style
DEFAULT_ROOT_RUN_DIR="~/lsst_runs"
DEFAULT_RUN_NAMES="lsst5_30k_4h200_bs192_ep50 distill_30k_4h200_bs192_ep50 comb_30k_4h200_bs144_ep50"
DEFAULT_SCORE_THRESHOLDS="0.65"
DEFAULT_NMS_THRESHOLDS="0.65"
DEFAULT_MAG_LIMIT="gold"
DEFAULT_BUFFERS="1"
DEFAULT_LINKING_LENGTHS="1.0 2.0"
DEFAULT_PREFIX="dd"
DEFAULT_MAG_MIN="18.0"
DEFAULT_MAG_MAX="28.0"
DEFAULT_MAG_BIN_WIDTH="0.5"
DEFAULT_MIN_COUNT="1"
DEFAULT_OUTPUT_DIR=""
DEFAULT_PLOT_SCRIPT="$HOME/DeepDISC_Roman_Rubin/classify_plot.py"

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Run Options:
    --run-names "R1 R2 ..."         Run names under root-run-dir (default: "$DEFAULT_RUN_NAMES")
    --no-lsst                        Skip LSST detection-catalog plots (default: included)
    --root-run-dir PATH             Root directory (default: $DEFAULT_ROOT_RUN_DIR)
    --score-thresholds "S1 ..."     Score threshold(s) (default: "$DEFAULT_SCORE_THRESHOLDS")
    --nms-thresholds "N1 ..."       NMS threshold(s)   (default: "$DEFAULT_NMS_THRESHOLDS")
    --combo                         Cartesian product of score x NMS thresholds
    --mag-limit NAME                gold | power_law | nominal (default: $DEFAULT_MAG_LIMIT)
    --buffers "B1 B2"               Buffer values 0|1|2 (default: "$DEFAULT_BUFFERS")
    --linking-lengths "L1 L2"       Linking lengths (default: "$DEFAULT_LINKING_LENGTHS")
    --prefix dd|lsst                Pipeline prefix (default: $DEFAULT_PREFIX)
    --mag-min FLOAT                 Mag bin lower edge (default: $DEFAULT_MAG_MIN)
    --mag-max FLOAT                 Mag bin upper edge (default: $DEFAULT_MAG_MAX)
    --mag-bin-width FLOAT           Mag bin width (default: $DEFAULT_MAG_BIN_WIDTH)
    --min-count N                   Min objects per bin (default: $DEFAULT_MIN_COUNT)
    --output-dir PATH               Output directory (default: {root_run_dir}/classifications/{run_tag})
    --legend-name-map "K=V ..."    Legend aliases (e.g. lsst5=DeepDISC-LSST comb=DeepDISC-Combined)
    --no-save                       Show plots interactively
    --summary-only                  Only produce combined summary figure
    --plot-script PATH              Path to classify_plot.py
    --dry-run                       Print command without executing
    -h, --help                      Show this message

Examples:
    # Four checkpoints, one combo, both LLs
    $0 --run-names "og_50ep flatten_15ep flatten_50ep supCon_15ep" \\
       --score-thresholds "0.5" --nms-thresholds "0.6" \\
       --linking-lengths "1.0 2.0" --buffers "1"

    # Two checkpoints, Cartesian product of two combos
    $0 --run-names "og_50ep flatten_50ep" \\
       --score-thresholds "0.4 0.5" --nms-thresholds "0.55 0.6" --combo
EOF
}

ROOT_RUN_DIR="$DEFAULT_ROOT_RUN_DIR"
RUN_NAMES_RAW="$DEFAULT_RUN_NAMES"
SCORE_THRESHOLDS="$DEFAULT_SCORE_THRESHOLDS"
NMS_THRESHOLDS="$DEFAULT_NMS_THRESHOLDS"
COMBO="false"
MAG_LIMIT="$DEFAULT_MAG_LIMIT"
BUFFERS_RAW="$DEFAULT_BUFFERS"
LINKING_LENGTHS="$DEFAULT_LINKING_LENGTHS"
PREFIX="$DEFAULT_PREFIX"
MAG_MIN="$DEFAULT_MAG_MIN"
MAG_MAX="$DEFAULT_MAG_MAX"
MAG_BIN_WIDTH="$DEFAULT_MAG_BIN_WIDTH"
MIN_COUNT="$DEFAULT_MIN_COUNT"
OUTPUT_DIR="$DEFAULT_OUTPUT_DIR"
NO_SAVE="false"
SUMMARY_ONLY="false"
PLOT_SCRIPT="$DEFAULT_PLOT_SCRIPT"
DRY_RUN="false"
INCLUDE_LSST="true"
LEGEND_NAME_MAP_RAW=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-names)          RUN_NAMES_RAW="$2";    shift 2 ;;
        --no-lsst)            INCLUDE_LSST="false";  shift ;;
        --root-run-dir)       ROOT_RUN_DIR="$2";      shift 2 ;;
        --score-thresholds)   SCORE_THRESHOLDS="$2";  shift 2 ;;
        --nms-thresholds)     NMS_THRESHOLDS="$2";    shift 2 ;;
        --combo)              COMBO="true";            shift ;;
        --mag-limit)          MAG_LIMIT="$2";          shift 2 ;;
        --buffers)            BUFFERS_RAW="$2";        shift 2 ;;
        --linking-lengths)    LINKING_LENGTHS="$2";    shift 2 ;;
        --prefix)             PREFIX="$2";             shift 2 ;;
        --mag-min)            MAG_MIN="$2";            shift 2 ;;
        --mag-max)            MAG_MAX="$2";            shift 2 ;;
        --mag-bin-width)      MAG_BIN_WIDTH="$2";      shift 2 ;;
        --min-count)          MIN_COUNT="$2";          shift 2 ;;
        --output-dir)         OUTPUT_DIR="$2";         shift 2 ;;
        --no-save)            NO_SAVE="true";          shift ;;
        --summary-only)       SUMMARY_ONLY="true";     shift ;;
        --plot-script)        PLOT_SCRIPT="$2";        shift 2 ;;
        --legend-name-map)    LEGEND_NAME_MAP_RAW="$2"; shift 2 ;;
        --dry-run)            DRY_RUN="true";          shift ;;
        -h|--help)            show_usage; exit 0 ;;
        *)
            echo "Error: Unknown option '$1'"
            show_usage
            exit 1
            ;;
    esac
done

if [[ ! "$MAG_LIMIT" =~ ^(gold|power_law|nominal)$ ]]; then
    echo "Error: --mag-limit must be: gold, power_law, or nominal"
    exit 1
fi
if [[ ! "$PREFIX" =~ ^(dd|lsst)$ ]]; then
    echo "Error: --prefix must be dd or lsst"
    exit 1
fi

ROOT_RUN_DIR_EXPANDED="${ROOT_RUN_DIR/#\~/$HOME}"
if [[ ! -d "$ROOT_RUN_DIR_EXPANDED" ]]; then
    echo "Error: root run dir does not exist: $ROOT_RUN_DIR_EXPANDED"
    exit 1
fi
if [[ ! -f "$PLOT_SCRIPT" ]]; then
    echo "Error: plot script not found: $PLOT_SCRIPT"
    exit 1
fi

read -ra RUN_NAMES        <<< "$RUN_NAMES_RAW"
read -ra SCORE_THRESH_ARR <<< "$SCORE_THRESHOLDS"
read -ra NMS_THRESH_ARR   <<< "$NMS_THRESHOLDS"
read -ra BUFFERS_ARR      <<< "$BUFFERS_RAW"
read -ra LINKING_LENGTHS_ARR <<< "$LINKING_LENGTHS"
read -ra LEGEND_NAME_MAP_ARR <<< "$LEGEND_NAME_MAP_RAW"

echo "Morph classification config:"
echo "  Root run dir   : $ROOT_RUN_DIR_EXPANDED"
echo "  Run names      : ${RUN_NAMES[*]}"
echo "  Include LSST baseline: $INCLUDE_LSST"
echo "  Score thresh   : ${SCORE_THRESH_ARR[*]}"
echo "  NMS thresh     : ${NMS_THRESH_ARR[*]}"
echo "  Cartesian      : $COMBO"
echo "  Mag limit      : $MAG_LIMIT"
echo "  Buffers        : ${BUFFERS_ARR[*]}"
echo "  Linking lengths: ${LINKING_LENGTHS_ARR[*]}"
echo "  Prefix         : $PREFIX"
echo "  Mag range/bin  : $MAG_MIN to $MAG_MAX (width $MAG_BIN_WIDTH)"
echo "  Min count      : $MIN_COUNT"
echo "  Output dir     : ${OUTPUT_DIR:-{root_run_dir}/classifications/{run_tag}}"
echo "  Legend map     : ${LEGEND_NAME_MAP_ARR[*]:-<default>}"
echo "  Summary only   : $SUMMARY_ONLY"

declare -a CMD
CMD=(
    python "$PLOT_SCRIPT"
    --root-run-dir "$ROOT_RUN_DIR_EXPANDED"
    --run-names "${RUN_NAMES[@]}"
    --score-thresholds "${SCORE_THRESH_ARR[@]}"
    --nms-thresholds "${NMS_THRESH_ARR[@]}"
    --mag-limit "$MAG_LIMIT"
    --buffers "${BUFFERS_ARR[@]}"
    --linking-lengths "${LINKING_LENGTHS_ARR[@]}"
    --prefix "$PREFIX"
    --mag-min "$MAG_MIN"
    --mag-max "$MAG_MAX"
    --mag-bin-width "$MAG_BIN_WIDTH"
    --min-count "$MIN_COUNT"
)

[[ "$COMBO"        == "true" ]] && CMD+=(--combo)
[[ "$NO_SAVE"      == "true" ]] && CMD+=(--no-save)
[[ "$SUMMARY_ONLY" == "true" ]] && CMD+=(--summary-only)
[[ -n "$OUTPUT_DIR"          ]] && CMD+=(--output-dir "$OUTPUT_DIR")
[[ "${#LEGEND_NAME_MAP_ARR[@]}" -gt 0 ]] && CMD+=(--legend-name-map "${LEGEND_NAME_MAP_ARR[@]}")
if [[ "$PREFIX" == "dd" ]] && [[ "$INCLUDE_LSST" == "true" ]]; then
    CMD+=(--include-lsst-baseline)
fi

echo "Command:$(printf ' %q' "${CMD[@]}")"

if [[ "$DRY_RUN" != "true" ]]; then
    "${CMD[@]}"
fi

if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run only."
fi