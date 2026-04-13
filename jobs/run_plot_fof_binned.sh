#!/usr/bin/env bash
set -euo pipefail

# Submit-style wrapper for DeepDISC_Roman_Rubin/plot_fof_binned.py.
# This mirrors submit_run_fof_classify.sh workflow options and dispatch style,
# but executes plotting directly (no sbatch) by default.
# ./run_plot_fof_binned.sh --run-names "clip5_30k_4h200_bs64_ep50 clip5_flatten_30k_4h200_bs64_ep15_resume" --comparison-pairs "clip5_30k_4h200_bs64_ep50,clip5_flatten_30k_4h200_bs64_ep15_resume" --dataset-preset test_8k --score-thresholds "0.4" --nms-thresholds "0.55" --dry-run
# ./run_plot_fof_binned.sh --run-names "clip5_30k_4h200_bs192_ep15 clip5_flatten_30k_4h200_bs64_ep15" --comparison-pairs "clip5_30k_4h200_bs192_ep15,clip5_flatten_30k_4h200_bs64_ep15" --dataset-preset test_8k --score-thresholds "0.4" --nms-thresholds "0.55" --only-comparison --dry-run
# ./run_plot_fof_binned.sh --run-names "lsst5_30k_4h200_bs192_ep50 clip5_30k_4h200_bs192_ep15" --comparison-pairs "lsst5_30k_4h200_bs192_ep50,clip5_30k_4h200_bs192_ep15" --dataset-preset test_8k --score-thresholds "0.4" --nms-thresholds "0.55" --only-comparison --dry-run

# ./run_plot_fof_binned.sh --run-names "lsst5_30k_4h200_bs192_ep50 distill_30k_4h200_bs192_ep50" --comparison-pairs "lsst5_30k_4h200_bs192_ep50,distill_30k_4h200_bs192_ep50" --dataset-preset test_8k --score-thresholds "0.4" --nms-thresholds "0.55" --only-comparison
# ./run_plot_fof_binned.sh --run-names "lsst5_30k_4h200_bs192_ep50 distill_30k_4h200_bs192_ep50" --comparison-pairs "lsst5_30k_4h200_bs192_ep50,distill_30k_4h200_bs192_ep50" --dataset-preset test_8k --score-thresholds "0.65" --nms-thresholds "0.65"
# ./jobs/submit_plot_fof_binned.sh --run-names "distill_30k_4h200_bs192_ep50" --dataset-preset test_8k --score-thresholds "0.65" --nms-thresholds "0.65"

# ./jobs/submit_plot_fof_binned.sh --run-names "comb_30k_4h200_bs144_ep50" --dataset-preset test_8k --score-thresholds "0.4 0.55 0.65" --nms-thresholds "0.55 0.65 0.65" --no-default-comparison-pairs
# ./run_plot_fof_binned.sh --run-names "lsst5_30k_4h200_bs192_ep50 comb_30k_4h200_bs144_ep50" --comparison-pairs "lsst5_30k_4h200_bs192_ep50,comb_30k_4h200_bs144_ep50" --dataset-preset test_8k --score-thresholds "0.4" --nms-thresholds "0.55" --only-comparison
# ./run_plot_fof_binned.sh --run-names "comb_30k_4h200_bs144_ep50 distill_30k_4h200_bs192_ep50" --comparison-pairs "comb_30k_4h200_bs144_ep50,distill_30k_4h200_bs192_ep50" --dataset-preset test_8k --score-thresholds "0.65" --nms-thresholds "0.65" --only-comparison --buffers "1"

# ./jobs/submit_plot_fof_binned.sh --run-names "lsst5_30k_4h200_bs192_ep50 clip5_30k_4h200_bs192_ep15_lprj" --comparison-pairs "lsst5_30k_4h200_bs192_ep50,clip5_30k_4h200_bs192_ep15_lprj" --dataset-preset test_8k --score-thresholds "0.4" --nms-thresholds "0.55" --only-comparison
# ./jobs/submit_plot_fof_binned.sh --run-names "clip5_30k_4h200_bs192_ep15 clip5_30k_4h200_bs192_ep15_lprj" --comparison-pairs "clip5_30k_4h200_bs192_ep15,clip5_30k_4h200_bs192_ep15_lprj" --dataset-preset test_8k --score-thresholds "0.4" --nms-thresholds "0.55" --only-comparison
DEFAULT_ROOT_RUN_DIR="~/lsst_runs"
DEFAULT_RUN_NAMES="lsst5_30k_4h200_bs192_ep50"
DEFAULT_DATASET_PRESET="test_8k"
DEFAULT_SCORE_THRESHOLDS="0.4 0.5"
DEFAULT_NMS_THRESHOLDS="0.55 0.6"
DEFAULT_MAG_LIMIT="gold"
DEFAULT_BUFFERS="1 2"
DEFAULT_LINKING_LENGTHS="1.0 2.0"
DEFAULT_OUTPUT_SUBDIR="plots_auto"
DEFAULT_COMPARISON_SUBDIR="comparisons_auto"
DEFAULT_PLOT_SCRIPT="/u/yse2/DeepDISC_Roman_Rubin/plot_fof_binned.py"

show_usage() {
    cat << EOF
Usage: $0 [OPTIONS]

Workflow Options:
    --run-names "R1 R2 ..."       Space-separated run names (default: "$DEFAULT_RUN_NAMES")
    --dataset-preset NAME          val_4k | val_all | test_8k | test_all | custom (default: $DEFAULT_DATASET_PRESET)
    --root-run-dir PATH            Root run directory (default: $DEFAULT_ROOT_RUN_DIR)
    --score-thresholds "S1 S2"     Space-separated score thresholds (default: "$DEFAULT_SCORE_THRESHOLDS")
    --nms-thresholds "N1 N2"       Space-separated NMS thresholds (default: "$DEFAULT_NMS_THRESHOLDS")
    --combo                        Cartesian product of score x NMS thresholds (default: paired by index)
    --mag-limit NAME               gold | power_law | nominal (default: $DEFAULT_MAG_LIMIT)
    --buffers "B1 B2"              Space-separated buffers in [0,1,2] (default: "$DEFAULT_BUFFERS")
    --linking-lengths "L1 L2"      Space-separated linking lengths (default: "$DEFAULT_LINKING_LENGTHS")
    --mag-min FLOAT                Magnitude min (default: 18.0)
    --mag-max FLOAT                Magnitude max (default: 28.0)
    --mag-bin-width FLOAT          Magnitude bin width (default: 0.5)
    --min-count N                  Min count per bin (default: 1)
    --skip-lsst                    Skip LSST curves on per-run plots

Output Options:
    --output-subdir NAME           Per-run output subdir inside each run dir (default: $DEFAULT_OUTPUT_SUBDIR)
    --comparison-subdir NAME       Comparison output subdir under root-run-dir (default: $DEFAULT_COMPARISON_SUBDIR)

Comparison Options:
    --comparison-pairs "A,B C,D"   Explicit run pairs. If omitted, defaults to:
                                    lsst5_30k_4h200_bs192_ep50,clip5_30k_4h200_bs64_ep50
                                    lsst5_30k_4h200_bs192_ep50,lsst5_all_4h200_bs192_ep20
    --no-default-comparison-pairs  Do not inject default comparison pairs when
                                    --comparison-pairs is omitted

Misc:
    --plot-script PATH             Path to plot_fof_binned.py
    --dry-run                      Print command without executing
    -h, --help                     Show this message

Examples:
    $0 --run-names "lsst5_30k_4h200_bs192_ep50 clip5_30k_4h200_bs64_ep50" \
       --dataset-preset test_8k --score-thresholds "0.4 0.5" --nms-thresholds "0.55 0.6"

    $0 --run-names "lsst5_30k_4h200_bs192_ep50 clip5_30k_4h200_bs64_ep50 lsst5_all_4h200_bs192_ep20" \
       --dataset-preset test_8k --score-thresholds "0.4 0.5" --nms-thresholds "0.55 0.6" \
       --buffers "1 2" --dry-run
EOF
}

ROOT_RUN_DIR="$DEFAULT_ROOT_RUN_DIR"
RUN_NAMES_RAW="$DEFAULT_RUN_NAMES"
DATASET_PRESET="$DEFAULT_DATASET_PRESET"
SCORE_THRESHOLDS="$DEFAULT_SCORE_THRESHOLDS"
NMS_THRESHOLDS="$DEFAULT_NMS_THRESHOLDS"
COMBO="false"
MAG_LIMIT="$DEFAULT_MAG_LIMIT"
BUFFERS_RAW="$DEFAULT_BUFFERS"
LINKING_LENGTHS="$DEFAULT_LINKING_LENGTHS"
MAG_MIN="18.0"
MAG_MAX="28.0"
MAG_BIN_WIDTH="0.5"
MIN_COUNT="1"
SKIP_LSST="false"
OUTPUT_SUBDIR="$DEFAULT_OUTPUT_SUBDIR"
COMPARISON_SUBDIR="$DEFAULT_COMPARISON_SUBDIR"
COMPARISON_PAIRS_RAW=""
NO_DEFAULT_COMPARISON_PAIRS="false"
PLOT_SCRIPT="$DEFAULT_PLOT_SCRIPT"
DRY_RUN="false"


ONLY_COMPARISON="false"
while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-names)            RUN_NAMES_RAW="$2";       shift 2 ;;
        --dataset-preset)       DATASET_PRESET="$2";      shift 2 ;;
        --root-run-dir)         ROOT_RUN_DIR="$2";        shift 2 ;;
        --score-thresholds)     SCORE_THRESHOLDS="$2";    shift 2 ;;
        --nms-thresholds)       NMS_THRESHOLDS="$2";      shift 2 ;;
        --combo)                COMBO="true";             shift ;;
        --mag-limit)            MAG_LIMIT="$2";           shift 2 ;;
        --buffers)              BUFFERS_RAW="$2";         shift 2 ;;
        --linking-lengths)      LINKING_LENGTHS="$2";     shift 2 ;;
        --mag-min)              MAG_MIN="$2";             shift 2 ;;
        --mag-max)              MAG_MAX="$2";             shift 2 ;;
        --mag-bin-width)        MAG_BIN_WIDTH="$2";       shift 2 ;;
        --min-count)            MIN_COUNT="$2";           shift 2 ;;
        --skip-lsst)            SKIP_LSST="true";         shift ;;
        --output-subdir)        OUTPUT_SUBDIR="$2";       shift 2 ;;
        --comparison-subdir)    COMPARISON_SUBDIR="$2";   shift 2 ;;
        --comparison-pairs)     COMPARISON_PAIRS_RAW="$2"; shift 2 ;;
        --no-default-comparison-pairs) NO_DEFAULT_COMPARISON_PAIRS="true"; shift ;;
        --plot-script)          PLOT_SCRIPT="$2";         shift 2 ;;
        --dry-run)              DRY_RUN="true";           shift ;;
        --only-comparison)      ONLY_COMPARISON="true";   shift ;;
        -h|--help)              show_usage; exit 0 ;;
        *)
            echo "Error: Unknown option '$1'"
            show_usage
            exit 1
            ;;
    esac
done

if [[ ! "$MAG_LIMIT" =~ ^(gold|power_law|nominal)$ ]]; then
    echo "Error: --mag-limit must be one of: gold, power_law, nominal"
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

read -ra RUN_NAMES <<< "$RUN_NAMES_RAW"
read -ra SCORE_THRESH_ARR <<< "$SCORE_THRESHOLDS"
read -ra NMS_THRESH_ARR <<< "$NMS_THRESHOLDS"
read -ra BUFFERS_ARR <<< "$BUFFERS_RAW"
read -ra LINKING_LENGTHS_ARR <<< "$LINKING_LENGTHS"

if [[ "${#RUN_NAMES[@]}" -eq 0 ]]; then
    echo "Error: --run-names did not contain any run names"
    exit 1
fi
if [[ "${#SCORE_THRESH_ARR[@]}" -eq 0 ]] || [[ "${#NMS_THRESH_ARR[@]}" -eq 0 ]]; then
    echo "Error: threshold lists cannot be empty"
    exit 1
fi
if [[ "${#BUFFERS_ARR[@]}" -eq 0 ]]; then
    echo "Error: --buffers cannot be empty"
    exit 1
fi
if [[ "${#LINKING_LENGTHS_ARR[@]}" -eq 0 ]]; then
    echo "Error: --linking-lengths cannot be empty"
    exit 1
fi

for b in "${BUFFERS_ARR[@]}"; do
    if [[ ! "$b" =~ ^[0-2]$ ]]; then
        echo "Error: buffer must be one of 0,1,2 (got '$b')"
        exit 1
    fi
done
for s in "${SCORE_THRESH_ARR[@]}"; do
    awk -v x="$s" 'BEGIN{exit !(x+0>=0 && x+0<=1)}' || {
        echo "Error: score threshold out of range [0,1]: $s"
        exit 1
    }
done
for n in "${NMS_THRESH_ARR[@]}"; do
    awk -v x="$n" 'BEGIN{exit !(x+0>=0 && x+0<=1)}' || {
        echo "Error: nms threshold out of range [0,1]: $n"
        exit 1
    }
done
for ll in "${LINKING_LENGTHS_ARR[@]}"; do
    awk -v x="$ll" 'BEGIN{exit !(x+0>0)}' || {
        echo "Error: linking length must be > 0: $ll"
        exit 1
    }
done

# dataset preset is kept for parity with classification workflow.
case "$DATASET_PRESET" in
    val_4k|val_all|test_8k|test_all|custom)
        ;;
    *)
        echo "Error: --dataset-preset must be one of: val_4k, val_all, test_8k, test_all, custom"
        exit 1
        ;;
esac

if [[ -z "$COMPARISON_PAIRS_RAW" ]] && [[ "$NO_DEFAULT_COMPARISON_PAIRS" != "true" ]]; then
    COMPARISON_PAIRS_RAW="lsst5_30k_4h200_bs192_ep50,clip5_30k_4h200_bs64_ep50 lsst5_30k_4h200_bs192_ep50,lsst5_all_4h200_bs192_ep20"
fi
read -ra COMPARISON_PAIRS_ARR <<< "$COMPARISON_PAIRS_RAW"

echo "Plot binned metrics configuration:"
echo "  Root run dir      : $ROOT_RUN_DIR_EXPANDED"
echo "  Dataset preset    : $DATASET_PRESET"
echo "  Run names         : ${RUN_NAMES[*]}"
echo "  Score thresholds  : ${SCORE_THRESH_ARR[*]}"
echo "  NMS thresholds    : ${NMS_THRESH_ARR[*]}"
echo "  Cartesian combos  : $COMBO"
echo "  Mag limit         : $MAG_LIMIT"
echo "  Buffers           : ${BUFFERS_ARR[*]}"
echo "  Linking lengths   : ${LINKING_LENGTHS_ARR[*]}"
echo "  Mag range/bin     : $MAG_MIN to $MAG_MAX (width $MAG_BIN_WIDTH)"
echo "  Min count         : $MIN_COUNT"
echo "  Skip LSST         : $SKIP_LSST"
echo "  Output subdir     : $OUTPUT_SUBDIR"
echo "  Compare subdir    : $COMPARISON_SUBDIR"
echo "  No default pairs  : $NO_DEFAULT_COMPARISON_PAIRS"
echo "  Comparison pairs  : ${COMPARISON_PAIRS_ARR[*]}"

declare -a CMD
CMD=(
    python "$PLOT_SCRIPT"
    --root-run-dir "$ROOT_RUN_DIR_EXPANDED"
    --run-names "${RUN_NAMES[@]}"
    --score-thresholds "${SCORE_THRESH_ARR[@]}"
    --nms-thresholds "${NMS_THRESH_ARR[@]}"
    --buffers "${BUFFERS_ARR[@]}"
    --linking-lengths "${LINKING_LENGTHS_ARR[@]}"
    --mag-limit "$MAG_LIMIT"
    --mag-min "$MAG_MIN"
    --mag-max "$MAG_MAX"
    --mag-bin-width "$MAG_BIN_WIDTH"
    --min-count "$MIN_COUNT"
    --output-subdir "$OUTPUT_SUBDIR"
    --comparison-subdir "$COMPARISON_SUBDIR"
)


if [[ "$COMBO" == "true" ]]; then
    CMD+=(--combo)
fi
if [[ "$SKIP_LSST" == "true" ]]; then
    CMD+=(--skip-lsst)
fi
if [[ "$ONLY_COMPARISON" == "true" ]]; then
    CMD+=(--only-comparison)
fi
if [[ "${#COMPARISON_PAIRS_ARR[@]}" -gt 0 ]]; then
    CMD+=(--comparison-pairs "${COMPARISON_PAIRS_ARR[@]}")
fi

echo "Command:$(printf ' %q' "${CMD[@]}")"

if [[ "$DRY_RUN" == "true" ]]; then
    echo "Dry run only."
    exit 0
fi

"${CMD[@]}"
