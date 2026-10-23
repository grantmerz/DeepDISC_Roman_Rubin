#!/bin/bash
#SBATCH -J grid_search
#SBATCH --output=/projects/bfhm/yse2/logs/grid_search/%x.%j.%N.out
#SBATCH --error=/projects/bfhm/yse2/logs/grid_search/%x.%j.%N.err
#SBATCH --account=bdsp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --constraint="work"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-user=yse2@illinois.edu
#SBATCH --mail-type=ALL
#
# Generic grid_search.py batch runner. Replaces grid_search.sh / grid_search_clip.sh /
# grid_search_all.sh -- one script instead of one copy-paste per run; every difference
# between those three (job name, array, mem, time, run-dir, test-cats-dir, mag-limit,
# buffer, ...) is now an sbatch/script-CLI override instead of a separate file.
#
# Usage:
#   sbatch jobs/gs.sh [model] [mag] [buffer] [options...]
#
# Positional (all optional):
#   model  : lsst30k | clip30k | comb30k | ups30k | lsstall   (default: ups30k)
#            resolves --run-dir and --test-cats-dir; pass --run-dir/--test-cats-dir
#            directly instead for a run not in this map (e.g. distill_30k, clip5_flatten).
#   mag    : gold | power_law | ... (any blend_metrics.MAG_LIMITS key)  (default: gold)
#   buffer : 0 | 1 | 2                       (default: $SLURM_ARRAY_TASK_ID, else 2)
#
# Submit examples (each reproduces one of the retired scripts):
### ========= PAPER RUNS =========
#   sbatch --job-name=lsst5_gs_gold jobs/gs.sh lsst30k gold 2
#   sbatch --job-name=lsst5_ups_gs_gold jobs/gs.sh ups30k gold 2
#   sbatch --job-name=comb5_gs_gold jobs/gs.sh comb30k gold 2
### ==============================
#   sbatch --job-name=lsst5_ups_gs_gold --array=1-2 --time=00:45:00 jobs/gs.sh ups30k gold
#   sbatch --job-name=lsst5_all_gs_gold --array=0-2 --mem=64G --time=02:00:00 jobs/gs.sh lsstall gold
#
#   # ad hoc run not in the model map
#   sbatch jobs/gs.sh --run-dir ~/lsst_runs/distill_30k_4h200_bs192_ep50 \
#       --test-cats-dir ~/lsst_data/test_cats_lvl5/val_4k/ --mag-limit gold --buffer 2
#
# --buffer defaults to $SLURM_ARRAY_TASK_ID when submitted as a job array (so one
# --array=0-2 submission sweeps buffers 0/1/2 as before); the positional or --buffer
# flag overrides it. --output defaults to <run-dir>/metrics/<prefix>_buf<buffer>/
# gs_metrics.csv (prefix: gold->gold, power_law->pl, else the mag-limit as given)
# matching the folder convention plot_grid_search.py/select_thresh.py already expect.
#
# Options (forwarded to grid_search.py; --run-dir/--test-cats-dir/--mag-limit/--buffer
# override the positional/model-map values above):
#   --run-dir DIR --test-cats-dir DIR --output PATH --mag-limit MAG --buffer N
#   --linking-lengths "LL1 LL2 ..." --score-thresholds "S1 S2 ..." --nms-thresholds "N1 N2 ..."
#   --preds-dir DIR --match-rad R --dmag-sig D --workers N --include-counts --skip-lsst

set -euo pipefail
cd /u/yse2/DeepDISC_Roman_Rubin/

echo "## Job ID: ${SLURM_JOB_ID:-none}"
echo "## Array Task ID: ${SLURM_ARRAY_TASK_ID:-none}"
echo "## Node: $(hostname)"
echo "## Started: $(date)"

VAL_TEST_CATS_DIR="$HOME/lsst_data/test_cats_lvl5/val_4k/"

declare -A MODEL_DIRS=(
    [lsst30k]="$HOME/lsst_runs/lsst5_30k_4h200_bs192_ep50"
    [ups30k]="$HOME/lsst_runs/lsst5_ups_30k_4h200_bs192_ep50"
    [comb30k]="$HOME/lsst_runs/comb_30k_4h200_bs144_ep50"
    [clip30k]="$HOME/lsst_runs/clip5_30k_4h200_bs64_ep50"
    [lsstall]="$HOME/lsst_runs/lsst5_all_4h200_bs192_ep20"
)
declare -A MODEL_TEST_CATS=(
    [lsst30k]="$VAL_TEST_CATS_DIR"
    [clip30k]="$VAL_TEST_CATS_DIR"
    [comb30k]="$VAL_TEST_CATS_DIR"
    [ups30k]="$VAL_TEST_CATS_DIR"
    [lsstall]="$HOME/lsst_data/test_cats_lvl5/val_all/"
)

# Positional args are consumed only while they don't look like a --flag, so
# `gs.sh --run-dir ...` (no model/mag/buffer given) falls straight through to the
# flag parser below instead of misreading "--run-dir" itself as the model name.
MODEL="ups30k"
[[ $# -gt 0 && "$1" != --* ]] && { MODEL="$1"; shift; }
if [[ -n "${MODEL_DIRS[$MODEL]+_}" ]]; then
    RUN_DIR="${MODEL_DIRS[$MODEL]}"
    TEST_CATS_DIR="${MODEL_TEST_CATS[$MODEL]}"
else
    echo "Unknown model '$MODEL'. Choose: lsst30k | clip30k | comb30k | ups30k | lsstall" \
        "(or pass --run-dir/--test-cats-dir directly for a run not in this map)" >&2
    exit 1
fi
MAG_LIMIT="gold"
[[ $# -gt 0 && "$1" != --* ]] && { MAG_LIMIT="$1"; shift; }
BUFFER="${SLURM_ARRAY_TASK_ID:-2}"
[[ $# -gt 0 && "$1" != --* ]] && { BUFFER="$1"; shift; }

OUTPUT=""
LINKING_LENGTHS=""
SCORE_THRESHOLDS=""
NMS_THRESHOLDS=""
PREDS_DIR=""
MATCH_RAD=""
DMAG_SIG=""
WORKERS=""
INCLUDE_COUNTS_FLAG=""
SKIP_LSST_FLAG=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir) RUN_DIR="$2"; shift 2 ;;
        --test-cats-dir) TEST_CATS_DIR="$2"; shift 2 ;;
        --output) OUTPUT="$2"; shift 2 ;;
        --mag-limit) MAG_LIMIT="$2"; shift 2 ;;
        --buffer) BUFFER="$2"; shift 2 ;;
        --linking-lengths) LINKING_LENGTHS="$2"; shift 2 ;;
        --score-thresholds) SCORE_THRESHOLDS="$2"; shift 2 ;;
        --nms-thresholds) NMS_THRESHOLDS="$2"; shift 2 ;;
        --preds-dir) PREDS_DIR="$2"; shift 2 ;;
        --match-rad) MATCH_RAD="$2"; shift 2 ;;
        --dmag-sig) DMAG_SIG="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --include-counts) INCLUDE_COUNTS_FLAG="--include-counts"; shift ;;
        --skip-lsst) SKIP_LSST_FLAG="--skip-lsst"; shift ;;
        *) echo "Unknown option: $1" >&2; exit 1 ;;
    esac
done

[[ -n "$RUN_DIR" ]] || { echo "--run-dir is required" >&2; exit 1; }
[[ -n "$TEST_CATS_DIR" ]] || { echo "--test-cats-dir is required" >&2; exit 1; }

if [[ -z "$OUTPUT" ]]; then
    case "$MAG_LIMIT" in
        gold) PFX="gold" ;;
        power_law) PFX="pl" ;;
        *) PFX="$MAG_LIMIT" ;;
    esac
    OUTPUT="${RUN_DIR}/metrics/${PFX}_buf${BUFFER}/gs_metrics.csv"
fi

echo "=========================================================="
echo "run-dir=$RUN_DIR  mag-limit=$MAG_LIMIT  buffer=$BUFFER"
echo "output=$OUTPUT"
echo "=========================================================="

python grid_search.py \
    --run-dir "$RUN_DIR" \
    --test-cats-dir "$TEST_CATS_DIR" \
    --output "$OUTPUT" \
    --mag-limit "$MAG_LIMIT" \
    --buffer "$BUFFER" \
    ${LINKING_LENGTHS:+--linking-lengths $LINKING_LENGTHS} \
    ${SCORE_THRESHOLDS:+--score-thresholds $SCORE_THRESHOLDS} \
    ${NMS_THRESHOLDS:+--nms-thresholds $NMS_THRESHOLDS} \
    ${PREDS_DIR:+--preds-dir "$PREDS_DIR"} \
    ${MATCH_RAD:+--match-rad "$MATCH_RAD"} \
    ${DMAG_SIG:+--dmag-sig "$DMAG_SIG"} \
    ${WORKERS:+--workers "$WORKERS"} \
    $INCLUDE_COUNTS_FLAG $SKIP_LSST_FLAG

echo "## Finished: $(date)"
