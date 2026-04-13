#!/bin/bash
#SBATCH -J fof_cls
#SBATCH --output=/projects/bfhm/yse2/logs/fof_classify/fof_cls.%j.%N.out
#SBATCH --error=/projects/bfhm/yse2/logs/fof_classify/fof_cls.%j.%N.err
#SBATCH --account=bfhm-delta-cpu
#SBATCH --partition=cpu
#SBATCH --exclude=cn001,cn002,cn003
#SBATCH --constraint="work"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --mail-user=yse2@illinois.edu
#SBATCH --mail-type=ALL

set -euo pipefail
SUBMIT_TIME=$(squeue -j "$SLURM_JOB_ID" -h -o "%V" || true)
START_TIME=$(date +"%Y-%m-%dT%H:%M:%S")

ROOT_RUN_DIR="${ROOT_RUN_DIR:-~/lsst_runs/}"
RUN_NAME="${RUN_NAME:-lsst5_30k_4h200_bs192_ep50}"
TEST_CATS_DIR="${TEST_CATS_DIR:-~/lsst_data/test_cats_lvl5/val_4k}"
PREDS_DIR="${PREDS_DIR:-eval}"
SCORE_THRESH="${SCORE_THRESH:-0.4}"
NMS_THRESH="${NMS_THRESH:-0.55}"
MAG_LIMIT="${MAG_LIMIT:-gold}"
BUFFER="${BUFFER:-1}"
LINKING_LENGTHS="${LINKING_LENGTHS:-1.0 2.0}"
MATCH_RAD="${MATCH_RAD:-}"
SKIP_LSST="${SKIP_LSST:-false}"
# only need expansion of ~ below to check if they exist or not later on in this script
ROOT_RUN_DIR="${ROOT_RUN_DIR/#\~/$HOME}"
TEST_CATS_DIR="${TEST_CATS_DIR/#\~/$HOME}"
RUN_DIR="$ROOT_RUN_DIR/$RUN_NAME"
if [[ -z "$SCORE_THRESH" ]] || [[ -z "$NMS_THRESH" ]]; then
    echo "Error: SCORE_THRESH and NMS_THRESH must be set" >&2
    exit 1
fi
if [[ ! -d "$RUN_DIR" ]]; then
    echo "Error: RUN_DIR does not exist: $RUN_DIR" >&2
    exit 1
fi
if [[ ! -d "$TEST_CATS_DIR" ]]; then
    echo "Error: TEST_CATS_DIR does not exist: $TEST_CATS_DIR" >&2
    exit 1
fi
PRED_FN="$RUN_DIR/preds/$PREDS_DIR/pred_s${SCORE_THRESH}_n${NMS_THRESH}.json"
if [[ ! -f "$PRED_FN" ]]; then
    echo "Error: Prediction file does not exist: $PRED_FN" >&2
    exit 1
fi

LOG_PREFIX="eval"
if [[ "$PREDS_DIR" == "test" ]]; then
    LOG_PREFIX="test"
fi
METRICS_LOG_DIR="$RUN_DIR/metrics/${MAG_LIMIT}_buf${BUFFER}"
mkdir -p "$METRICS_LOG_DIR"
METRICS_LOG_FILE="${METRICS_LOG_FILE:-$METRICS_LOG_DIR/${LOG_PREFIX}_s${SCORE_THRESH}_n${NMS_THRESH}.log}"
# Mirror job output into the per-run metrics log (same style as manual redirection)
exec > >(tee -a "$METRICS_LOG_FILE") 2>&1
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "Metrics log file: $METRICS_LOG_FILE"

echo "Run dir: $RUN_DIR"
echo "Root run dir: $ROOT_RUN_DIR"
echo "Run name: $RUN_NAME"
echo "Test cats dir: $TEST_CATS_DIR"
echo "Preds dir: $PREDS_DIR"
echo "Score thresh: $SCORE_THRESH"
echo "NMS thresh: $NMS_THRESH"
echo "Mag limit: $MAG_LIMIT"
echo "Buffer: $BUFFER"
echo "Linking lengths: $LINKING_LENGTHS"
echo "Match rad: ${MATCH_RAD:-<per-linking-length>}"
echo "Skip LSST: $SKIP_LSST"
echo "Prediction file: $PRED_FN"

cd /u/yse2/DeepDISC_Roman_Rubin/
read -ra LINKING_LENGTH_ARR <<< "$LINKING_LENGTHS"

CMD=(
    python run_fof_classify.py
    --root-run-dir "$ROOT_RUN_DIR"
    --run-name "$RUN_NAME"
    --preds-dir "$PREDS_DIR"
    --test-cats-dir "$TEST_CATS_DIR"
    --score-thresh "$SCORE_THRESH"
    --nms-thresh "$NMS_THRESH"
    --mag-limit "$MAG_LIMIT"
    --buffer "$BUFFER"
    --linking-lengths "${LINKING_LENGTH_ARR[@]}"
)
if [[ -n "$MATCH_RAD" ]]; then
    CMD+=(--match-rad "$MATCH_RAD")
fi
if [[ "$SKIP_LSST" == "true" ]]; then
    CMD+=(--skip-lsst)
fi

echo "Command:$(printf ' %q' "${CMD[@]}")"
"${CMD[@]}"

END_TIME=$(date +"%Y-%m-%dT%H:%M:%S")
ELAPSED_SEC=$(( $(date -d "$END_TIME" +%s) - $(date -d "$START_TIME" +%s) ))
ELAPSED_FMT=$(printf "%02d hours, %02d minutes, %02d seconds" \
    $((ELAPSED_SEC/3600)) \
    $(((ELAPSED_SEC%3600)/60)) \
    $((ELAPSED_SEC%60)) )

echo "SUBMIT TIME          START TIME           END TIME"
echo "${SUBMIT_TIME:-NA}  $START_TIME  $END_TIME"
echo "ELAPSED TIME: $ELAPSED_FMT"
echo "============================================================"
