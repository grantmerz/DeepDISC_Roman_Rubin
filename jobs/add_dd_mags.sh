#!/bin/bash
#SBATCH -J s30k_eval_mags
#SBATCH --output=/projects/bfhm/yse2/logs/add_mags/s30k_eval_mags.%j.%N.out
#SBATCH --error=/projects/bfhm/yse2/logs/add_mags/s30k_eval_mags.%j.%N.err
#SBATCH --account=bfhm-delta-cpu
#SBATCH --partition=cpu
#SBATCH --constraint="work"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH --mail-user=yse2@illinois.edu
#SBATCH --mail-type=ALL

echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

cd ~/DeepDISC_Roman_Rubin/
# lsst5_all: ~2.2 hrs w/ 16 32C, 64G for val of 15k
RUN_NAME="${RUN_NAME:-lsst5_30k_4h200_bs192_ep50}"
ROOT_RUN_DIR="${ROOT_RUN_DIR:-~/lsst_runs/}"
PREDS_DIR="${PREDS_DIR:-eval}"
FILENAMES_CSV="${FILENAMES_CSV:-$HOME/val4k_fns.csv}"
IO_THREADS="${IO_THREADS:-16}"
N_PROCESSES="${N_PROCESSES:-16}"

DEFAULT_SCORE_THRESHOLDS="0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"
DEFAULT_NMS_THRESHOLDS="0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75"
IFS=' ' read -ra SCORE_THRESHOLDS <<< "${SCORE_THRESHOLDS_STR:-$DEFAULT_SCORE_THRESHOLDS}"
IFS=' ' read -ra NMS_THRESHOLDS <<< "${NMS_THRESHOLDS_STR:-$DEFAULT_NMS_THRESHOLDS}"

EXTRA_FLAGS=()
if [[ "${RESUME:-false}" == "true" ]]; then
    EXTRA_FLAGS+=(--resume)
fi
if [[ "${NO_COMBO:-false}" == "true" ]]; then
    EXTRA_FLAGS+=(--no_combo)
fi

echo "Running add_dd_mags.py"
echo "Run name: ${RUN_NAME}"
echo "Root run dir: ${ROOT_RUN_DIR}"
echo "Preds dir: ${PREDS_DIR}"
echo "Filenames CSV: ${FILENAMES_CSV}"
echo "IO threads: ${IO_THREADS}"
echo "Worker processes: ${N_PROCESSES}"
echo "Score thresholds: ${SCORE_THRESHOLDS[*]}"
echo "NMS thresholds: ${NMS_THRESHOLDS[*]}"
echo "No combo: ${NO_COMBO:-false}"
echo "Resume: ${RESUME:-false}"

python add_dd_mags.py \
    --run_name "${RUN_NAME}" \
    --root_run_dir "${ROOT_RUN_DIR}" \
    --preds_dir "${PREDS_DIR}" \
    --filenames_csv "${FILENAMES_CSV}" \
    --io_threads "${IO_THREADS}" \
    --n_processes "${N_PROCESSES}" \
    --score_thresholds "${SCORE_THRESHOLDS[@]}" \
    --nms_thresholds "${NMS_THRESHOLDS[@]}" \
    "${EXTRA_FLAGS[@]}"

echo "Adding magnitudes script finished."
echo "Job Finished: $(date)"
