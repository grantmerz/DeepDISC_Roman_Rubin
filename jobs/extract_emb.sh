#!/bin/bash
#SBATCH -J test_extract_s30k
#SBATCH --output=/projects/bfhm/yse2/logs/extract_emb/test/test_extract_s30k.%j.%N.out
#SBATCH --error=/projects/bfhm/yse2/logs/extract_emb/test/test_extract_s30k.%j.%N.err
#SBATCH --account=bdsp-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:4
#SBATCH --constraint="projects,work"
#SBATCH --exclude=gpua003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32       # 64 CPUs divided by GPUs for data loading
#SBATCH --mem=64G
#SBATCH --time=00:45:00
#SBATCH --mail-user=yse2@illinois.edu
#SBATCH --mail-type=ALL

SUBMIT_TIME=$(squeue -j $SLURM_JOB_ID -h -o "%V")
START_TIME=$(date +"%Y-%m-%dT%H:%M:%S")
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# Use a job-unique rendezvous port for torch.distributed.
# Keep it in an unprivileged range and stable within this job.
if [ -n "$SLURM_JOB_ID" ]; then
    export MASTER_PORT=$((15000 + (SLURM_JOB_ID % 45000)))
else
    export MASTER_PORT=$((15000 + (RANDOM % 45000)))
fi
echo "MASTER_PORT: $MASTER_PORT"

cd ~
# GPU and CUDA checks
# clears GPU cache to avoid Cublas errors
echo "Checking CUDA and GPU status..."
python -c "import torch; torch.cuda.empty_cache(); print('PyTorch CUDA version: ', torch.version.cuda); print('CuDNN version:', torch.backends.cudnn.version())"

# output current GPUs
echo "Running on GPUs:"
nvidia-smi --query-gpu=gpu_name,gpu_bus_id,memory.total,memory.used --format=csv
echo "Defining inference parameters..."

# Set up variables for extract_pre_nms_embeddings.py
# RUN_NAME is required. CFGFILE and TEST_DATA_FN are optional:
#   - If RUN_NAME is in the script's RUN_DEFAULTS, config is auto-resolved.
#   - If RUN_NAME is not in RUN_DEFAULTS, you must set CFGFILE.
#   - To override the default config, set CFGFILE explicitly.
#   - TEST_DATA_FN is optional; if not set, the Python script will auto-resolve based on DATA_SPLIT.
# You can override these via environment variables if needed.
# RUN_NAMES can be set as a comma-separated list to execute multiple runs in one job.
RUN_NAME="${RUN_NAME:-clip5_30k_4h200_bs192_ep15}"
RUN_NAMES="${RUN_NAMES:-$RUN_NAME}"
# Use override env vars if set (from submit_extract_emb.sh), else fall back to CFGFILE/TEST_DATA_FN
CFGFILE="${CFG_FILE_OVERRIDE:-${CFGFILE}}"
DATA_SPLIT="${DATA_SPLIT:-test}"
SAVE_REPR="${SAVE_REPR:-z_q}"
NUM_GPUS="${NUM_GPUS:-4}"
TEST_DATA_FN="${TEST_DATA_FN_OVERRIDE:-${TEST_DATA_FN}}"
EMB_CHUNK_PROP_BUDGET=250000
export EMB_CHUNK_PROP_BUDGET
echo "Launching extract_pre_nms_embeddings.py job..."
echo "Run names: ${RUN_NAMES}"
if [ -n "$CFGFILE" ]; then
    echo "Config file: ${CFGFILE}"
else
    echo "Config file: (auto-resolved by script if needed)"
fi
echo "Data split: ${DATA_SPLIT}"
echo "Save repr: ${SAVE_REPR}"
if [ -n "$TEST_DATA_FN" ]; then
    echo "Test data file: ${TEST_DATA_FN}"
else
    echo "Test data file: (auto-resolved by script if needed)"
fi
echo "Num GPUs: ${NUM_GPUS}"

ulimit -n 131072

# Run each requested run sequentially within this single SLURM job
IFS=',' read -r -a RUN_ARRAY <<< "$RUN_NAMES"
TOTAL_RUNS=0
for RUN in "${RUN_ARRAY[@]}"; do
    RUN_TRIMMED="${RUN#${RUN%%[![:space:]]*}}"
    RUN_TRIMMED="${RUN_TRIMMED%${RUN_TRIMMED##*[![:space:]]}}"
    if [ -n "$RUN_TRIMMED" ]; then
        TOTAL_RUNS=$((TOTAL_RUNS + 1))
    fi
done

CURRENT_RUN=0
for RUN in "${RUN_ARRAY[@]}"; do
    RUN_TRIMMED="${RUN#${RUN%%[![:space:]]*}}"
    RUN_TRIMMED="${RUN_TRIMMED%${RUN_TRIMMED##*[![:space:]]}}"
    if [ -z "$RUN_TRIMMED" ]; then
        continue
    fi
    CURRENT_RUN=$((CURRENT_RUN + 1))
    echo "============================================================"
    echo "Starting run ${CURRENT_RUN}/${TOTAL_RUNS}: ${RUN_TRIMMED}"
    PY_ARGS=(--run_name "${RUN_TRIMMED}" --data_split "${DATA_SPLIT}" --save_repr "${SAVE_REPR}" --num_gpus "${NUM_GPUS}")
    if [ -n "$CFGFILE" ]; then
        PY_ARGS+=(--cfgfile "${CFGFILE}")
    fi
    if [ -n "$TEST_DATA_FN" ]; then
        PY_ARGS+=(--test_data_fn "${TEST_DATA_FN}")
    fi
    python extract_pre_nms_embeddings.py "${PY_ARGS[@]}"
    echo "Finished run ${CURRENT_RUN}/${TOTAL_RUNS}: ${RUN_TRIMMED}"
done

END_TIME=$(date +"%Y-%m-%dT%H:%M:%S")
ELAPSED_SEC=$(( $(date -d "$END_TIME" +%s) - $(date -d "$START_TIME" +%s) ))
ELAPSED_FMT=$(printf "%02d hours, %02d minutes, %02d seconds" \
    $((ELAPSED_SEC/3600)) \
    $(( (ELAPSED_SEC%3600)/60 )) \
    $((ELAPSED_SEC%60)) )
echo "SUBMIT TIME          START TIME           END TIME"
echo "$SUBMIT_TIME  $START_TIME  $END_TIME"
echo 
echo "ELAPSED TIME: $ELAPSED_FMT" 
echo "============================================================"
 
