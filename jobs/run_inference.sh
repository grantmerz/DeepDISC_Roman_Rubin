#!/bin/bash
#SBATCH -J eval_infer_s30k
#SBATCH --output=/projects/bfhm/yse2/logs/inference/eval/eval_infer_s30k.%j.%N.out
#SBATCH --error=/projects/bfhm/yse2/logs/inference/eval/eval_infer_s30k.%j.%N.err
#SBATCH --account=bdsp-delta-gpu
#SBATCH --partition=gpuA100x4
#SBATCH --gres=gpu:4
#SBATCH --constraint="projects,work"
#SBATCH --exclude=gpua003
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32       # 64 CPUs divided by GPUs for data loading
#SBATCH --mem=64G
#SBATCH --time=04:30:00
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
# Parameters are read from environment variables when set by submit_run_inference.sh,
# falling back to these defaults when the script is run directly.
# MODEL_TYPE="standard_30k"
MODEL_TYPE="${MODEL_TYPE:-standard_30k}"
# MODEL_TYPE="clip_30k"
# DATA_SPLIT="eval"
DATA_SPLIT="${DATA_SPLIT:-eval}"
TOPK_PER_IMG="${TOPK_PER_IMG:-2000}"
DATA_ROOT_DIR="${DATA_ROOT_DIR:-~/lsst_data}"
ANNS_FOLDER="${ANNS_FOLDER:-annotations_lvl5}"

# locations of config and data files are determined by model type defaults if not provided
case $MODEL_TYPE in
    "standard_30k")
        CFG_FILE="/u/yse2/deepdisc/configs/solo/swin_lsst_30k.py"
        RUN_NAME="lsst5_30k_4h200_bs192_ep50"
        EVAL_DATA_FN="${DATA_ROOT_DIR}/${ANNS_FOLDER}/val_4k_keypoints.json"
        TEST_DATA_FN="${DATA_ROOT_DIR}/${ANNS_FOLDER}/test_8k_keypoints.json"
        ;;
    "standard_all")
        CFG_FILE="/u/yse2/deepdisc/configs/solo/swin_lsst_100k.py"
        RUN_NAME="lsst5_all_4h200_bs192_ep20"
        EVAL_DATA_FN="${DATA_ROOT_DIR}/${ANNS_FOLDER}/val_keypoints.json"
        TEST_DATA_FN="${DATA_ROOT_DIR}/${ANNS_FOLDER}/test_keypoints.json"
        ;;
    "clip_30k")
        CFG_FILE="/u/yse2/deepdisc/configs/solo/swin_clip_lsst_roman_30k.py"
        RUN_NAME="clip5_30k_4h200_bs64_ep50"
        EVAL_DATA_FN="${DATA_ROOT_DIR}/${ANNS_FOLDER}/val_4k_keypoints.json"
        TEST_DATA_FN="${DATA_ROOT_DIR}/${ANNS_FOLDER}/test_8k_keypoints.json"
        ;;
    "clip_30k_emb")
        CFG_FILE="/u/yse2/deepdisc/configs/solo/swin_clip_lsst_roman_30k.py"
        RUN_NAME="clip5_30k_4h200_bs192_ep15_lprj"
        EVAL_DATA_FN="${DATA_ROOT_DIR}/${ANNS_FOLDER}/val_4k_keypoints.json"
        TEST_DATA_FN="${DATA_ROOT_DIR}/${ANNS_FOLDER}/test_8k_keypoints.json"
        ;;
    "clip_all")
        CFG_FILE="/u/yse2/deepdisc/configs/solo/swin_clip_lsst_roman_100k.py"
        RUN_NAME="clip5_all_4h200_bs64_ep20"
        EVAL_DATA_FN="${DATA_ROOT_DIR}/${ANNS_FOLDER}/val_keypoints.json"
        TEST_DATA_FN="${DATA_ROOT_DIR}/${ANNS_FOLDER}/test_keypoints.json"
        ;;
    "comb_30k")
        CFG_FILE="/u/yse2/deepdisc/configs/solo/swin_comb_lsst_roman_30k.py"
        RUN_NAME="comb_30k_4h200_bs144_ep50"
        EVAL_DATA_FN="${DATA_ROOT_DIR}/${ANNS_FOLDER}/val_4k_keypoints_wcs.json"
        TEST_DATA_FN="${DATA_ROOT_DIR}/${ANNS_FOLDER}/test_8k_keypoints_wcs.json"
        ;;
    "distill_30k")
        CFG_FILE="/u/yse2/deepdisc/configs/solo/swin_distill_lsst_roman_30k.py"
        RUN_NAME="distill_30k_4h200_bs192_ep50"
        EVAL_DATA_FN="${DATA_ROOT_DIR}/${ANNS_FOLDER}/val_4k_keypoints.json"
        TEST_DATA_FN="${DATA_ROOT_DIR}/${ANNS_FOLDER}/test_8k_keypoints.json"
        ;;
    *)
        echo "Unknown model type: $MODEL_TYPE"
        exit 1
        ;;
esac

# Optional override from submit_run_inference.sh
RUN_NAME="${RUN_NAME_OVERRIDE:-$RUN_NAME}"

if [ "$DATA_SPLIT" = "eval" ]; then
    DATA_FN=$EVAL_DATA_FN
else
    DATA_FN=$TEST_DATA_FN
fi

# Thresholds are read from space-separated env var strings set by submit_run_inference.sh.
# When running the script directly, edit the values after the :- below
DEFAULT_SCORE_THRESHOLDS="0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9"
DEFAULT_NMS_THRESHOLDS="0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75"
IFS=' ' read -ra SCORE_THRESHOLDS <<< "${SCORE_THRESHOLDS_STR:-$DEFAULT_SCORE_THRESHOLDS}"
IFS=' ' read -ra NMS_THRESHOLDS <<< "${NMS_THRESHOLDS_STR:-$DEFAULT_NMS_THRESHOLDS}"

# Build optional flags
EXTRA_FLAGS=""
[[ -n "${CFGFILE:-}" ]]      && EXTRA_FLAGS="$EXTRA_FLAGS --cfgfile ${CFGFILE}"
[[ "${RESUME:-false}"   == "true" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --resume"
[[ "${NO_COMBO:-false}" == "true" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --no_combo"
[[ "${EXTRACT_EMBEDDINGS:-false}" == "true" ]] && EXTRA_FLAGS="$EXTRA_FLAGS --extract_embeddings"

echo "Launching DeepDISC inference job..."
echo "Model type: ${MODEL_TYPE}"
echo "Data split: ${DATA_SPLIT}"
[[ -n "${CFGFILE:-}" ]] && echo "Config override: ${CFGFILE}"
echo "Config file: ${CFG_FILE}"
echo "Run name: ${RUN_NAME}"
echo "Data file: ${DATA_FN}"
echo "Top K per image: ${TOPK_PER_IMG}"
echo "Score thresholds: ${SCORE_THRESHOLDS[@]}"
echo "NMS thresholds: ${NMS_THRESHOLDS[@]}"
echo "Extract embeddings: ${EXTRACT_EMBEDDINGS:-false}"

ulimit -n 131072
# Run inference with all threshold combos in a single call.
# Pass multiple values to --score_thresholds / --nms_thresholds for a grid search,
# or use --no_combo to pair them by index instead of the full cartesian product
python run_inference.py \
    --model_type ${MODEL_TYPE} \
    --run_name "${RUN_NAME}" \
    --data_split ${DATA_SPLIT} \
    --topk_per_img ${TOPK_PER_IMG} \
    --score_thresholds "${SCORE_THRESHOLDS[@]}" \
    --nms_thresholds "${NMS_THRESHOLDS[@]}" \
    --num_gpus "${NUM_GPUS:-4}" \
    ${EXTRA_FLAGS}

# To override model paths explicitly (bypassing the per-model-type defaults):
# python run_inference.py \
#     --model_type ${MODEL_TYPE} \
#     --data_split ${DATA_SPLIT} \
#     --cfgfile ${CFG_FILE} \
#     --run_name ${RUN_NAME} \
#     --test_data_fn ${DATA_FN} \
#     --topk_per_img ${TOPK_PER_IMG} \
#     --score_thresholds "${SCORE_THRESHOLDS[@]}" \
#     --nms_thresholds "${NMS_THRESHOLDS[@]}" \
#     --num_gpus "${NUM_GPUS:-4}" \
#     ${EXTRA_FLAGS}

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