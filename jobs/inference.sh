#!/bin/bash
#SBATCH -J hp_eval
#SBATCH --output=/projects/bfhm/yse2/logs/inference/hp_infer_out.%j.%N.out
#SBATCH --error=/projects/bfhm/yse2/logs/inference/hp_infer_err.%j.%N.err
#SBATCH --account=bdsp-delta-gpu
#SBATCH --partition=gpuA100x8
#SBATCH --gres=gpu:4
#SBATCH --constraint="projects,work"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64       # 64 CPUs divided by GPUs for data loading
#SBATCH --mem=128G
#SBATCH --time=06:30:00
#SBATCH --mail-user=yse2@illinois.edu
#SBATCH --mail-type=ALL

echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

cd ~
# GPU and CUDA checks
# clears GPU cache to avoid Cublas errors
echo "Checking CUDA and GPU status..."
python -c "import torch; torch.cuda.empty_cache(); print('PyTorch CUDA version: ', torch.version.cuda); print('CuDNN version:', torch.backends.cudnn.version())"

# output current GPUs
echo "Running on GPUs:"
nvidia-smi --query-gpu=gpu_name,gpu_bus_id,memory.total,memory.used --format=csv
echo "Defining inference parameters..."
# locations of config and data files
CFG_FILE="/u/yse2/deepdisc/configs/solo/swin_lsst_job.py"
RUN_NAME="lsst5_30k_4h200_bs192_ep50"
TEST_DATA_FN="/u/yse2/lsst_data/annotations_lvl5/test_8k.json"
TOPK_PER_IMG=2000
# SCORE_THRESHOLDS=(0.4)
# NMS_THRESHOLDS=(0.3)
SCORE_THRESHOLDS=(0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9)
NMS_THRESHOLDS=(0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75)

echo "Launching DeepDISC training job..."
echo "Config file: ${CFG_FILE}"
echo "Run name: ${RUN_NAME}"
echo "Test data file: ${TEST_DATA_FN}"
echo "Top K per image: ${TOPK_PER_IMG}"
echo "Score thresholds: ${SCORE_THRESHOLDS[@]}"
echo "NMS thresholds: ${NMS_THRESHOLDS[@]}"

# with default params w/ varying thresholds
python preprocess_eval_data_gpu.py \
    --score_thresholds ${SCORE_THRESHOLDS[@]} \
    --nms_thresholds ${NMS_THRESHOLDS[@]} \
    --num_gpus 4
# with all params
# python preprocess_eval_data_gpu.py \
#     --cfgfile ${CFG_FILE} \
#     --run_name ${RUN_NAME} \
#     --test_data_fn ${TEST_DATA_FN} \
#     --topk_per_img ${TOPK_PER_IMG} \
#     --score_thresholds ${SCORE_THRESHOLDS[@]} \
#     --nms_thresholds ${NMS_THRESHOLDS[@]} \
#     --num_gpus 4 
 