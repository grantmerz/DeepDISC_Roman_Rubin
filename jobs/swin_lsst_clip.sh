#!/bin/bash
#SBATCH -J clip_30k_4h200_bs64_ep50
#SBATCH --output=/projects/bdsp/g4merz/logs/training_logs/clip_30k_4h200_bs64_ep50.%j.out
#SBATCH --error=/projects/bdsp/g4merz/logs/training_logs/clip_30k_4h200_bs64_ep50.%j.err
#SBATCH --account=bdsp-delta-gpu
#SBATCH --partition=gpuH200x8
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4         # Using 4 GPUs on the node
#SBATCH --mem=128G    
#SBATCH --ntasks-per-node=1        # one process per GPU for distributed training
#SBATCH --cpus-per-task=64       # 64 CPUs divided by GPUs for data loading
#SBATCH --constraint="work"
#SBATCH --time=15:00:00
#SBATCH --mail-user=gmerz3@illinois.edu
#SBATCH --mail-type=ALL

### Environment and Setup
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# GPU and CUDA checks
# clears GPU cache to avoid Cublas errors
echo "Checking CUDA and GPU status..."
python -c "import torch; torch.cuda.empty_cache(); print('PyTorch CUDA version: ', torch.version.cuda); print('CuDNN version:', torch.backends.cudnn.version())"

# output current GPUs
echo "Running on GPUs:"
nvidia-smi --query-gpu=gpu_name,gpu_bus_id,memory.total,memory.used --format=csv
echo "Defining training parameters..."
# locations of config and data files
CFG_FILE="/work/hdd/bdsp/g4merz/RubinRoman/configs/solo/swin_lsst_job_keypoint_clip_roi_30k.py"
TRAIN_FILE="/work/hdd/bdsp/yse2/lsst_data/annotations_lvl5/train_30k_keypoints.json"
EVAL_FILE="/work/hdd/bdsp/yse2/lsst_data/annotations_lvl5/val_4k_keypoints.json"

RUN_NAME="clip_30k_4h200_bs64_ep50_new"
#RUN_NAME="roman_rubin_30k_clip_roi_test"
OUTPUT_DIR="/work/hdd/bdsp/yse2/lsst_runs/${RUN_NAME}"

echo "Launching DeepDISC training job..."


python run_model_job_30k.py \
    --cfgfile ${CFG_FILE} \
    --train-metadata ${TRAIN_FILE} \
    --eval-metadata ${EVAL_FILE} \
    --num-gpus 4 \
    --run-name ${RUN_NAME} \
    --output-dir ${OUTPUT_DIR}

echo "Training script finished."
echo "Job Finished: $(date)"
