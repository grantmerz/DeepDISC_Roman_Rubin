#!/bin/bash
#SBATCH -J anchor_recall
#SBATCH --output=/projects/bfhm/yse2/logs/inference/anchor_recall_out.%j.%N.out
#SBATCH --error=/projects/bfhm/yse2/logs/inference/anchor_recall_err.%j.%N.err
#SBATCH --account=bdsp-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --gres=gpu:1
#SBATCH --constraint="projects,work"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --mail-user=yse2@illinois.edu
#SBATCH --mail-type=ALL

# Per-GT best-anchor IoU and RPN proposal recall for the native baseline,
# the upsampled (patch 4) model and the PS13 (patch 13) model.
# compute_anchor_recall.py is single-process, so one GPU is enough.
# Outputs: ~/lsst_runs/<run_name>/metrics/anchor_recall/anchor_recall_<run_name>.csv

SUBMIT_TIME=$(squeue -j $SLURM_JOB_ID -h -o "%V")
START_TIME=$(date +"%Y-%m-%dT%H:%M:%S")
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPUs: $CUDA_VISIBLE_DEVICES"

# source /projects/bdsp/miniconda3/etc/profile.d/conda.sh
# conda activate /projects/bdsp/miniconda3/envs/ddbtknv

cd "$SLURM_SUBMIT_DIR"

# GPU and CUDA checks
# clears GPU cache to avoid Cublas errors
echo "Checking CUDA and GPU status..."
python -c "import torch; torch.cuda.empty_cache(); print('PyTorch CUDA version: ', torch.version.cuda); print('CuDNN version:', torch.backends.cudnn.version())"

# output current GPUs
echo "Running on GPUs:"
nvidia-smi --query-gpu=gpu_name,gpu_bus_id,memory.total,memory.used --format=csv

# 0 = the full 4285-image validation set; bright, large objects are rare,
# so the brightest mag/size bins need the full set to be populated
NUM_IMAGES=0
MODEL_TYPES=(base_30k ups_30k ups_30k_ps13)

for MT in "${MODEL_TYPES[@]}"; do
    echo "===== ${MT}: $(date) ====="
    python compute_anchor_recall.py \
        --model-type ${MT} \
        --num-images ${NUM_IMAGES} \
        --batch-size 8 \
        --num-workers 16
done

echo "Job Finished: $(date)"
