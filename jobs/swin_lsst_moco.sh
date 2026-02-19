#!/bin/bash
#SBATCH -J moco5_30k_test
#SBATCH --output=/projects/bfhm/yse2/logs/training_logs/moco5_30k_test_out.%j.out
#SBATCH --error=/projects/bfhm/yse2/logs/training_logs/moco5_30k_test_err.%j.err
#SBATCH --account=bdsp-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=4         # Using 4 GPUs on the node
#SBATCH --mem=128G    
#SBATCH --ntasks-per-node=1        # one process per GPU for distributed training
#SBATCH --cpus-per-task=64       # 64 CPUs divided by GPUs for data loading
#SBATCH --reservation=RH8
#SBATCH --constraint="work"
#SBATCH --time=02:00:00
#SBATCH --mail-user=yse2@illinois.edu
#SBATCH --mail-type=ALL

### Environment and Setup
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
echo "Defining training parameters..."
# locations of config and data files
CFG_FILE="/u/yse2/deepdisc/configs/solo/swin_lsst_job_moco.py"
TRAIN_FILE="/u/yse2/lsst_data/annotations_lvl5/train_30k.json" 
EVAL_FILE="/u/yse2/lsst_data/annotations_lvl5/val_4k.json"  

# RUN_NAME="moco5_30k_4h200_bs256_ep50"
# RUN_NAME="moco5_30k_4h200_bs256_ep50_frz"
# RUN_NAME="moco5_30k_4h200_bs256_ep50_frz_4moco"
RUN_NAME="moco5_30k_test"
OUTPUT_DIR="./lsst_runs/${RUN_NAME}"

echo "Launching DeepDISC training job..."

# python run_model_job_moco.py --cfgfile /u/yse2/deepdisc/configs/solo/swin_lsst_job_moco.py --train-metadata /u/yse2/lsst_data/annotations_lvl5/train_30k.json --eval-metadata /u/yse2/lsst_data/annotations_lvl5/val_4k.json --num-gpus 2 --run-name moco5_30k_test --output-dir ./lsst_runs/moco5_30k_test

python run_model_job_moco.py \
    --cfgfile ${CFG_FILE} \
    --train-metadata ${TRAIN_FILE} \
    --eval-metadata ${EVAL_FILE} \
    --num-gpus 4 \
    --run-name ${RUN_NAME} \
    --output-dir ${OUTPUT_DIR}

# python run_model_job_moco.py --cfgfile /u/yse2/deepdisc/configs/solo/swin_lsst_job_moco.py --train-metadata /u/yse2/lsst_data/annotations_lvl5/train_30k.json --eval-metadata /u/yse2/lsst_data/annotations_lvl5/val_4k.json --num-gpus 4 --run-name moco5_30k_test_4g --output-dir ./lsst_runs/moco5_30k_test_4g
# python run_model_job_moco.py --cfgfile /u/yse2/deepdisc/configs/solo/swin_lsst_job_moco.py --train-metadata /u/yse2/lsst_data/annotations_lvl5/train_30k.json --eval-metadata /u/yse2/lsst_data/annotations_lvl5/val_4k.json --num-gpus 2 --run-name moco5_30k_test_npt_molo --output-dir ./lsst_runs/moco5_30k_test_npt_molo
# python run_model_job_moco.py --cfgfile /u/yse2/deepdisc/configs/solo/swin_lsst_job_moco.py --train-metadata /u/yse2/lsst_data/annotations_lvl5/train_30k.json --eval-metadata /u/yse2/lsst_data/annotations_lvl5/val_4k.json --num-gpus 2 --run-name moco5_30k_test_freeze_molo --output-dir ./lsst_runs/moco5_30k_test_freeze_molo
# >> /projects/bfhm/yse2/logs/training_logs/moco5_30k_test_npt_molo.log
# >> /projects/bfhm/yse2/logs/training_logs/moco5_30k_test_freeze_molo.log

# python run_model_job_moco.py --cfgfile /u/yse2/deepdisc/configs/solo/swin_lsst_job_moco.py --train-metadata /u/yse2/lsst_data/annotations_lvl5/train_30k.json --eval-metadata /u/yse2/lsst_data/annotations_lvl5/val_4k.json --num-gpus 2 --run-name moco5_30k_test_freeze --output-dir ./lsst_runs/moco5_30k_test_freeze
echo "Training script finished."
echo "Job Finished: $(date)"
