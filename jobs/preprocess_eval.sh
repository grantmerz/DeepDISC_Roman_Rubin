#!/bin/bash
#SBATCH --job-name="evaluate_model"
#SBATCH --output="eval_out.%j.%N.out"
#SBATCH --error="eval_err.%j.%N.err"
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=90G
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --time=0:50:00
#SBATCH --mail-user=yse2@illinois.edu
#SBATCH --mail-type=ALL

module load conda_base
conda activate deepdisc
cd ~

echo "LD_LIBRARY_PATH BEFORE: $LD_LIBRARY_PATH\n"

# gets rid of CUBLAS_GEMM_UNSUPPORTED ERRORS by removing incompatible cuda versions
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "cuda-11.3" | tr '\n' ':')
export LD_LIBRARY_PATH=$(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep -v "cuda-12" | tr '\n' ':')

echo "LD_LIBRARY_PATH AFTER: $LD_LIBRARY_PATH\n"

# clears GPU cache to avoid Cublas errors
python -c "import torch; torch.cuda.empty_cache(); print('cuda version: ', torch.version.cuda); print(torch.backends.cudnn.version())" 
# Driver Version: 550.54.15      CUDA Version: 12.4  on node6 and node16 and node 14
# output current GPUs
echo "Running on GPUs:"
nvidia-smi  | head -n10
nvidia-smi --query-gpu=gpu_name,gpu_bus_id --format=csv

# RUN 5 - Lvl5

# python preprocess_eval_data.py --folder annotations --scale False --run_dir run5_sm_dlvl5 --config_file ./deepdisc/configs/solo/swin_lsst.py --model_path ./lsst_runs/run5_sm_dlvl5/lsst_dlvl5.pth

# python preprocess_eval_data.py --folder annotationsc-ups --scale True --run_dir run5_ups_roman_dlvl5 --config_file ./deepdisc/configs/solo/swinc_lsst_ups.py --model_path ./lsst_runs/run5_ups_roman_dlvl5/lsstc_ups_dlvl5.pth

# RUN 6 - Lvl6

python preprocess_eval_data.py --folder annotations-lvl2 --scale False --run_dir run6_sm_dlvl2 --config_file ./deepdisc/configs/solo/swin_lsst.py --model_path ./lsst_runs/run6_sm_dlvl2/lsst_dlvl2.pth

python preprocess_eval_data.py --folder annotationsc-ups-lvl2 --scale True --run_dir run6_ups_roman_dlvl2 --config_file ./deepdisc/configs/solo/swinc_lsst_ups.py --model_path ./lsst_runs/run6_ups_roman_dlvl2/lsstc_ups_dlvl2.pth


# -test_score_thresh 0.45 --nms 0.5
# exclude=hal01,hal02,hal09

# --exclude=hal09,hal11,hal14
# --nodelist=hal09, hal14
 

 