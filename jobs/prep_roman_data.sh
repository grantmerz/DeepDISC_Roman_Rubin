#!/bin/bash
#SBATCH --job-name="roman"
#SBATCH --output="prep_roman_out.%j.%N.out"
#SBATCH --error="prep_roman_err.%j.%N.err"
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=96
#SBATCH --time=02:30:00
#SBATCH --array=0-4

echo "Job ID: $SLURM_JOB_ID, Task ID: $SLURM_ARRAY_TASK_ID"

module load conda_base
conda activate deepdisc
cd ~

# Run this in a terminal before submitting job and running the script
# echo "Splitting tiles into 5 files for the job array..."
# mkdir -p "tile_lists_split"
# split -d -n l/5 "700_tiles.txt" "tile_lists_split/tile_list_"

CURRENT_TILE_FILE="tile_lists_split/tile_list_$SLURM_ARRAY_TASK_ID"
echo "This task will process 140 (700/5) tiles from: $CURRENT_TILE_FILE"

python prepare_roman_data.py --tile_list "$CURRENT_TILE_FILE"

echo "Task $SLURM_ARRAY_TASK_ID has completed."