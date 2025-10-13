#!/bin/bash
#SBATCH --job-name=download_arr
#SBATCH --output=download_arr_%A_%a.log
#SBATCH --error=download_arr_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8     
#SBATCH --time=08:00:00
#SBATCH --array=0-20

cd ~
# echo $PATH
# # Format the task ID to match the two-digit filename (e.g., 00, 01, 02...)
TASK_ID=$(printf "%02d" $SLURM_ARRAY_TASK_ID)
URL_FILE="url_chunks_${TASK_ID}"

DEST_DIR="/home/shared/hsc/roman_lsst/original_fits"
echo "Job $SLURM_ARRAY_TASK_ID: Starting download of URLs from $URL_FILE"
# # aria2c on this job's dedicated chunk file
# # parallelism is now across jobs
aria2c \
    --dir="$DEST_DIR" \
    --input-file="$URL_FILE" \
    -j 16 \
    -x 16 \
    --file-allocation=none

echo "Job $SLURM_ARRAY_TASK_ID: Finished download at $(date)"
