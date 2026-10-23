#!/bin/bash
#SBATCH -J lsst5_ups_gs_gold
#SBATCH --output=/projects/bfhm/yse2/logs/grid_search/lsst5_ups_gs_gold.%j.%N.out
#SBATCH --error=/projects/bfhm/yse2/logs/grid_search/lsst5_ups_gs_gold.%j.%N.err
#SBATCH --account=bdsp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --constraint="work"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --mail-user=yse2@illinois.edu
#SBATCH --mail-type=ALL

# echo "## Starting Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "## Job ID: $SLURM_JOB_ID"
echo "## Node: $(hostname)"
echo "## Started: $(date)"

cd /u/yse2/DeepDISC_Roman_Rubin/

BUFFER=2
MAG=25.3
TRUTH_MAG_LIMIT=$(echo "$MAG + $BUFFER" | bc)
echo "=========================================================="
echo "Running Gold + Buffer $BUFFER (Truth Mag Limit: $TRUTH_MAG_LIMIT)"
echo "=========================================================="

# python grid_search.py \
#     --run-dir ~/lsst_runs/lsst5_30k_4h200_bs192_ep50 \
#     --test-cats-dir ~/lsst_data/test_cats_lvl5/val_4k/ \
#     --output ~/lsst_runs/lsst5_30k_4h200_bs192_ep50/metrics/gold_buf${BUFFER}/gs_metrics.csv \
#     --mag-limit gold \
#     --buffer $BUFFER
python grid_search.py \
    --run-dir ~/lsst_runs/lsst5_ups_30k_4h200_bs192_ep50 \
    --test-cats-dir ~/lsst_data/test_cats_lvl5/val_4k/ \
    --output ~/lsst_runs/lsst5_ups_30k_4h200_bs192_ep50/metrics/gold_buf${BUFFER}/gs_metrics.csv \
    --mag-limit gold \
    --buffer $BUFFER

echo "## Finished: $(date)"

# MAG=26.22
# --array=0-2
# --output=/projects/bfhm/yse2/logs/grid_search/lsst5_gs_gold.%A_%a.out
# --error=/projects/bfhm/yse2/logs/grid_search/lsst5_gs_gold.%A_%a.err
# echo "## Job Array Task ID: $SLURM_ARRAY_TASK_ID finished."
# BUFFER=$SLURM_ARRAY_TASK_ID