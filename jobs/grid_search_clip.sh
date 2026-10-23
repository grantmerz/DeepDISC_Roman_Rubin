#!/bin/bash
#SBATCH -J lsst5_ups_gs_gold
#SBATCH --output=/projects/bfhm/yse2/logs/grid_search/lsst5_ups_gs_gold.%A_%a.out
#SBATCH --error=/projects/bfhm/yse2/logs/grid_search/lsst5_ups_gs_gold.%A_%a.err
#SBATCH --account=bdsp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --constraint="work"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --array=1-2
#SBATCH --mail-user=yse2@illinois.edu
#SBATCH --mail-type=ALL

echo "## Starting Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "## Job ID: $SLURM_JOB_ID"
echo "## Node: $(hostname)"
echo "## Started: $(date)"

cd ~/DeepDISC_Roman_Rubin/

BUFFER=$SLURM_ARRAY_TASK_ID
# BUFFER=1
MAG=25.3
# MAG=26.22
TRUTH_MAG_LIMIT=$(echo "$MAG + $BUFFER" | bc)
echo "=========================================================="
echo "Running Gold + Buffer(s) $BUFFER (Truth Mag Limit: $TRUTH_MAG_LIMIT)"
echo "=========================================================="

# python grid_search.py \
#     --run-dir ~/lsst_runs/comb_30k_4h200_bs144_ep50 \
#     --test-cats-dir ~/lsst_data/test_cats_lvl5/val_4k/ \
#     --output ~/lsst_runs/comb_30k_4h200_bs144_ep50/metrics/gold_buf${BUFFER}/gs_metrics.csv \
#     --mag-limit gold \
#     --buffer $BUFFER \

python grid_search.py \
    --run-dir ~/lsst_runs/lsst5_ups_30k_4h200_bs192_ep50 \
    --test-cats-dir ~/lsst_data/test_cats_lvl5/val_4k/ \
    --output ~/lsst_runs/lsst5_ups_30k_4h200_bs192_ep50/metrics/gold_buf${BUFFER}/gs_metrics.csv \
    --mag-limit gold \
    --buffer $BUFFER \
    # --score-thresholds 0.45 0.5 0.55 0.6 0.65 \
    # --nms-thresholds 0.45 0.5 0.55 0.6 0.65

# python grid_search.py \
#     --run-dir ~/lsst_runs/distill_30k_4h200_bs192_ep50 \
#     --test-cats-dir ~/lsst_data/test_cats_lvl5/val_4k/ \
#     --output ~/lsst_runs/distill_30k_4h200_bs192_ep50/metrics/gold_buf${BUFFER}/gs_metrics.csv \
#     --mag-limit gold \
#     --buffer $BUFFER
# python grid_search.py \
#     --run-dir ~/lsst_runs/clip5_flatten_30k_4h200_bs64_ep15_resume \
#     --test-cats-dir ~/lsst_data/test_cats_lvl5/val_4k/ \
#     --output ~/lsst_runs/clip5_flatten_30k_4h200_bs64_ep15_resume/metrics/gold_buf${BUFFER}/gs_metrics.csv \
#     --mag-limit gold \
#     --buffer $BUFFER

echo "## Job Array Task ID: $SLURM_ARRAY_TASK_ID finished."
echo "## Finished: $(date)"

