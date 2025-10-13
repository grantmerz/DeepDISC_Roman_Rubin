#!/bin/bash
#SBATCH -J lsst_anns
#SBATCH --output=/projects/bfhm/yse2/logs/anns_log/annsf_out.%A_%a.out
#SBATCH --error=/projects/bfhm/yse2/logs/anns_log/annsf_err.%A_%a.err
#SBATCH --account=bfhm-delta-cpu
#SBATCH --partition=cpu
#SBATCH --constraint="work"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=64g
#SBATCH --time=0:45:00
#SBATCH --array=1-10
#SBATCH --mail-user=yse2@illinois.edu
#SBATCH --mail-type=ALL

TILES_PER_TASK=17 # ~4 mins for 6 tiles to finish and 6 for 700 bc 700/6=117 tasks
TILE_LIST_FILE="/u/yse2/tiles_remaining.txt"

echo "## Starting Job Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "## Node: $(hostname)"
cd ~

readarray -t ALL_TILES < "$TILE_LIST_FILE"
NUM_TOTAL_TILES=${#ALL_TILES[@]}

## TILE BATCH FOR THIS TASK
END_INDEX=$((SLURM_ARRAY_TASK_ID * TILES_PER_TASK))
START_INDEX=$((END_INDEX - TILES_PER_TASK))

## PROCESS BATCH OF TILES
for (( i=$START_INDEX; i<$END_INDEX; i++ )); do
    if [[ $i -lt $NUM_TOTAL_TILES ]]; then
        CURRENT_TILE=${ALL_TILES[$i]}
        echo "--> Processing Tile #${i}: $CURRENT_TILE"
        python lsst_anns.py $CURRENT_TILE
        echo "--> Finished Tile #${i}: $CURRENT_TILE"
    fi
done

echo "## Job Array Task ID: $SLURM_ARRAY_TASK_ID finished."


# SINGLE TILE CASE
# TILES=("50.94_-39.9")

# # specific tile for this job array task
# CURRENT_TILE=${TILES[0]}
# # echo "Starting job $SLURM_ARRAY_TASK_ID for tile $CURRENT_TILE"

# cd ~
# # run script script with the correct tile
# python lsst_anns.py $CURRENT_TILE
# echo "Finished job for tile $CURRENT_TILE"

# SNR_LEVELS=(1 2 3 4 5)

# TILE_INDEX=$((SLURM_ARRAY_TASK_ID / 5))
# SNR_INDEX=$((SLURM_ARRAY_TASK_ID % 5))

# CURRENT_TILE=${TILES[$TILE_INDEX]}
# CURRENT_SNR=${SNR_LEVELS[$SNR_INDEX]}

# #!/bin/bash
# #SBATCH --job-name="anns"
# #SBATCH --output="anns_out.%j.%N.out"
# #SBATCH --error="anns_err.%j.%N.err"
# #SBATCH --partition=cpu
# #SBATCH --nodes=1
# #SBATCH --ntasks-per-node=1
# #SBATCH --cpus-per-task=32
# #SBATCH --time=0:50:00
# #SBATCH --mail-user=yse2@illinois.edu
# #SBATCH --mail-type=ALL

# module load conda_base
# conda activate btknv
# cd ~
# # python lsst_anns.py # old annotation code w/ more permissive masks (no dilation, lower threshold, no PSF area scaling)
# # python lsst_anns_v2d.py # dilated masks 
# # python lsst_anns_v3d.py # correctly dilated masks and image convolve with PSF
# # python lsst_anns_v3d_lvl2.py # with lvl 2 instead of lvl 5
# python lsst_anns_v4d.py

