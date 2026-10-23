#!/bin/bash
#SBATCH -J pemb_lprj_z
#SBATCH --output=/projects/bfhm/yse2/logs/plot_emb/pemb_lprj_z.%j.%N.out
#SBATCH --error=/projects/bfhm/yse2/logs/plot_emb/pemb_lprj_z.%j.%N.err
#SBATCH --account=bdsp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --constraint="projects,work"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --mail-user=yse2@illinois.edu
#SBATCH --mail-type=ALL

SUBMIT_TIME=$(squeue -j $SLURM_JOB_ID -h -o "%V")
START_TIME=$(date +"%Y-%m-%dT%H:%M:%S")
echo "Job Started: $(date)"
echo "Node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Usage: sbatch plot_emb.sh [mode1 [mode2 ...]]
# If no modes are provided, defaults to pooled_q
EMB_MODES=("$@")
if [[ ${#EMB_MODES[@]} -eq 0 ]]; then
    EMB_MODES=(pooled_q)
fi

for EMB_MODE in "${EMB_MODES[@]}"; do
    if [[ "$EMB_MODE" != "z_q" && "$EMB_MODE" != "z_k" && "$EMB_MODE" != "pooled_q" && "$EMB_MODE" != "pooled_k" && "$EMB_MODE" != "both" ]]; then
        echo "Invalid EMB_MODE: $EMB_MODE (expected: z_q, z_k, pooled_q, pooled_k, both)"
        exit 1
    fi
done

echo "EMB_MODES: ${EMB_MODES[*]}"

cd "$SLURM_SUBMIT_DIR"

for EMB_MODE in "${EMB_MODES[@]}"; do
    export EMB_MODE
    echo "Running EMB_MODE: $EMB_MODE"
    python embed_analysis/coherence.py
done

END_TIME=$(date +"%Y-%m-%dT%H:%M:%S")
ELAPSED_SEC=$(( $(date -d "$END_TIME" +%s) - $(date -d "$START_TIME" +%s) ))
ELAPSED_FMT=$(printf "%02d hours, %02d minutes, %02d seconds" \
    $((ELAPSED_SEC/3600)) \
    $(( (ELAPSED_SEC%3600)/60 )) \
    $((ELAPSED_SEC%60)) )
echo "SUBMIT TIME          START TIME           END TIME"
echo "$SUBMIT_TIME  $START_TIME  $END_TIME"
echo 
echo "ELAPSED TIME: $ELAPSED_FMT" 
echo "============================================================"
 
