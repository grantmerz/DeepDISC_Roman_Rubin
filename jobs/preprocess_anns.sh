#!/bin/bash
#SBATCH -J lsst_anns_split
#SBATCH --output=/projects/bfhm/yse2/logs/anns_train_out.%j.out
#SBATCH --error=/projects/bfhm/yse2/logs/anns_train_err.%j.err
#SBATCH --account=bdsp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32g
#SBATCH --time=0:30:00
#SBATCH --mail-user=yse2@illinois.edu
#SBATCH --mail-type=ALL

echo "## Starting Job"
echo "## Node: $(hostname)"
cd ~
python preprocess_anns.py --plots
echo "## Job finished."
