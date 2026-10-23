#!/bin/bash
#SBATCH -J pixel_mean
#SBATCH --output=/projects/bfhm/yse2/logs/pixel_mean_out.%A_%a.out
#SBATCH --error=/projects/bfhm/yse2/logs/pixel_mean_err.%A_%a.err
#SBATCH --account=bfhm-delta-cpu
#SBATCH --partition=cpu
#SBATCH --constraint="work"
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=01:00:00
#SBATCH --mail-user=yse2@illinois.edu
#SBATCH --mail-type=ALL

cd "$SLURM_SUBMIT_DIR"
# python data_processing/calc_pixel_mean_std.py --root_dir /u/yse2/lsst_data/ --anns_dir annotations --snr_lvl 5 --num_channels 3 --max_workers 32 --using_roman
python data_processing/calc_pixel_mean_std.py --root_dir /u/yse2/lsst_data/ --anns_dir annotations --snr_lvl 5 --num_channels 6 --max_workers 32 --upsample_lsst