#!/bin/bash
#SBATCH -J cache_wcs_maps
#SBATCH --output=/projects/bdsp/g4merz/logs/training_logs/cache_wcs_maps.%j.out
#SBATCH --error=/projects/bdsp/g4merz/logs/training_logs/cache_wcs_maps.%j.err
#SBATCH --account=bdsp-delta-cpu
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --mem=128G    
#SBATCH --ntasks-per-node=1        # one process per GPU for distributed training
#SBATCH --cpus-per-task=32       
#SBATCH --constraint="work"
#SBATCH --time=3:00:00
#SBATCH --mail-user=gmerz3@illinois.edu
#SBATCH --mail-type=ALL


python cache_wcs_maps.py --dset test_8k_keypoints_wcs --anns-dir annotations

python cache_wcs_maps.py --dset train_30k_keypoints_wcs --anns-dir annotations