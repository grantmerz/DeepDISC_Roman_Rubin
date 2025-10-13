#!/bin/bash
#SBATCH --image=lsstdesc/stack-jupyter:weekly
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=256
#SBATCH --qos=debug
#SBATCH --constraint=cpu
#SBATCH --account=xxxxx
#SBATCH -J multi_patch
#SBATCH --mail-user=your_email_address
#SBATCH --mail-type=ALL
#SBATCH --time=00:10:00
#SBATCH --output="/pscratch/sd/y/yaswante/MyQuota/roman_lsst/logs/multi_patch.%j.%N.out"
#SBATCH --error="/pscratch/sd/y/yaswante/MyQuota/roman_lsst/logs/multi_patch.%j.%N.err"

# OpenMP settings:
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread
export KMP_AFFINITY=disabled

echo "Hello"
# srun -n <num_mpi_processes> -c <cpus_per_task> a.out
srun -n 1 -c 256 shifter /pscratch/sd/y/yaswante/MyQuota/roman_lsst/run_lsst_multipatch.sh






