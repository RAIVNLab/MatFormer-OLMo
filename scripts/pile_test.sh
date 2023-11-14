#!/bin/bash
#SBATCH --job-name=<job>
#SBATCH --account=<account>
#SBATCH --output=<logdir>/%j.log
#SBATCH --nodes=8            # Total number of nodes 
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4       # Allocate one gpu per MPI rank
#SBATCH --cpus-per-task=2
#SBATCH --time=120:00:00
#SBATCH --mem=190G			# All memory on the node
#SBATCH --partition=<partition>


srun \
  --cpus-per-task=$SLURM_CPUS_PER_TASK \
  --distribution=block:block \
  --kill-on-bad-exit \
  scripts/env_pile_test.sh \
    python scripts/train.py configs/pile-tiny.yaml --run_name=${SLURM_JOB_ID} ${@}
