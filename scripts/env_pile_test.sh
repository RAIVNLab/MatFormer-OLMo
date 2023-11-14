#!/bin/bash

# Note: This script does not run inside the container. It runs on the bare compute node.


# Redirect stdout and stderr so that we get a prefix with the node name
export NODENAME=$(hostname -s)

export MASTER_ADDR=$(scontrol show hostnames | head -n 1)
export MASTER_PORT=39591
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
export LOCAL_RANK=$SLURM_LOCALID
export NODE_RANK=$((($RANK - $LOCAL_RANK) / $LOCAL_WORLD_SIZE))
export HF_HOME=<HF_HOME>

exec $*
