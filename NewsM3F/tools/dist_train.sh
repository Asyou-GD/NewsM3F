#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${PET_NNODES:-1}
NODE_RANK=${PET_NODE_RANK:-0}
PORT=${PET_MASTER_PORT:-29500}
MASTER_ADDR=${PET_MASTER_ADDR:-"127.0.0.1"}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --launcher pytorch ${@:3}
