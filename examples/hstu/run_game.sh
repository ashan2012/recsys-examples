#!/bin/bash
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliate

# batch_size = 16GPUs * 8 = 128
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_TIMEOUT=3600
export NCCL_ASYNC_ERROR_HANDLING=1
PYTHONPATH=${PYTHONPATH}:$(realpath ../) torchrun --nproc_per_node 8 --nnodes 1 --node_rank 0 --master_addr localhost --master_port 6000  ./training/pretrain_gr_retrieval.py --gin-config-file ./training/configs/gameid_retrieval.gin