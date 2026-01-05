#!/bin/bash

# 评估 gameid 召回模型的shell脚本
# 用法: bash eval_game.sh

# 配置参数
GIN_CONFIG="./configs/gameid_retrieval.gin"
CHECKPOINT_DIR="./ckpt_re_test/iter100"
NUM_GPUS=4

# 运行评估
torchrun --nproc_per_node=${NUM_GPUS} \
    eval_retrieval.py \
    --gin_config_file ${GIN_CONFIG} \
    --checkpoint_dir ${CHECKPOINT_DIR}

# 可选参数示例：
# --max_eval_iters 100        # 限制评估迭代数
# --eval_batch_size 256       # 指定评估batch size

