# 模型评估说明文档

本文档说明如何使用训练好的checkpoint对测试集进行评估。

## 概述

`eval_retrieval.py` 脚本用于评估训练好的召回（retrieval）模型，它会：

1. ✅ 加载指定的checkpoint
2. ✅ 使用与训练相同的数据格式和配置
3. ✅ 在整个测试集（或指定数量的batch）上评估
4. ✅ 计算所有配置的评估指标（NDCG@10, NDCG@20, HR@10, HR@50等）
5. ✅ 输出详细的评估结果

## 快速开始

### 方法一：使用Shell脚本 (推荐)

```bash
# 修改 eval_game.sh 中的配置
vim eval_game.sh

# 设置以下参数：
# GIN_CONFIG="./configs/gameid_retrieval.gin"     # 配置文件（与训练时相同）
# CHECKPOINT_DIR="./ckpt_re_test/iter100"         # checkpoint目录
# NUM_GPUS=4                                      # GPU数量

# 运行评估
bash eval_game.sh
```

### 方法二：直接使用torchrun

```bash
# 基本用法（使用配置文件中的所有设置）
torchrun --nproc_per_node=4 eval_retrieval.py \
    --gin_config_file ./configs/gameid_retrieval.gin \
    --checkpoint_dir ./ckpt_re_test/iter100

# 限制评估batch数量（快速测试）
torchrun --nproc_per_node=4 eval_retrieval.py \
    --gin_config_file ./configs/gameid_retrieval.gin \
    --checkpoint_dir ./ckpt_re_test/iter100 \
    --max_eval_iters 100

# 自定义评估batch size
torchrun --nproc_per_node=4 eval_retrieval.py \
    --gin_config_file ./configs/gameid_retrieval.gin \
    --checkpoint_dir ./ckpt_re_test/iter100 \
    --eval_batch_size 256
```

## 参数说明

### 必需参数

| 参数 | 说明 | 示例 |
|------|------|------|
| `--gin_config_file` | gin配置文件路径（与训练时使用的相同） | `./configs/gameid_retrieval.gin` |
| `--checkpoint_dir` | checkpoint目录路径 | `./ckpt_re_test/iter100` |

### 可选参数

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--max_eval_iters` | 最大评估迭代数，用于快速测试 | None (评估全部) | `100` |
| `--eval_batch_size` | 评估batch size | 配置文件中的值 | `256` |

## 输出说明

### 控制台输出

脚本运行时会输出：

```
================================================================================
Model Evaluation Script
================================================================================
Config file: ./configs/gameid_retrieval.gin
Checkpoint: ./ckpt_re_test/iter100
================================================================================

================================================================================
Configuration Summary
================================================================================
Dataset: gameid
Eval batch size: 128
Max eval iters: All
Pipeline type: native
================================================================================

Initializing distributed environment...
Distributed environment initialized

Creating model configuration...
Evaluation metrics: ('NDCG@10', 'NDCG@20', 'HR@10', 'HR@50')

Creating model...
Model created

Setting up dynamic embeddings...
Dynamic embeddings setup complete

Loading checkpoint from ./ckpt_re_test/iter100...
Checkpoints loaded successfully

Creating metric module...
Metric types: ('NDCG@10', 'NDCG@20', 'HR@10', 'HR@50')

Creating data loaders...
Test dataloader created with 250 batches

Creating evaluation pipeline...
Pipeline created: JaggedMegatronTrainPipelineSparseDist

Setting model to evaluation mode...

================================================================================
Starting Evaluation on Test Set
================================================================================
[eval] [eval 32000 users]:
    Metrics:
    NDCG@10: 0.1234
    NDCG@20: 0.1567
    HR@10: 0.2345
    HR@50: 0.4567

================================================================================
Evaluation Complete
================================================================================
Elapsed time: 125.67 seconds
================================================================================

Evaluation finished successfully!
```

### 评估指标说明

| 指标 | 全称 | 说明 | 取值范围 |
|------|------|------|----------|
| **NDCG@K** | Normalized Discounted Cumulative Gain | 归一化折损累积增益，考虑排序位置的准确性 | 0.0 - 1.0，越高越好 |
| **HR@K** | Hit Rate | 命中率，top-K推荐中是否包含用户真实交互的item | 0.0 - 1.0，越高越好 |
| **Recall@K** | Recall | 召回率，top-K中召回的真实items比例 | 0.0 - 1.0，越高越好 |
| **Precision@K** | Precision | 精确率，top-K中正确items的比例 | 0.0 - 1.0，越高越好 |

**K值**: 表示推荐列表的长度（如K=10表示top-10推荐）

## 使用场景

### 1. 模型选择（Model Selection）

在不同的checkpoint之间选择最佳模型：

```bash
# 评估不同迭代的checkpoint
for iter in 100 200 300 400 500; do
    echo "Evaluating iteration ${iter}..."
    torchrun --nproc_per_node=4 eval_retrieval.py \
        --gin_config_file ./configs/gameid_retrieval.gin \
        --checkpoint_dir ./ckpt_re_test/iter${iter} \
        | tee eval_results_iter${iter}.log
done

# 比较结果
grep "NDCG@10" eval_results_iter*.log
```

### 2. 超参数调优验证

验证不同超参数配置的效果：

```bash
# 评估不同配置的模型
for config in config_v1 config_v2 config_v3; do
    torchrun --nproc_per_node=4 eval_retrieval.py \
        --gin_config_file ./configs/${config}.gin \
        --checkpoint_dir ./ckpts/${config}/best_model \
        | tee eval_${config}.log
done
```

### 3. 快速验证（Sanity Check）

在少量batch上快速验证checkpoint是否正常：

```bash
# 只评估前50个batch
torchrun --nproc_per_node=4 eval_retrieval.py \
    --gin_config_file ./configs/gameid_retrieval.gin \
    --checkpoint_dir ./ckpt_re_test/iter100 \
    --max_eval_iters 50
```

### 4. 不同数据集评估

在不同的测试集上评估模型泛化能力：

```bash
# 评估不同的数据集
for dataset in gameid_test1 gameid_test2; do
    # 修改配置文件中的数据集路径
    torchrun --nproc_per_node=4 eval_retrieval.py \
        --gin_config_file ./configs/${dataset}.gin \
        --checkpoint_dir ./ckpt_re_test/iter100
done
```

## 配置文件要求

评估脚本使用的gin配置文件应该与训练时使用的配置文件**相同或兼容**。

### 必需配置项

```python
# 数据集配置
DatasetArgs.dataset_name = 'gameid'
DatasetArgs.dataset_path = '/path/to/dataset'
DatasetArgs.max_sequence_length = 200
DatasetArgs.max_num_candidates = 64
DatasetArgs.train_split_ratio = 0.7  # 确保与训练时一致

# 训练器配置
TrainerArgs.eval_batch_size = 128
TrainerArgs.seed = 2025

# 网络配置
NetworkArgs.dtype_str = "bfloat16"
NetworkArgs.num_layers = 4
NetworkArgs.num_attention_heads = 4
NetworkArgs.hidden_size = 256
# ... 其他网络参数

# 评估指标配置
RetrievalArgs.eval_metrics = ("NDCG@10", "NDCG@20", "HR@10", "HR@50")
```

### 可选配置项

```python
# 限制评估batch数（用于快速测试）
TrainerArgs.max_eval_iters = 100

# Pipeline类型
TrainerArgs.pipeline_type = 'native'  # 或 'prefetch'
```

## 与训练的区别

| 方面 | 训练 | 评估 |
|------|------|------|
| **数据集** | 训练集（train split） | 测试集（eval split） |
| **模式** | `model.train()` | `model.eval()` |
| **梯度** | 计算梯度，更新参数 | `torch.no_grad()`，不更新参数 |
| **随机性** | Dropout等生效 | Dropout等关闭 |
| **输出** | Loss、指标（周期性） | 完整指标（一次性） |
| **Dynamic Embeddings** | `training=True` | `training=False` |

## 注意事项

### 1. 数据集分割一致性

⚠️ **重要**: 确保 `train_split_ratio` 与训练时完全一致，否则测试集会不同。

```python
# 在配置文件中
DatasetArgs.train_split_ratio = 0.7  # 必须与训练时相同
```

### 2. GPU数量

- 评估时的GPU数量（`--nproc_per_node`）应该与训练时相同
- 如果GPU数量不同，可能导致数据分布不一致

### 3. Checkpoint路径

确保checkpoint目录包含以下文件：
```
ckpt_re_test/iter100/
├── dynamicemb_module/           # Dynamic embeddings
│   └── model._embedding_collection._model_parallel_embedding_collection/
│       ├── item_id_emb_keys.rank_*.world_size_*
│       ├── item_id_emb_values.rank_*.world_size_*
│       ├── user_id_emb_keys.rank_*.world_size_*
│       └── user_id_emb_values.rank_*.world_size_*
└── torch_module/                # Dense模型参数
    └── mp_rank_*_000000/
        └── model_optim_rng.pt
```

### 4. 内存考虑

评估时仍需加载完整模型和embeddings到GPU，确保有足够的GPU内存。

## 故障排查

### 问题1: Checkpoint加载失败

```
Error: ckpt_load_dir ./ckpt_re_test/iter100 does not exist
```

**解决方案**:
- 检查checkpoint路径是否正确
- 确认checkpoint目录存在且包含必要文件

### 问题2: 配置不匹配

```
Error: size mismatch for ...
```

**解决方案**:
- 确保使用与训练时相同的配置文件
- 检查模型结构配置（hidden_size, num_layers等）是否一致

### 问题3: GPU数量不足

```
RuntimeError: CUDA out of memory
```

**解决方案**:
- 减少 `eval_batch_size`
- 使用更多GPU: 增加 `--nproc_per_node`
- 释放GPU内存: `nvidia-smi` 查看并关闭其他进程

### 问题4: 评估指标为0或异常

**可能原因**:
- Checkpoint未正确加载
- 数据集分割不一致
- 配置文件错误

**解决方案**:
1. 检查日志中是否有 "Checkpoints loaded!!" 信息
2. 验证 `train_split_ratio` 配置
3. 用少量batch测试: `--max_eval_iters 10`

## 高级用法

### 批量评估多个Checkpoint

```bash
#!/bin/bash

# 批量评估脚本
CHECKPOINT_BASE="./ckpt_re_test"
CONFIG_FILE="./configs/gameid_retrieval.gin"
RESULTS_DIR="./eval_results"

mkdir -p ${RESULTS_DIR}

# 评估所有checkpoint
for ckpt_dir in ${CHECKPOINT_BASE}/iter*; do
    iter_name=$(basename ${ckpt_dir})
    echo "Evaluating ${iter_name}..."
    
    torchrun --nproc_per_node=4 eval_retrieval.py \
        --gin_config_file ${CONFIG_FILE} \
        --checkpoint_dir ${ckpt_dir} \
        2>&1 | tee ${RESULTS_DIR}/${iter_name}.log
done

# 提取并比较结果
echo "Summary of Results:"
echo "===================="
for log_file in ${RESULTS_DIR}/*.log; do
    iter_name=$(basename ${log_file} .log)
    ndcg=$(grep "NDCG@10" ${log_file} | tail -1 | awk '{print $2}')
    hr=$(grep "HR@10" ${log_file} | tail -1 | awk '{print $2}')
    echo "${iter_name}: NDCG@10=${ndcg}, HR@10=${hr}"
done
```

### 保存评估结果到JSON

修改脚本以保存结构化结果（可在脚本中添加）:

```python
import json

# 在evaluate函数调用后添加
eval_results = {
    "checkpoint": args.checkpoint_dir,
    "config": args.gin_config_file,
    "metrics": eval_metric_dict,
    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    "elapsed_time": elapsed_time,
}

with open("eval_results.json", "w") as f:
    json.dump(eval_results, f, indent=2)
```

## 性能优化

### 1. 使用Prefetch Pipeline

如果训练时使用了prefetch pipeline，评估时也应使用以加速：

```python
# 在配置文件中
TrainerArgs.pipeline_type = 'prefetch'
```

### 2. 调整Batch Size

更大的batch size可以提高GPU利用率：

```bash
torchrun --nproc_per_node=4 eval_retrieval.py \
    --gin_config_file ./configs/gameid_retrieval.gin \
    --checkpoint_dir ./ckpt_re_test/iter100 \
    --eval_batch_size 256  # 增大batch size
```

### 3. 部分评估（用于调试）

在开发阶段，可以只评估部分数据以加快迭代：

```bash
# 只评估前100个batch
torchrun --nproc_per_node=4 eval_retrieval.py \
    --gin_config_file ./configs/gameid_retrieval.gin \
    --checkpoint_dir ./ckpt_re_test/iter100 \
    --max_eval_iters 100
```

## 总结

评估脚本提供了完整的模型评估能力：

✅ **易用性**: 使用与训练相同的配置，无需额外设置  
✅ **准确性**: 调用训练过程中的evaluate函数，确保一致性  
✅ **灵活性**: 支持自定义评估参数和批量评估  
✅ **全面性**: 输出所有配置的评估指标

使用此脚本可以方便地评估和比较不同checkpoint的性能，选择最佳模型用于生产环境。

## 相关文档

- [训练说明](README.md)
- [Embedding导出说明](EMBEDDINGS_README.md)
- [快速入门](EMBEDDINGS_QUICKSTART.txt)

