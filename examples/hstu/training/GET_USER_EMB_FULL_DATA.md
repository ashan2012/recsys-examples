# 使用全部数据提取 User Embedding

本文档说明如何使用 `--use_full_data` 参数提取所有数据的 user embedding，而不仅仅是测试集。

## 问题说明

默认情况下，`get_useremb.py` 脚本会根据 `train_split_ratio` 将数据分为训练集和测试集，然后只对测试集提取 embedding。

例如，如果 `train_split_ratio = 0.7`，数据会被分为：
- 训练集：70% 的数据
- 测试集：30% 的数据

**问题**：如果您想要提取所有用户的 embedding（包括训练集和测试集），默认方式只能获取 30% 的数据。

## 解决方案

使用 `--use_full_data` 参数，脚本会临时将 `train_split_ratio` 设置为 0，这样所有数据都会作为"测试集"加载。

## 使用方法

### 1. 统计全部数据的样本数

```bash
torchrun --nproc_per_node=4 get_useremb.py \
    --gin-config-file ./configs/gameid_retrieval.gin \
    --use_full_data \
    --count_only
```

**输出示例**：
```
================================================================================
Using FULL DATA as test set (--use_full_data mode)
Original train_split_ratio: 0.7
Temporarily setting train_split_ratio to 0.0
All data will be loaded as test set
================================================================================

================================================================================
Dataset Information
================================================================================
MODE: Using FULL DATA (all samples)
Total batches: 850
Batch size: 128
Estimated total samples: ~108800
  (actual count may be slightly different due to incomplete last batch)
================================================================================

================================================================================
Exact Count Result
================================================================================
Total samples in test set: 106150
Total batches: 850
Average batch size: 124.88
================================================================================
```

### 2. 提取全部数据的 embedding

```bash
# 保存为 JSON 格式
torchrun --nproc_per_node=4 get_useremb.py \
    --gin-config-file ./configs/gameid_retrieval.gin \
    --use_full_data \
    --output_file user_embeddings_all.json \
    --max_samples 1000

# 保存为 NPZ 格式（推荐，更省空间）
torchrun --nproc_per_node=4 get_useremb.py \
    --gin-config-file ./configs/gameid_retrieval.gin \
    --use_full_data \
    --output_file user_embeddings_all.npz \
    --max_samples 100000  # 设置一个足够大的数
```

### 3. 对比：只使用测试集

```bash
# 不使用 --use_full_data 参数
torchrun --nproc_per_node=4 get_useremb.py \
    --gin-config-file ./configs/gameid_retrieval.gin \
    --count_only
```

**输出示例**：
```
================================================================================
Dataset Information
================================================================================
MODE: Using TEST SET only
Train/Test split ratio: 70.0% / 30.0%
Train samples: ~74536
Test samples: ~31945
Test batches: 250
Test batch size: 128
================================================================================
```

## 参数对比表

| 场景 | 命令 | 数据量 | 说明 |
|------|------|--------|------|
| **只提取测试集** | 不加 `--use_full_data` | 30% (假设 split=0.7) | 默认行为 |
| **提取全部数据** | 加 `--use_full_data` | 100% | 包含训练集+测试集 |

## 工作原理

### 内部实现

```python
# 1. 保存原始的 train_split_ratio
original_train_split_ratio = dataset_args.train_split_ratio  # 例如 0.7

# 2. 如果使用 --use_full_data，临时修改为 0
if args.use_full_data:
    dataset_args.train_split_ratio = 0.0  # 所有数据作为测试集

# 3. 加载数据
train_dataloader, test_dataloader = get_data_loader(...)
# 此时 test_dataloader 包含所有数据

# 4. 恢复原始值
dataset_args.train_split_ratio = original_train_split_ratio
```

### 数据分割说明

**train_split_ratio 的含义**：
- `0.0`: 所有数据作为测试集（0% 训练，100% 测试）
- `0.7`: 70% 训练，30% 测试
- `1.0`: 所有数据作为训练集（100% 训练，0% 测试）

## 完整工作流程示例

### 场景：提取所有用户的 embedding 用于推荐系统

```bash
# 步骤1: 统计全部数据量
echo "Step 1: Counting all samples..."
torchrun --nproc_per_node=4 get_useremb.py \
    --gin-config-file ./configs/gameid_retrieval.gin \
    --use_full_data \
    --count_only

# 假设输出：Total samples: 106150

# 步骤2: 先测试提取少量样本
echo "Step 2: Testing with 100 samples..."
torchrun --nproc_per_node=4 get_useremb.py \
    --gin-config-file ./configs/gameid_retrieval.gin \
    --use_full_data \
    --output_file user_emb_test.json \
    --max_samples 100

# 步骤3: 提取所有数据
echo "Step 3: Extracting all samples..."
torchrun --nproc_per_node=4 get_useremb.py \
    --gin-config-file ./configs/gameid_retrieval.gin \
    --use_full_data \
    --output_file user_embeddings_full.npz \
    --max_samples 110000  # 略大于实际数量

echo "Done! All user embeddings saved to user_embeddings_full.npz"
```

## 输出文件说明

### NPZ 格式（推荐）

```python
import numpy as np
import json

# 加载数据
data = np.load('user_embeddings_full.npz', allow_pickle=True)

embeddings = data['embeddings']     # shape: (106150, 256)
sample_ids = data['sample_ids']     # shape: (106150,)
metadata = json.loads(str(data['metadata']))

print(f"Total users: {len(embeddings)}")
print(f"Embedding dimension: {embeddings.shape[1]}")

# 访问第一个用户
first_user = metadata[0]
print(f"User ID: {first_user['user_ids']}")
print(f"Item IDs: {first_user['item_ids']}")
print(f"Embedding: {embeddings[0][:5]}...")
```

### JSON 格式

```python
import json

with open('user_embeddings_full.json', 'r') as f:
    data = json.load(f)

print(f"Total users: {len(data)}")

# 访问第一个用户
first_user = data[0]
print(f"Sample ID: {first_user['sample_id']}")
print(f"User ID: {first_user['user_ids']}")
print(f"Item IDs: {first_user['item_ids']}")
print(f"Embedding: {first_user['embedding'][:5]}...")
```

## 使用场景

### 1. 构建完整的用户画像库

```python
# 提取所有用户的 embedding
# 用于构建用户画像系统

import numpy as np
import json

data = np.load('user_embeddings_full.npz', allow_pickle=True)
embeddings = data['embeddings']
metadata = json.loads(str(data['metadata']))

# 构建 user_id 到 embedding 的映射
user_profiles = {}
for i, meta in enumerate(metadata):
    user_id = meta['user_ids'][0] if meta['user_ids'] else None
    if user_id:
        user_profiles[user_id] = {
            'embedding': embeddings[i],
            'items': meta['item_ids'],
            'norm': meta['embedding_norm']
        }

print(f"Built profiles for {len(user_profiles)} users")
```

### 2. 用户聚类分析（全量）

```python
from sklearn.cluster import KMeans
import numpy as np

# 加载所有用户的 embedding
data = np.load('user_embeddings_full.npz')
embeddings = data['embeddings']

# 对所有用户进行聚类
n_clusters = 20
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# 分析每个聚类
from collections import Counter
cluster_sizes = Counter(clusters)

print("Cluster distribution:")
for cluster_id, size in sorted(cluster_sizes.items()):
    print(f"  Cluster {cluster_id}: {size} users ({size/len(embeddings)*100:.1f}%)")
```

### 3. 用户相似度矩阵（全量）

```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# 加载所有用户的 embedding
data = np.load('user_embeddings_full.npz')
embeddings = data['embeddings']

# 计算相似度矩阵（注意：如果用户数很多，会很大）
# 建议分批计算或使用近似方法
similarity_matrix = cosine_similarity(embeddings)

print(f"Similarity matrix shape: {similarity_matrix.shape}")
print(f"Memory usage: {similarity_matrix.nbytes / 1024**2:.2f} MB")
```

## 注意事项

### 1. 数据一致性

⚠️ **重要**：确保配置文件中的 `train_split_ratio` 与训练时相同，这样数据分割才一致。

```python
# 在 gin 配置文件中
DatasetArgs.train_split_ratio = 0.7  # 必须与训练时相同
```

### 2. 内存使用

- **全量数据**：如果数据集很大（如百万用户），确保有足够的存储空间
- **NPZ 文件大小估算**：
  - 100K 用户 × 256 维 × 4 bytes ≈ 100 MB
  - 1M 用户 × 256 维 × 4 bytes ≈ 1 GB

### 3. 处理时间

- **统计样本数**（--count_only）：几秒到几十秒
- **提取 embedding**：取决于样本数，通常需要几分钟到几十分钟

### 4. GPU 要求

- 确保有足够的 GPU 内存加载模型和批次数据
- `--nproc_per_node` 应与训练时相同

### 5. 分批处理（大数据集）

如果数据集非常大，可以分批处理：

```bash
# 处理前 50K 样本
torchrun --nproc_per_node=4 get_useremb.py \
    --gin-config-file ./configs/gameid_retrieval.gin \
    --use_full_data \
    --output_file user_emb_part1.npz \
    --max_samples 50000

# 然后手动合并文件（需要自己写脚本）
```

## 常见问题

### Q1: --use_full_data 和不使用有什么区别？

**A**: 
- **不使用**：只获取测试集的 embedding（例如 30% 的数据）
- **使用**：获取所有数据的 embedding（100% 的数据）

### Q2: 会影响训练的模型吗？

**A**: 不会。脚本只是临时修改数据加载参数，不会影响已保存的模型或配置文件。

### Q3: 可以只提取训练集吗？

**A**: 可以，但需要修改代码。将 `dataset_args.train_split_ratio = 0.0` 改为 `1.0`，并使用 `train_dataloader` 而不是 `test_dataloader`。

### Q4: 提取的 embedding 顺序是什么？

**A**: 按照数据文件中的顺序。如果设置了 `shuffle=True`，顺序会被打乱。

### Q5: 如何验证提取了全部数据？

**A**: 使用 `--count_only` 查看样本数，应该接近数据文件的总行数。

## 对比总结

| 维度 | 默认模式（测试集） | --use_full_data（全量） |
|------|-------------------|------------------------|
| **数据范围** | 仅测试集（如 30%） | 所有数据（100%） |
| **样本数** | ~31,945（示例） | ~106,150（示例） |
| **适用场景** | 模型评估、验证 | 生产环境、完整画像 |
| **处理时间** | 较短 | 较长（约 3.3 倍） |
| **文件大小** | 较小 | 较大（约 3.3 倍） |
| **推荐使用** | 开发调试 | 生产部署 |

## 相关文档

- [完整使用文档](GET_USER_EMBEDDING_README.md)
- [快速使用指南](GET_USER_EMBEDDING_USAGE.txt)
- [模型评估](EVAL_README.md)

## 总结

使用 `--use_full_data` 参数可以轻松提取所有用户的 embedding，而无需修改配置文件或核心代码。这对于构建完整的用户画像系统、进行全量分析非常有用。

**推荐工作流程**：
1. ✅ 先用 `--count_only` 统计样本数
2. ✅ 用少量样本测试（`--max_samples 100`）
3. ✅ 确认无误后提取全部数据（`--max_samples` 设为足够大）
4. ✅ 使用 NPZ 格式节省空间

**快速开始**：
```bash
torchrun --nproc_per_node=4 get_useremb.py \
    --gin-config-file ./configs/gameid_retrieval.gin \
    --use_full_data \
    --output_file user_embeddings_all.npz \
    --max_samples 200000
```

