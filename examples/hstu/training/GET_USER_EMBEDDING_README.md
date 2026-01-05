# 用户 Embedding 提取说明

本文档说明如何使用 `get_useremb.py` 脚本提取每个用户样本的 embedding。

## 功能说明

`get_useremb.py` 脚本可以：

1. ✅ 加载训练好的模型
2. ✅ 对测试集进行推理
3. ✅ 提取每个样本的详细信息：
   - User ID（用户ID）
   - Item IDs（交互的物品ID序列）
   - User Embedding（用户的embedding向量）
   - Embedding统计信息（维度、范数等）
4. ✅ 支持保存结果到文件（JSON或NPZ格式）

## 快速使用

### 基本用法（打印到控制台）

```bash
torchrun --nproc_per_node=4 get_useremb.py \
    --gin-config-file ./configs/gameid_retrieval.gin
```

### 保存结果到文件

```bash
# 保存为 JSON 格式（可读性好）
torchrun --nproc_per_node=4 get_useremb.py \
    --gin-config-file ./configs/gameid_retrieval.gin \
    --output_file user_embeddings.json

# 保存为 NPZ 格式（节省空间）
torchrun --nproc_per_node=4 get_useremb.py \
    --gin-config-file ./configs/gameid_retrieval.gin \
    --output_file user_embeddings.npz
```

### 限制处理样本数（快速测试）

```bash
# 只处理前100个样本
torchrun --nproc_per_node=4 get_useremb.py \
    --gin-config-file ./configs/gameid_retrieval.gin \
    --output_file user_embeddings.json \
    --max_samples 100
```

## 参数说明

| 参数 | 说明 | 必需 | 默认值 | 示例 |
|------|------|------|--------|------|
| `--gin-config-file` | gin配置文件路径 | 是 | - | `./configs/gameid_retrieval.gin` |
| `--output_file` | 输出文件路径 | 否 | None | `user_embeddings.json` |
| `--max_samples` | 最大处理样本数 | 否 | None (前3个batch) | `100` |

## 输出格式

### 控制台输出

```
================================================================================
Starting inference and extracting embeddings for each sample
Results will be saved to: user_embeddings.json
Maximum samples to process: 100
================================================================================

================================================================================
Batch 1 - Processing 128 samples
================================================================================

--- Sample 1 (Batch 1, Index 0) ---
User ID: 12345
Item IDs (sequence): [101, 203, 456, 789, 1024] ... [5678, 9012] (total: 50)
Embedding shape: (256,)
Embedding dtype: torch.bfloat16
Embedding (first 10 dims): [ 0.0234 -0.0156  0.0421 ... ]
Embedding norm: 1.234567

--- Sample 2 (Batch 1, Index 1) ---
...

================================================================================
Inference complete. Total samples processed: 100
================================================================================

Saving results to user_embeddings.json...
Saved 100 samples to user_embeddings.json (JSON format)
```

### JSON 输出格式

```json
[
  {
    "sample_id": 1,
    "batch_idx": 1,
    "batch_inner_idx": 0,
    "user_ids": [12345],
    "item_ids": [101, 203, 456, 789, 1024, ..., 5678, 9012],
    "embedding": [0.0234, -0.0156, 0.0421, ...],
    "embedding_norm": 1.234567
  },
  {
    "sample_id": 2,
    ...
  }
]
```

### NPZ 输出格式

```python
import numpy as np

# 加载 NPZ 文件
data = np.load('user_embeddings.npz', allow_pickle=True)

# 访问数据
embeddings = data['embeddings']     # shape: (N, embedding_dim)
sample_ids = data['sample_ids']     # shape: (N,)
metadata = json.loads(str(data['metadata']))  # 包含 user_ids, item_ids 等

print(f"Loaded {len(embeddings)} embeddings")
print(f"Embedding dimension: {embeddings.shape[1]}")
```

## 配置文件要求

确保配置文件中包含 checkpoint 路径：

```python
# 在 gin 配置文件中
TrainerArgs.ckpt_load_dir = './ckpt_re_test/iter100'
```

或者直接在训练配置文件中已经设置好。

## 使用场景

### 1. 用户画像分析

提取所有用户的 embedding，进行聚类分析：

```python
import numpy as np
import json
from sklearn.cluster import KMeans

# 加载 JSON 格式
with open('user_embeddings.json', 'r') as f:
    data = json.load(f)

# 提取所有 embeddings
embeddings = np.array([sample['embedding'] for sample in data])
user_ids = [sample['user_ids'][0] if sample['user_ids'] else None for sample in data]

# 聚类
kmeans = KMeans(n_clusters=10, random_state=42)
clusters = kmeans.fit_predict(embeddings)

# 分析每个聚类
for cluster_id in range(10):
    mask = (clusters == cluster_id)
    cluster_users = [uid for uid, m in zip(user_ids, mask) if m]
    print(f"Cluster {cluster_id}: {len(cluster_users)} users")
```

### 2. 用户相似度计算

计算两个用户之间的相似度：

```python
import numpy as np
import json

# 加载数据
with open('user_embeddings.json', 'r') as f:
    data = json.load(f)

# 构建 user_id 到 embedding 的映射
user_emb_dict = {}
for sample in data:
    if sample['user_ids']:
        user_id = sample['user_ids'][0]
        user_emb_dict[user_id] = np.array(sample['embedding'])

# 计算余弦相似度
def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

user_id1 = 12345
user_id2 = 67890

if user_id1 in user_emb_dict and user_id2 in user_emb_dict:
    sim = cosine_similarity(user_emb_dict[user_id1], user_emb_dict[user_id2])
    print(f"Similarity between user {user_id1} and {user_id2}: {sim:.4f}")
```

### 3. 用户兴趣建模

根据用户的历史行为（item_ids）和 embedding 分析用户兴趣：

```python
import numpy as np
import json

# 加载数据
with open('user_embeddings.json', 'r') as f:
    data = json.load(f)

# 分析某个用户
for sample in data:
    user_id = sample['user_ids'][0] if sample['user_ids'] else None
    item_ids = sample['item_ids']
    embedding = np.array(sample['embedding'])
    
    if user_id == 12345:  # 目标用户
        print(f"User {user_id}:")
        print(f"  Interacted with {len(item_ids)} items: {item_ids[:5]}...")
        print(f"  Embedding norm: {sample['embedding_norm']:.4f}")
        print(f"  Top embedding dims: {np.argsort(np.abs(embedding))[-5:][::-1]}")
```

### 4. 推荐系统 - 基于用户的协同过滤

找到相似用户并推荐其喜欢的物品：

```python
import numpy as np
import json

# 加载数据
with open('user_embeddings.json', 'r') as f:
    data = json.load(f)

# 构建用户数据
users = []
for sample in data:
    if sample['user_ids'] and sample['item_ids']:
        users.append({
            'user_id': sample['user_ids'][0],
            'embedding': np.array(sample['embedding']),
            'items': set(sample['item_ids']),
        })

def find_similar_users(target_user_id, k=10):
    """找到最相似的k个用户"""
    target_user = next((u for u in users if u['user_id'] == target_user_id), None)
    if not target_user:
        return []
    
    target_emb = target_user['embedding']
    
    # 计算与所有其他用户的相似度
    similarities = []
    for user in users:
        if user['user_id'] == target_user_id:
            continue
        sim = np.dot(target_emb, user['embedding']) / (
            np.linalg.norm(target_emb) * np.linalg.norm(user['embedding'])
        )
        similarities.append((user['user_id'], sim, user['items']))
    
    # 排序并返回 top-k
    similarities.sort(key=lambda x: x[1], reverse=True)
    return similarities[:k]

def recommend_items(target_user_id, k=20):
    """推荐物品"""
    target_user = next((u for u in users if u['user_id'] == target_user_id), None)
    if not target_user:
        return []
    
    target_items = target_user['items']
    similar_users = find_similar_users(target_user_id, k=10)
    
    # 统计相似用户喜欢的物品
    item_scores = {}
    for sim_user_id, similarity, items in similar_users:
        for item in items:
            if item not in target_items:  # 排除已交互的物品
                item_scores[item] = item_scores.get(item, 0) + similarity
    
    # 排序并返回 top-k
    recommendations = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)[:k]
    return recommendations

# 使用
recommendations = recommend_items(12345, k=20)
print(f"Recommended items for user 12345:")
for item_id, score in recommendations[:10]:
    print(f"  Item {item_id}: score={score:.4f}")
```

## 数据格式说明

### User IDs
- 如果一个样本只有一个 user_id，显示为：`User ID: 12345`
- 如果有多个（序列），显示为：`User IDs (sequence): [12345, 12346, ...] (total: 10)`

### Item IDs
- 显示该用户交互过的所有 item_ids（历史行为序列）
- 格式：`Item IDs (sequence): [101, 203, 456, 789, 1024] ... [5678, 9012] (total: 50)`

### Embedding
- 用户的 embedding 向量（经过模型编码后的用户表示）
- 维度通常为 256 或 512（取决于模型配置）
- 使用 BFloat16 格式存储，输出时转换为 float32

## 性能优化

### 1. 批量处理

默认处理前 3 个 batch，可以通过参数控制：

```bash
# 处理所有数据（移除代码中的 batch 限制）
# 或使用 --max_samples 参数
torchrun --nproc_per_node=4 get_useremb.py \
    --gin-config-file ./configs/gameid_retrieval.gin \
    --max_samples 10000
```

### 2. 使用 NPZ 格式

对于大量数据，使用 NPZ 格式更高效：

```bash
# NPZ 格式比 JSON 小 3-5 倍
torchrun --nproc_per_node=4 get_useremb.py \
    --gin-config-file ./configs/gameid_retrieval.gin \
    --output_file user_embeddings.npz \
    --max_samples 100000
```

## 常见问题

### Q: 为什么 embedding dtype 是 BFloat16？
A: 模型使用 BFloat16 进行训练以节省内存和加速计算。输出到 numpy 时会自动转换为 float32。

### Q: 如何处理所有测试集数据？
A: 两种方法：
   1. 设置 `--max_samples` 为一个很大的数字
   2. 修改代码中的 `if batch_idx >= 2` 这一行

### Q: 输出文件太大怎么办？
A: 使用 NPZ 格式而不是 JSON，或者分批处理：
```bash
# 分批处理
for i in {0..9}; do
    torchrun --nproc_per_node=4 get_useremb.py \
        --gin-config-file ./configs/gameid_retrieval.gin \
        --output_file user_embeddings_part${i}.npz \
        --max_samples 10000
done
```

### Q: User ID 显示为 Not found？
A: 检查数据集中是否包含 user_id 字段，或者字段名是否匹配。

### Q: 如何与 item embedding 结合使用？
A: 结合使用 `get_item_embedding.py` 导出的 item embeddings：
```python
# 加载 user embeddings
with open('user_embeddings.json') as f:
    user_data = json.load(f)

# 加载 item embeddings
item_data = np.load('item_embeddings_arrays.npz')
item_keys = item_data['keys']
item_values = item_data['values']

# 计算 user-item 相似度
user_emb = np.array(user_data[0]['embedding'])
for item_id in item_keys[:10]:
    idx = np.where(item_keys == item_id)[0][0]
    item_emb = item_values[idx]
    sim = np.dot(user_emb, item_emb) / (
        np.linalg.norm(user_emb) * np.linalg.norm(item_emb)
    )
    print(f"User-Item {item_id} similarity: {sim:.4f}")
```

## 相关文档

- [Item Embedding 导出](EMBEDDINGS_README.md)
- [模型评估](EVAL_README.md)
- [训练说明](README.md)

## 总结

`get_useremb.py` 提供了提取用户 embedding 的完整功能：

✅ **详细输出**: 每个样本的 user_id, item_ids, embedding  
✅ **灵活存储**: 支持 JSON 和 NPZ 两种格式  
✅ **易于使用**: 一键运行，自动加载模型和数据  
✅ **可扩展**: 支持各种下游应用（推荐、聚类、相似度计算等）

使用此工具可以方便地分析用户行为、构建推荐系统、进行用户画像等任务。

