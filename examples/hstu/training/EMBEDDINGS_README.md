# Item Embeddings 使用说明

本文档说明如何读取和使用从训练好的模型中导出的 item embeddings。

## 文件格式

导出的 embeddings 保存为 NumPy 的 `.npz` 格式（压缩的 numpy 数组），包含两个文件：

### 1. `item_embeddings_arrays.npz` (推荐使用)
**数组格式**，包含以下字段：
- `keys`: 所有 item_id 的数组，shape: `(N,)`，dtype: `int64`
- `values`: 所有 embedding 向量的数组，shape: `(N, embedding_dim)`，dtype: `float32`
- `table_name`: embedding 表名称（字符串）

**优点**: 高效，易于批量处理，支持快速查询和相似度计算

### 2. `item_embeddings.npz`
**字典格式**，每个 item 作为单独的 key：
- `item_0`: item_id=0 的 embedding
- `item_1`: item_id=1 的 embedding
- ...

**优点**: 直接通过 item_id 访问，适合单个查询

## 快速开始

### 方法一：使用提供的脚本 (推荐)

```bash
# 1. 查看文件基本信息
python read_item_embeddings.py --file item_embeddings_arrays.npz --info

# 输出示例：
# ============================================================
# Embedding File Information
# ============================================================
# Format: Array format (keys + values)
# Number of items: 12345
# Embedding dimension: 256
# Keys dtype: int64
# Values dtype: float32
# Keys shape: (12345,)
# Values shape: (12345, 256)
#
# Key statistics:
#   Min item_id: 0
#   Max item_id: 99999
#   Unique keys: 12345
#
# First 10 item_ids: [0, 1, 5, 8, 12, 15, 20, 23, 27, 30]
# ============================================================

# 2. 查询单个 item 的 embedding
python read_item_embeddings.py --file item_embeddings_arrays.npz --item_id 123

# 3. 查询多个 items
python read_item_embeddings.py --file item_embeddings_arrays.npz --item_ids 123,456,789

# 4. 计算两个 item 的相似度
python read_item_embeddings.py --file item_embeddings_arrays.npz --similarity 123,456

# 5. 查找相似的 items (推荐系统)
python read_item_embeddings.py --file item_embeddings_arrays.npz --similar 123 --top_k 10
```

### 方法二：在 Python 代码中使用

#### 加载 embeddings

```python
import numpy as np

# 加载数组格式（推荐）
data = np.load("item_embeddings_arrays.npz")
keys = data['keys']      # shape: (N,) - 所有 item_ids
values = data['values']  # shape: (N, embedding_dim) - 所有 embeddings

print(f"加载了 {len(keys)} 个 item embeddings")
print(f"Embedding 维度: {values.shape[1]}")
```

#### 查询单个 item

```python
def get_embedding(keys, values, item_id):
    """获取指定 item_id 的 embedding"""
    mask = (keys == item_id)
    if mask.any():
        idx = np.where(mask)[0][0]
        return values[idx]
    else:
        return None

# 使用
item_id = 123
embedding = get_embedding(keys, values, item_id)

if embedding is not None:
    print(f"Item {item_id} embedding shape: {embedding.shape}")
    print(f"First 5 values: {embedding[:5]}")
else:
    print(f"Item {item_id} not found")
```

#### 批量查询

```python
# 查询多个 item_ids
query_ids = [123, 456, 789, 1000]

# 方法1: 使用 isin
mask = np.isin(keys, query_ids)
found_keys = keys[mask]
found_embeddings = values[mask]

print(f"查询 {len(query_ids)} 个 items，找到 {len(found_keys)} 个")

# 方法2: 构建字典进行查询（更快）
embedding_dict = {int(k): v for k, v in zip(keys, values)}

for item_id in query_ids:
    if item_id in embedding_dict:
        emb = embedding_dict[item_id]
        print(f"Item {item_id}: {emb[:3]}...")
    else:
        print(f"Item {item_id}: Not found")
```

#### 计算相似度

```python
def cosine_similarity(emb1, emb2):
    """计算余弦相似度"""
    return np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))

def euclidean_distance(emb1, emb2):
    """计算欧氏距离"""
    return np.linalg.norm(emb1 - emb2)

# 使用
emb1 = get_embedding(keys, values, 123)
emb2 = get_embedding(keys, values, 456)

if emb1 is not None and emb2 is not None:
    cos_sim = cosine_similarity(emb1, emb2)
    euc_dist = euclidean_distance(emb1, emb2)
    
    print(f"余弦相似度: {cos_sim:.6f}")
    print(f"欧氏距离: {euc_dist:.6f}")
```

#### 批量计算相似度（高效）

```python
def find_similar_items(target_item_id, keys, values, top_k=10):
    """查找与指定 item 最相似的 top-k 个 items"""
    # 获取目标 embedding
    target_emb = get_embedding(keys, values, target_item_id)
    if target_emb is None:
        return []
    
    # 归一化
    target_norm = target_emb / np.linalg.norm(target_emb)
    values_norm = values / np.linalg.norm(values, axis=1, keepdims=True)
    
    # 计算所有相似度（向量化操作，非常快）
    similarities = np.dot(values_norm, target_norm)
    
    # 获取 top-k 索引（排除自己）
    top_indices = np.argsort(similarities)[::-1]
    
    results = []
    for idx in top_indices:
        if keys[idx] == target_item_id:
            continue  # 跳过自己
        results.append((int(keys[idx]), float(similarities[idx])))
        if len(results) >= top_k:
            break
    
    return results

# 使用
similar_items = find_similar_items(123, keys, values, top_k=10)
print(f"与 item 123 最相似的 10 个 items:")
for item_id, similarity in similar_items:
    print(f"  Item {item_id}: {similarity:.6f}")
```

## 常见应用场景

### 1. 推荐系统 - 基于 Item 的协同过滤

```python
def recommend_similar_items(user_history, keys, values, top_k=20):
    """
    根据用户历史行为推荐相似的 items
    
    Args:
        user_history: 用户历史交互的 item_ids 列表
        keys: 所有 item_ids
        values: 所有 embeddings
        top_k: 推荐数量
    """
    # 计算用户兴趣向量（历史 items 的平均 embedding）
    user_embeddings = []
    for item_id in user_history:
        emb = get_embedding(keys, values, item_id)
        if emb is not None:
            user_embeddings.append(emb)
    
    if not user_embeddings:
        return []
    
    user_vector = np.mean(user_embeddings, axis=0)
    
    # 计算与所有 items 的相似度
    user_norm = user_vector / np.linalg.norm(user_vector)
    values_norm = values / np.linalg.norm(values, axis=1, keepdims=True)
    similarities = np.dot(values_norm, user_norm)
    
    # 获取 top-k（排除已交互的 items）
    top_indices = np.argsort(similarities)[::-1]
    
    recommendations = []
    for idx in top_indices:
        item_id = int(keys[idx])
        if item_id not in user_history:
            recommendations.append((item_id, float(similarities[idx])))
            if len(recommendations) >= top_k:
                break
    
    return recommendations

# 使用
user_history = [123, 456, 789]
recommendations = recommend_similar_items(user_history, keys, values, top_k=20)
print("推荐结果:")
for item_id, score in recommendations:
    print(f"  Item {item_id}: score={score:.6f}")
```

### 2. Item 聚类分析

```python
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# 对 embeddings 进行聚类
n_clusters = 10
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
cluster_labels = kmeans.fit_predict(values)

# 统计每个聚类的大小
from collections import Counter
cluster_counts = Counter(cluster_labels)
print("聚类分布:")
for cluster_id, count in sorted(cluster_counts.items()):
    print(f"  Cluster {cluster_id}: {count} items")

# 查看某个 item 所属的聚类
item_id = 123
idx = np.where(keys == item_id)[0][0]
cluster = cluster_labels[idx]
print(f"\nItem {item_id} 属于 Cluster {cluster}")

# 找到同一聚类中的其他 items
same_cluster_mask = (cluster_labels == cluster)
same_cluster_items = keys[same_cluster_mask]
print(f"同一聚类中的 items (前10个): {same_cluster_items[:10].tolist()}")
```

### 3. 降维可视化

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# 使用 PCA 降到 2 维
pca = PCA(n_components=2, random_state=42)
embeddings_2d = pca.fit_transform(values)

# 可视化
plt.figure(figsize=(12, 8))
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], alpha=0.5, s=10)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('Item Embeddings Visualization (PCA)')
plt.savefig('embeddings_visualization.png', dpi=150)
print(f"可视化已保存到 embeddings_visualization.png")

# 高亮显示特定的 items
highlight_ids = [123, 456, 789]
for item_id in highlight_ids:
    idx = np.where(keys == item_id)[0]
    if len(idx) > 0:
        plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], 
                   s=100, label=f'Item {item_id}', marker='x')
plt.legend()
plt.savefig('embeddings_visualization_highlighted.png', dpi=150)
```

### 4. Embedding 质量评估

```python
# 计算 embedding 的统计信息
print("Embedding 质量评估:")
print(f"  维度: {values.shape[1]}")
print(f"  均值: {values.mean():.6f}")
print(f"  标准差: {values.std():.6f}")
print(f"  最小值: {values.min():.6f}")
print(f"  最大值: {values.max():.6f}")

# 计算向量的模
norms = np.linalg.norm(values, axis=1)
print(f"\n向量模的统计:")
print(f"  均值: {norms.mean():.6f}")
print(f"  标准差: {norms.std():.6f}")
print(f"  最小值: {norms.min():.6f}")
print(f"  最大值: {norms.max():.6f}")

# 检查是否有零向量或异常值
zero_vectors = np.sum(norms < 1e-6)
print(f"\n零向量数量: {zero_vectors}")

# 计算平均余弦相似度（随机采样）
n_samples = min(1000, len(values))
indices = np.random.choice(len(values), n_samples, replace=False)
sample_values = values[indices]
sample_norm = sample_values / np.linalg.norm(sample_values, axis=1, keepdims=True)
similarity_matrix = np.dot(sample_norm, sample_norm.T)
# 排除对角线
mask = ~np.eye(n_samples, dtype=bool)
avg_similarity = similarity_matrix[mask].mean()
print(f"\n平均余弦相似度 (采样 {n_samples} 个): {avg_similarity:.6f}")
```

## 高级用法

### 在线服务中使用

```python
class EmbeddingService:
    """Embedding 查询服务"""
    
    def __init__(self, npz_file):
        """加载 embeddings 并构建索引"""
        data = np.load(npz_file)
        self.keys = data['keys']
        self.values = data['values']
        
        # 构建 dict 索引加速查询
        self.embedding_dict = {int(k): v for k, v in zip(self.keys, self.values)}
        
        # 预计算归一化的 embeddings（用于快速相似度计算）
        self.values_norm = self.values / np.linalg.norm(
            self.values, axis=1, keepdims=True
        )
        
        print(f"Loaded {len(self.keys)} embeddings")
    
    def get_embedding(self, item_id):
        """获取单个 embedding"""
        return self.embedding_dict.get(item_id)
    
    def get_embeddings_batch(self, item_ids):
        """批量获取 embeddings"""
        return [self.embedding_dict.get(iid) for iid in item_ids]
    
    def find_similar(self, item_id, top_k=10, exclude_ids=None):
        """查找相似 items"""
        emb = self.get_embedding(item_id)
        if emb is None:
            return []
        
        # 归一化查询向量
        query_norm = emb / np.linalg.norm(emb)
        
        # 计算相似度
        similarities = np.dot(self.values_norm, query_norm)
        
        # 排序
        top_indices = np.argsort(similarities)[::-1]
        
        # 收集结果
        results = []
        exclude_set = set(exclude_ids or [])
        exclude_set.add(item_id)  # 排除自己
        
        for idx in top_indices:
            iid = int(self.keys[idx])
            if iid not in exclude_set:
                results.append((iid, float(similarities[idx])))
                if len(results) >= top_k:
                    break
        
        return results

# 使用
service = EmbeddingService("item_embeddings_arrays.npz")

# 查询
emb = service.get_embedding(123)
print(f"Embedding shape: {emb.shape}")

# 查找相似
similar = service.find_similar(123, top_k=10)
print("Similar items:", similar)
```

## 注意事项

1. **内存使用**: 如果 embeddings 数量很大（百万级），加载到内存可能需要较多 RAM
   - 1M items × 256 维 × 4 bytes (float32) ≈ 1GB
   - 考虑使用内存映射: `np.load(file, mmap_mode='r')`

2. **性能优化**:
   - 使用向量化操作（numpy）而不是循环
   - 预先归一化 embeddings 以加速余弦相似度计算
   - 对于超大规模检索，考虑使用 FAISS 等向量检索库

3. **数据一致性**: 确保使用的 item_ids 与训练时一致

## 故障排查

### 问题: 找不到某个 item_id

```python
# 检查 item_id 是否存在
item_id = 123
if item_id in keys:
    print(f"Item {item_id} exists")
else:
    print(f"Item {item_id} NOT found")
    print(f"Available item_ids range: {keys.min()} - {keys.max()}")
    print(f"Total items: {len(keys)}")
```

### 问题: 内存不足

```python
# 使用内存映射模式
data = np.load("item_embeddings_arrays.npz", mmap_mode='r')
keys = data['keys']  # 不会立即加载到内存
values = data['values']  # 按需加载
```

## 更多资源

- NumPy 文档: https://numpy.org/doc/
- 向量检索库 FAISS: https://github.com/facebookresearch/faiss
- 推荐系统教程: https://github.com/grahamjenson/list_of_recommender_systems

## 联系支持

如有问题，请查看项目 README 或提交 issue。

