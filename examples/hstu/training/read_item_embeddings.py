#!/usr/bin/env python3
"""
读取和查询导出的 item_id embeddings

用法示例:
    # 1. 查看文件信息
    python read_item_embeddings.py --file item_embeddings_arrays.npz --info
    
    # 2. 查询单个 item_id
    python read_item_embeddings.py --file item_embeddings_arrays.npz --item_id 123
    
    # 3. 查询多个 item_ids
    python read_item_embeddings.py --file item_embeddings_arrays.npz --item_ids 123,456,789
    
    # 4. 导出为文本文件（前100个）
    python read_item_embeddings.py --file item_embeddings_arrays.npz --export_txt output.txt --limit 100
"""

import argparse
import numpy as np
from typing import Optional, List, Dict


def load_embeddings(file_path: str) -> Dict[str, np.ndarray]:
    """
    加载 npz 格式的 embedding 文件
    
    Args:
        file_path: npz 文件路径
        
    Returns:
        包含 keys, values 等数据的字典
    """
    print(f"Loading embeddings from: {file_path}")
    data = np.load(file_path)
    
    # 检查文件格式
    file_keys = list(data.keys())
    print(f"File contains: {file_keys}")
    
    result = {}
    for key in file_keys:
        result[key] = data[key]
    
    return result


def show_info(data: Dict[str, np.ndarray]):
    """显示 embedding 文件的基本信息"""
    print("\n" + "=" * 60)
    print("Embedding File Information")
    print("=" * 60)
    
    if 'keys' in data and 'values' in data:
        # 数组格式
        keys = data['keys']
        values = data['values']
        
        print(f"Format: Array format (keys + values)")
        print(f"Number of items: {len(keys)}")
        print(f"Embedding dimension: {values.shape[1] if len(values.shape) > 1 else 'N/A'}")
        print(f"Keys dtype: {keys.dtype}")
        print(f"Values dtype: {values.dtype}")
        print(f"Keys shape: {keys.shape}")
        print(f"Values shape: {values.shape}")
        print(f"\nKey statistics:")
        print(f"  Min item_id: {keys.min()}")
        print(f"  Max item_id: {keys.max()}")
        print(f"  Unique keys: {len(np.unique(keys))}")
        
        if 'table_name' in data:
            print(f"\nTable name: {data['table_name']}")
        
        # 显示前几个 item_ids
        print(f"\nFirst 10 item_ids: {keys[:10].tolist()}")
        
    else:
        # 字典格式
        print(f"Format: Dictionary format")
        item_keys = [k for k in data.keys() if k.startswith('item_')]
        print(f"Number of items: {len(item_keys)}")
        
        if item_keys:
            first_key = item_keys[0]
            embedding = data[first_key]
            print(f"Embedding dimension: {len(embedding)}")
            print(f"Embedding dtype: {embedding.dtype}")
            print(f"\nFirst 10 item_ids: {[int(k.replace('item_', '')) for k in item_keys[:10]]}")
    
    print("=" * 60)


def get_embedding(data: Dict[str, np.ndarray], item_id: int) -> Optional[np.ndarray]:
    """
    获取指定 item_id 的 embedding
    
    Args:
        data: 加载的 npz 数据
        item_id: 要查询的 item_id
        
    Returns:
        embedding 向量，如果不存在则返回 None
    """
    if 'keys' in data and 'values' in data:
        # 数组格式
        keys = data['keys']
        values = data['values']
        
        # 查找 item_id 在 keys 中的位置
        mask = (keys == item_id)
        if mask.any():
            idx = np.where(mask)[0][0]
            return values[idx]
        else:
            return None
    else:
        # 字典格式
        key = f'item_{item_id}'
        if key in data:
            return data[key]
        else:
            return None


def query_items(data: Dict[str, np.ndarray], item_ids: List[int]):
    """查询多个 item_ids 的 embeddings"""
    print("\n" + "=" * 60)
    print(f"Querying {len(item_ids)} item(s)")
    print("=" * 60)
    
    for item_id in item_ids:
        embedding = get_embedding(data, item_id)
        if embedding is not None:
            print(f"\nItem ID: {item_id}")
            print(f"  Embedding shape: {embedding.shape}")
            print(f"  Embedding norm: {np.linalg.norm(embedding):.6f}")
            print(f"  First 10 values: {embedding[:10].tolist()}")
        else:
            print(f"\nItem ID: {item_id} - NOT FOUND")
    
    print("=" * 60)


def export_to_text(data: Dict[str, np.ndarray], output_file: str, limit: Optional[int] = None):
    """
    导出 embeddings 到文本文件
    
    Args:
        data: 加载的 npz 数据
        output_file: 输出文件路径
        limit: 最多导出多少条，None 表示全部导出
    """
    print(f"\nExporting embeddings to text file: {output_file}")
    
    if 'keys' in data and 'values' in data:
        keys = data['keys']
        values = data['values']
        
        if limit is not None:
            keys = keys[:limit]
            values = values[:limit]
        
        with open(output_file, 'w') as f:
            # 写入头部信息
            f.write(f"# Item embeddings\n")
            f.write(f"# Format: item_id embedding_dim value1 value2 ...\n")
            f.write(f"# Total items: {len(keys)}\n")
            f.write(f"# Embedding dimension: {values.shape[1]}\n")
            f.write(f"\n")
            
            # 写入每个 item 的 embedding
            for item_id, embedding in zip(keys, values):
                f.write(f"{item_id} {len(embedding)}")
                for val in embedding:
                    f.write(f" {val:.6f}")
                f.write("\n")
        
        print(f"Exported {len(keys)} embeddings to {output_file}")
    else:
        print("Error: Text export only supports array format files (*_arrays.npz)")


def compute_similarity(data: Dict[str, np.ndarray], item_id1: int, item_id2: int):
    """计算两个 item 之间的相似度"""
    emb1 = get_embedding(data, item_id1)
    emb2 = get_embedding(data, item_id2)
    
    if emb1 is None:
        print(f"Item {item_id1} not found")
        return
    if emb2 is None:
        print(f"Item {item_id2} not found")
        return
    
    # 计算余弦相似度
    cosine_sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2))
    
    # 计算欧氏距离
    euclidean_dist = np.linalg.norm(emb1 - emb2)
    
    print("\n" + "=" * 60)
    print(f"Similarity between item {item_id1} and item {item_id2}")
    print("=" * 60)
    print(f"Cosine similarity: {cosine_sim:.6f}")
    print(f"Euclidean distance: {euclidean_dist:.6f}")
    print("=" * 60)


def find_similar_items(data: Dict[str, np.ndarray], item_id: int, top_k: int = 10):
    """查找与指定 item 最相似的 top-k 个 items"""
    target_emb = get_embedding(data, item_id)
    if target_emb is None:
        print(f"Item {item_id} not found")
        return
    
    if 'keys' not in data or 'values' not in data:
        print("Error: Similar search only supports array format files (*_arrays.npz)")
        return
    
    keys = data['keys']
    values = data['values']
    
    print(f"\nComputing similarities for {len(keys)} items...")
    
    # 计算余弦相似度
    # 归一化
    target_norm = target_emb / np.linalg.norm(target_emb)
    values_norm = values / np.linalg.norm(values, axis=1, keepdims=True)
    
    # 计算相似度
    similarities = np.dot(values_norm, target_norm)
    
    # 获取 top-k（排除自己）
    top_indices = np.argsort(similarities)[::-1]
    
    print("\n" + "=" * 60)
    print(f"Top {top_k} similar items to item {item_id}")
    print("=" * 60)
    
    count = 0
    for idx in top_indices:
        if keys[idx] == item_id:
            continue  # 跳过自己
        
        print(f"{count + 1}. Item {keys[idx]}: similarity = {similarities[idx]:.6f}")
        count += 1
        
        if count >= top_k:
            break
    
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="读取和查询 item embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--file",
        type=str,
        required=True,
        help="npz 文件路径（通常是 *_arrays.npz）",
    )
    
    parser.add_argument(
        "--info",
        action="store_true",
        help="显示文件信息",
    )
    
    parser.add_argument(
        "--item_id",
        type=int,
        help="查询单个 item_id 的 embedding",
    )
    
    parser.add_argument(
        "--item_ids",
        type=str,
        help="查询多个 item_ids（逗号分隔，如: 123,456,789）",
    )
    
    parser.add_argument(
        "--export_txt",
        type=str,
        help="导出为文本文件",
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        help="导出时限制条数",
    )
    
    parser.add_argument(
        "--similarity",
        type=str,
        help="计算两个 item 之间的相似度（格式: item_id1,item_id2）",
    )
    
    parser.add_argument(
        "--similar",
        type=int,
        help="查找与指定 item_id 最相似的 items",
    )
    
    parser.add_argument(
        "--top_k",
        type=int,
        default=10,
        help="查找相似 items 时返回的数量（默认: 10）",
    )
    
    args = parser.parse_args()
    
    # 加载数据
    data = load_embeddings(args.file)
    
    # 执行操作
    if args.info:
        show_info(data)
    
    if args.item_id is not None:
        query_items(data, [args.item_id])
    
    if args.item_ids:
        item_ids = [int(x.strip()) for x in args.item_ids.split(',')]
        query_items(data, item_ids)
    
    if args.export_txt:
        export_to_text(data, args.export_txt, args.limit)
    
    if args.similarity:
        parts = args.similarity.split(',')
        if len(parts) == 2:
            item_id1 = int(parts[0].strip())
            item_id2 = int(parts[1].strip())
            compute_similarity(data, item_id1, item_id2)
        else:
            print("Error: --similarity format should be: item_id1,item_id2")
    
    if args.similar is not None:
        find_similar_items(data, args.similar, args.top_k)
    
    # 如果没有指定任何操作，默认显示信息
    if not any([args.info, args.item_id, args.item_ids, args.export_txt, 
                args.similarity, args.similar is not None]):
        show_info(data)


if __name__ == "__main__":
    main()

