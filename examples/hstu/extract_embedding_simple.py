#!/usr/bin/env python3
"""
简化版嵌入向量提取工具 - 专门针对HSTU检索模型

用法:
    python extract_embedding_simple.py --item_ids 12345,67890 --model_dir ./checkpoints
"""

import os
import sys
import json
import argparse
from typing import List, Optional, Dict, Any
import numpy as np

# 尝试导入torchrec相关模块
try:
    import torch
    from torchrec.distributed.embedding import ShardedEmbeddingCollection
    TORCHREC_AVAILABLE = True
except ImportError:
    print("警告: torchrec不可用，将使用模拟模式")
    TORCHREC_AVAILABLE = False


class SimpleEmbeddingExtractor:
    """简化的嵌入向量提取器"""
    
    def __init__(self, model_dir: str):
        """
        初始化提取器
        
        Args:
            model_dir: 模型目录路径
        """
        self.model_dir = model_dir
        self.embedding_tables = {}
        self._load_embeddings()
        
    def _load_embeddings(self):
        """加载嵌入表"""
        print(f"正在从 {self.model_dir} 加载嵌入表...")
        
        # 首先尝试在model_dir直接查找二进制文件
        if self._load_binary_embeddings(self.model_dir):
            return  # 成功加载就返回
            
        # 然后检查常见的嵌入表文件位置
        possible_paths = [
            os.path.join(self.model_dir, "item_embeddings"),
            os.path.join(self.model_dir, "embeddings", "item_table"),
            os.path.join(self.model_dir, "embedding_collection", "item_table"),
            os.path.join(self.model_dir, "retrieval_gr", "item_table"),
        ]
        
        # 查找二进制文件（key-value对）
        for path in possible_paths:
            if os.path.exists(path):
                if self._load_binary_embeddings(path):
                    return
                
        # 最后尝试从模型检查点加载
        self._load_from_checkpoint()
            
    def _load_binary_embeddings(self, path: str) -> bool:
        """从二进制文件加载嵌入表"""
        print(f"从二进制文件加载嵌入表: {path}")
        
        key_file = os.path.join(path, "keys.bin")
        value_file = os.path.join(path, "values.bin")
        
        print(f"检查文件: {key_file} (存在: {os.path.exists(key_file)})")
        print(f"检查文件: {value_file} (存在: {os.path.exists(value_file)})")
        
        if os.path.exists(key_file) and os.path.exists(value_file):
            try:
                keys = np.fromfile(key_file, dtype=np.int64)
                values = np.fromfile(value_file, dtype=np.float32)
                
                print(f"加载了 {len(keys)} 个键和 {len(values)} 个值")
                
                # 重塑values数组，每一行是一个嵌入向量
                embedding_dim = values.shape[0] // keys.shape[0]
                values = values.reshape(keys.shape[0], embedding_dim)
                
                self.embedding_tables['item_table'] = {
                    'keys': keys,
                    'values': values
                }
                print(f"成功加载 {len(keys)} 个item的嵌入向量，维度: {embedding_dim}")
                return True
                
            except Exception as e:
                print(f"加载二进制文件失败: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"未找到键值文件: {key_file}, {value_file}")
            
            # 如果直接路径没有，尝试在当前目录查找
            key_file_current = os.path.join(self.model_dir, "keys.bin")
            value_file_current = os.path.join(self.model_dir, "values.bin")
            
            print(f"尝试检查当前目录: {key_file_current} (存在: {os.path.exists(key_file_current)})")
            print(f"尝试检查当前目录: {value_file_current} (存在: {os.path.exists(value_file_current)})")
            
            if os.path.exists(key_file_current) and os.path.exists(value_file_current):
                try:
                    keys = np.fromfile(key_file_current, dtype=np.int64)
                    values = np.fromfile(value_file_current, dtype=np.float32)
                    
                    print(f"从当前目录加载了 {len(keys)} 个键和 {len(values)} 个值")
                    
                    # 重塑values数组
                    embedding_dim = values.shape[0] // keys.shape[0]
                    values = values.reshape(keys.shape[0], embedding_dim)
                    
                    self.embedding_tables['item_table'] = {
                        'keys': keys,
                        'values': values
                    }
                    print(f"成功从当前目录加载 {len(keys)} 个item的嵌入向量，维度: {embedding_dim}")
                    return True
                    
                except Exception as e:
                    print(f"从当前目录加载二进制文件失败: {e}")
                    import traceback
                    traceback.print_exc()
                    
        return False
            
    def _load_from_checkpoint(self):
        """从模型检查点加载"""
        checkpoint_files = []
        for file in os.listdir(self.model_dir):
            if file.endswith(('.pth', '.pt', '.pkl')):
                checkpoint_files.append(os.path.join(self.model_dir, file))
                
        for checkpoint_path in checkpoint_files:
            try:
                print(f"尝试加载检查点: {checkpoint_path}")
                checkpoint = torch.load(checkpoint_path, map_location='cpu')
                
                # 查找嵌入相关的参数
                if isinstance(checkpoint, dict):
                    for key, value in checkpoint.items():
                        if 'embedding' in key.lower() and isinstance(value, torch.Tensor):
                            # 假设这是item embedding矩阵
                            if len(value.shape) == 2:
                                self.embedding_tables['item_table'] = {
                                    'keys': np.arange(value.shape[0]),
                                    'values': value.detach().cpu().numpy()
                                }
                                print(f"从检查点加载嵌入表: {key}, 形状: {value.shape}")
                                return
                                
                # 如果checkpoint就是模型
                elif hasattr(checkpoint, 'state_dict'):
                    state_dict = checkpoint.state_dict()
                    for key, value in state_dict.items():
                        if 'embedding' in key.lower() and isinstance(value, torch.Tensor):
                            if len(value.shape) == 2:
                                self.embedding_tables['item_table'] = {
                                    'keys': np.arange(value.shape[0]),
                                    'values': value.detach().cpu().numpy()
                                }
                                print(f"从模型加载嵌入表: {key}, 形状: {value.shape}")
                                return
                                
            except Exception as e:
                print(f"加载检查点 {checkpoint_path} 失败: {e}")
                continue
                
        print("未能在检查点中找到嵌入表")
        
    def get_item_embedding(self, item_id: int) -> Optional[np.ndarray]:
        """
        获取指定item_id的嵌入向量
        
        Args:
            item_id: item的ID
            
        Returns:
            嵌入向量或None（如果未找到）
        """
        if 'item_table' not in self.embedding_tables:
            print("未找到item embedding表")
            return None
            
        table = self.embedding_tables['item_table']
        keys = table['keys']
        values = table['values']
        
        # 查找item_id
        if item_id >= 0 and item_id < len(keys):
            if keys[item_id] == item_id:
                return values[item_id]
            else:
                # 需要在keys中搜索
                mask = keys == item_id
                if np.any(mask):
                    idx = np.where(mask)[0][0]
                    return values[idx]
                    
        print(f"未找到 item_id {item_id}")
        return None
        
    def get_items_embeddings(self, item_ids: List[int]) -> Dict[int, Optional[np.ndarray]]:
        """
        批量获取item嵌入向量
        
        Args:
            item_ids: item_id列表
            
        Returns:
            item_id到嵌入向量的字典
        """
        results = {}
        for item_id in item_ids:
            results[item_id] = self.get_item_embedding(item_id)
        return results
        
    def search_similar_items(self, query_embedding: np.ndarray, top_k: int = 10) -> List[tuple]:
        """
        查找相似item（基于余弦相似度）
        
        Args:
            query_embedding: 查询嵌入向量
            top_k: 返回前k个相似item
            
        Returns:
            (item_id, 相似度)的列表
        """
        if 'item_table' not in self.embedding_tables:
            return []
            
        table = self.embedding_tables['item_table']
        keys = table['keys']
        values = table['values']
        
        # 归一化向量用于余弦相似度计算
        query_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
        values_norm = values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-8)
        
        # 计算余弦相似度
        similarities = np.dot(values_norm, query_norm)
        
        # 获取top_k个最相似的item
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            item_id = int(keys[idx])
            similarity = float(similarities[idx])
            results.append((item_id, similarity))
            
        return results


def create_test_data():
    """创建测试数据用于演示"""
    print("创建测试数据...")
    
    # 生成模拟的item embedding数据
    num_items = 1000
    embedding_dim = 64
    
    # 创建随机嵌入向量
    np.random.seed(42)
    item_ids = np.arange(num_items)
    embeddings = np.random.randn(num_items, embedding_dim).astype(np.float32)
    
    # 保存到文件
    test_dir = "/tmp/test_embeddings"
    os.makedirs(test_dir, exist_ok=True)
    
    key_file = os.path.join(test_dir, "keys.bin")
    value_file = os.path.join(test_dir, "values.bin")
    
    item_ids.tofile(key_file)
    embeddings.tofile(value_file)
    
    print(f"测试数据已创建: {test_dir}")
    print(f"  - Item数量: {num_items}")
    print(f"  - 嵌入维度: {embedding_dim}")
    
    return test_dir


def main():
    parser = argparse.ArgumentParser(description="提取item嵌入向量 - HSTU检索模型专用")
    parser.add_argument("--model_dir", type=str, help="模型目录路径")
    parser.add_argument("--item_ids", type=str, help="item_id列表，用逗号分隔，如: 1,2,3,4,5")
    parser.add_argument("--item_id", type=int, help="单个item_id")
    parser.add_argument("--output", type=str, help="输出文件路径(JSON格式)")
    parser.add_argument("--search", action="store_true", help="启用相似搜索模式")
    parser.add_argument("--top_k", type=int, default=10, help="相似搜索返回的top_k结果")
    parser.add_argument("--create_test", action="store_true", help="创建测试数据")
    parser.add_argument("--embedding_dim", type=int, default=64, help="测试数据的嵌入维度")
    
    args = parser.parse_args()
    
    # 创建测试数据
    if args.create_test:
        model_dir = create_test_data()
        args.model_dir = model_dir
    elif not args.model_dir:
        parser.error("请指定 --model_dir 或使用 --create_test 创建测试数据")
        
    # 初始化提取器
    extractor = SimpleEmbeddingExtractor(args.model_dir)
    
    # 相似搜索模式
    if args.search and args.item_id is not None:
        print(f"执行相似搜索，查询item_id: {args.item_id}")
        
        # 获取查询item的嵌入向量
        query_embedding = extractor.get_item_embedding(args.item_id)
        if query_embedding is not None:
            similar_items = extractor.search_similar_items(query_embedding, args.top_k)
            print(f"\n与item {args.item_id} 最相似的前{args.top_k}个items:")
            for item_id, similarity in similar_items:
                print(f"  Item {item_id}: 相似度 = {similarity:.4f}")
        else:
            print(f"未找到item {args.item_id}")
            
    # 提取嵌入向量
    elif args.item_ids or args.item_id:
        if args.item_id:
            item_ids = [args.item_id]
        else:
            item_ids = [int(x.strip()) for x in args.item_ids.split(",")]
            
        print(f"提取嵌入向量: {item_ids}")
        
        embeddings = extractor.get_items_embeddings(item_ids)
        
        # 显示结果
        for item_id, embedding in embeddings.items():
            print(f"\nItem ID: {item_id}")
            if embedding is not None:
                print(f"  嵌入向量形状: {embedding.shape}")
                print(f"  前10个元素: {embedding[:10]}")
                print(f"  L2范数: {np.linalg.norm(embedding):.4f}")
            else:
                print("  未找到")
                
        # 保存结果
        if args.output:
            results = {}
            for item_id, embedding in embeddings.items():
                if embedding is not None:
                    results[str(item_id)] = {
                        'embedding': embedding.tolist(),
                        'shape': list(embedding.shape),
                        'norm': float(np.linalg.norm(embedding))
                    }
                else:
                    results[str(item_id)] = None
                    
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2)
                
            print(f"\n结果已保存到: {args.output}")
            
    else:
        parser.print_help()


if __name__ == "__main__":
    main()