#!/usr/bin/env python3
"""
提取模型中特定item_id对应的嵌入向量

用法:
    python extract_embedding.py --item_id 12345 --model_path /path/to/model.pt --table_name item_table
    python extract_embedding.py --item_ids 12345,67890 --model_path /path/to/model.pt --table_name item_table
"""

import argparse
import os
import sys
from typing import List, Optional, Union, Dict, Tuple
import json
import numpy as np

import torch
import torch.nn as nn

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from hstu.modules.embedding import ShardedEmbedding


class EmbeddingExtractor:
    """嵌入向量提取器"""
    
    def __init__(self, model_path: str, device: str = "cpu"):
        """
        初始化嵌入提取器
        
        Args:
            model_path: 模型文件路径
            device: 设备类型 ('cpu' 或 'cuda')
        """
        self.model_path = model_path
        self.device = device
        self.model = None
        self.embedding_tables = {}
        
    def load_model(self) -> nn.Module:
        """加载模型"""
        print(f"正在加载模型: {self.model_path}")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型文件不存在: {self.model_path}")
            
        try:
            # 加载模型检查点
            checkpoint = torch.load(self.model_path, map_location=self.device)
            
            # 如果是state_dict，需要重新构建模型结构
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model_state = checkpoint['model_state_dict']
            elif isinstance(checkpoint, dict) and any(key.startswith('embedding') for key in checkpoint.keys()):
                model_state = checkpoint
            else:
                # 假设整个对象就是模型
                self.model = checkpoint.to(self.device)
                self._extract_embedding_tables()
                return self.model
                
            # 这里需要根据实际模型结构来重构模型
            # 暂时提供一个通用框架
            print("模型加载成功")
            return self.model
            
        except Exception as e:
            print(f"加载模型时出错: {e}")
            raise
            
    def _extract_embedding_tables(self):
        """从模型中提取嵌入表"""
        if self.model is None:
            return
            
        def find_embedding_modules(module, prefix=""):
            embedding_tables = {}
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                if isinstance(child, ShardedEmbedding):
                    embedding_tables[full_name] = child
                elif hasattr(child, 'embedding_collection'):
                    # 处理包含embedding_collection的模块
                    embedding_tables[full_name] = child.embedding_collection
                else:
                    # 递归搜索
                    embedding_tables.update(find_embedding_modules(child, full_name))
            return embedding_tables
            
        self.embedding_tables = find_embedding_modules(self.model)
        print(f"找到 {len(self.embedding_tables)} 个嵌入表")
        
    def extract_embedding_by_id(self, item_id: Union[int, str], table_name: Optional[str] = None) -> Optional[np.ndarray]:
        """
        根据item_id提取嵌入向量
        
        Args:
            item_id: item的ID
            table_name: 表名（如果为None则使用第一个表）
            
        Returns:
            嵌入向量numpy数组，如果未找到则返回None
        """
        if not self.embedding_tables:
            print("未找到嵌入表")
            return None
            
        # 选择嵌入表
        if table_name is None:
            table_names = list(self.embedding_tables.keys())
            if not table_names:
                print("没有可用的嵌入表")
                return None
            table_name = table_names[0]
            print(f"使用默认表: {table_name}")
            
        if table_name not in self.embedding_tables:
            print(f"表 {table_name} 不存在，可用表: {list(self.embedding_tables.keys())}")
            return None
            
        embedding_table = self.embedding_tables[table_name]
        print(f"正在从表 {table_name} 提取 item_id {item_id} 的嵌入向量")
        
        try:
            # 方法1: 使用export_local_embedding方法（如果有的话）
            if hasattr(embedding_table, 'export_local_embedding'):
                keys, values = embedding_table.export_local_embedding(table_name)
                item_id_np = np.array([int(item_id)], dtype=np.int64)
                
                # 查找对应的embedding
                mask = keys == item_id_np
                if np.any(mask):
                    embedding = values[mask][0]
                    return embedding.numpy() if hasattr(embedding, 'numpy') else embedding
                    
            # 方法2: 直接访问模型参数
            if hasattr(embedding_table, 'state_dict'):
                state_dict = embedding_table.state_dict()
                # 这里需要根据实际的参数名来查找
                for param_name, param in state_dict.items():
                    if 'weight' in param_name.lower() or 'embedding' in param_name.lower():
                        # 假设参数的第一维是item_id
                        if param.shape[0] > int(item_id):
                            embedding = param[int(item_id)]
                            return embedding.detach().cpu().numpy()
                            
            # 方法3: 如果是ShardedEmbedding
            if isinstance(embedding_table, ShardedEmbedding):
                # 尝试使用ShardedEmbedding的方法
                try:
                    local_emb = embedding_table.export_local_embedding(table_name)
                    if local_emb is not None:
                        keys, values = local_emb
                        item_id_np = np.array([int(item_id)], dtype=np.int64)
                        mask = keys == item_id_np
                        if np.any(mask):
                            idx = np.where(mask)[0][0]
                            embedding = values[idx]
                            return embedding.detach().cpu().numpy() if hasattr(embedding, 'detach') else embedding
                except Exception as e:
                    print(f"使用export_local_embedding方法失败: {e}")
                    
            print(f"在表 {table_name} 中未找到 item_id {item_id}")
            return None
            
        except Exception as e:
            print(f"提取嵌入向量时出错: {e}")
            return None
            
    def extract_embeddings_by_ids(self, item_ids: List[Union[int, str]], table_name: Optional[str] = None) -> Dict[Union[int, str], Optional[np.ndarray]]:
        """
        批量提取嵌入向量
        
        Args:
            item_ids: item_id列表
            table_name: 表名
            
        Returns:
            item_id到嵌入向量的字典
        """
        results = {}
        for item_id in item_ids:
            embedding = self.extract_embedding_by_id(item_id, table_name)
            results[item_id] = embedding
        return results
        
    def list_embedding_tables(self):
        """列出所有可用的嵌入表"""
        if not self.embedding_tables:
            print("未找到嵌入表")
            return
            
        print("可用的嵌入表:")
        for name, table in self.embedding_tables.items():
            print(f"  - {name}: {type(table).__name__}")
            
    def save_embeddings_to_file(self, item_ids: List[Union[int, str]], output_file: str, table_name: Optional[str] = None):
        """
        将提取的嵌入向量保存到文件
        
        Args:
            item_ids: item_id列表
            output_file: 输出文件路径
            table_name: 表名
        """
        embeddings = self.extract_embeddings_by_ids(item_ids, table_name)
        
        # 转换为可序列化的格式
        serializable_embeddings = {}
        for item_id, embedding in embeddings.items():
            if embedding is not None:
                serializable_embeddings[str(item_id)] = {
                    'embedding': embedding.tolist(),
                    'shape': embedding.shape,
                    'dtype': str(embedding.dtype)
                }
            else:
                serializable_embeddings[str(item_id)] = None
                
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_embeddings, f, indent=2, ensure_ascii=False)
            
        print(f"嵌入向量已保存到: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="提取模型中特定item_id的嵌入向量")
    parser.add_argument("--model_path", type=str, required=True, help="模型文件路径")
    parser.add_argument("--item_id", type=str, help="单个item_id")
    parser.add_argument("--item_ids", type=str, help="多个item_id，用逗号分隔，如: 12345,67890")
    parser.add_argument("--table_name", type=str, help="嵌入表名称（可选）")
    parser.add_argument("--output_file", type=str, help="输出文件路径（JSON格式）")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="设备类型")
    parser.add_argument("--list_tables", action="store_true", help="列出所有可用的嵌入表")
    
    args = parser.parse_args()
    
    try:
        # 创建提取器
        extractor = EmbeddingExtractor(args.model_path, args.device)
        
        # 加载模型
        extractor.load_model()
        
        # 列出嵌入表
        if args.list_tables or not any([args.item_id, args.item_ids]):
            extractor.list_embedding_tables()
            
        # 处理item_id
        if args.item_id or args.item_ids:
            if args.item_id:
                item_ids = [args.item_id]
            else:
                item_ids = [id.strip() for id in args.item_ids.split(",")]
                
            print(f"正在提取 {len(item_ids)} 个item的嵌入向量...")
            
            # 提取嵌入向量
            embeddings = extractor.extract_embeddings_by_ids(item_ids, args.table_name)
            
            # 显示结果
            for item_id, embedding in embeddings.items():
                print(f"\nItem ID: {item_id}")
                if embedding is not None:
                    print(f"  嵌入向量形状: {embedding.shape}")
                    print(f"  嵌入向量前10个元素: {embedding[:10]}")
                    print(f"  嵌入向量类型: {embedding.dtype}")
                else:
                    print("  未找到对应的嵌入向量")
                    
            # 保存到文件
            if args.output_file:
                extractor.save_embeddings_to_file(item_ids, args.output_file, args.table_name)
                
        else:
            parser.print_help()
            
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()