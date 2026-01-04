#!/usr/bin/env python3
"""
从保存的checkpoint中获取指定item_id的embedding

用法:
    python get_item_embedding.py --checkpoint_dir ./checkpoints/iter100 --item_id 12345 --gin_config_file ./training/configs/gameid_retrieval.gin
    python get_item_embedding.py --checkpoint_dir ./checkpoints/iter100 --item_ids 12345,67890 --gin_config_file ./training/configs/gameid_retrieval.gin
"""

import argparse
import os
import sys
from typing import List, Optional, Dict, Tuple
import numpy as np

import torch
import torch.distributed as dist
import gin

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import commons.utils.initialize as init
from commons.checkpoint import get_unwrapped_module, load
from configs import RetrievalConfig
from model import get_retrieval_model
from trainer.utils import (
    create_embedding_configs,
    create_hstu_config,
    get_dataset_and_embedding_args,
)
from utils import (
    DatasetArgs,
    NetworkArgs,
    TensorModelParallelArgs,
)


def load_model_from_checkpoint(
    checkpoint_dir: str,
    gin_config_file: str,
    device: str = "cuda",
) -> torch.nn.Module:
    """
    从checkpoint加载模型和embedding
    
    Args:
        checkpoint_dir: checkpoint目录路径
        gin_config_file: gin配置文件路径
        device: 设备类型
        
    Returns:
        加载的模型
    """
    # 解析gin配置
    gin.parse_config_file(gin_config_file)
    
    # 获取配置参数
    dataset_args, embedding_args = get_dataset_and_embedding_args()
    network_args = NetworkArgs()
    tp_args = TensorModelParallelArgs()
    
    # 初始化分布式环境（单进程模式）
    # 对于单进程模式，设置必要的环境变量
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "12355"
    
    # 初始化分布式环境
    if not dist.is_initialized():
        try:
            init.initialize_distributed()
        except KeyError as e:
            # 如果仍然缺少环境变量，使用单进程初始化
            print(f"Warning: {e}, using single rank initialization")
            init.initialize_single_rank()
    
    init.initialize_model_parallel(
        tensor_model_parallel_size=tp_args.tensor_model_parallel_size
    )
    
    # 创建模型配置
    hstu_config = create_hstu_config(network_args, tp_args)
    task_config = RetrievalConfig(
        embedding_configs=create_embedding_configs(
            dataset_args, network_args, embedding_args
        ),
        temperature=1.0,
        l2_norm_eps=1e-8,
        num_negatives=128,
        eval_metrics=("NDCG@10",),
    )
    
    # 创建模型
    model = get_retrieval_model(hstu_config=hstu_config, task_config=task_config)
    
    # 将模型移到指定设备
    if device == "cuda" and torch.cuda.is_available():
        model = model.cuda()
    
    # 加载checkpoint
    print(f"Loading checkpoint from {checkpoint_dir}")
    load(checkpoint_dir, model, dense_optimizer=None, include_optim_state=False)
    print("Checkpoint loaded successfully")
    
    return model


def get_item_embedding_from_model(
    model: torch.nn.Module,
    item_id: int,
    table_name: Optional[str] = None,
) -> Optional[np.ndarray]:
    """
    从模型中获取指定item_id的embedding
    
    Args:
        model: 加载的模型
        item_id: item的ID
        table_name: embedding表名称，如果为None则自动查找
        
    Returns:
        embedding向量或None
    """
    # 获取unwrapped model
    unwrapped_model = get_unwrapped_module(model)
    
    # 如果是RetrievalGR模型，获取item feature table name
    if hasattr(unwrapped_model, "get_item_feature_table_name"):
        if table_name is None:
            table_name = unwrapped_model.get_item_feature_table_name()
        print(f"Using table name: {table_name}")
    
    # 获取embedding collection
    if not hasattr(unwrapped_model, "_embedding_collection"):
        print("Model does not have _embedding_collection attribute")
        return None
    
    embedding_collection = unwrapped_model._embedding_collection
    
    # 尝试使用export_local_embedding方法
    try:
        if hasattr(embedding_collection, "export_local_embedding"):
            print(f"Exporting local embeddings from table: {table_name}")
            embedding_export = embedding_collection.export_local_embedding(table_name)
            
            # export_local_embedding返回一个元组 (keys, values)
            if isinstance(embedding_export, tuple) and len(embedding_export) == 2:
                keys, values = embedding_export
                
                # 将keys和values转换为numpy数组
                if isinstance(keys, torch.Tensor):
                    keys = keys.cpu().numpy()
                if isinstance(values, torch.Tensor):
                    values = values.cpu().numpy()
                
                # 确保keys和values是numpy数组
                keys = np.asarray(keys)
                values = np.asarray(values)
                
                print(f"Exported {len(keys)} embeddings")
                print(f"Key range: {keys.min()} to {keys.max()}")
                
                # 查找item_id对应的embedding
                mask = keys == item_id
                
                if np.any(mask):
                    idx = np.where(mask)[0][0]
                    embedding = values[idx]
                    print(f"Found embedding for item_id {item_id} at index {idx}")
                    return embedding
                else:
                    print(f"Item_id {item_id} not found in keys")
                    print(f"Available key range: {keys.min()} to {keys.max()}")
                    # 显示最接近的keys
                    if len(keys) > 0:
                        closest_idx = np.argmin(np.abs(keys - item_id))
                        print(f"Closest key: {keys[closest_idx]} (difference: {abs(keys[closest_idx] - item_id)})")
                    return None
            else:
                print(f"Unexpected export format, got type: {type(embedding_export)}")
                if isinstance(embedding_export, tuple):
                    print(f"Tuple length: {len(embedding_export)}")
                return None
        else:
            print("embedding_collection does not have export_local_embedding method")
            return None
    except Exception as e:
        print(f"Error exporting embeddings: {e}")
        import traceback
        traceback.print_exc()
        return None


def get_items_embeddings(
    model: torch.nn.Module,
    item_ids: List[int],
    table_name: Optional[str] = None,
) -> Dict[int, Optional[np.ndarray]]:
    """
    批量获取多个item_id的embedding
    
    Args:
        model: 加载的模型
        item_ids: item_id列表
        table_name: embedding表名称
        
    Returns:
        item_id到embedding的字典
    """
    results = {}
    for item_id in item_ids:
        results[item_id] = get_item_embedding_from_model(model, item_id, table_name)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="从checkpoint中获取item embedding"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="checkpoint目录路径（包含dynamicemb_module和torch_module）",
    )
    parser.add_argument(
        "--gin_config_file",
        type=str,
        required=True,
        help="gin配置文件路径",
    )
    parser.add_argument(
        "--item_id",
        type=int,
        default=None,
        help="单个item_id",
    )
    parser.add_argument(
        "--item_ids",
        type=str,
        default=None,
        help="多个item_id，用逗号分隔，例如: 12345,67890",
    )
    parser.add_argument(
        "--table_name",
        type=str,
        default=None,
        help="embedding表名称（可选，如果不指定则自动查找）",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="输出文件路径（保存embedding到numpy文件）",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="设备类型",
    )
    
    args = parser.parse_args()
    
    # 检查checkpoint目录是否存在
    if not os.path.exists(args.checkpoint_dir):
        print(f"Error: Checkpoint directory {args.checkpoint_dir} does not exist")
        sys.exit(1)
    
    # 检查gin配置文件是否存在
    if not os.path.exists(args.gin_config_file):
        print(f"Error: Gin config file {args.gin_config_file} does not exist")
        sys.exit(1)
    
    # 解析item_ids
    item_ids = []
    if args.item_id is not None:
        item_ids = [args.item_id]
    elif args.item_ids is not None:
        item_ids = [int(x.strip()) for x in args.item_ids.split(",")]
    else:
        print("Error: Please provide either --item_id or --item_ids")
        sys.exit(1)
    
    try:
        # 加载模型
        print("=" * 60)
        print("Loading model from checkpoint...")
        print("=" * 60)
        model = load_model_from_checkpoint(
            args.checkpoint_dir,
            args.gin_config_file,
            device=args.device,
        )
        
        # 获取embeddings
        print("=" * 60)
        print(f"Extracting embeddings for {len(item_ids)} items...")
        print("=" * 60)
        embeddings = get_items_embeddings(
            model,
            item_ids,
            table_name=args.table_name,
        )
        
        # 显示结果
        print("\n" + "=" * 60)
        print("Results:")
        print("=" * 60)
        for item_id, embedding in embeddings.items():
            print(f"\nItem ID: {item_id}")
            if embedding is not None:
                print(f"  Embedding shape: {embedding.shape}")
                print(f"  Embedding dtype: {embedding.dtype}")
                print(f"  Embedding norm: {np.linalg.norm(embedding):.6f}")
                print(f"  First 10 elements: {embedding[:10]}")
            else:
                print("  Embedding not found")
        
        # 保存到文件
        if args.output_file:
            print(f"\nSaving embeddings to {args.output_file}")
            np.savez(args.output_file, **{f"item_{item_id}": emb for item_id, emb in embeddings.items() if emb is not None})
            print("Embeddings saved successfully")
        
        print("\n" + "=" * 60)
        print("Done!")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # 清理分布式环境
        if dist.is_initialized():
            init.destroy_global_state()


if __name__ == "__main__":
    main()

