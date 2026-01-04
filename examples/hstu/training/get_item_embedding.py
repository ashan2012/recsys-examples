#!/usr/bin/env python3
"""
从保存的checkpoint中导出所有item_id的embedding

用法:
    python get_item_embedding.py --checkpoint_dir ./checkpoints/iter100 --gin_config_file ./training/configs/gameid_retrieval.gin --output_file item_embeddings.npz
"""

import argparse
import os
import sys
from typing import Dict
import numpy as np

import torch
import torch.distributed as dist
import gin

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import commons.utils.initialize as init
from commons.checkpoint import get_unwrapped_module
from configs import RetrievalConfig
from model import get_retrieval_model
from trainer.utils import (
    create_embedding_configs,
    create_hstu_config,
    get_dataset_and_embedding_args,
)
from utils import (
    NetworkArgs,
    TensorModelParallelArgs,
)


def load_model_from_checkpoint(
    checkpoint_dir: str,
    gin_config_file: str,
) -> torch.nn.Module:
    """
    从checkpoint加载模型和embedding
    
    Args:
        checkpoint_dir: checkpoint目录路径
        gin_config_file: gin配置文件路径
        
    Returns:
        加载的模型
    """
    # 解析gin配置
    gin.parse_config_file(gin_config_file)
    
    # 获取配置参数
    dataset_args, embedding_args = get_dataset_and_embedding_args()
    network_args = NetworkArgs()
    tp_args = TensorModelParallelArgs()
    
    # 初始化分布式环境
    if not dist.is_initialized():
        init.initialize_distributed()
    init.initialize_model_parallel(
        tensor_model_parallel_size=tp_args.tensor_model_parallel_size
    )
    
    # 设置CUDA设备
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
    
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
    print("Creating model...")
    model = get_retrieval_model(hstu_config=hstu_config, task_config=task_config)
    
    # 加载checkpoint
    unwrapped_model = get_unwrapped_module(model)
    
    # 1. 加载dynamic embedding表
    from dynamicemb.dump_load import DynamicEmbLoad as dynamic_emb_load
    save_dir = os.path.join(checkpoint_dir, "dynamicemb_module")
    if os.path.exists(save_dir):
        print("Loading dynamic embedding tables...")
        dynamic_emb_load(save_dir, unwrapped_model, optim=False)
        print("Dynamic embedding tables loaded")
    else:
        print(f"Warning: Dynamic embedding directory {save_dir} not found")
    
    # 2. 加载dense模型参数（过滤掉dynamic embedding表的参数）
    save_path = os.path.join(
        checkpoint_dir, "torch_module", "model.{}.pth".format(dist.get_rank())
    )
    if os.path.exists(save_path):
        print("Loading dense model parameters...")
        state_dict = torch.load(save_path, map_location="cpu")
        if "model_state_dict" in state_dict:
            model_state_dict = state_dict["model_state_dict"]
            new_state_dict = {}
            
            # 获取dynamic embedding表名
            dynamic_table_names = set()
            if hasattr(unwrapped_model, "_embedding_collection"):
                embedding_collection = unwrapped_model._embedding_collection
                if hasattr(embedding_collection, "_dynamic_embedding_collection"):
                    dynamic_emb_collection = embedding_collection._dynamic_embedding_collection
                    if hasattr(dynamic_emb_collection, "_embedding_tables"):
                        dynamic_tables = dynamic_emb_collection._embedding_tables
                        if hasattr(dynamic_tables, "table_names"):
                            dynamic_table_names = set(dynamic_tables.table_names)
            
            # 过滤掉dynamic embedding表的参数
            for key, value in model_state_dict.items():
                # 跳过dynamic embedding表的参数
                is_dynamic_emb = False
                for table_name in dynamic_table_names:
                    if (f".{table_name}." in key or 
                        f".{table_name}_" in key or
                        f"embeddings.{table_name}." in key or
                        ".item_id." in key or 
                        ".user_id." in key):
                        is_dynamic_emb = True
                        break
                
                # 检查是否是[1,1]形状的占位符
                if "_model_parallel_embedding_collection.embeddings" in key:
                    if isinstance(value, torch.Tensor) and value.shape == torch.Size([1, 1]):
                        is_dynamic_emb = True
                
                if is_dynamic_emb:
                    continue
                
                # 处理表名不匹配
                new_key = key.replace("interaction_weights", "interaction")
                new_key = new_key.replace("action_weights", "interaction")
                new_state_dict[new_key] = value
                
                # 同时添加带和不带.weight的版本（处理interaction表）
                if "interaction" in new_key and not new_key.endswith(".weight"):
                    new_state_dict[new_key + ".weight"] = value
            
            # 加载dense参数
            missing_keys, unexpected_keys = unwrapped_model.load_state_dict(
                new_state_dict, strict=False
            )
            
            # 过滤掉dynamic embedding相关的missing keys
            filtered_missing = [k for k in missing_keys if not any(
                f".{tn}." in k or f".{tn}_" in k or ".item_id." in k or ".user_id." in k
                for tn in dynamic_table_names
            )]
            if filtered_missing:
                print(f"Warning: Missing keys (non-dynamic): {filtered_missing[:3]}...")
            
            print("Dense model parameters loaded")
    
    # 移动到CUDA
    if torch.cuda.is_available():
        model = model.cuda()
    
    print("Checkpoint loaded successfully")
    return model


def export_all_item_embeddings(
    model: torch.nn.Module,
    output_file: str,
) -> Dict[int, np.ndarray]:
    """
    导出所有item_id的embedding
    
    Args:
        model: 加载的模型
        output_file: 输出文件路径
        
    Returns:
        item_id到embedding的字典
    """
    unwrapped_model = get_unwrapped_module(model)
    
    # 获取item feature table name
    if not hasattr(unwrapped_model, "get_item_feature_table_name"):
        print("Error: Model does not have get_item_feature_table_name method")
        return {}
    
    table_name = unwrapped_model.get_item_feature_table_name()
    print(f"Exporting all embeddings from table: {table_name}")
    
    # 获取embedding collection
    if not hasattr(unwrapped_model, "_embedding_collection"):
        print("Error: Model does not have _embedding_collection attribute")
        return {}
    
    embedding_collection = unwrapped_model._embedding_collection
    
    try:
        if hasattr(embedding_collection, "export_local_embedding"):
            embedding_export = embedding_collection.export_local_embedding(table_name)
            
            if isinstance(embedding_export, tuple) and len(embedding_export) == 2:
                keys, values = embedding_export
                
                # 转换为numpy数组
                if isinstance(keys, torch.Tensor):
                    keys = keys.cpu().numpy()
                if isinstance(values, torch.Tensor):
                    values = values.cpu().numpy()
                
                keys = np.asarray(keys)
                values = np.asarray(values)
                
                print(f"Exported {len(keys)} embeddings from table {table_name}")
                print(f"Key range: {keys.min()} to {keys.max()}")
                
                # 创建字典
                embeddings_dict = {int(keys[i]): values[i] for i in range(len(keys))}
                
                # 保存到文件
                print(f"Saving all embeddings to {output_file}")
                # 保存为字典格式
                np.savez(output_file, **{f"item_{k}": v for k, v in embeddings_dict.items()})
                # 同时保存keys和values数组（更方便加载）
                arrays_file = output_file.replace(".npz", "_arrays.npz") if output_file.endswith(".npz") else f"{output_file}_arrays.npz"
                np.savez(arrays_file, keys=keys, values=values, table_name=table_name)
                print(f"Embeddings saved to {output_file} and {arrays_file}")
                
                return embeddings_dict
            else:
                print(f"Error: Unexpected export format")
                return {}
        else:
            print("Error: embedding_collection does not have export_local_embedding method")
            return {}
    except Exception as e:
        print(f"Error exporting embeddings: {e}")
        import traceback
        traceback.print_exc()
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="从checkpoint中导出所有item_id的embedding"
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
        "--output_file",
        type=str,
        default="item_embeddings.npz",
        help="输出文件路径（默认: item_embeddings.npz）",
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
    
    try:
        # 加载模型
        print("=" * 60)
        print("Loading model from checkpoint...")
        print("=" * 60)
        model = load_model_from_checkpoint(
            args.checkpoint_dir,
            args.gin_config_file,
        )
        
        # 导出所有item_id的embedding
        print("\n" + "=" * 60)
        print("Exporting all item embeddings...")
        print("=" * 60)
        all_embeddings = export_all_item_embeddings(
            model,
            args.output_file,
        )
        
        print("\n" + "=" * 60)
        print(f"Successfully exported {len(all_embeddings)} item embeddings")
        if len(all_embeddings) > 0:
            sample_keys = list(all_embeddings.keys())[:5]
            print(f"Sample item_ids: {sample_keys}")
            sample_item_id = sample_keys[0]
            sample_emb = all_embeddings[sample_item_id]
            print(f"Sample embedding (item_id={sample_item_id}):")
            print(f"  Shape: {sample_emb.shape}")
            print(f"  Dtype: {sample_emb.dtype}")
            print(f"  Norm: {np.linalg.norm(sample_emb):.6f}")
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
