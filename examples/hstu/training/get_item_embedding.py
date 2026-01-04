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
from pipelines import make_optimizer_and_shard
from trainer.utils import (
    create_dynamic_optitons_dict,
    create_embedding_configs,
    create_hstu_config,
    create_optimizer_params,
    get_dataset_and_embedding_args,
    get_embedding_vector_storage_multiplier,
)
from utils import (
    NetworkArgs,
    OptimizerArgs,
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
    optimizer_args = OptimizerArgs()
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
    
    # 设置dynamic embeddings (这一步很关键！)
    print("Setting up dynamic embeddings...")
    dynamic_options_dict = create_dynamic_optitons_dict(
        embedding_args,
        network_args.hidden_size,
        training=False,  # 推理模式
        embedding_dim_multiplier=get_embedding_vector_storage_multiplier(
            optimizer_args.optimizer_str
        ),
    )
    optimizer_param = create_optimizer_params(optimizer_args)
    model_sharded, _ = make_optimizer_and_shard(
        model,
        config=hstu_config,
        sparse_optimizer_param=optimizer_param,
        dense_optimizer_param=optimizer_param,
        dynamicemb_options_dict=dynamic_options_dict,
        pipeline_type="native",  # 使用简单的pipeline
    )
    print("Dynamic embeddings setup complete")
    
    # 加载checkpoint
    unwrapped_model = get_unwrapped_module(model_sharded)
    
    # 1. 加载dynamic embedding表
    from dynamicemb.dump_load import DynamicEmbLoad as dynamic_emb_load
    from dynamicemb.dump_load import get_dynamic_emb_module as check_dynamic_modules
    
    save_dir = os.path.join(checkpoint_dir, "dynamicemb_module")
    print(f"Checking dynamic embedding directory: {save_dir}")
    print(f"Directory exists: {os.path.exists(save_dir)}")
    
    if os.path.exists(save_dir):
        # 列出目录内容
        print(f"Contents of {save_dir}:")
        for item in os.listdir(save_dir):
            item_path = os.path.join(save_dir, item)
            if os.path.isdir(item_path):
                print(f"  [DIR] {item}/")
                for subitem in os.listdir(item_path):
                    print(f"    - {subitem}")
            else:
                print(f"  [FILE] {item}")
        
        print("\nBefore loading - checking for dynamic embedding modules...")
        modules_before = check_dynamic_modules(unwrapped_model)
        print(f"Found {len(modules_before)} dynamic embedding modules before loading")
        
        print("\nLoading dynamic embedding tables...")
        dynamic_emb_load(save_dir, unwrapped_model, optim=False)
        print("Dynamic embedding tables loaded")
        
        print("\nAfter loading - checking for dynamic embedding modules...")
        modules_after = check_dynamic_modules(unwrapped_model)
        print(f"Found {len(modules_after)} dynamic embedding modules after loading")
        for i, m in enumerate(modules_after):
            print(f"  Module {i}: {type(m)}, tables: {getattr(m, 'table_names', 'N/A')}")
    else:
        print(f"Warning: Dynamic embedding directory {save_dir} not found")
    
    # 2. 跳过dense模型参数的加载（我们只需要item_id的embedding，不需要其他参数）
    # 注意：如果需要完整的模型推理，可以取消下面的注释
    # 但为了只导出item_id embedding，我们跳过这一步以避免形状不匹配错误
    print("Skipping dense model parameters loading (only need item_id embeddings)")
    
    # 修复所有meta tensor（因为跳过了dense参数加载，这些参数仍然是meta tensor）
    def fix_meta_tensors(module, prefix=""):
        """将meta tensor初始化为零tensor"""
        fixed_count = 0
        for name, param in list(module.named_parameters(recurse=False)):
            if param.is_meta:
                full_name = f"{prefix}.{name}" if prefix else name
                try:
                    # 创建相同形状的零tensor
                    new_param = torch.nn.Parameter(torch.zeros(param.shape, dtype=param.dtype, device="cpu"))
                    # 直接替换参数
                    if hasattr(module, name):
                        delattr(module, name)
                    setattr(module, name, new_param)
                    fixed_count += 1
                except Exception as e:
                    print(f"  Warning: Failed to fix {full_name}: {e}")
        for child_name, child in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            fixed_count += fix_meta_tensors(child, child_prefix)
        return fixed_count
    
    # 检查并修复meta tensor
    meta_params = []
    for name, param in unwrapped_model.named_parameters():
        if param.is_meta:
            meta_params.append(name)
    
    if meta_params:
        print(f"Found {len(meta_params)} meta tensor parameters, initializing them...")
        fixed_count = fix_meta_tensors(unwrapped_model)
        print(f"Fixed {fixed_count} meta tensor parameters")
    
    # 注意：导出embedding不需要模型在CUDA上，export_local_embedding会将数据移到CPU
    # 所以我们可以保持在CPU上，避免meta tensor问题
    print("Model ready (staying on CPU for embedding export)")
    
    print("Checkpoint loaded successfully")
    return model_sharded


def export_all_item_embeddings(
    model: torch.nn.Module,
    output_file: str,
    table_name: str = "item_id",
) -> Dict[int, np.ndarray]:
    """
    导出所有item_id的embedding
    
    Args:
        model: 加载的模型
        output_file: 输出文件路径
        table_name: embedding表名称，默认为"item_id"
        
    Returns:
        item_id到embedding的字典
    """
    unwrapped_model = get_unwrapped_module(model)
    print(f"Exporting all embeddings from table: {table_name}")
    
    # 获取embedding collection
    if not hasattr(unwrapped_model, "_embedding_collection"):
        print("Error: Model does not have _embedding_collection attribute")
        return {}
    
    embedding_collection = unwrapped_model._embedding_collection
    
    # 直接从dynamic embedding module导出
    try:
        from dynamicemb.dump_load import get_dynamic_emb_module
        
        # 打印embedding_collection的结构
        print(f"Embedding collection type: {type(embedding_collection)}")
        print(f"Embedding collection attributes: {dir(embedding_collection)}")
        
        # 检查model_parallel和data_parallel collections
        has_mp = hasattr(embedding_collection, "_model_parallel_embedding_collection")
        has_dp = hasattr(embedding_collection, "_data_parallel_embedding_collection")
        print(f"Has model_parallel_embedding_collection: {has_mp}")
        print(f"Has data_parallel_embedding_collection: {has_dp}")
        
        if not has_mp:
            print("Error: embedding_collection does not have _model_parallel_embedding_collection")
            return {}
        
        model_parallel_collection = embedding_collection._model_parallel_embedding_collection
        if model_parallel_collection is None:
            print("Error: model_parallel_embedding_collection is None")
            return {}
        
        print(f"Model parallel collection type: {type(model_parallel_collection)}")
        
        # 获取dynamic embedding modules
        dynamicemb_modules = get_dynamic_emb_module(model_parallel_collection)
        
        print(f"Found {len(dynamicemb_modules)} dynamic embedding modules")
        
        if len(dynamicemb_modules) == 0:
            print("Error: No dynamic embedding modules found")
            print("Trying to inspect model_parallel_collection structure...")
            
            # 尝试打印更多信息
            if hasattr(model_parallel_collection, '_lookups'):
                print(f"  Has _lookups: {type(model_parallel_collection._lookups)}")
            if hasattr(model_parallel_collection, '_embedding_configs'):
                print(f"  Has _embedding_configs: {model_parallel_collection._embedding_configs}")
            if hasattr(model_parallel_collection, 'embedding_configs'):
                print(f"  Has embedding_configs(): {model_parallel_collection.embedding_configs()}")
            
            # 尝试查找所有子模块
            print("All named modules:")
            for name, module in model_parallel_collection.named_modules():
                print(f"  {name}: {type(module)}")
            
            return {}
        
        print(f"Found {len(dynamicemb_modules)} dynamic embedding modules")
        
        # 查找指定的table
        for m in dynamicemb_modules:
            print(f"Checking module with tables: {m.table_names}")
            if table_name in set(m.table_names):
                print(f"Found table {table_name}, exporting keys and values...")
                keys_tensor, values_tensor = m.export_keys_values(
                    table_name, device=torch.device("cpu")
                )
                
                # 转换为numpy
                keys = keys_tensor.numpy()
                values = values_tensor.numpy()
                
                print(f"Exported {len(keys)} embeddings from table {table_name}")
                print(f"Keys shape: {keys.shape}, Values shape: {values.shape}")
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
        
        print(f"Error: Table {table_name} not found in any dynamic embedding module")
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
    parser.add_argument(
        "--table_name",
        type=str,
        default="item_id",
        help="embedding表名称（默认: item_id）",
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
            args.table_name,
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
