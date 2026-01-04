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
from commons.checkpoint import get_unwrapped_module
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
    
    # 初始化分布式环境
    if not dist.is_initialized():
        init.initialize_distributed()
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
    
    # 设置CUDA设备（load函数中的barrier需要）
    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(0)
    
    # 创建模型（此时模型参数可能是meta tensor，这是正常的）
    model = get_retrieval_model(hstu_config=hstu_config, task_config=task_config)
    
    # 加载checkpoint
    # 注意：dynamic embedding表通过dynamic_emb_load加载
    # dense embedding表通过load_state_dict加载，但需要使用strict=False处理表名不匹配
    print(f"Loading checkpoint from {checkpoint_dir}")
    
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
            # 处理表名不匹配：interaction vs interaction_weights
            model_state_dict = state_dict["model_state_dict"]
            new_state_dict = {}
            
            # 获取dynamic embedding表名列表
            dynamic_table_names = set()
            if hasattr(unwrapped_model, "_embedding_collection"):
                embedding_collection = unwrapped_model._embedding_collection
                if hasattr(embedding_collection, "_dynamic_embedding_collection"):
                    dynamic_emb_collection = embedding_collection._dynamic_embedding_collection
                    if hasattr(dynamic_emb_collection, "_embedding_tables"):
                        dynamic_tables = dynamic_emb_collection._embedding_tables
                        if hasattr(dynamic_tables, "table_names"):
                            dynamic_table_names = set(dynamic_tables.table_names)
                            print(f"Dynamic embedding table names: {dynamic_table_names}")
            
            # 过滤掉dynamic embedding表的参数，只保留dense模型参数
            for key, value in model_state_dict.items():
                # 跳过dynamic embedding表的参数
                # 检查是否包含dynamic embedding表名
                is_dynamic_emb = False
                for table_name in dynamic_table_names:
                    # 匹配各种可能的key格式
                    if (f".{table_name}." in key or 
                        f".{table_name}_" in key or
                        f"embeddings.{table_name}." in key or
                        f"embeddings.{table_name}_" in key):
                        is_dynamic_emb = True
                        break
                
                # 额外检查：如果key包含_model_parallel_embedding_collection.embeddings
                # 且形状是[1,1]，很可能是dynamic embedding的占位符
                # 或者key明确包含item_id或user_id
                if "_model_parallel_embedding_collection.embeddings" in key:
                    if isinstance(value, torch.Tensor) and value.shape == torch.Size([1, 1]):
                        is_dynamic_emb = True
                    # 明确检查item_id和user_id
                    if ".item_id." in key or ".user_id." in key:
                        is_dynamic_emb = True
                
                if is_dynamic_emb:
                    print(f"Skipping dynamic embedding parameter: {key} (shape: {value.shape if hasattr(value, 'shape') else 'N/A'})")
                    continue
                
                # 替换表名：interaction_weights -> interaction
                # 同时处理可能的key格式差异
                new_key = key.replace("interaction_weights", "interaction")
                new_key = new_key.replace("action_weights", "interaction")
                
                # 如果key是 "interaction" 而不是 "interaction.weight"，需要添加 .weight
                # 或者如果key是 "interaction.weight" 但模型期望 "interaction"，需要处理
                # 先尝试原key
                new_state_dict[new_key] = value
                
                # 如果key没有.weight后缀，也尝试添加.weight后缀的版本
                if not new_key.endswith(".weight") and "embeddings.interaction" in new_key:
                    weight_key = new_key + ".weight"
                    # 同时添加带.weight和不带.weight的版本，让load_state_dict选择
                    new_state_dict[weight_key] = value
            
            print(f"Loading {len(new_state_dict)} dense model parameters...")
            
            # 打印一些key示例用于调试
            if len(new_state_dict) > 0:
                sample_keys = list(new_state_dict.keys())[:3]
                print(f"Sample keys to load: {sample_keys}")
            
            # 使用strict=False加载
            missing_keys, unexpected_keys = unwrapped_model.load_state_dict(
                new_state_dict, strict=False
            )
            
            # 处理missing keys：尝试修复表名不匹配
            if missing_keys:
                filtered_missing = [k for k in missing_keys if not any(
                    f".{tn}." in k or f".{tn}_" in k for tn in dynamic_table_names
                )]
                if filtered_missing:
                    print(f"Warning: Missing keys: {filtered_missing}")
                    # 尝试从unexpected keys或new_state_dict中找到匹配的
                    for missing_key in filtered_missing:
                        # 尝试从unexpected keys中找到匹配的
                        for unexpected_key in unexpected_keys:
                            if "interaction" in missing_key and "interaction" in unexpected_key:
                                print(f"  Attempting to map: {unexpected_key} -> {missing_key}")
                                # 尝试手动加载
                                try:
                                    if unexpected_key in new_state_dict:
                                        value = new_state_dict[unexpected_key]
                                        # 使用递归方式设置参数
                                        keys_parts = missing_key.split(".")
                                        obj = unwrapped_model
                                        for part in keys_parts[:-1]:
                                            if hasattr(obj, part):
                                                obj = getattr(obj, part)
                                            else:
                                                break
                                        else:
                                            param_name = keys_parts[-1]
                                            if hasattr(obj, param_name):
                                                param = getattr(obj, param_name)
                                                if param.shape == value.shape:
                                                    with torch.no_grad():
                                                        param.data.copy_(value)
                                                    print(f"  ✓ Manually loaded {missing_key}")
                                except Exception as e:
                                    print(f"  ✗ Failed to manually load {missing_key}: {e}")
            
            if unexpected_keys:
                print(f"Warning: Unexpected keys: {unexpected_keys}")
                # 检查是否有interaction相关的key需要处理
                for unexpected_key in unexpected_keys:
                    if "interaction" in unexpected_key and not unexpected_key.endswith(".weight"):
                        # 尝试添加.weight后缀
                        weight_key = unexpected_key + ".weight"
                        if weight_key in new_state_dict:
                            print(f"  Found matching key: {unexpected_key} -> {weight_key}")
            
            print("Dense model parameters loaded")
            
            # 检查是否还有meta tensor
            def check_meta_tensors(module, prefix=""):
                meta_params = []
                for name, param in module.named_parameters(recurse=False):
                    if param.is_meta:
                        meta_params.append(f"{prefix}.{name}" if prefix else name)
                for name, child in module.named_children():
                    child_meta = check_meta_tensors(child, f"{prefix}.{name}" if prefix else name)
                    meta_params.extend(child_meta)
                return meta_params
            
            meta_params = check_meta_tensors(unwrapped_model)
            if meta_params:
                print(f"Warning: Found {len(meta_params)} meta tensor parameters:")
                for mp in meta_params[:10]:
                    print(f"  {mp}")
                print("These parameters may cause errors when moving to CUDA")
        else:
            print(f"Warning: No model_state_dict in checkpoint file")
    else:
        print(f"Warning: Checkpoint file {save_path} not found")
    
    print("Checkpoint loaded successfully")
    
    # 检查并处理meta tensor
    unwrapped_model = get_unwrapped_module(model)
    
    def fix_meta_tensors(module, prefix=""):
        """将meta tensor初始化为零tensor"""
        fixed_count = 0
        for name, param in list(module.named_parameters(recurse=False)):
            if param.is_meta:
                full_name = f"{prefix}.{name}" if prefix else name
                print(f"Initializing meta tensor: {full_name}, shape: {param.shape}")
                try:
                    # 创建相同形状的零tensor
                    new_param = torch.nn.Parameter(torch.zeros(param.shape, dtype=param.dtype, device="cpu"))
                    # 直接替换参数
                    if hasattr(module, name):
                        delattr(module, name)
                    setattr(module, name, new_param)
                    fixed_count += 1
                except Exception as e:
                    print(f"  Failed to fix {full_name}: {e}")
        for child_name, child in module.named_children():
            child_prefix = f"{prefix}.{child_name}" if prefix else child_name
            fixed_count += fix_meta_tensors(child, child_prefix)
        return fixed_count
    
    # 检查是否有meta tensor
    meta_params = []
    for name, param in unwrapped_model.named_parameters():
        if param.is_meta:
            meta_params.append(name)
    
    if meta_params:
        print(f"Found {len(meta_params)} meta tensor parameters, initializing them...")
        fixed_count = fix_meta_tensors(unwrapped_model)
        print(f"Fixed {fixed_count} meta tensor parameters")
        
        # 再次检查是否还有meta tensor
        remaining_meta = []
        for name, param in unwrapped_model.named_parameters():
            if param.is_meta:
                remaining_meta.append(name)
        if remaining_meta:
            print(f"Warning: Still have {len(remaining_meta)} meta tensors: {remaining_meta[:5]}")
        else:
            print("All meta tensors have been fixed")
    
    # 加载完checkpoint后，将模型移到指定设备
    # 此时所有参数都应该是实际tensor，不再是meta tensor
    if device == "cuda" and torch.cuda.is_available():
        try:
            model = model.cuda()
        except NotImplementedError as e:
            if "meta tensor" in str(e).lower():
                print("Error: Still have meta tensors after loading checkpoint")
                print("This usually means some parameters were not loaded correctly")
                print("Trying to initialize remaining meta tensors...")
                # 再次尝试修复
                fix_meta_tensors(unwrapped_model)
                model = model.cuda()
            else:
                raise
    
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


def check_loaded_embeddings(model: torch.nn.Module) -> Dict[str, any]:
    """
    检查已加载的embedding表信息
    
    Args:
        model: 加载的模型
        
    Returns:
        包含embedding表信息的字典
    """
    unwrapped_model = get_unwrapped_module(model)
    info = {}
    
    if not hasattr(unwrapped_model, "_embedding_collection"):
        print("Model does not have _embedding_collection attribute")
        return info
    
    embedding_collection = unwrapped_model._embedding_collection
    
    # 检查dynamic embedding表
    if hasattr(embedding_collection, "_dynamic_embedding_collection"):
        dynamic_emb_collection = embedding_collection._dynamic_embedding_collection
        if hasattr(dynamic_emb_collection, "_embedding_tables"):
            dynamic_tables = dynamic_emb_collection._embedding_tables
            if hasattr(dynamic_tables, "table_names"):
                info["dynamic_tables"] = list(dynamic_tables.table_names)
                print(f"Dynamic embedding tables: {info['dynamic_tables']}")
                
                # 尝试导出每个表的信息
                for table_name in dynamic_tables.table_names:
                    try:
                        if hasattr(embedding_collection, "export_local_embedding"):
                            keys, values = embedding_collection.export_local_embedding(table_name)
                            if isinstance(keys, torch.Tensor):
                                keys = keys.cpu().numpy()
                            if isinstance(values, torch.Tensor):
                                values = values.cpu().numpy()
                            info[f"{table_name}_keys"] = keys
                            info[f"{table_name}_values"] = values
                            info[f"{table_name}_count"] = len(keys)
                            print(f"  {table_name}: {len(keys)} embeddings, key range: {keys.min()} to {keys.max()}")
                    except Exception as e:
                        print(f"  {table_name}: Failed to export - {e}")
    
    # 检查dense embedding表
    if hasattr(embedding_collection, "_data_parallel_embedding_collection"):
        print("Data parallel embedding collection found")
    
    return info


def export_all_item_embeddings(
    model: torch.nn.Module,
    table_name: Optional[str] = None,
    output_file: Optional[str] = None,
) -> Dict[int, np.ndarray]:
    """
    导出所有item_id的embedding
    
    Args:
        model: 加载的模型
        table_name: embedding表名称，如果为None则自动查找
        output_file: 输出文件路径（可选）
        
    Returns:
        item_id到embedding的字典
    """
    unwrapped_model = get_unwrapped_module(model)
    
    # 获取item feature table name
    if hasattr(unwrapped_model, "get_item_feature_table_name"):
        if table_name is None:
            table_name = unwrapped_model.get_item_feature_table_name()
        print(f"Exporting all embeddings from table: {table_name}")
    
    # 获取embedding collection
    if not hasattr(unwrapped_model, "_embedding_collection"):
        print("Model does not have _embedding_collection attribute")
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
                
                # 创建字典
                embeddings_dict = {int(keys[i]): values[i] for i in range(len(keys))}
                
                # 保存到文件
                if output_file:
                    print(f"Saving all embeddings to {output_file}")
                    np.savez(output_file, **{f"item_{k}": v for k, v in embeddings_dict.items()})
                    # 同时保存keys和values数组
                    np.savez(
                        output_file.replace(".npz", "_arrays.npz") if output_file.endswith(".npz") else f"{output_file}_arrays.npz",
                        keys=keys,
                        values=values,
                        table_name=table_name
                    )
                    print("Embeddings saved successfully")
                
                return embeddings_dict
            else:
                print(f"Unexpected export format")
                return {}
        else:
            print("embedding_collection does not have export_local_embedding method")
            return {}
    except Exception as e:
        print(f"Error exporting all embeddings: {e}")
        import traceback
        traceback.print_exc()
        return {}


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
    parser.add_argument(
        "--check_embeddings",
        action="store_true",
        help="检查已加载的embedding表信息",
    )
    parser.add_argument(
        "--export_all",
        action="store_true",
        help="导出所有item_id的embedding",
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
    
    # 解析item_ids（如果不需要检查或导出所有）
    item_ids = []
    if not args.check_embeddings and not args.export_all:
        if args.item_id is not None:
            item_ids = [args.item_id]
        elif args.item_ids is not None:
            item_ids = [int(x.strip()) for x in args.item_ids.split(",")]
        else:
            print("Error: Please provide either --item_id, --item_ids, --check_embeddings, or --export_all")
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
        
        # 检查已加载的embedding表
        if args.check_embeddings:
            print("\n" + "=" * 60)
            print("Checking loaded embeddings...")
            print("=" * 60)
            emb_info = check_loaded_embeddings(model)
            print("\nEmbedding tables information:")
            for key, value in emb_info.items():
                if key.endswith("_count"):
                    print(f"  {key}: {value}")
            print("=" * 60)
        
        # 导出所有item_id的embedding
        elif args.export_all:
            print("\n" + "=" * 60)
            print("Exporting all item embeddings...")
            print("=" * 60)
            all_embeddings = export_all_item_embeddings(
                model,
                table_name=args.table_name,
                output_file=args.output_file or "all_item_embeddings.npz",
            )
            print(f"\nExported {len(all_embeddings)} item embeddings")
            if len(all_embeddings) > 0:
                sample_keys = list(all_embeddings.keys())[:5]
                print(f"Sample item_ids: {sample_keys}")
            print("=" * 60)
        
        # 获取指定item_ids的embeddings
        else:
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

