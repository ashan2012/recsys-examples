# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import warnings

# Ignore all FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=SyntaxWarning)
import argparse
from typing import List, Union

import numpy as np
import commons.utils.initialize as init
import gin
import torch  # pylint: disable-unused-import
from configs import RetrievalConfig
from distributed.sharding import make_optimizer_and_shard
from model import get_retrieval_model
from modules.metrics import RetrievalTaskMetricWithSampling
from pipeline.train_pipeline import (
    JaggedMegatronPrefetchTrainPipelineSparseDist,
    JaggedMegatronTrainNonePipeline,
    JaggedMegatronTrainPipelineSparseDist,
)
from trainer.training import maybe_load_ckpts, train_with_pipeline
from trainer.utils import (
    create_dynamic_optitons_dict,
    create_embedding_configs,
    create_hstu_config,
    create_optimizer_params,
    get_data_loader,
    get_dataset_and_embedding_args,
    get_embedding_vector_storage_multiplier,
)
from utils import (  # from hstu.utils
    BenchmarkDatasetArgs,
    DatasetArgs,
    EmbeddingArgs,
    NetworkArgs,
    OptimizerArgs,
    RetrievalArgs,
    TensorModelParallelArgs,
    TrainerArgs,
)
from commons.checkpoint import get_unwrapped_module

def create_retrieval_config(
    dataset_args: Union[DatasetArgs, BenchmarkDatasetArgs],
    network_args: NetworkArgs,
    embedding_args: List[EmbeddingArgs],
) -> RetrievalConfig:
    retrieval_args = RetrievalArgs()

    return RetrievalConfig(
        embedding_configs=create_embedding_configs(
            dataset_args, network_args, embedding_args
        ),
        temperature=retrieval_args.temperature,
        l2_norm_eps=retrieval_args.l2_norm_eps,
        num_negatives=retrieval_args.num_negatives,
        eval_metrics=retrieval_args.eval_metrics,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Distributed GR Arguments", allow_abbrev=False
    )
    parser.add_argument("--gin-config-file", type=str, required=True, help="Path to gin config file")
    parser.add_argument("--output_file", type=str, default=None, help="Output file to save embeddings (optional)")
    parser.add_argument("--max_samples", type=int, default=None, help="Maximum number of samples to process (for testing)")
    args = parser.parse_args()
    gin.parse_config_file(args.gin_config_file)
    trainer_args = TrainerArgs()
    dataset_args, embedding_args = get_dataset_and_embedding_args()
    network_args = NetworkArgs()
    optimizer_args = OptimizerArgs()
    tp_args = TensorModelParallelArgs()

    init.initialize_distributed()
    init.initialize_model_parallel(
        tensor_model_parallel_size=tp_args.tensor_model_parallel_size
    )
    init.set_random_seed(trainer_args.seed)

    hstu_config = create_hstu_config(network_args, tp_args)
    task_config = create_retrieval_config(dataset_args, network_args, embedding_args)
    model = get_retrieval_model(hstu_config=hstu_config, task_config=task_config)

    dynamic_options_dict = create_dynamic_optitons_dict(
        embedding_args,
        network_args.hidden_size,
        training=True,
        embedding_dim_multiplier=get_embedding_vector_storage_multiplier(
            optimizer_args.optimizer_str
        ),
    )
    optimizer_param = create_optimizer_params(optimizer_args)
    model_train, dense_optimizer = make_optimizer_and_shard(
        model,
        config=hstu_config,
        sparse_optimizer_param=optimizer_param,
        dense_optimizer_param=optimizer_param,
        dynamicemb_options_dict=dynamic_options_dict,
        pipeline_type=trainer_args.pipeline_type,
    )
    train_dataloader, test_dataloader = get_data_loader(
        "retrieval", dataset_args, trainer_args, 0
    )
    maybe_load_ckpts(trainer_args.ckpt_load_dir, model, dense_optimizer)

    model_train.eval()
    
    print("=" * 80)
    print("Starting inference and extracting embeddings for each sample")
    if args.output_file:
        print(f"Results will be saved to: {args.output_file}")
    if args.max_samples:
        print(f"Maximum samples to process: {args.max_samples}")
    print("=" * 80)
    
    # 用于保存所有样本的数据
    all_samples_data = []
    sample_count = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_dataloader):
            # 将 batch 移到 GPU
            batch_cuda = batch.to(torch.device("cuda", torch.cuda.current_device()))
            
            # 获取 embedding
            embedding, _, _, _ = get_unwrapped_module(model_train).get_logit_and_labels(batch_cuda)
            
            # 获取 batch 中的 features
            features_dict = batch.features.to_dict()
            
            # 提取 user_id 和 item_id（JaggedTensor 格式）
            # JaggedTensor 包含 values, lengths, offsets
            user_id_data = None
            item_id_data = None
            
            if 'user_id' in features_dict:
                user_jt = features_dict['user_id']
                user_id_data = {
                    'values': user_jt.values().cpu().numpy() if hasattr(user_jt.values(), 'cpu') else user_jt.values(),
                    'lengths': user_jt.lengths().cpu().numpy() if hasattr(user_jt.lengths(), 'cpu') else user_jt.lengths(),
                }
            
            if 'item_id' in features_dict:
                item_jt = features_dict['item_id']
                item_id_data = {
                    'values': item_jt.values().cpu().numpy() if hasattr(item_jt.values(), 'cpu') else item_jt.values(),
                    'lengths': item_jt.lengths().cpu().numpy() if hasattr(item_jt.lengths(), 'cpu') else item_jt.lengths(),
                }
            
            # 将 embedding 转换为 numpy（注意 BFloat16 需要先转 float32）
            embedding_numpy = embedding.cpu().float().numpy()
            
            # 获取 batch size（使用 lengths 的实际长度）
            # JaggedTensor 的 lengths 表示每个样本的序列长度
            if user_id_data is not None:
                batch_size = len(user_id_data['lengths'])
            elif item_id_data is not None:
                batch_size = len(item_id_data['lengths'])
            else:
                batch_size = embedding_numpy.shape[0]
            
            # 确保 batch_size 不超过 embedding 的数量
            batch_size = min(batch_size, embedding_numpy.shape[0])
            
            print(f"\n{'='*80}")
            print(f"Batch {batch_idx + 1} - Processing {batch_size} samples")
            print(f"  Embedding shape: {embedding_numpy.shape}")
            if user_id_data is not None:
                print(f"  User lengths shape: {user_id_data['lengths'].shape}")
            if item_id_data is not None:
                print(f"  Item lengths shape: {item_id_data['lengths'].shape}")
            print(f"{'='*80}")
            
            # 逐条打印每个样本的信息
            user_offset = 0
            item_offset = 0
            
            for i in range(batch_size):
                sample_count += 1
                
                print(f"\n--- Sample {sample_count} (Batch {batch_idx + 1}, Index {i}) ---")
                
                # 初始化变量（防止未定义错误）
                user_values = None
                item_values = None
                sample_embedding = None
                
                # 打印 user_id（根据 lengths 提取该样本的 user_id）
                if user_id_data is not None and i < len(user_id_data['lengths']):
                    user_length = int(user_id_data['lengths'][i])
                    user_values = user_id_data['values'][user_offset:user_offset + user_length]
                    user_offset += user_length
                    
                    if len(user_values) == 1:
                        print(f"User ID: {user_values[0]}")
                    elif len(user_values) > 0:
                        print(f"User IDs (sequence): {user_values[:10]}{'...' if len(user_values) > 10 else ''} (total: {len(user_values)})")
                    else:
                        print(f"User ID: Empty sequence")
                else:
                    print(f"User ID: Not found in features (i={i}, lengths={len(user_id_data['lengths']) if user_id_data else 'N/A'})")
                
                # 打印 item_id（根据 lengths 提取该样本的 item_id 序列）
                if item_id_data is not None and i < len(item_id_data['lengths']):
                    item_length = int(item_id_data['lengths'][i])
                    item_values = item_id_data['values'][item_offset:item_offset + item_length]
                    item_offset += item_length
                    
                    if len(item_values) == 0:
                        print(f"Item IDs: Empty sequence")
                    elif len(item_values) <= 5:
                        print(f"Item IDs (sequence): {item_values.tolist()}")
                    else:
                        print(f"Item IDs (sequence): {item_values[:5].tolist()} ... {item_values[-2:].tolist()} (total: {len(item_values)})")
                else:
                    print(f"Item IDs: Not found in features (i={i}, lengths={len(item_id_data['lengths']) if item_id_data else 'N/A'})")
                
                # 打印 embedding（检查索引是否有效）
                if i < len(embedding_numpy):
                    sample_embedding = embedding_numpy[i]
                    print(f"Embedding shape: {sample_embedding.shape}")
                    print(f"Embedding dtype: {embedding.dtype}")
                    print(f"Embedding (first 10 dims): {sample_embedding[:10]}")
                    print(f"Embedding norm: {np.linalg.norm(sample_embedding):.6f}")
                else:
                    print(f"Embedding: Index {i} out of bounds (total: {len(embedding_numpy)})")
                
                # 保存数据到列表（用于后续写入文件）
                if args.output_file and sample_embedding is not None:
                    sample_data = {
                        'sample_id': sample_count,
                        'batch_idx': batch_idx + 1,
                        'batch_inner_idx': i,
                        'user_ids': user_values.tolist() if user_values is not None and hasattr(user_values, 'tolist') else None,
                        'item_ids': item_values.tolist() if item_values is not None and hasattr(item_values, 'tolist') else None,
                        'embedding': sample_embedding.tolist(),
                        'embedding_norm': float(np.linalg.norm(sample_embedding)),
                    }
                    all_samples_data.append(sample_data)
                
                # 检查是否达到最大样本数
                if args.max_samples and sample_count >= args.max_samples:
                    print(f"\n{'='*80}")
                    print(f"Reached maximum samples limit: {args.max_samples}")
                    print(f"{'='*80}")
                    break
            
            # 检查是否达到最大样本数（外层循环）
            if args.max_samples and sample_count >= args.max_samples:
                break
            
            # 默认只处理前3个 batch 作为示例（可以通过 --max_samples 参数覆盖）
            if args.max_samples is None and batch_idx >= 2:
                print(f"\n{'='*80}")
                print(f"Processed {sample_count} samples from {batch_idx + 1} batches (demo mode)")
                print(f"Use --max_samples to process more, or remove the limit in code")
                print(f"{'='*80}")
                break
    
    print(f"\n{'='*80}")
    print(f"Inference complete. Total samples processed: {sample_count}")
    print(f"{'='*80}")
    
    # 保存结果到文件
    if args.output_file and len(all_samples_data) > 0:
        import json
        
        print(f"\nSaving results to {args.output_file}...")
        
        # 保存为 JSON 格式
        if args.output_file.endswith('.json'):
            with open(args.output_file, 'w') as f:
                json.dump(all_samples_data, f, indent=2)
            print(f"Saved {len(all_samples_data)} samples to {args.output_file} (JSON format)")
        
        # 保存为 NPZ 格式（更节省空间）
        elif args.output_file.endswith('.npz'):
            # 提取所有数据
            embeddings = np.array([s['embedding'] for s in all_samples_data])
            sample_ids = np.array([s['sample_id'] for s in all_samples_data])
            
            # 保存其他信息为 JSON 字符串
            metadata = [{
                'sample_id': s['sample_id'],
                'user_ids': s['user_ids'],
                'item_ids': s['item_ids'],
                'embedding_norm': s['embedding_norm'],
            } for s in all_samples_data]
            
            np.savez(
                args.output_file,
                embeddings=embeddings,
                sample_ids=sample_ids,
                metadata=json.dumps(metadata),
            )
            print(f"Saved {len(all_samples_data)} samples to {args.output_file} (NPZ format)")
            print(f"  - embeddings shape: {embeddings.shape}")
            print(f"  - Use np.load('{args.output_file}') to load")
        
        # 默认保存为 JSON
        else:
            output_file_json = args.output_file if args.output_file.endswith('.json') else args.output_file + '.json'
            with open(output_file_json, 'w') as f:
                json.dump(all_samples_data, f, indent=2)
            print(f"Saved {len(all_samples_data)} samples to {output_file_json} (JSON format)")
    
    init.destroy_global_state()


if __name__ == "__main__":
    main()