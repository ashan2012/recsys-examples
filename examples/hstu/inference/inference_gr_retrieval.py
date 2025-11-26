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
import argparse
import enum
import math
import os
import sys
import time
import torch
from commons.utils.stringify import stringify_dict
from configs import (
    InferenceEmbeddingConfig,
    PositionEncodingConfig,
    RetrievalConfig,
    get_inference_hstu_config,
    get_kvcache_config,
)
from dataset import get_data_loader
from dataset.inference_dataset import InferenceDataset
from dataset.sequence_dataset import get_dataset
from preprocessor import get_common_preprocessors
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor
from utils import DatasetArgs, NetworkArgs, RetrievalArgs

sys.path.append("./model/")
from inference_retrieval_gr import InferenceRetrievalGR

class RunningMode(enum.Enum):
    EVAL = "eval"
    EMBEDDING = "embedding"

    def __str__(self):
        return self.value

def get_inference_dataset_and_embedding_configs(
    disable_contextual_features: bool = False,
):
    dataset_args = DatasetArgs()
    embedding_dim = NetworkArgs().hidden_size
    HASH_SIZE = 10_000_000
    if dataset_args.dataset_name == "gameid":
        # 根据gameid_retrieval.gin配置调整embedding配置
        embedding_configs = [
            InferenceEmbeddingConfig(
                feature_names=["user_id"],
                table_name="user_id",
                vocab_size=HASH_SIZE,
                dim=embedding_dim,
                use_dynamicemb=True,
            ),
            # 可以根据需要添加更多特征配置
        ]
        return (
            dataset_args,
            embedding_configs,
        )

    raise ValueError(f"dataset {dataset_args.dataset_name} is not supported")

def get_inference_hstu_model(
    emb_configs,
    max_batch_size,
    num_contextual_features,
    total_max_seqlen,
    checkpoint_dir,
):
    network_args = NetworkArgs()
    if network_args.dtype_str == "bfloat16":
        inference_dtype = torch.bfloat16
    else:
        raise ValueError(
            f"Inference data type {network_args.dtype_str} is not supported"
        )

    position_encoding_config = PositionEncodingConfig(
        num_position_buckets=8192,
        num_time_buckets=2048,
        use_time_encoding=False,
        static_max_seq_len=math.ceil(total_max_seqlen / 32) * 32,
    )

    hstu_config = get_inference_hstu_config(
        hidden_size=network_args.hidden_size,
        num_layers=network_args.num_layers,
        num_attention_heads=network_args.num_attention_heads,
        head_dim=network_args.kv_channels,
        dtype=inference_dtype,
        position_encoding_config=position_encoding_config,
        contextual_max_seqlen=num_contextual_features,
        scaling_seqlen=network_args.scaling_seqlen,
    )

    kvcache_args = {
        "blocks_in_primary_pool": 10240,
        "page_size": 32,
        "offload_chunksize": 1024,
        "max_batch_size": max_batch_size,
        "max_seq_len": math.ceil(total_max_seqlen / 32) * 32,
    }
    kv_cache_config = get_kvcache_config(**kvcache_args)

    retrieval_args = RetrievalArgs()
    task_config = RetrievalConfig(
        embedding_configs=emb_configs,
        temperature=retrieval_args.temperature,
        l2_norm_eps=retrieval_args.l2_norm_eps,
        num_negatives=retrieval_args.num_negatives,
        eval_metrics=retrieval_args.eval_metrics,
    )

    hstu_cudagraph_configs = {
        "batch_size": [1],
        "length_per_sequence": [128] + [i * 256 for i in range(1, 34)],
    }

    model = InferenceRetrievalGR(
        hstu_config=hstu_config,
        kvcache_config=kv_cache_config,
        task_config=task_config,
        use_cudagraph=True,
        cudagraph_configs=hstu_cudagraph_configs,
    )
    if hstu_config.bf16:
        model.bfloat16()
    elif hstu_config.fp16:
        model.half()
    model.load_checkpoint(checkpoint_dir)
    model.eval()

    return model

def run_retrieval_gr_evaluate(
    checkpoint_dir: str,
    disable_contextual_features: bool = False,
):
    dataset_args, emb_configs = get_inference_dataset_and_embedding_configs(
        disable_contextual_features
    )

    dataproc = get_common_preprocessors("")[dataset_args.dataset_name]
    num_contextual_features = (
        len(dataproc._contextual_feature_names)
        if not disable_contextual_features
        else 0
    )

    max_batch_size = 1
    total_max_seqlen = dataset_args.max_sequence_length * 2 + num_contextual_features
    print("total_max_seqlen", total_max_seqlen)

    def strip_padding_batch(batch, unpadded_batch_size):
        batch.batch_size = unpadded_batch_size
        kjt_dict = batch.features.to_dict()
        for k in kjt_dict:
            kjt_dict[k] = JaggedTensor.from_dense_lengths(
                kjt_dict[k].to_padded_dense()[: batch.batch_size],
                kjt_dict[k].lengths()[: batch.batch_size].long(),
            )
        batch.features = KeyedJaggedTensor.from_jt_dict(kjt_dict)
        if hasattr(batch, 'num_candidates') and batch.num_candidates is not None:
            batch.num_candidates = batch.num_candidates[: batch.batch_size]
        return batch

    with torch.inference_mode():
        model = get_inference_hstu_model(
            emb_configs,
            max_batch_size,
            num_contextual_features,
            total_max_seqlen,
            checkpoint_dir,
        )

        # 创建评估指标模块
        from modules.metrics import RetrievalTaskMetricWithSampling
        eval_module = RetrievalTaskMetricWithSampling(
            metric_types=model._task_config.eval_metrics,
            MAX_K=500
        )

        # 获取数据集
        _, eval_dataset = get_dataset(
            dataset_name=dataset_args.dataset_name,
            dataset_path=dataset_args.dataset_path,
            max_sequence_length=dataset_args.max_sequence_length,
            max_num_candidates=dataset_args.max_num_candidates,
            num_tasks=0,  # 召回任务不需要多任务
            batch_size=max_batch_size,
            rank=0,
            world_size=1,
            shuffle=False,
            random_seed=0,
            eval_batch_size=max_batch_size,
        )

        dataloader = get_data_loader(dataset=eval_dataset)
        dataloader_iter = iter(dataloader)

        while True:
            try:
                batch = next(dataloader_iter)
                batch = batch.to(device=torch.cuda.current_device())
                
                # 从批次中提取用户ID和序列起始位置
                d = batch.features.to_dict()
                user_ids = d["user_id"].values().cpu().int()
                seq_startpos = torch.zeros_like(user_ids)

                if user_ids.shape[0] != batch.batch_size:
                    batch = strip_padding_batch(batch, user_ids.shape[0])

                # 前向传播，获取用户嵌入和物品嵌入
                user_embeddings, item_embeddings = model.forward(batch, user_ids, seq_startpos)
                
                # 使用嵌入进行评估
                eval_module(user_embeddings, item_embeddings, batch.labels)
            except StopIteration:
                break

        eval_metric_dict = eval_module.compute()
        print(
            f"[eval]:\n    "
            + stringify_dict(eval_metric_dict, prefix="Metrics", sep="\n    ")
        )

def run_retrieval_gr_embedding(
    checkpoint_dir: str,
    output_dir: str,
    disable_contextual_features: bool = False,
):
    dataset_args, emb_configs = get_inference_dataset_and_embedding_configs(
        disable_contextual_features
    )

    dataproc = get_common_preprocessors("")[dataset_args.dataset_name]
    num_contextual_features = (
        len(dataproc._contextual_feature_names)
        if not disable_contextual_features
        else 0
    )

    max_batch_size = 1
    total_max_seqlen = dataset_args.max_sequence_length * 2 + num_contextual_features
    print("total_max_seqlen", total_max_seqlen)

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    def strip_padding_batch(batch, unpadded_batch_size):
        batch.batch_size = unpadded_batch_size
        kjt_dict = batch.features.to_dict()
        for k in kjt_dict:
            kjt_dict[k] = JaggedTensor.from_dense_lengths(
                kjt_dict[k].to_padded_dense()[: batch.batch_size],
                kjt_dict[k].lengths()[: batch.batch_size].long(),
            )
        batch.features = KeyedJaggedTensor.from_jt_dict(kjt_dict)
        if hasattr(batch, 'num_candidates') and batch.num_candidates is not None:
            batch.num_candidates = batch.num_candidates[: batch.batch_size]
        return batch

    with torch.inference_mode():
        model = get_inference_hstu_model(
            emb_configs,
            max_batch_size,
            num_contextual_features,
            total_max_seqlen,
            checkpoint_dir,
        )

        # 获取数据集
        _, eval_dataset = get_dataset(
            dataset_name=dataset_args.dataset_name,
            dataset_path=dataset_args.dataset_path,
            max_sequence_length=dataset_args.max_sequence_length,
            max_num_candidates=dataset_args.max_num_candidates,
            num_tasks=0,
            batch_size=max_batch_size,
            rank=0,
            world_size=1,
            shuffle=False,
            random_seed=0,
            eval_batch_size=max_batch_size,
        )

        dataloader = get_data_loader(dataset=eval_dataset)
        dataloader_iter = iter(dataloader)

        user_embeddings_dict = {}
        item_embeddings_dict = {}
        processed_count = 0

        start_time = time.time()
        while True:
            try:
                batch = next(dataloader_iter)
                batch = batch.to(device=torch.cuda.current_device())
                
                # 从批次中提取用户ID和序列起始位置
                d = batch.features.to_dict()
                user_ids = d["user_id"].values().cpu().int()
                seq_startpos = torch.zeros_like(user_ids)

                if user_ids.shape[0] != batch.batch_size:
                    batch = strip_padding_batch(batch, user_ids.shape[0])

                # 前向传播，获取用户嵌入和物品嵌入
                user_embeddings, item_embeddings = model.forward(batch, user_ids, seq_startpos)
                
                # 保存用户嵌入到字典
                for i, user_id in enumerate(user_ids):
                    user_embeddings_dict[user_id.item()] = user_embeddings[i].cpu().numpy()
                
                # 如果批次包含物品ID，也保存物品嵌入
                if hasattr(batch, 'item_ids') and batch.item_ids is not None:
                    for i, item_id in enumerate(batch.item_ids):
                        item_embeddings_dict[item_id.item()] = item_embeddings[i].cpu().numpy()
                
                processed_count += user_ids.shape[0]
                if processed_count % 1000 == 0:
                    print(f"Processed {processed_count} users...")
                    
            except StopIteration:
                break
        end_time = time.time()
        
        print(f"Total processed users: {processed_count}")
        print(f"Total time: {end_time - start_time:.2f} seconds")
        
        # 保存用户嵌入到文件
        user_embeddings_file = os.path.join(output_dir, "user_embeddings.pt")
        torch.save(user_embeddings_dict, user_embeddings_file)
        print(f"User embeddings saved to: {user_embeddings_file}")
        
        # 如果有物品嵌入，也保存
        if item_embeddings_dict:
            item_embeddings_file = os.path.join(output_dir, "item_embeddings.pt")
            torch.save(item_embeddings_dict, item_embeddings_file)
            print(f"Item embeddings saved to: {item_embeddings_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieval Inference Example")
    parser.add_argument("--gin_config_file", type=str, required=True)
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument(
        "--mode", type=RunningMode, choices=list(RunningMode), required=True
    )
    parser.add_argument("--output_dir", type=str, default="./output_embeddings")
    parser.add_argument("--disable_context", action="store_true")

    args = parser.parse_args()
    gin.parse_config_file(args.gin_config_file)

    if args.mode == RunningMode.EVAL:
        if args.disable_context:
            print("disable_context is ignored in Eval mode.")
        run_retrieval_gr_evaluate(
            checkpoint_dir=args.checkpoint_dir,
        )
    elif args.mode == RunningMode.EMBEDDING:
        run_retrieval_gr_embedding(
            checkpoint_dir=args.checkpoint_dir,
            output_dir=args.output_dir,
            disable_contextual_features=args.disable_context,
        )