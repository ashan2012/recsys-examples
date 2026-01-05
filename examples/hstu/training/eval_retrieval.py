#!/usr/bin/env python3
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

"""
评估训练好的召回模型

用法:
    torchrun --nproc_per_node=4 eval_retrieval.py \\
        --gin_config_file ./configs/gameid_retrieval.gin \\
        --checkpoint_dir ./ckpt_re_test/iter100

说明:
    - 使用与训练相同的数据格式和配置
    - 加载指定的checkpoint
    - 在整个测试集上评估
    - 输出所有配置的评估指标
"""

import argparse
import sys

import gin
import torch
from commons.utils import initialize as init
from commons.utils.logger import print_rank_0
from configs import RetrievalConfig
from distributed.sharding import make_optimizer_and_shard
from model import get_retrieval_model
from modules.metrics import RetrievalTaskMetricWithSampling
from pipeline.train_pipeline import (
    JaggedMegatronPrefetchTrainPipelineSparseDist,
    JaggedMegatronTrainNonePipeline,
    JaggedMegatronTrainPipelineSparseDist,
)
from trainer.training import evaluate, maybe_load_ckpts
from trainer.utils import (
    create_dynamic_optitons_dict,
    create_embedding_configs,
    create_hstu_config,
    create_optimizer_params,
    create_retrieval_config,
    get_data_loader,
    get_dataset_and_embedding_args,
    get_embedding_vector_storage_multiplier,
)
from utils import (
    BenchmarkDatasetArgs,
    DatasetArgs,
    EmbeddingArgs,
    NetworkArgs,
    OptimizerArgs,
    RetrievalArgs,
    TensorModelParallelArgs,
    TrainerArgs,
)


def main():
    parser = argparse.ArgumentParser(
        description="评估召回模型",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "--gin_config_file",
        type=str,
        required=True,
        help="gin配置文件路径（与训练时相同）"
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="checkpoint目录路径（例如: ./ckpt_re_test/iter100）"
    )
    parser.add_argument(
        "--max_eval_iters",
        type=int,
        default=None,
        help="最大评估迭代数，None表示评估整个测试集（默认: None）"
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="评估batch size，None表示使用配置文件中的值（默认: None）"
    )
    
    args = parser.parse_args()
    
    print_rank_0("=" * 80)
    print_rank_0("Model Evaluation Script")
    print_rank_0("=" * 80)
    print_rank_0(f"Config file: {args.gin_config_file}")
    print_rank_0(f"Checkpoint: {args.checkpoint_dir}")
    print_rank_0("=" * 80)
    
    # 解析gin配置
    gin.parse_config_file(args.gin_config_file)
    
    # 获取配置参数
    trainer_args = TrainerArgs()
    
    # 覆盖评估参数（如果提供）
    if args.max_eval_iters is not None:
        trainer_args.max_eval_iters = args.max_eval_iters
    if args.eval_batch_size is not None:
        trainer_args.eval_batch_size = args.eval_batch_size
    
    dataset_args, embedding_args = get_dataset_and_embedding_args()
    network_args = NetworkArgs()
    optimizer_args = OptimizerArgs()
    tp_args = TensorModelParallelArgs()
    
    print_rank_0("\n" + "=" * 80)
    print_rank_0("Configuration Summary")
    print_rank_0("=" * 80)
    print_rank_0(f"Dataset: {dataset_args.dataset_name}")
    print_rank_0(f"Eval batch size: {trainer_args.eval_batch_size}")
    print_rank_0(f"Max eval iters: {trainer_args.max_eval_iters or 'All'}")
    print_rank_0(f"Pipeline type: {trainer_args.pipeline_type}")
    print_rank_0("=" * 80)
    
    # 初始化分布式环境
    print_rank_0("\nInitializing distributed environment...")
    init.initialize_distributed()
    init.initialize_model_parallel(
        tensor_model_parallel_size=tp_args.tensor_model_parallel_size
    )
    init.set_random_seed(trainer_args.seed)
    print_rank_0("Distributed environment initialized")
    
    # 创建模型配置
    print_rank_0("\nCreating model configuration...")
    hstu_config = create_hstu_config(network_args, tp_args)
    task_config = create_retrieval_config(dataset_args, network_args, embedding_args)
    
    print_rank_0(f"Evaluation metrics: {task_config.eval_metrics}")
    
    # 创建模型
    print_rank_0("\nCreating model...")
    model = get_retrieval_model(hstu_config=hstu_config, task_config=task_config)
    print_rank_0("Model created")
    
    # 设置dynamic embeddings
    print_rank_0("\nSetting up dynamic embeddings...")
    dynamic_options_dict = create_dynamic_optitons_dict(
        embedding_args,
        network_args.hidden_size,
        training=False,  # 评估模式
        embedding_dim_multiplier=get_embedding_vector_storage_multiplier(
            optimizer_args.optimizer_str
        ),
    )
    optimizer_param = create_optimizer_params(optimizer_args)
    model_eval, dense_optimizer = make_optimizer_and_shard(
        model,
        config=hstu_config,
        sparse_optimizer_param=optimizer_param,
        dense_optimizer_param=optimizer_param,
        dynamicemb_options_dict=dynamic_options_dict,
        pipeline_type=trainer_args.pipeline_type,
    )
    print_rank_0("Dynamic embeddings setup complete")
    
    # 加载checkpoint
    print_rank_0(f"\nLoading checkpoint from {args.checkpoint_dir}...")
    maybe_load_ckpts(args.checkpoint_dir, model, dense_optimizer)
    print_rank_0("Checkpoint loaded successfully")
    
    # 创建评估指标模块
    print_rank_0("\nCreating metric module...")
    stateful_metric_module = RetrievalTaskMetricWithSampling(
        metric_types=task_config.eval_metrics, MAX_K=500
    )
    print_rank_0(f"Metric types: {task_config.eval_metrics}")
    
    # 创建数据加载器
    print_rank_0("\nCreating data loaders...")
    train_dataloader, test_dataloader = get_data_loader(
        "retrieval", dataset_args, trainer_args, 0
    )
    print_rank_0(f"Test dataloader created with {len(test_dataloader)} batches")
    
    # 创建pipeline
    print_rank_0("\nCreating evaluation pipeline...")
    if trainer_args.pipeline_type in ["prefetch", "native"]:
        pipeline_factory = (
            JaggedMegatronPrefetchTrainPipelineSparseDist
            if trainer_args.pipeline_type == "prefetch"
            else JaggedMegatronTrainPipelineSparseDist
        )
        pipeline = pipeline_factory(
            model_eval,
            dense_optimizer,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
    else:
        pipeline = JaggedMegatronTrainNonePipeline(
            model_eval,
            dense_optimizer,
            device=torch.device("cuda", torch.cuda.current_device()),
        )
    print_rank_0(f"Pipeline created: {type(pipeline).__name__}")
    
    # 设置为评估模式
    print_rank_0("\nSetting model to evaluation mode...")
    model_eval.eval()
    
    # 运行评估
    print_rank_0("\n" + "=" * 80)
    print_rank_0("Starting Evaluation on Test Set")
    print_rank_0("=" * 80)
    
    import time
    start_time = time.time()
    
    evaluate(
        pipeline,
        stateful_metric_module,
        trainer_args=trainer_args,
        eval_loader=test_dataloader,
    )
    
    elapsed_time = time.time() - start_time
    
    print_rank_0("\n" + "=" * 80)
    print_rank_0("Evaluation Complete")
    print_rank_0("=" * 80)
    print_rank_0(f"Elapsed time: {elapsed_time:.2f} seconds")
    print_rank_0("=" * 80)
    
    # 清理
    init.destroy_global_state()
    print_rank_0("\nEvaluation finished successfully!")


if __name__ == "__main__":
    main()

