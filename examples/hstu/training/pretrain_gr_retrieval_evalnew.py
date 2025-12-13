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

import commons.utils.initialize as init
from commons.utils.logger import print_rank_0
from commons.utils.stringify import stringify_dict
import gin
import torch  # pylint: disable-unused-import
from configs import RetrievalConfig
from distributed.sharding import make_optimizer_and_shard
from megatron.core import parallel_state
from itertools import chain, count, cycle, islice
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
import logging

# 设置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def evaluate(
    pipeline: Union[
        JaggedMegatronPrefetchTrainPipelineSparseDist,
        JaggedMegatronTrainNonePipeline,
        JaggedMegatronTrainPipelineSparseDist,
    ],
    stateful_metric_module: torch.nn.Module,
    trainer_args: TrainerArgs,
    eval_loader: torch.utils.data.DataLoader,):
    eval_iter = 0
    torch.cuda.nvtx.range_push(f"#evaluate")
    print_rank_0("=== 开始评估过程 ===")
    
    # 获取评估配置信息
    max_eval_iters = trainer_args.max_eval_iters or len(eval_loader)
    max_eval_iters = min(max_eval_iters, len(eval_loader))
    print_rank_0(f"评估配置: max_eval_iters={max_eval_iters}, eval_loader长度={len(eval_loader)}")
    
    # make a copy of eval_loader to avoid modifying the original loader
    iterated_eval_loader = islice(eval_loader, len(eval_loader))
    
    with torch.no_grad():
        print_rank_0("进入评估循环...")
        for i in range(max_eval_iters):
            eval_iter += 1
            print_rank_0(f"--- 评估迭代 {eval_iter}/{max_eval_iters} ---")
            
            # 记录pipeline.progress开始
            print_rank_0("调用pipeline.progress处理数据...")
            reporting_loss, (batch_data, logits, labels, additional_info) = pipeline.progress(
                iterated_eval_loader
            )
            
            print_rank_0(f"批次数据信息: reporting_loss={reporting_loss}")
            print_rank_0(f"logits形状: {logits.shape if logits is not None else 'None'}")
            print_rank_0(f"labels形状: {labels.shape if labels is not None else 'None'}")
            print_rank_0(f"additional_info: {additional_info}")
            
            # 记录batch_data的详细信息
            if hasattr(batch_data, '__dict__'):
                print_rank_0(f"batch_data属性: {batch_data.__dict__}")
            else:
                print_rank_0(f"batch_data类型: {type(batch_data)}")
                print_rank_0(f"batch_data内容: {batch_data}")
            
            # 获取用户嵌入和TopK推荐结果
            if isinstance(stateful_metric_module, RetrievalTaskMetricWithSampling):
                print_rank_0("获取用户嵌入和TopK推荐结果...")
                retrieval_gr = get_unwrapped_module(pipeline._model)
                
                # 尝试获取用户嵌入
                try:
                    if hasattr(retrieval_gr, 'get_user_embeddings'):
                        user_embeddings = retrieval_gr.get_user_embeddings(batch_data)
                        print_rank_0(f"用户嵌入形状: {user_embeddings.shape}")
                        print_rank_0(f"用户嵌入前3个样本:")
                        for i in range(min(3, user_embeddings.shape[0])):
                            print_rank_0(f"  用户 {i}: 平均值={user_embeddings[i].mean().item():.4f}, 标准差={user_embeddings[i].std().item():.4f}")
                    else:
                        print_rank_0("模型不支持直接获取用户嵌入")
                except Exception as e:
                    print_rank_0(f"获取用户嵌入时出错: {e}")
                
                # 计算TopK推荐
                try:
                    if logits is not None:
                        # 获取TopK推荐结果 (假设K=10)
                        top_k = 10
                        top_scores, top_indices = torch.topk(logits, top_k, dim=1)
                        
                        print_rank_0(f"批次大小: {logits.shape[0]}, 每个用户推荐{top_k}个item")
                        print_rank_0(f"TopK分数形状: {top_scores.shape}")
                        print_rank_0(f"TopK索引形状: {top_indices.shape}")
                        
                        # 获取item特征表名称和item embeddings
                        export_table_name = retrieval_gr.get_item_feature_table_name()
                        embedding_export = retrieval_gr._embedding_collection.export_local_embedding(export_table_name)
                        
                        # 假设第一个embedding是item embedding
                        if len(embedding_export) > 0:
                            item_embeddings = embedding_export[0]  # item embedding matrix
                            print_rank_0(f"Item嵌入矩阵形状: {item_embeddings.shape}")
                            
                            # 显示每个用户的TopK推荐详情
                            for user_idx in range(min(3, logits.shape[0])):  # 只显示前3个用户
                                print_rank_0(f"--- 用户 {user_idx} 的Top{top_k}推荐 ---")
                                user_top_indices = top_indices[user_idx]  # TopK item indices for this user
                                user_top_scores = top_scores[user_idx]   # 对应的分数
                                
                                for rank, (item_idx, score) in enumerate(zip(user_top_indices, user_top_scores)):
                                    # 获取该item的embedding
                                    item_embedding = item_embeddings[item_idx]
                                    print_rank_0(f"  排名 {rank+1}: Item ID={item_idx}, 分数={score:.4f}, "
                                              f"Embedding均值={item_embedding.mean().item():.4f}")
                                    
                                    # 显示item embedding的前几个维度
                                    embedding_preview = item_embedding[:5].cpu().numpy()
                                    print_rank_0(f"    Embedding前5维: {embedding_preview}")
                        else:
                            print_rank_0("未找到item embedding信息")
                     
                except Exception as e:
                    print_rank_0(f"计算TopK推荐时出错: {e}")
            
            # metric module forward
            print_rank_0("调用stateful_metric_module进行前向计算...")
            stateful_metric_module(logits, labels)
            
            # 每隔几个迭代记录一次状态
            if eval_iter % 5 == 0 or eval_iter == 1:
                print_rank_0(f"已完成 {eval_iter} 个迭代")
        
        print_rank_0("评估循环完成，开始计算最终指标...")
        
        # compute will reset the states
        if isinstance(stateful_metric_module, RetrievalTaskMetricWithSampling):
            print_rank_0("使用RetrievalTaskMetricWithSampling进行评估")
            retrieval_gr = get_unwrapped_module(pipeline._model)
            export_table_name = retrieval_gr.get_item_feature_table_name()
            print_rank_0(f"获取到的item_feature_table_name: {export_table_name}")
            
            # 记录embedding导出过程
            embedding_export = retrieval_gr._embedding_collection.export_local_embedding(
                export_table_name
            )
            print_rank_0(f"导出的embedding信息: 长度={len(embedding_export)}")
            for idx, emb in enumerate(embedding_export):
                print_rank_0(f"  embedding[{idx}]: 形状={emb.shape if hasattr(emb, 'shape') else 'N/A'}")
            
            eval_metric_dict, _, _ = stateful_metric_module.compute(
                *embedding_export,
            )
        else:
            print_rank_0(f"使用通用metric模块进行评估, 类型: {type(stateful_metric_module)}")
            eval_metric_dict = stateful_metric_module.compute()
        
        dp_size = parallel_state.get_data_parallel_world_size()
        print_rank_0(f"数据并行大小: {dp_size}")
    
    # TODO, fix the samples when there is incomplete batch
    print_rank_0("=== 评估完成，输出结果 ===")
    print_rank_0(f"评估了 {eval_iter * dp_size * trainer_args.eval_batch_size} 个用户")
    print_rank_0(f"指标字典: {eval_metric_dict}")
    
    print_rank_0(
        f"[eval] [eval {eval_iter * dp_size * trainer_args.eval_batch_size} users]:\n    "
        + stringify_dict(eval_metric_dict, prefix="Metrics", sep="\n    ")
    )
    torch.cuda.nvtx.range_pop()
    print_rank_0("=== 评估过程结束 ===")

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
    print_rank_0("=== 程序开始启动 ===")
    
    parser = argparse.ArgumentParser(
        description="Distributed GR Arguments", allow_abbrev=False
    )
    parser.add_argument("--gin-config-file", type=str)
    args = parser.parse_args()
    print_rank_0(f"解析命令行参数: {args}")
    
    gin.parse_config_file(args.gin_config_file)
    print_rank_0("Gin配置文件解析完成")
    
    trainer_args = TrainerArgs()
    print_rank_0(f"训练器参数: max_eval_iters={trainer_args.max_eval_iters}, eval_batch_size={trainer_args.eval_batch_size}")
    
    dataset_args, embedding_args = get_dataset_and_embedding_args()
    print_rank_0(f"数据集参数: {dataset_args}")
    print_rank_0(f"嵌入参数数量: {len(embedding_args)}")
    
    network_args = NetworkArgs()
    optimizer_args = OptimizerArgs()
    tp_args = TensorModelParallelArgs()
    print_rank_0(f"网络参数: hidden_size={network_args.hidden_size}")
    print_rank_0(f"张量并行大小: {tp_args.tensor_model_parallel_size}")

    print_rank_0("初始化分布式环境...")
    init.initialize_distributed()
    init.initialize_model_parallel(
        tensor_model_parallel_size=tp_args.tensor_model_parallel_size
    )
    init.set_random_seed(trainer_args.seed)
    print_rank_0("分布式环境初始化完成")

    print_rank_0("创建HSTU配置...")
    hstu_config = create_hstu_config(network_args, tp_args)
    task_config = create_retrieval_config(dataset_args, network_args, embedding_args)
    print_rank_0("配置创建完成")

    print_rank_0("创建检索模型...")
    model = get_retrieval_model(hstu_config=hstu_config, task_config=task_config)
    print_rank_0("模型创建完成")

    print_rank_0("创建动态选项字典...")
    dynamic_options_dict = create_dynamic_optitons_dict(
        embedding_args,
        network_args.hidden_size,
        training=True,
        embedding_dim_multiplier=get_embedding_vector_storage_multiplier(
            optimizer_args.optimizer_str
        ),
    )
    
    optimizer_param = create_optimizer_params(optimizer_args)
    print_rank_0("创建优化器和分片...")
    model_train, dense_optimizer = make_optimizer_and_shard(
        model,
        config=hstu_config,
        sparse_optimizer_param=optimizer_param,
        dense_optimizer_param=optimizer_param,
        dynamicemb_options_dict=dynamic_options_dict,
        pipeline_type=trainer_args.pipeline_type,
    )
    print_rank_0("优化器和分片创建完成")

    print_rank_0("获取数据加载器...")
    train_dataloader, test_dataloader = get_data_loader(
        "retrieval", dataset_args, trainer_args, 0
    )
    print_rank_0(f"训练数据加载器长度: {len(train_dataloader) if train_dataloader else 'None'}")
    print_rank_0(f"测试数据加载器长度: {len(test_dataloader)}")

    print_rank_0("创建状态化指标模块...")
    stateful_metric_module = RetrievalTaskMetricWithSampling(
        metric_types=task_config.eval_metrics, MAX_K=500
    )
    print_rank_0(f"评估指标类型: {task_config.eval_metrics}")

    print_rank_0("加载检查点...")
    maybe_load_ckpts(trainer_args.ckpt_load_dir, model, dense_optimizer)
    print_rank_0("检查点加载完成")

    print_rank_0("=== 开始执行评估 ===")
    evaluate(
        pipeline=model_train,
        stateful_metric_module=stateful_metric_module,
        trainer_args=trainer_args,
        eval_loader=test_dataloader,
    )
    print_rank_0("=== 评估执行完成 ===")

    init.destroy_global_state()


if __name__ == "__main__":
    main()