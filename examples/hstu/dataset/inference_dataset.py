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
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
from collections import defaultdict
from typing import Dict, Iterator, List, Optional

import numpy as np
import pandas as pd
import torch
from dataset.utils import Batch, RetrievalBatch
from torch.utils.data.dataset import IterableDataset
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from dataclasses import dataclass


def load_seq(x: str):
    if isinstance(x, str):
        y = json.loads(x)
    else:
        y = x
    return y


def maybe_truncate_seq(
    y: List[int],
    max_seq_len: int,
) -> List[int]:
    y_len = len(y)
    if y_len > max_seq_len:
        y = y[:max_seq_len]
    return y


@dataclass
class UserItemDatasetConfig:
    """配置用于推理的用户-物品数据集"""
    seq_logs_file: str
    batch_logs_file: str
    batch_size: int
    max_seqlen: int
    item_feature_name: str
    contextual_feature_names: List[str]
    action_feature_name: str
    max_num_candidates: int
    item_vocab_size: int
    userid_name: str
    date_name: str
    sequence_endptr_name: str
    timestamp_names: List[str]
    random_seed: int = 0
    seq_nrows: Optional[int] = None
    batch_nrows: Optional[int] = None


class UserItemDataset(IterableDataset[RetrievalBatch]):
    """
    用于召回推理的用户-物品数据集类，保持与训练数据集相同的数据结构。
    该类继承自IterableDataset，专门设计用于提供用户嵌入向量提取所需的数据。

    Args:
        config (UserItemDatasetConfig): 包含数据集配置信息的对象。
    """

    def __init__(self, config: UserItemDatasetConfig) -> None:
        super().__init__()
        self._device = torch.cuda.current_device()
        
        # 从配置对象中获取参数
        self.seq_logs_file = config.seq_logs_file
        self.batch_logs_file = config.batch_logs_file
        self.batch_size = config.batch_size
        self.max_seqlen = config.max_seqlen
        self.item_feature_name = config.item_feature_name
        self.contextual_feature_names = config.contextual_feature_names
        self.action_feature_name = config.action_feature_name
        self.max_num_candidates = config.max_num_candidates
        self.item_vocab_size = config.item_vocab_size
        self.userid_name = config.userid_name
        self.date_name = config.date_name
        self.sequence_endptr_name = config.sequence_endptr_name
        self.timestamp_names = config.timestamp_names
        self.random_seed = config.random_seed
        self.seq_nrows = config.seq_nrows
        self.batch_nrows = config.batch_nrows
        
        # 读取序列日志文件，用于构建用户序列数据
        self.seq_logs_df = pd.read_json(
            self.seq_logs_file,
            lines=True,
            nrows=self.seq_nrows,
        )
        
        # 读取批次日志文件，用于获取候选物品数据
        self.batch_logs_df = pd.read_json(
            self.batch_logs_file,
            lines=True,
            nrows=self.batch_nrows,
        )
        
        # 设置样本索引
        self._num_samples = len(self.batch_logs_df)
        self._sample_ids = np.arange(self._num_samples)

    # We do batching in our own
    def __len__(self) -> int:
        return math.ceil(self._num_samples / self._batch_size)

    def __iter__(self) -> Iterator[RetrievalBatch]:
        """返回一个检索批处理数据，格式与训练一致"""
        for i in range(len(self)):
            batch_start = i * self._batch_size
            batch_end = min(
                (i + 1) * self._batch_size,
                self._num_samples,
            )
            sample_ids = self._sample_ids[batch_start:batch_end]
            user_ids: List[int] = []
            dates: List[int] = []
            seq_endptrs: List[int] = []
            
            for sample_id in sample_ids:
                seq_endptr = self.batch_logs_df.iloc[sample_id][self.sequence_endptr_name]
                if seq_endptr > self.max_seqlen:
                    continue
                user_ids.append(
                    self.batch_logs_df.iloc[sample_id][self.userid_name]
                )
                dates.append(self.batch_logs_df.iloc[sample_id][self.date_name])
                seq_endptrs.append(seq_endptr)
            
            if len(user_ids) == 0:
                continue
            
            # 转换数据为tensor并传递给get_input_batch
            user_ids_tensor = torch.tensor(user_ids)
            dates_tensor = torch.tensor(dates)
            seq_endptrs_tensor = torch.tensor(seq_endptrs)
            seq_startptrs_tensor = torch.zeros_like(seq_endptrs_tensor)  # 从序列开始位置
            
            # 获取包含特征数据的batch
            batch = self.get_input_batch(
                user_ids=user_ids_tensor,
                dates=dates_tensor,
                sequence_endptrs=seq_endptrs_tensor,
                sequence_startptrs=seq_startptrs_tensor,
                with_contextual_features=True,  # 保留特征数据
                with_ranking_labels=False,  # 推理时不需要标签
            )
            
            # 如果batch有效，则yield，否则跳过
            if batch is not None:
                yield batch

    def get_input_batch(
        self,
        user_ids,
        dates,
        sequence_endptrs,
        sequence_startptrs,
        with_contextual_features=False,
        with_ranking_labels=False,
    ):
        """获取一个批次的输入数据，确保格式与训练一致"""
        contextual_features: Dict[str, List[int]] = defaultdict(list)
        contextual_features_seqlen: Dict[str, List[int]] = defaultdict(list)
        item_features: List[int] = []
        item_features_seqlen: List[int] = []
        action_features: List[int] = []
        action_features_seqlen: List[int] = []
        num_candidates: List[int] = []
        labels: List[int] = []
        user_ids_list: List[int] = []

        if len(user_ids) == 0:
            return None

        # 确保序列长度不超过设定的最大值
        sequence_endptrs = torch.clip(sequence_endptrs, 0, self.max_seqlen)
        
        for idx in range(len(user_ids)):
            uid = user_ids[idx].item()
            date = dates[idx].item()
            end_pos = sequence_endptrs[idx].item()  # 序列结束位置
            start_pos = sequence_startptrs[idx].item()  # 序列开始位置

            # 从序列日志中获取用户数据
            user_data = self.seq_logs_df[
                (self.seq_logs_df[self.userid_name] == uid)
                & (self.seq_logs_df[self.date_name] == date)
            ]
            
            # 如果找不到该用户的数据，跳过
            if len(user_data) == 0:
                continue
                
            data = user_data.iloc[0]  # 取第一条记录
            
            # 添加上下文特征（如果需要）
            if with_contextual_features:
                for contextual_feature_name in self.contextual_feature_names:
                    contextual_features[contextual_feature_name].append(
                        data[contextual_feature_name]
                    )
                    contextual_features_seqlen[contextual_feature_name].append(1)

            # 获取物品和动作序列
            item_seq = load_seq(data[self.item_feature_name])[start_pos:end_pos]
            action_seq = load_seq(data[self.action_feature_name])[start_pos:end_pos]
            num_candidate = 0
            
            # 如果需要候选物品
            if self.max_num_candidates > 0:
                # 生成随机候选物品
                if not with_ranking_labels:
                    num_candidate = self.max_num_candidates
                    candidate_seq = torch.randint(
                        self.item_vocab_size, (num_candidate,)
                    ).tolist()
                # 从后续序列中提取候选物品
                else:
                    future_seqs = self.seq_logs_df[
                        (self.seq_logs_df[self.userid_name] == uid)
                        & (self.seq_logs_df[self.date_name] >= date)
                    ]
                    candidate_seq = sum(
                        [
                            load_seq(future_seqs.iloc[i][self.item_feature_name])
                            for i in range(len(future_seqs))
                        ],
                        start=[],
                    )[end_pos : end_pos + self.max_num_candidates]
                    num_candidate = len(candidate_seq)
                    label_seq = sum(
                        [
                            load_seq(future_seqs.iloc[i][self.action_feature_name])
                            for i in range(len(future_seqs))
                        ],
                        start=[],
                    )[end_pos : end_pos + self.max_num_candidates]

                # 合并历史序列和候选物品序列
                all_item_seq = item_seq + candidate_seq
            else:
                all_item_seq = item_seq

            # 将物品序列添加到批处理
            item_features.extend(all_item_seq)
            item_features_seqlen.append(len(all_item_seq))
            num_candidates.append(num_candidate)
            
            # 如果需要标签（训练时使用）
            if with_ranking_labels:
                labels.extend(label_seq)

            # 添加动作序列
            action_features.extend(action_seq)
            action_features_seqlen.append(len(action_seq))
            
            # 添加用户ID
            user_ids_list.append(uid)

        # 如果没有有效数据，返回None
        if len(user_ids_list) == 0:
            return None

        # 计算每个特征的最大序列长度
        feature_to_max_seqlen = {}
        for name in self.contextual_feature_names:
            feature_to_max_seqlen[name] = max(
                contextual_features_seqlen[name], default=0
            )

        feature_to_max_seqlen[self.item_feature_name] = max(item_features_seqlen)
        feature_to_max_seqlen[self.action_feature_name] = max(action_features_seqlen)

        # 将上下文特征转换为tensor
        if with_contextual_features:
            contextual_features_tensor = torch.tensor(
                [contextual_features[name] for name in self.contextual_feature_names],
            ).view(-1)
            contextual_features_lengths_tensor = torch.tensor(
                [
                    contextual_features_seqlen[name]
                    for name in self.contextual_feature_names
                ]
            ).view(-1)
        else:
            contextual_features_tensor = torch.empty((0,), dtype=torch.int64)
            contextual_features_lengths_tensor = torch.tensor(
                [0 for name in self.contextual_feature_names]
            ).view(-1)
            
        # 构建KeyedJaggedTensor
        features = KeyedJaggedTensor.from_lengths_sync(
            keys=self.contextual_feature_names
            + [self.item_feature_name, self.action_feature_name],
            values=torch.concat(
                [
                    contextual_features_tensor.to(device=self._device),
                    torch.tensor(item_features, device=self._device),
                    torch.tensor(action_features, device=self._device),
                ]
            ).long(),
            lengths=torch.concat(
                [
                    contextual_features_lengths_tensor.to(device=self._device),
                    torch.tensor(item_features_seqlen, device=self._device),
                    torch.tensor(action_features_seqlen, device=self._device),
                ]
            ).long(),
        )
        
        # 准备批处理关键字参数
        batch_kwargs = dict(
            features=features,
            batch_size=len(user_ids_list),  # 使用实际批处理大小
            feature_to_max_seqlen=feature_to_max_seqlen,
            contextual_feature_names=self.contextual_feature_names,
            item_feature_name=self.item_feature_name,
            action_feature_name=self.action_feature_name,
            max_num_candidates=self.max_num_candidates,
            num_candidates=torch.tensor(num_candidates, device=self._device)
            if self.max_num_candidates > 0
            else None,
        )
        
        # 如果需要标签（训练时）
        if with_ranking_labels:
            labels = torch.tensor(labels, dtype=torch.int64, device=self._device)
            batch_kwargs['labels'] = labels
            return Batch(**batch_kwargs)
        
        # 返回与训练格式一致的RetrievalBatch
        return RetrievalBatch(**batch_kwargs)
