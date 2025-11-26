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
from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class InferenceDatasetConfig:
    """
    Configuration for inference dataset, similar to the training dataset.

    Args:
        seq_logs_file (str): Path to the sequence logs file.
        batch_logs_file (str): Path to the batch logs file.
        batch_size (int): The batch size for inference.
        max_seqlen (int): The maximum sequence length.
        item_feature_name (str): The name of the item feature.
        contextual_feature_names (List[str]): List of contextual feature names.
        action_feature_name (str): The name of the action feature.
        max_num_candidates (int): The maximum number of candidate items.
        item_vocab_size (int): The size of the item vocabulary.
        userid_name (str): The name of the user ID column.
        date_name (str): The name of the date column.
        sequence_endptr_name (str): The name of the sequence end pointer column.
        timestamp_names (List[str]): The names of timestamp columns.
        random_seed (int): The random seed.
        seq_nrows (int): The number of rows to read from the sequence file.
        batch_nrows (int): The number of rows to read from the batch file.
    """

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
    seq_nrows: int = None
    batch_nrows: int = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert the configuration to a dictionary."""
        return {
            "seq_logs_file": self.seq_logs_file,
            "batch_logs_file": self.batch_logs_file,
            "batch_size": self.batch_size,
            "max_seqlen": self.max_seqlen,
            "item_feature_name": self.item_feature_name,
            "contextual_feature_names": self.contextual_feature_names,
            "action_feature_name": self.action_feature_name,
            "max_num_candidates": self.max_num_candidates,
            "item_vocab_size": self.item_vocab_size,
            "userid_name": self.userid_name,
            "date_name": self.date_name,
            "sequence_endptr_name": self.sequence_endptr_name,
            "timestamp_names": self.timestamp_names,
            "random_seed": self.random_seed,
            "seq_nrows": self.seq_nrows,
            "batch_nrows": self.batch_nrows,
        }

def create_inference_dataset_config(config_data: Dict[str, Any]) -> InferenceDatasetConfig:
    """
    Create an InferenceDatasetConfig from a dictionary.

    Args:
        config_data (Dict[str, Any]): Dictionary containing configuration data.

    Returns:
        InferenceDatasetConfig: The created configuration object.
    """
    return InferenceDatasetConfig(
        seq_logs_file=config_data.get("seq_logs_file", ""),
        batch_logs_file=config_data.get("batch_logs_file", ""),
        batch_size=config_data.get("batch_size", 1024),
        max_seqlen=config_data.get("max_seqlen", 100),
        item_feature_name=config_data.get("item_feature_name", "item_id"),
        contextual_feature_names=config_data.get("contextual_feature_names", []),
        action_feature_name=config_data.get("action_feature_name", "action"),
        max_num_candidates=config_data.get("max_num_candidates", 0),
        item_vocab_size=config_data.get("item_vocab_size", 100000),
        userid_name=config_data.get("userid_name", "user_id"),
        date_name=config_data.get("date_name", "date"),
        sequence_endptr_name=config_data.get("sequence_endptr_name", "seq_endptr"),
        timestamp_names=config_data.get("timestamp_names", ["timestamp"]),
        random_seed=config_data.get("random_seed", 0),
        seq_nrows=config_data.get("seq_nrows", None),
        batch_nrows=config_data.get("batch_nrows", None),
    )