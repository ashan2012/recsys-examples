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
import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from configs import (
    InferenceHSTUConfig,
    InferenceEmbeddingConfig,
    KVCacheConfig,
    RetrievalConfig,
)
from dataset import get_data_loader
from dataset.utils import RetrievalBatch
from dataset.inference_dataset import UserItemDataset

# Add project root to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from examples.hstu.model.inference_retrieval_gr import InferenceRetrievalGR

def create_inference_dataset_config(dataset_args: Dict, batch_size: int) -> "UserItemDatasetConfig":
    """Create inference dataset configuration from gin config."""
    # 创建一个简单的配置对象用于推理
    from dataset.inference_dataset_config import UserItemDatasetConfig
    
    return UserItemDatasetConfig(
        seq_logs_file=dataset_args.get('seq_logs_file', 'data/seq_logs.json'),
        batch_logs_file=dataset_args.get('batch_logs_file', 'data/batch_logs.json'),
        userid_name=dataset_args.get('userid_name', 'user_id'),
        date_name=dataset_args.get('date_name', 'date'),
        item_feature_name=dataset_args.get('item_feature_name', 'item_sequence'),
        action_feature_name=dataset_args.get('action_feature_name', 'action_sequence'),
        contextual_feature_names=dataset_args.get('contextual_feature_names', []),
        max_seqlen=dataset_args.get('max_seqlen', 1024),
        item_vocab_size=dataset_args.get('item_vocab_size', 100000),
        max_num_candidates=dataset_args.get('max_num_candidates', 10),
        batch_size=batch_size,
        num_epochs=dataset_args.get('num_epochs', 1),
        shuffle=dataset_args.get('shuffle', False),
        drop_last=dataset_args.get('drop_last', False),
        num_workers=dataset_args.get('num_workers', 0),
        pin_memory=dataset_args.get('pin_memory', False),
        prefetch_factor=dataset_args.get('prefetch_factor', 2),
        persistent_workers=dataset_args.get('persistent_workers', False)
    )

def parse_arguments():
    """
    Parse command line arguments.
    """
    parser = argparse.ArgumentParser(description="Inference for RetrievalGR model")
    parser.add_argument(
        "--config_file",
        type=str,
        default="inference/configs/gameid_retrieval.gin",
        help="Path to the configuration file",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to the model directory containing checkpoints",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Path to save the embeddings and results",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["EVAL", "EMBEDDING"],
        default="EMBEDDING",
        help="Run mode: EVAL for evaluation, EMBEDDING for embedding extraction",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1024,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
        help="Device to run inference on",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode",
    )
    return parser.parse_args()

def create_inference_config(config_file: str) -> Dict:
    """
    Create inference configuration from gin config file.
    
    Args:
        config_file: Path to the gin configuration file
        
    Returns:
        Dictionary with configuration parameters
    """
    import gin
    
    # Clear any existing gin configurations
    gin.clear_config()
    
    # Parse the gin configuration file
    gin.parse_config_file(config_file)
    
    # Extract configuration parameters
    config = {
        "trainer": gin.query_parameter("TrainerArgs"),
        "dataset": gin.query_parameter("DatasetArgs"),
        "model": gin.query_parameter("ModelArgs"),
    }
    
    return config

def load_inference_model(model_dir: str, config: Dict, device: str) -> InferenceRetrievalGR:
    """
    Load the InferenceRetrievalGR model from checkpoint.
    
    Args:
        model_dir: Path to the model directory
        config: Configuration dictionary
        device: Device to load the model on
        
    Returns:
        Loaded InferenceRetrievalGR model
    """
    # Create model configuration
    task_config = RetrievalConfig(
        embedding_configs=config["model"].embedding_configs,
        l2_norm_eps=config["model"].l2_norm_eps,
    )
    
    # Create inference model instance
    model = InferenceRetrievalGR(
        hstu_config=config["model"].hstu_config,
        kvcache_config=config["model"].kvcache_config,
        task_config=task_config,
    )
    
    # Set correct precision based on configuration
    if config["model"].hstu_config.bf16:
        model = model.bfloat16()
    elif config["model"].hstu_config.fp16:
        model = model.half()
    
    # Load model checkpoint
    model.load_checkpoint(model_dir)
    
    return model

def extract_user_embeddings(
    model: InferenceRetrievalGR,
    dataset: UserItemDataset,
    batch_size: int,
    device: str,
    output_file: str,
    debug: bool = False
) -> Dict[int, List[float]]:
    """
    Extract user embeddings from the model.
    
    Args:
        model: The InferenceRetrievalGR model
        dataset: The dataset to extract embeddings from
        batch_size: Batch size for inference
        device: Device to run inference on
        output_file: File to save the embeddings to
        debug: Whether to enable debug mode
        
    Returns:
        Dictionary mapping user IDs to embeddings
    """
    user_embeddings = {}
    # UserItemDataset 已经是可迭代的，不需要 DataLoader
    dataloader = dataset
    
    # Prepare output directory
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            start_time = time.time()
            
            # Move batch to device
            batch = batch.to(device)
            
            # Get user IDs
            user_ids = batch.user_ids
            
            # Create user start positions (assuming each user starts at 0)
            user_start_pos = torch.zeros_like(user_ids, dtype=torch.int32)
            
            # Forward pass to get user embeddings
            embeddings = model(
                batch=batch,
                user_ids=user_ids,
                user_start_pos=user_start_pos,
            )
            
            # Convert to numpy and store embeddings
            user_ids_np = user_ids.cpu().numpy()
            embeddings_np = embeddings.cpu().numpy()
            
            for user_id, embedding in zip(user_ids_np, embeddings_np):
                user_embeddings[int(user_id)] = embedding.tolist()  # Convert to list for JSON serialization
            
            # Print progress
            if debug and (i + 1) % 10 == 0:
                batch_time = time.time() - start_time
                embeddings_per_sec = batch_size / batch_time
                print(f"Processed batch {i + 1}/{len(dataloader)}")
                print(f"Batch time: {batch_time:.4f}s, Embeddings/sec: {embeddings_per_sec:.2f}")
    
    # Save embeddings to file
    with open(output_file, 'w') as f:
        json.dump(user_embeddings, f)
    
    print(f"Extracted embeddings for {len(user_embeddings)} users")
    print(f"Embeddings saved to {output_file}")
    
    return user_embeddings

def evaluate_model(
    model: InferenceRetrievalGR,
    dataset: UserItemDataset,
    batch_size: int,
    device: str,
    output_file: str,
    debug: bool = False
):
    """
    Evaluate the model performance.
    
    Args:
        model: The InferenceRetrievalGR model
        dataset: The dataset to evaluate on
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        output_file: File to save the evaluation results
        debug: Whether to enable debug mode
    """
    # This is a placeholder for evaluation logic
    print("Running evaluation...")
    
    # Extract user embeddings first
    embeddings_file = os.path.join(os.path.dirname(output_file), "eval_user_embeddings.json")
    user_embeddings = extract_user_embeddings(
        model,
        dataset,
        batch_size,
        device,
        embeddings_file,
        debug,
    )
    
    # TODO: Implement actual evaluation metrics
    results = {
        "total_users": len(user_embeddings),
        "embedding_dim": len(next(iter(user_embeddings.values()))),
        "evaluation_time": time.time(),
        "metrics": {
            "recall@1": 0.0,  # Placeholder
            "recall@5": 0.0,  # Placeholder
            "recall@10": 0.0,  # Placeholder
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Evaluation completed. Results saved to {output_file}")
    return results

def main():
    """
    Main function to run inference for RetrievalGR model.
    """
    # Parse arguments
    args = parse_arguments()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    print(f"Loading configuration from {args.config_file}")
    config = create_inference_config(args.config_file)
    
    # Load model
    print(f"Loading model from {args.model_dir}")
    model = load_inference_model(args.model_dir, config, args.device)
    
    # Create dataset configuration
    dataset_config = create_inference_dataset_config(
        config["dataset"],
        batch_size=args.batch_size,
    )
    
    # Load dataset
    print("Loading dataset...")
    dataset = UserItemDataset(dataset_config)
    
    # Run in the specified mode
    if args.mode == "EMBEDDING":
        # Extract user embeddings
        output_file = os.path.join(args.output_dir, "user_embeddings.json")
        user_embeddings = extract_user_embeddings(
            model,
            dataset,
            args.batch_size,
            args.device,
            output_file,
            args.debug,
        )
    elif args.mode == "EVAL":
        # Run evaluation
        output_file = os.path.join(args.output_dir, "evaluation_results.json")
        results = evaluate_model(
            model,
            dataset,
            args.batch_size,
            args.device,
            output_file,
            args.debug,
        )
    else:
        raise ValueError(f"Unknown mode: {args.mode}")
    
    print("Inference completed successfully!")

if __name__ == "__main__":
    main()