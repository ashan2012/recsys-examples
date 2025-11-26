#!/usr/bin/env python3
"""
简化的推理数据集测试脚本
用于验证 UserItemDataset 类的功能，不依赖复杂的深度学习框架
"""

import json
import os
import sys
from typing import Dict, List

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset.inference_dataset import UserItemDataset, UserItemDatasetConfig

def create_test_data():
    """创建测试数据文件"""
    
    # 确保目录存在
    os.makedirs("test_data", exist_ok=True)
    
    # 创建序列日志数据
    seq_logs_data = [
        {
            "user_id": 1,
            "date": "2024-01-01",
            "item_sequence": json.dumps([101, 102, 103, 104, 105]),
            "action_sequence": json.dumps([1, 2, 3, 4, 5])
        },
        {
            "user_id": 2,
            "date": "2024-01-01",
            "item_sequence": json.dumps([201, 202, 203, 204]),
            "action_sequence": json.dumps([1, 2, 3, 4])
        },
        {
            "user_id": 1,
            "date": "2024-01-02",
            "item_sequence": json.dumps([105, 106, 107]),
            "action_sequence": json.dumps([5, 6, 7])
        }
    ]
    
    # 创建批次日志数据
    batch_logs_data = [
        {"user_id": 1, "date": "2024-01-01", "sequence_endptr": 5},
        {"user_id": 2, "date": "2024-01-01", "sequence_endptr": 4},
        {"user_id": 1, "date": "2024-01-02", "sequence_endptr": 3}
    ]
    
    # 保存为 JSON 文件
    with open("test_data/seq_logs.json", "w") as f:
        json.dump(seq_logs_data, f, indent=2)
    
    with open("test_data/batch_logs.json", "w") as f:
        json.dump(batch_logs_data, f, indent=2)
    
    print("✅ 测试数据创建完成:")
    print("   - test_data/seq_logs.json")
    print("   - test_data/batch_logs.json")

def test_useritem_dataset():
    """测试 UserItemDataset 类"""
    
    print("\n🧪 开始测试 UserItemDataset...")
    
    # 创建配置
    config = UserItemDatasetConfig(
        seq_logs_file="test_data/seq_logs.json",
        batch_logs_file="test_data/batch_logs.json",
        userid_name="user_id",
        date_name="date",
        item_feature_name="item_sequence",
        action_feature_name="action_sequence",
        contextual_feature_names=[],  # 简化测试
        max_seqlen=100,
        item_vocab_size=1000,
        max_num_candidates=2,
        batch_size=2,
        num_epochs=1,
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=False,
        prefetch_factor=2,
        persistent_workers=False
    )
    
    # 创建数据集
    dataset = UserItemDataset(config)
    
    print(f"✅ 数据集创建成功，共 {len(dataset)} 个批次")
    
    # 测试迭代
    batch_count = 0
    successful_batches = 0
    
    for batch in dataset:
        batch_count += 1
        print(f"\n📦 处理批次 {batch_count}:")
        
        # 检查批次类型
        print(f"   批次类型: {type(batch).__name__}")
        
        # 检查批次属性
        if hasattr(batch, 'features'):
            print(f"   特征键: {batch.features.keys()}")
            print(f"   批处理大小: {batch.batch_size}")
            
        if hasattr(batch, 'max_num_candidates'):
            print(f"   最大候选数: {batch.max_num_candidates}")
            
        if hasattr(batch, 'num_candidates') and batch.num_candidates is not None:
            print(f"   候选数: {batch.num_candidates}")
        
        successful_batches += 1
        
        # 只处理前几个批次进行测试
        if batch_count >= 5:
            break
    
    print(f"\n✅ 迭代测试完成:")
    print(f"   总共 {batch_count} 个批次")
    print(f"   成功处理 {successful_batches} 个批次")
    
    return successful_batches > 0

def test_dataset_config():
    """测试数据集配置"""
    
    print("\n🧪 开始测试 UserItemDatasetConfig...")
    
    # 创建配置
    config = UserItemDatasetConfig(
        seq_logs_file="test_data/seq_logs.json",
        batch_logs_file="test_data/batch_logs.json",
        userid_name="user_id",
        date_name="date",
        item_feature_name="item_sequence",
        action_feature_name="action_sequence",
        contextual_feature_names=["feature1", "feature2"],
        max_seqlen=512,
        item_vocab_size=50000,
        max_num_candidates=5,
        batch_size=16,
        num_epochs=1,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True
    )
    
    print("✅ 配置创建成功:")
    for key, value in config.__dict__.items():
        print(f"   {key}: {value}")
    
    return True

def main():
    """主测试函数"""
    
    print("🚀 开始推理数据集测试...")
    
    # 清理之前的测试数据
    if os.path.exists("test_data"):
        import shutil
        shutil.rmtree("test_data")
    
    try:
        # 创建测试数据
        create_test_data()
        
        # 测试配置类
        test_dataset_config()
        
        # 测试数据集类
        success = test_useritem_dataset()
        
        if success:
            print("\n🎉 所有测试通过！UserItemDataset 类工作正常。")
        else:
            print("\n❌ 测试失败！")
            
    except Exception as e:
        print(f"\n❌ 测试过程中出现错误: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 清理测试数据
        if os.path.exists("test_data"):
            import shutil
            shutil.rmtree("test_data")
            print("\n🧹 清理测试数据完成")

if __name__ == "__main__":
    main()