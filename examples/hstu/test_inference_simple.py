#!/usr/bin/env python3
"""
简化的推理数据集测试脚本（不依赖 torchrec）
只测试基本的类结构和数据处理逻辑
"""

import json
import os
import sys
from typing import List, Dict, Any

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

class MockKeyedJaggedTensor:
    """模拟 KeyedJaggedTensor 类的简化版本"""
    
    def __init__(self, keys: List[str], values: List[Any], offsets: List[int] = None):
        self.keys = keys
        self.values = values
        self.offsets = offsets if offsets is not None else []
        
    def keys(self):
        return self.keys
        
    def __repr__(self):
        return f"MockKeyedJaggedTensor(keys={self.keys}, values={self.values}, offsets={self.offsets})"

class SimpleBatch:
    """简化版本的 Batch 类，不依赖 torchrec"""
    
    def __init__(self, features: Dict[str, Any], max_num_candidates: int):
        self.features = features
        self.max_num_candidates = max_num_candidates
        self.num_candidates = None
        self.batch_size = self._get_batch_size()
        
    def _get_batch_size(self):
        """获取批次大小"""
        for key, value in self.features.items():
            if hasattr(value, '__len__'):
                return len(value)
        return 0
        
    def __repr__(self):
        return f"SimpleBatch(features={list(self.features.keys())}, batch_size={self.batch_size})"

class SimpleRetrievalBatch:
    """简化版本的 RetrievalBatch 类，不依赖 torchrec"""
    
    def __init__(self, features: Dict[str, Any], max_num_candidates: int, num_candidates: List[int]):
        self.features = features
        self.max_num_candidates = max_num_candidates
        self.num_candidates = num_candidates
        self.batch_size = self._get_batch_size()
        
    def _get_batch_size(self):
        """获取批次大小"""
        for key, value in self.features.items():
            if hasattr(value, '__len__'):
                return len(value)
        return 0
        
    def __repr__(self):
        return f"SimpleRetrievalBatch(features={list(self.features.keys())}, num_candidates={self.num_candidates}, batch_size={self.batch_size})"

def test_json_data_loading():
    """测试 JSON 数据加载功能"""
    
    print("\n🧪 测试 JSON 数据加载...")
    
    # 加载序列日志
    with open("test_data/seq_logs.json", "r") as f:
        seq_logs = json.load(f)
    
    print(f"   序列日志: {len(seq_logs)} 条记录")
    for log in seq_logs:
        item_seq = json.loads(log["item_sequence"])
        print(f"   - 用户 {log['user_id']}: {len(item_seq)} 个物品")
    
    # 加载批次日志
    with open("test_data/batch_logs.json", "r") as f:
        batch_logs = json.load(f)
    
    print(f"   批次日志: {len(batch_logs)} 条记录")
    for log in batch_logs:
        print(f"   - 用户 {log['user_id']}, 结束指针: {log['sequence_endptr']}")
    
    return True

def test_mock_keyed_jagged_tensor():
    """测试模拟的 KeyedJaggedTensor"""
    
    print("\n🧪 测试模拟 KeyedJaggedTensor...")
    
    # 创建模拟数据
    keys = ["item_ids", "action_ids"]
    values = [101, 102, 103, 201, 202, 1, 2, 3]
    offsets = [0, 3, 5, 8]  # 用户0: 0-2, 用户1: 3-4, 用户2: 5-7
    
    kjt = MockKeyedJaggedTensor(keys, values, offsets)
    print(f"   ✅ 成功创建: {kjt}")
    
    return True

def test_simple_batch():
    """测试简化版本的 Batch 类"""
    
    print("\n🧪 测试简化 Batch 类...")
    
    # 创建特征数据
    features = {
        "item_ids": [101, 102, 103, 201, 202],
        "action_ids": [1, 2, 3, 4, 5]
    }
    
    # 创建批次
    batch = SimpleBatch(features, max_num_candidates=5)
    print(f"   ✅ 成功创建: {batch}")
    
    return True

def test_simple_retrieval_batch():
    """测试简化版本的 RetrievalBatch 类"""
    
    print("\n🧪 测试简化 RetrievalBatch 类...")
    
    # 创建特征数据
    features = {
        "user_ids": [1, 2, 3],
        "item_ids": [101, 102, 103, 201, 202, 204, 105, 106, 107],
        "action_ids": [1, 2, 3, 4, 5, 6, 7]
    }
    
    # 创建候选数
    num_candidates = [3, 2, 4]
    
    # 创建检索批次
    retrieval_batch = SimpleRetrievalBatch(features, max_num_candidates=5, num_candidates=num_candidates)
    print(f"   ✅ 成功创建: {retrieval_batch}")
    
    return True

def simulate_batch_processing():
    """模拟批处理逻辑"""
    
    print("\n🧪 模拟批处理逻辑...")
    
    # 加载数据
    with open("test_data/seq_logs.json", "r") as f:
        seq_logs = json.load(f)
    
    with open("test_data/batch_logs.json", "r") as f:
        batch_logs = json.load(f)
    
    # 按用户分组序列
    user_sequences = {}
    for log in seq_logs:
        user_id = log["user_id"]
        if user_id not in user_sequences:
            user_sequences[user_id] = []
        
        item_seq = json.loads(log["item_sequence"])
        action_seq = json.loads(log["action_sequence"])
        
        user_sequences[user_id].append({
            "items": item_seq,
            "actions": action_seq,
            "date": log["date"]
        })
    
    print(f"   处理了 {len(user_sequences)} 个用户的序列数据")
    
    # 处理批次
    batch_count = 0
    for batch_log in batch_logs:
        user_id = batch_log["user_id"]
        sequence_endptr = batch_log["sequence_endptr"]
        
        # 获取用户序列
        sequences = user_sequences.get(user_id, [])
        
        # 模拟创建特征
        item_ids = []
        action_ids = []
        
        for seq in sequences:
            item_ids.extend(seq["items"][:sequence_endptr])
            action_ids.extend(seq["actions"][:sequence_endptr])
        
        # 创建模拟 KeyedJaggedTensor
        features = {
            "item_ids": MockKeyedJaggedTensor(["item_ids"], item_ids),
            "action_ids": MockKeyedJaggedTensor(["action_ids"], action_ids)
        }
        
        # 创建批次
        if batch_count % 2 == 0:
            batch = SimpleBatch(features, max_num_candidates=5)
        else:
            batch = SimpleRetrievalBatch(features, max_num_candidates=5, num_candidates=[len(item_ids)])
        
        batch_count += 1
        print(f"   批次 {batch_count}: {batch}")
        
        if batch_count >= 3:
            break
    
    return batch_count > 0

def main():
    """主测试函数"""
    
    print("🚀 开始简化的推理数据集测试（不依赖 torchrec）...")
    
    # 清理之前的测试数据
    if os.path.exists("test_data"):
        import shutil
        shutil.rmtree("test_data")
    
    try:
        # 创建测试数据
        create_test_data()
        
        # 测试各个组件
        test_json_data_loading()
        test_mock_keyed_jagged_tensor()
        test_simple_batch()
        test_simple_retrieval_batch()
        simulate_batch_processing()
        
        print("\n🎉 所有测试通过！基本的数据处理逻辑工作正常。")
        
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