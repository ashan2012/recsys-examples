#!/usr/bin/env python3
"""
用户行为序列分析脚本

功能:
1. 分析history flag下的用户item序列长度分布
2. 分析target flag下用户的target数量分布

输入文件格式: item_id,user_id,timestamp,flag
其中flag包括: target, history

用法:
    python analyze_user_sequences.py --input data.csv --output analysis_results.json
    python analyze_user_sequences.py --input data.csv --print_summary
"""

import argparse
import pandas as pd
import numpy as np
import json
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
import os


class UserSequenceAnalyzer:
    """用户序列分析器"""
    
    def __init__(self, input_file: str):
        """
        初始化分析器
        
        Args:
            input_file: 输入CSV文件路径
        """
        self.input_file = input_file
        self.data = None
        self.history_data = None
        self.target_data = None
        
    def load_data(self) -> pd.DataFrame:
        """加载数据"""
        print(f"正在加载数据文件: {self.input_file}")
        
        if not os.path.exists(self.input_file):
            raise FileNotFoundError(f"文件不存在: {self.input_file}")
            
        # 读取CSV文件
        try:
            self.data = pd.read_csv(self.input_file, names=['item_id', 'user_id', 'timestamp', 'flag'])
            print(f"成功加载 {len(self.data)} 条记录")
            print(f"数据预览:")
            print(self.data.head())
            
            # 数据基本统计
            print(f"\n数据基本信息:")
            print(f"- 总记录数: {len(self.data)}")
            print(f"- 用户数: {self.data['user_id'].nunique()}")
            print(f"- Item数: {self.data['item_id'].nunique()}")
            print(f"- Flag分布:")
            print(self.data['flag'].value_counts())
            
            return self.data
            
        except Exception as e:
            print(f"加载数据失败: {e}")
            raise
            
    def analyze_history_sequences(self) -> Dict:
        """
        分析history序列长度分布
        
        Returns:
            分析结果字典
        """
        print("\n=== 分析History序列长度分布 ===")
        
        # 过滤history数据
        self.history_data = self.data[self.data['flag'] == 'history'].copy()
        print(f"History记录数: {len(self.history_data)}")
        
        if len(self.history_data) == 0:
            print("没有找到history记录")
            return {}
            
        # 按用户聚合，计算每个用户的item序列长度
        user_sequence_lengths = self.history_data.groupby('user_id')['item_id'].count()
        print(f"有history记录的用户数: {len(user_sequence_lengths)}")
        
        # 计算序列长度统计
        sequence_stats = {
            'total_users_with_history': len(user_sequence_lengths),
            'total_history_records': len(self.history_data),
            'min_sequence_length': int(user_sequence_lengths.min()),
            'max_sequence_length': int(user_sequence_lengths.max()),
            'mean_sequence_length': float(user_sequence_lengths.mean()),
            'median_sequence_length': float(user_sequence_lengths.median()),
            'std_sequence_length': float(user_sequence_lengths.std())
        }
        
        print(f"序列长度统计:")
        print(f"- 最短序列: {sequence_stats['min_sequence_length']}")
        print(f"- 最长序列: {sequence_stats['max_sequence_length']}")
        print(f"- 平均长度: {sequence_stats['mean_sequence_length']:.2f}")
        print(f"- 中位数长度: {sequence_stats['median_sequence_length']:.2f}")
        print(f"- 标准差: {sequence_stats['std_sequence_length']:.2f}")
        
        # 计算长度分布阈值
        thresholds = [5, 10, 20, 30, 50, 100, 200]
        distribution = {}
        
        for threshold in thresholds:
            count = (user_sequence_lengths <= threshold).sum()
            percentage = (count / len(user_sequence_lengths)) * 100
            distribution[f'≤{threshold}'] = {
                'user_count': int(count),
                'percentage': round(percentage, 2)
            }
            print(f"长度 ≤{threshold} 的用户: {count} ({percentage:.2f}%)")
            
        # 添加更详细的分位数信息
        percentiles = [10, 25, 50, 75, 90, 95, 99]
        percentile_stats = {}
        for p in percentiles:
            value = np.percentile(user_sequence_lengths, p)
            percentile_stats[f'P{p}'] = int(value)
            print(f"P{p}: {int(value)}")
            
        sequence_stats['distribution'] = distribution
        sequence_stats['percentiles'] = percentile_stats
        
        return sequence_stats
        
    def analyze_target_distribution(self) -> Dict:
        """
        分析target数量分布
        
        Returns:
            分析结果字典
        """
        print("\n=== 分析Target数量分布 ===")
        
        # 过滤target数据
        self.target_data = self.data[self.data['flag'] == 'target'].copy()
        print(f"Target记录数: {len(self.target_data)}")
        
        if len(self.target_data) == 0:
            print("没有找到target记录")
            return {}
            
        # 计算每个用户的target数量
        user_target_counts = self.target_data.groupby('user_id')['item_id'].count()
        print(f"有target记录的用户数: {len(user_target_counts)}")
        
        # Target数量统计
        target_stats = {
            'total_users_with_target': len(user_target_counts),
            'total_target_records': len(self.target_data),
            'min_targets_per_user': int(user_target_counts.min()),
            'max_targets_per_user': int(user_target_counts.max()),
            'mean_targets_per_user': float(user_target_counts.mean()),
            'median_targets_per_user': float(user_target_counts.median()),
            'std_targets_per_user': float(user_target_counts.std())
        }
        
        print(f"每用户Target数量统计:")
        print(f"- 最少: {target_stats['min_targets_per_user']}")
        print(f"- 最多: {target_stats['max_targets_per_user']}")
        print(f"- 平均: {target_stats['mean_targets_per_user']:.2f}")
        print(f"- 中位数: {target_stats['median_targets_per_user']:.2f}")
        print(f"- 标准差: {target_stats['std_targets_per_user']:.2f}")
        
        # 统计特定target数量的用户占比
        target_count_distribution = user_target_counts.value_counts().sort_index()
        distribution = {}
        
        # 重点关注1个和2个target的情况
        users_with_1_target = target_count_distribution.get(1, 0)
        users_with_2_targets = target_count_distribution.get(2, 0)
        users_with_3_targets = target_count_distribution.get(3, 0)
        
        total_users = len(user_target_counts)
        
        distribution['1_target'] = {
            'user_count': int(users_with_1_target),
            'percentage': round((users_with_1_target / total_users) * 100, 2)
        }
        distribution['2_targets'] = {
            'user_count': int(users_with_2_targets),
            'percentage': round((users_with_2_targets / total_users) * 100, 2)
        }
        distribution['3_targets'] = {
            'user_count': int(users_with_3_targets),
            'percentage': round((users_with_3_targets / total_users) * 100, 2)
        }
        
        print(f"\nTarget数量分布:")
        print(f"- 1个target的用户: {users_with_1_target} ({(users_with_1_target/total_users)*100:.2f}%)")
        print(f"- 2个target的用户: {users_with_2_targets} ({(users_with_2_targets/total_users)*100:.2f}%)")
        print(f"- 3个target的用户: {users_with_3_targets} ({(users_with_3_targets/total_users)*100:.2f}%)")
        
        # 显示所有target数量分布
        print(f"\n完整Target数量分布:")
        for target_count, user_count in target_count_distribution.items():
            percentage = (user_count / total_users) * 100
            print(f"- {target_count}个target: {user_count} 用户 ({percentage:.2f}%)")
            
        target_stats['distribution'] = distribution
        target_stats['complete_distribution'] = {
            str(k): {'user_count': int(v), 'percentage': round((v/total_users)*100, 2)}
            for k, v in target_count_distribution.items()
        }
        
        return target_stats
        
    def generate_summary_report(self) -> Dict:
        """生成完整的分析报告"""
        print("\n=== 开始生成分析报告 ===")
        
        # 加载数据
        self.load_data()
        
        # 分析history序列
        history_analysis = self.analyze_history_sequences()
        
        # 分析target分布
        target_analysis = self.analyze_target_distribution()
        
        # 合并结果
        report = {
            'data_summary': {
                'input_file': self.input_file,
                'total_records': len(self.data) if self.data is not None else 0,
                'unique_users': self.data['user_id'].nunique() if self.data is not None else 0,
                'unique_items': self.data['item_id'].nunique() if self.data is not None else 0,
                'flag_distribution': self.data['flag'].value_counts().to_dict() if self.data is not None else {}
            },
            'history_sequence_analysis': history_analysis,
            'target_distribution_analysis': target_analysis,
            'analysis_timestamp': pd.Timestamp.now().isoformat()
        }
        
        return report
        
    def save_report(self, report: Dict, output_file: str):
        """保存分析报告到文件"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        print(f"\n分析报告已保存到: {output_file}")
        
    def print_summary(self, report: Dict):
        """打印分析摘要"""
        print("\n" + "="*60)
        print("                用户序列分析摘要")
        print("="*60)
        
        # 数据概览
        data_summary = report.get('data_summary', {})
        print(f"\n📊 数据概览:")
        print(f"   输入文件: {data_summary.get('input_file', 'N/A')}")
        print(f"   总记录数: {data_summary.get('total_records', 0):,}")
        print(f"   用户数: {data_summary.get('unique_users', 0):,}")
        print(f"   Item数: {data_summary.get('unique_items', 0):,}")
        
        # Flag分布
        flag_dist = data_summary.get('flag_distribution', {})
        print(f"\n🏷️  Flag分布:")
        for flag, count in flag_dist.items():
            print(f"   {flag}: {count:,}")
            
        # History序列分析
        history_analysis = report.get('history_sequence_analysis', {})
        if history_analysis:
            print(f"\n📈 History序列长度分布:")
            print(f"   有history的用户数: {history_analysis.get('total_users_with_history', 0):,}")
            print(f"   平均序列长度: {history_analysis.get('mean_sequence_length', 0):.2f}")
            print(f"   中位数长度: {history_analysis.get('median_sequence_length', 0):.2f}")
            
            print(f"\n   序列长度阈值分布:")
            distribution = history_analysis.get('distribution', {})
            for threshold, stats in distribution.items():
                print(f"   ≤{threshold}: {stats['user_count']:,} 用户 ({stats['percentage']}%)")
                
        # Target分布分析
        target_analysis = report.get('target_distribution_analysis', {})
        if target_analysis:
            print(f"\n🎯 Target数量分布:")
            print(f"   有target的用户数: {target_analysis.get('total_users_with_target', 0):,}")
            print(f"   平均每用户target数: {target_analysis.get('mean_targets_per_user', 0):.2f}")
            
            print(f"\n   重点关注分布:")
            distribution = target_analysis.get('distribution', {})
            for target_type, stats in distribution.items():
                print(f"   {target_type}: {stats['user_count']:,} 用户 ({stats['percentage']}%)")
                
        print("\n" + "="*60)


def create_sample_data(output_file: str = "sample_user_data.csv", num_users: int = 1000, num_items: int = 100):
    """创建示例数据用于测试"""
    print(f"创建示例数据...")
    
    np.random.seed(42)
    
    data_records = []
    
    for user_id in range(1, num_users + 1):
        # 为每个用户生成history序列
        num_history = np.random.poisson(lam=10) + 1  # 平均10个history，但至少1个
        history_items = np.random.choice(num_items, size=num_history, replace=False)
        
        # 添加history记录（按时间排序）
        for i, item_id in enumerate(history_items):
            data_records.append({
                'item_id': item_id + 1,  # Item ID从1开始
                'user_id': user_id,
                'timestamp': i + 1,
                'flag': 'history'
            })
            
        # 为每个用户生成target
        num_targets = np.random.choice([1, 2, 3], p=[0.7, 0.25, 0.05])  # 70%用户1个target，25%用户2个target，5%用户3个target
        target_items = np.random.choice(num_items, size=num_targets, replace=False)
        
        for item_id in target_items:
            data_records.append({
                'item_id': item_id + 1,
                'user_id': user_id,
                'timestamp': num_history + 1,
                'flag': 'target'
            })
            
    # 创建DataFrame并保存
    df = pd.DataFrame(data_records)
    df.to_csv(output_file, index=False, header=False)
    
    print(f"示例数据已创建: {output_file}")
    print(f"总记录数: {len(df):,}")
    print(f"用户数: {df['user_id'].nunique():,}")
    print(f"Item数: {df['item_id'].nunique():,}")
    print(f"Flag分布:")
    print(df['flag'].value_counts())


def main():
    parser = argparse.ArgumentParser(description="用户行为序列分析工具")
    parser.add_argument("--input", type=str, help="输入CSV文件路径")
    parser.add_argument("--output", type=str, help="输出JSON报告文件路径")
    parser.add_argument("--print_summary", action="store_true", help="打印分析摘要")
    parser.add_argument("--create_sample", action="store_true", help="创建示例数据文件")
    parser.add_argument("--sample_users", type=int, default=1000, help="示例数据的用户数")
    parser.add_argument("--sample_items", type=int, default=100, help="示例数据的Item数")
    parser.add_argument("--sample_output", type=str, default="sample_user_data.csv", help="示例数据输出文件")
    
    args = parser.parse_args()
    
    try:
        # 创建示例数据
        if args.create_sample:
            create_sample_data(args.sample_output, args.sample_users, args.sample_items)
            print("示例数据创建完成，使用 --input 参数指定该文件进行分析")
            return
            
        if not args.input:
            parser.error("请指定输入文件 (--input)")
            
        # 创建分析器
        analyzer = UserSequenceAnalyzer(args.input)
        
        # 生成报告
        report = analyzer.generate_summary_report()
        
        # 打印摘要
        if args.print_summary:
            analyzer.print_summary(report)
            
        # 保存报告
        if args.output:
            analyzer.save_report(report, args.output)
        else:
            # 默认输出文件名
            output_file = os.path.splitext(args.input)[0] + "_analysis.json"
            analyzer.save_report(report, output_file)
            
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()