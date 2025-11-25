#!/usr/bin/env python3
# 根据configs目录下的gin配置文件计算模型参数量
import sys
import os
import re

# 解析gin配置文件的函数
def parse_gin_config(config_file):
    config = {
        'num_layers': 4,
        'num_attention_heads': 4,
        'hidden_size': 256,
        'kv_channels': 64
    }
    
    try:
        with open(config_file, 'r') as f:
            content = f.read()
            
            # 使用正则表达式匹配配置值
            num_layers_match = re.search(r'NetworkArgs\.num_layers\s*=\s*(\d+)', content)
            if num_layers_match:
                config['num_layers'] = int(num_layers_match.group(1))
            
            num_heads_match = re.search(r'NetworkArgs\.num_attention_heads\s*=\s*(\d+)', content)
            if num_heads_match:
                config['num_attention_heads'] = int(num_heads_match.group(1))
            
            hidden_size_match = re.search(r'NetworkArgs\.hidden_size\s*=\s*(\d+)', content)
            if hidden_size_match:
                config['hidden_size'] = int(hidden_size_match.group(1))
            
            kv_channels_match = re.search(r'NetworkArgs\.kv_channels\s*=\s*(\d+)', content)
            if kv_channels_match:
                config['kv_channels'] = int(kv_channels_match.group(1))
            
            # 如果没有找到kv_channels，可以从hidden_size和num_attention_heads计算
            if 'kv_channels' not in config or config['kv_channels'] == 64:
                if config['hidden_size'] % config['num_attention_heads'] == 0:
                    config['kv_channels'] = config['hidden_size'] // config['num_attention_heads']
    
    except Exception as e:
        print(f"警告: 无法读取或解析配置文件 {config_file}: {e}")
        print("使用默认配置参数")
    
    return config

# 获取配置文件路径
if len(sys.argv) > 1:
    config_file = sys.argv[1]
else:
    # 默认使用examples/hstu/training/configs/gameid_retrieval.gin
    config_file = os.path.join(os.path.dirname(__file__), 'examples/hstu/training/configs/gameid_retrieval.gin')

# 读取配置参数
config = parse_gin_config(config_file)
num_layers = config['num_layers']
num_attention_heads = config['num_attention_heads']
hidden_size = config['hidden_size']
kv_channels = config['kv_channels']

# 计算每一层HSTULayer的参数量
def calculate_hstu_layer_params():
    # _linear_uvqk: 输入size=hidden_size, 输出size=sum([u, v, q, k]) * num_heads
    # u, v, q, k每个的维度都是kv_channels
    split_arg_list = [kv_channels, kv_channels, kv_channels, kv_channels]
    linear_uvqk_input = hidden_size
    linear_uvqk_output = sum(split_arg_list) * num_attention_heads
    linear_uvqk_params = linear_uvqk_input * linear_uvqk_output  # 假设bias=False
    
    # _linear_proj: 输入size=kv_channels * num_heads, 输出size=hidden_size
    linear_proj_input = kv_channels * num_attention_heads
    linear_proj_output = hidden_size
    linear_proj_params = linear_proj_input * linear_proj_output  # 代码中明确bias=False
    
    # 输入层归一化参数
    input_layernorm_params = 2 * hidden_size  # weight和bias各hidden_size个参数
    
    # 输出层归一化参数
    output_layernorm_params = 2 * (kv_channels * num_attention_heads)  # weight和bias
    
    total_per_layer = linear_uvqk_params + linear_proj_params + input_layernorm_params + output_layernorm_params
    return {
        'linear_uvqk': linear_uvqk_params,
        'linear_proj': linear_proj_params,
        'input_layernorm': input_layernorm_params,
        'output_layernorm': output_layernorm_params,
        'total_per_layer': total_per_layer
    }

# 计算所有层的总参数量
layer_params = calculate_hstu_layer_params()
total_params = layer_params['total_per_layer'] * num_layers

# 打印计算结果
print(f'=== 模型参数量计算（基于配置文件: {config_file}）===')
print('配置参数:')
print(f'  层数: {num_layers}')
print(f'  注意力头数: {num_attention_heads}')
print(f'  隐藏层大小: {hidden_size}')
print(f'  KV通道数: {kv_channels}')
print('\n每HSTULayer层参数量:')
for key, value in layer_params.items():
    if key != 'total_per_layer':
        print(f'  {key}: {value:,}')
print(f'  每层总参数: {layer_params["total_per_layer"]:,}')
print('\n模型总参数量:')
print(f'  所有层参数: {total_params:,}')
print(f'  约等于: {total_params/1e6:.2f}M')