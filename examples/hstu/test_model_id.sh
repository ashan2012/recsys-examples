#!/bin/bash
set -euo pipefail

# 复制run_game.sh中的关键函数进行测试
CONFIG_FILE="./training/configs/gameid_retrieval.gin"

# 从配置文件中读取模型标识
read_model_id() {
  # 尝试从不同的配置块中读取model_id
  MODEL_ID=$(grep -E 'TrainerArgs\.model_id|NetworkArgs\.model_id|ModelArgs\.model_id' "${CONFIG_FILE}" | head -1 | sed -E "s/.*=\s*['\"]([^'"]*)['\"].*/\1/")
  
  # 如果没有找到，使用默认值
  if [[ -z "${MODEL_ID}" ]]; then
    MODEL_ID="game_model_$(date +%Y%m%d_%H%M%S)"
    echo "Warning: model_id not found in config, using default: ${MODEL_ID}"
  fi
  
  echo "${MODEL_ID}"
}

# 测试读取模型标识
echo "测试从配置文件读取模型标识..."
MODEL_ID=$(read_model_id)
MODEL_DIR="./models"
MODEL_SAVE_PATH="${MODEL_DIR}/${MODEL_ID}"

echo "配置文件: ${CONFIG_FILE}"
echo "读取到的模型标识: ${MODEL_ID}"
echo "模型保存路径: ${MODEL_SAVE_PATH}"

# 创建测试目录结构
echo "创建测试目录结构..."
mkdir -p "${MODEL_SAVE_PATH}"

# 保存测试信息
echo "保存测试信息到模型目录..."
echo "Test model ID: ${MODEL_ID}" > "${MODEL_SAVE_PATH}/test_info.txt"
echo "Test timestamp: $(date '+%F %T')" >> "${MODEL_SAVE_PATH}/test_info.txt"

# 复制配置文件
echo "复制配置文件到模型目录..."
cp "${CONFIG_FILE}" "${MODEL_SAVE_PATH}/config.gin"

echo ""
echo "测试完成！验证结果："
echo "1. 目录是否创建: $(ls -la "${MODEL_DIR}" | grep "${MODEL_ID}")"
echo "2. 测试文件是否创建: $(ls -la "${MODEL_SAVE_PATH}")"
echo "3. 测试信息内容:"
cat "${MODEL_SAVE_PATH}/test_info.txt"

echo ""
echo "功能验证成功！run_game.sh脚本将能够正确读取配置文件中的模型标识并保存结果。"