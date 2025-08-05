#!/bin/bash
# 海事GAT-PPO联邦学习服务器启动脚本

echo "🚢 启动海事GAT-PPO联邦学习服务器"
echo "================================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 未找到，请安装Python3"
    exit 1
fi

# 检查配置文件
CONFIG_FILE="config/maritime_fedml_config.yaml"
if [ ! -f "$CONFIG_FILE" ]; then
    echo "❌ 配置文件不存在: $CONFIG_FILE"
    exit 1
fi

# 创建日志目录
mkdir -p logs

# 设置环境变量
export FEDML_CONFIG_PATH="$CONFIG_FILE"
export FEDML_ROLE="server"
export FEDML_TRAINING_TYPE="cross_silo"
export FEDML_BACKEND="MQTT_S3"
export PYTHONPATH="$PYTHONPATH:../../..:../../../FedML/python"

echo "📋 配置文件: $CONFIG_FILE"
echo "🔧 启动服务器..."

# 启动服务器 (使用FedML one-line API)
python3 maritime_server.py

echo "✅ 服务器已停止"
