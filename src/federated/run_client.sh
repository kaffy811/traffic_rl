#!/bin/bash
# 海事GAT-PPO联邦学习客户端启动脚本

# 检查参数
if [ $# -lt 1 ]; then
    echo "用法: $0 <client_rank> [run_id]"
    echo "示例: $0 1 maritime_test_001"
    echo ""
    echo "客户端rank映射:"
    echo "  1 -> new_orleans (新奥尔良港)"
    echo "  2 -> south_louisiana (南路易斯安那港)"
    echo "  3 -> baton_rouge (巴吞鲁日港)"
    echo "  4 -> gulfport (海湾港)"
    exit 1
fi

CLIENT_RANK=$1
RUN_ID=${2:-"maritime_test_$(date +%Y%m%d_%H%M%S)"}

echo "🚢 启动海事GAT-PPO联邦学习客户端"
echo "================================="
echo "🆔 客户端Rank: $CLIENT_RANK"
echo "🔖 运行ID: $RUN_ID"

# 检查rank范围
if [ "$CLIENT_RANK" -lt 1 ] || [ "$CLIENT_RANK" -gt 4 ]; then
    echo "❌ 客户端rank必须在1-4之间"
    exit 1
fi

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
export FEDML_ROLE="client"
export FEDML_CLIENT_RANK="$CLIENT_RANK"
export FEDML_TRAINING_TYPE="cross_silo"
export FEDML_BACKEND="MQTT_S3"
export PYTHONPATH="$PYTHONPATH:../../..:../../../FedML/python"

# 映射rank到节点名称
case $CLIENT_RANK in
    1) NODE_NAME="new_orleans" ;;
    2) NODE_NAME="south_louisiana" ;;
    3) NODE_NAME="baton_rouge" ;;
    4) NODE_NAME="gulfport" ;;
esac

echo "📍 节点名称: $NODE_NAME"
echo "📋 配置文件: $CONFIG_FILE"
echo "🔧 启动客户端..."

# 设置客户端rank环境变量
export FEDML_CLIENT_RANK="$CLIENT_RANK"

# 启动客户端 (使用FedML one-line API)
python3 maritime_client.py

echo "✅ 客户端已停止"
