#!/bin/bash
"""
启动South Louisiana港口客户端
在终端3运行此脚本
"""

echo "🏭 启动South Louisiana港口客户端"
echo "=========================================="
echo "港口ID: 1"
echo "港口名称: south_louisiana"
echo "服务器: localhost:8888"
echo "=========================================="

cd "$(dirname "$0")/../../.."

# 启动South Louisiana港口客户端
python src/federated/distributed_port_client.py \
    --port_id 1 \
    --port_name south_louisiana \
    --server_host localhost \
    --server_port 8888 \
    --topology 3x3 \
    --rounds 10 \
    --episodes 3

echo "🔒 South Louisiana港口客户端已停止"