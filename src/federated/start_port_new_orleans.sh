#!/bin/bash
"""
启动New Orleans港口客户端
在终端2运行此脚本
"""

echo "🏭 启动New Orleans港口客户端"
echo "=========================================="
echo "港口ID: 0"
echo "港口名称: new_orleans"
echo "服务器: localhost:8888"
echo "=========================================="

cd "$(dirname "$0")/../../.."

# 启动New Orleans港口客户端
python src/federated/distributed_port_client.py \
    --port_id 0 \
    --port_name new_orleans \
    --server_host localhost \
    --server_port 8888 \
    --topology 3x3 \
    --rounds 10 \
    --episodes 3

echo "🔒 New Orleans港口客户端已停止"