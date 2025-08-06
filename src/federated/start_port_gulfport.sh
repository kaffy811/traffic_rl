#!/bin/bash
"""
启动Gulfport港口客户端
在终端5运行此脚本
"""

echo "🏭 启动Gulfport港口客户端"
echo "=========================================="
echo "港口ID: 3"
echo "港口名称: gulfport"
echo "服务器: localhost:8888"
echo "=========================================="

cd "$(dirname "$0")/../../.."

# 启动Gulfport港口客户端
python src/federated/distributed_port_client.py \
    --port_id 3 \
    --port_name gulfport \
    --server_host localhost \
    --server_port 8888 \
    --topology 3x3 \
    --rounds 10 \
    --episodes 3

echo "🔒 Gulfport港口客户端已停止"