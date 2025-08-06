#!/bin/bash
"""
启动Baton Rouge港口客户端
在终端4运行此脚本
"""

echo "🏭 启动Baton Rouge港口客户端"
echo "=========================================="
echo "港口ID: 2"
echo "港口名称: baton_rouge"
echo "服务器: localhost:8888"
echo "=========================================="

cd "$(dirname "$0")/../../.."

# 启动Baton Rouge港口客户端
python src/federated/distributed_port_client.py \
    --port_id 2 \
    --port_name baton_rouge \
    --server_host localhost \
    --server_port 8888 \
    --topology 3x3 \
    --rounds 10 \
    --episodes 3

echo "🔒 Baton Rouge港口客户端已停止"