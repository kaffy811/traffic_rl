#!/usr/bin/env bash
# Smoke Test for GAT-FedPPO Release 2025-08-08-rc1

# 确保路径独立性 - 无论在哪个目录执行都能找到源码
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

echo "🧪 开始冒烟测试..."
echo "📍 工作目录: $(pwd)"
echo "📊 测试所有港口 (400样本×3种子)"
python3 scripts/nightly_ci.py --ports gulfport,baton_rouge,new_orleans,south_louisiana --samples 400 --seeds 42,123,2025 --lb-slack 0.04
echo "✅ 冒烟测试完成"
