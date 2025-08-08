#!/usr/bin/env bash
# Smoke Test for GAT-FedPPO Release 2025-08-08-rc1
echo "🧪 开始冒烟测试..."
echo "📊 测试所有港口 (400样本×3种子)"
python3 config/nightly_ci.py --ports gulfport,baton_rouge,new_orleans,south_louisiana --samples 400 --seeds 42,123,2025 --lb-slack 0.04
echo "✅ 冒烟测试完成"
