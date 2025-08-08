#!/usr/bin/env bash
# New Orleans 紧急补针预案
# 仅在晨间评估后需要时执行

set -euo pipefail

echo "🚨 New Orleans 紧急补针开始..."
echo "📍 目标: 中级阶段 +3pp 提升"
echo "⏰ 开始时间: $(date)"

# 备份当前模型
echo "💾 备份当前模型..."
cp models/curriculum_v2/new_orleans/stage_中级阶段_best.pt \
   models/curriculum_v2/new_orleans/stage_中级阶段_best.pt.backup_$(date +%Y%m%d_%H%M%S)

# 执行补针训练
echo "🎯 执行补针训练..."
python3 fine_tune_stages.py \
    --port new_orleans \
    --stage 中级阶段 \
    --target-improvement 0.03 \
    --max-episodes 90 \
    --device cpu

# 立即验证
echo "🧪 立即验证结果..."
python3 scripts/nightly_ci.py \
    --ports new_orleans \
    --samples 400 \
    --seeds 42,123,2025 \
    --lb-slack 0.04

echo "✅ New Orleans 补针完成: $(date)"
echo "📊 请检查最新的测试结果"