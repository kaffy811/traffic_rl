#!/bin/bash
# 后晚阈值回退脚本 - BR中级阶段阈值 0.500 (night 3, final rollback)

echo "=== 后晚阈值回退执行 ==="
echo "时间: $(date)"
echo "目标: BR中级阶段阈值 0.497 → 0.500"

# 检查当前状态
echo "当前阈值配置:"
grep -A 2 "中级阶段:" configs/thresholds.yaml

# 执行阈值调整
if [[ "$OSTYPE" == "darwin"* ]]; then
    # macOS
    sed -i '' 's/中级阶段: *0\.497/中级阶段: 0.500/' configs/thresholds.yaml
else
    # Linux
    sed -i 's/中级阶段: *0\.497/中级阶段: 0.500/' configs/thresholds.yaml
fi

echo "阈值调整完成:"
grep -A 2 "中级阶段:" configs/thresholds.yaml

# Git操作
git add configs/thresholds.yaml
git commit -m "revert: BR 中级阶段阈值回到 0.500 (night 3)"
git push origin master

echo "✅ 后晚阈值回退完成: 0.497 → 0.500"
echo "建议: 运行夜测验证最终阈值效果"
echo "命令: python scripts/nightly_ci.py --ports baton_rouge --samples 800 --seeds 42,123,2025 --no-cache"
echo ""
echo "⚠️  注意: 如果seed=2025仍<49.0%且LB不足，建议回滚到0.497"
echo "回滚命令: sed -i 's/中级阶段: *0\.500/中级阶段: 0.497/' configs/thresholds.yaml" 