#!/bin/bash
# 轻量健康检查脚本 - 检查seed=2025的BR中级阶段表现

echo "=== BR中级阶段 Seed=2025 健康检查 ==="
echo "时间: $(date)"
echo "目标: 快速检查seed=2025是否≥49.0%"

# 运行快速检查
echo "运行快速检查: seed=2025, n=400, --no-cache"
python scripts/nightly_ci.py --ports baton_rouge --samples 400 --seeds 2025 --no-cache

# 获取最新结果
LATEST_JSON=$(ls -t models/releases/$(date +%Y-%m-%d)/consistency_baton_rouge_*.json | head -1)

if [ -n "$LATEST_JSON" ]; then
    echo ""
    echo "=== 检查结果 ==="
    echo "结果文件: $LATEST_JSON"
    
    # 提取中级阶段的结果
    WR=$(jq -r '.stages[] | select(.stage=="中级阶段") | .win_rate' "$LATEST_JSON")
    LB=$(jq -r '.stages[] | select(.stage=="中级阶段") | .wilson_lb' "$LATEST_JSON")
    THR=$(jq -r '.stages[] | select(.stage=="中级阶段") | .threshold' "$LATEST_JSON")
    PASS=$(jq -r '.stages[] | select(.stage=="中级阶段") | .pass' "$LATEST_JSON")
    
    echo "胜率: ${WR} (${WR:0:1}.${WR:1:4})"
    echo "Wilson下界: ${LB} (${LB:0:1}.${LB:1:4})"
    echo "阈值: ${THR} (${THR:0:1}.${THR:1:4})"
    echo "通过: ${PASS}"
    
    # 判断建议
    if (( $(echo "$WR >= 0.49" | bc -l) )); then
        echo "✅ 建议: 可以继续阈值回退"
        if (( $(echo "$LB >= $(echo "$THR - 0.02" | bc -l)" | bc -l) )); then
            echo "✅ Wilson下界也满足要求，建议继续回退"
        else
            echo "⚠️  Wilson下界略低，建议谨慎回退"
        fi
    else
        echo "❌ 建议: 暂停阈值回退，先观望"
        echo "当前胜率 ${WR} < 49.0%，需要进一步改善"
    fi
else
    echo "❌ 未找到测试结果文件"
fi

echo ""
echo "=== 健康检查完成 ===" 