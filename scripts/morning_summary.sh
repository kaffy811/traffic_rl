#!/usr/bin/env bash
# 夜测后一键汇总脚本
# 生成完整的晨间状态报告

set -euo pipefail

echo "🌅 生成晨间汇总报告..."
echo "📊 时间: $(date)"
echo ""

# 创建汇总文件
SUMMARY_FILE="logs/nightly/morning_dashboard_$(date +%Y%m%d_%H%M%S).txt"
mkdir -p logs/nightly

{
    echo "================= 晨间状态汇总 ================="
    echo "生成时间: $(date)"
    echo "RC版本: 2025-08-08-rc1"
    echo ""
    
    echo "================= 夜测结果 ================="
    python3 scripts/monitoring_dashboard.py
    echo ""
    
    echo "================= 阈值基准 ================="
    cat releases/2025-08-08-rc1/config/thresholds.txt
    echo ""
    
    echo "================= 最新测试文件 ================="
    echo "最近5个测试结果:"
    ls -lt models/releases/2025-08-08/consistency_*.json | head -5
    echo ""
    
    echo "================= Git状态 ================="
    echo "当前分支: $(git branch --show-current)"
    echo "最新提交: $(git log -1 --oneline)"
    echo "标签: $(git tag --list | tail -3)"
    echo ""
    
    echo "================= 决策建议 ================="
    echo "1. 检查上述夜测结果"
    echo "2. 如果GP+BR稳定 → 宣布RC1为预部署版本"
    echo "3. NO状态评估:"
    echo "   - LB ≥ 阈值-2pp → 随版本发布为'Degraded allowed'"
    echo "   - LB < 阈值-2pp → 执行中级阶段补针"
    echo ""
    echo "================= 快速命令 ================="
    echo "# 部署RC版本:"
    echo "cd deploy/2025-08-08-rc1 && docker-compose up -d"
    echo ""
    echo "# NO补针(如需要):"
    echo "python3 fine_tune_stages.py --port new_orleans --stage 中级阶段 --target-improvement 0.03 --max-episodes 90"
    echo ""
    echo "# 验证完整性:"
    echo "shasum -c releases/2025-08-08-rc1/manifest.sha256"
    echo ""
    
} > "$SUMMARY_FILE"

# 同时输出到控制台
cat "$SUMMARY_FILE"

echo "✅ 汇总报告已保存: $SUMMARY_FILE"
echo "📧 可直接复制内容发送状态报告"