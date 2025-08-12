#!/bin/bash
# 联邦学习启动示例脚本

set -e

echo "🚀 联邦学习启动示例"
echo "===================="

# 设置环境变量
export FL_TAG=fl_$(date +%Y%m%d_%H%M%S)
echo "FL_TAG: $FL_TAG"

# 1. 检查准备工作
echo "📋 检查准备工作..."
if [ ! -f "models/fl/$FL_TAG/global_init.pt" ]; then
    echo "❌ 全局初始化模型不存在，请先运行初始化脚本"
    exit 1
fi

# 2. 启动联邦学习
echo "🚀 启动联邦学习..."
python scripts/fl_train.py \
    --algo fedavg \
    --rounds 30 \
    --clients "gulfport,new_orleans,south_louisiana,baton_rouge" \
    --global-init "models/fl/$FL_TAG/global_init.pt" \
    --save-dir "models/fl/$FL_TAG" \
    --local-episodes 8 \
    --lr-schedule "0:3e-4,5:1.5e-4,25:7.5e-5" \
    --batch-size 64 \
    --ppo-epochs 4 \
    --entropy-coef 0.01 \
    --log-file "logs/fl/$FL_TAG/train.log"

echo "✅ 联邦学习启动完成！"
echo ""
echo "📊 后续操作建议："
echo "1. 每5轮运行评测回调："
echo "   python scripts/fl_eval_callback.py --round 5 --no-cache"
echo "   python scripts/fl_eval_callback.py --round 10 --no-cache"
echo "   python scripts/fl_eval_callback.py --round 15 --no-cache"
echo "   python scripts/fl_eval_callback.py --round 20 --no-cache"
echo "   python scripts/fl_eval_callback.py --round 25 --no-cache"
echo "   python scripts/fl_eval_callback.py --round 30 --no-cache"
echo ""
echo "2. 监控训练日志："
echo "   tail -f logs/fl/$FL_TAG/train.log"
echo ""
echo "3. 检查模型保存："
echo "   ls -la models/fl/$FL_TAG/"
echo ""
echo "4. 如果触发早停，回滚到上一最优权重："
echo "   cp models/fl/$FL_TAG/global_round_XX.pt models/fl/$FL_TAG/global_best.pt" 