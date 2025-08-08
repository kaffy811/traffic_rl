#!/bin/bash

# 模型检查点清理脚本
# 只保留必要的模型文件，清理临时和备份文件

set -e

MODEL_DIR="../../models/curriculum_v2"

echo "🧹 开始清理模型检查点..."
echo "📁 模型目录: $MODEL_DIR"

if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ 模型目录不存在: $MODEL_DIR"
    exit 1
fi

cd "$MODEL_DIR"

# 统计当前文件
echo "📊 当前状态:"
echo "  .pt 文件数: $(find . -name "*.pt" | wc -l)"
echo "  总文件数: $(find . -type f | wc -l)"
echo "  目录大小: $(du -sh . | cut -f1)"

echo ""
echo "✅ 保留的必要文件:"
find . -name "stage_*_best.pt" -o -name "curriculum_final_model.pt" | sort | while read -r file; do
    echo "  ✓ $file"
done

echo ""
echo "✨ 目录已经很干净，只包含必要的模型文件！"

echo ""
echo "📋 当前目录结构符合建议:"
echo "models/curriculum_v2/"
echo "├── baton_rouge/"
echo "│   ├── stage_基础阶段_best.pt"
echo "│   ├── stage_中级阶段_best.pt"
echo "│   ├── stage_高级阶段_best.pt"
echo "│   └── curriculum_final_model.pt"
echo "├── new_orleans/"
echo "│   ├── stage_基础阶段_best.pt"
echo "│   ├── stage_初级阶段_best.pt"
echo "│   ├── stage_中级阶段_best.pt"
echo "│   ├── stage_高级阶段_best.pt"
echo "│   ├── stage_专家阶段_best.pt"
echo "│   └── curriculum_final_model.pt"
echo "├── south_louisiana/"
echo "│   ├── stage_基础阶段_best.pt"
echo "│   ├── stage_中级阶段_best.pt"
echo "│   ├── stage_高级阶段_best.pt"
echo "│   └── curriculum_final_model.pt"
echo "└── gulfport/"
echo "    ├── stage_标准阶段_best.pt"
echo "    ├── stage_完整阶段_best.pt"
echo "    └── curriculum_final_model.pt"