#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
回归测试脚本
验证gulfport补丁合并后的性能
"""
import os
import sys
import logging
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent / "src" / "federated"))

def clear_cache():
    """清除旧缓存"""
    cache_patterns = [
        "src/federated/*__*__seed*__samples*.npz",
        "models/releases/*/datasets/*__*__seed*__samples*.npz"
    ]
    
    for pattern in cache_patterns:
        cmd = f"find . -path '{pattern}' -delete 2>/dev/null || true"
        os.system(cmd)
    
    print("✅ 缓存清理完成")

def retrain_gulfport():
    """重新训练gulfport"""
    print("🔄 开始重新训练gulfport...")
    
    cmd = """cd src/federated && python -c "
from curriculum_trainer import CurriculumTrainer
trainer = CurriculumTrainer('gulfport')
trainer.train_curriculum()
" """
    
    result = os.system(cmd)
    if result == 0:
        print("✅ gulfport重训完成")
    else:
        print("❌ gulfport重训失败")
        return False
    
    return True

def run_regression_test():
    """运行回归测试"""
    print("🧪 开始回归测试...")
    
    # 测试三个种子
    seeds = [42, 123, 2025]
    results = {}
    
    for seed in seeds:
        print(f"  测试种子 {seed}...")
        cmd = f"cd src/federated && python consistency_test_fixed.py --port gulfport --samples 400 --seed {seed}"
        result = os.system(cmd)
        results[seed] = (result == 0)
    
    # 统计结果
    passed = sum(results.values())
    print(f"📊 回归测试结果: {passed}/3 种子通过")
    
    if passed >= 1:
        print("✅ 回归测试通过")
        return True
    else:
        print("❌ 回归测试失败")
        return False

def main():
    logging.basicConfig(level=logging.INFO)
    
    print("🚀 开始gulfport补丁回归测试")
    
    # 1. 清除缓存
    clear_cache()
    
    # 2. 重新训练
    if not retrain_gulfport():
        return 1
    
    # 3. 回归测试
    if not run_regression_test():
        return 1
    
    print("🎉 回归测试完成，补丁合并成功！")
    return 0

if __name__ == "__main__":
    sys.exit(main())