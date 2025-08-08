#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析四港口基线测试结果，生成论文表格
"""
import json
import numpy as np
from pathlib import Path

def wilson_lb(k, n, z=1.96):
    """计算Wilson置信区间下界"""
    if n == 0: return 0.0
    p = k / n
    denom = 1 + z*z/n
    centre = p + z*z/(2*n)
    margin = z*sqrt((p*(1-p) + z*z/(4*n))/n)
    lb = (centre - margin)/denom
    return max(0.0, min(1.0, lb))

def analyze_results():
    """分析测试结果"""
    # 加载结果
    with open('logs/nightly/monitoring_status.json', 'r') as f:
        data = json.load(f)
    
    print("=" * 80)
    print("表1: 四港口各阶段性能统计 (800样本, 3种子)")
    print("=" * 80)
    print(f"{'港口':<15} {'阶段':<12} {'胜率均值±std':<15} {'Wilson下界':<12} {'阈值':<8} {'通过':<6}")
    print("-" * 80)
    
    # 分析每个港口
    for port_name, port_data in data['ports'].items():
        seeds_data = port_data['seeds']
        
        # 按阶段聚合数据
        stage_stats = {}
        for seed_result in seeds_data:
            for stage in seed_result['data']['stages']:
                stage_name = stage['stage']
                if stage_name not in stage_stats:
                    stage_stats[stage_name] = {
                        'win_rates': [],
                        'threshold': stage['threshold']
                    }
                stage_stats[stage_name]['win_rates'].append(stage['win_rate'])
        
        # 计算统计量
        for stage_name, stats in stage_stats.items():
            win_rates = np.array(stats['win_rates'])
            mean_wr = np.mean(win_rates)
            std_wr = np.std(win_rates, ddof=1) if len(win_rates) > 1 else 0.0
            threshold = stats['threshold']
            
            # 计算最坏情况的Wilson下界
            worst_wr = np.min(win_rates)
            worst_k = int(worst_wr * 800)
            worst_wilson_lb = wilson_lb(worst_k, 800)
            
            # 判断是否通过 (Wilson下界 >= 阈值 - 3pp)
            passes = worst_wilson_lb >= (threshold - 0.03)
            pass_mark = "✅" if passes else "❌"
            
            print(f"{port_name:<15} {stage_name:<12} {mean_wr:.1%}±{std_wr:.1%}    {worst_wilson_lb:.1%}      {threshold:.1%}    {pass_mark}")
        
        print("-" * 80)
    
    print("\n" + "=" * 80)
    print("表1数据说明:")
    print("- 胜率均值±std: 三种子的平均胜率和标准差")
    print("- Wilson下界: 最坏种子的Wilson 95%置信区间下界")
    print("- 通过标准: Wilson下界 ≥ 阈值-3pp (安全边界)")
    print("- 样本数: 每种子800样本")
    print("- 种子: [42, 123, 2025]")
    print("=" * 80)

if __name__ == "__main__":
    from math import sqrt
    analyze_results()