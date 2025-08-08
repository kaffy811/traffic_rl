#!/usr/bin/env python3
"""
GAT-PPO单港训练结果汇总
"""

import os
import json
from pathlib import Path

def summarize_results():
    """汇总GAT-PPO训练结果"""
    
    print("="*60)
    print("GAT-PPO单港训练结果汇总")
    print("="*60)
    
    # 检查模型保存目录
    models_dir = Path("../../models/single_port")
    
    if not models_dir.exists():
        print("未找到模型保存目录")
        return
    
    ports = ['gulfport', 'baton_rouge', 'new_orleans', 'south_louisiana']
    
    results = {}
    
    for port in ports:
        port_dir = models_dir / port
        if port_dir.exists():
            print(f"\n{port.upper()} 港口:")
            print("-" * 40)
            
            # 统计模型文件
            model_files = list(port_dir.glob("*.pt"))
            print(f"  保存的模型数量: {len(model_files)}")
            
            # 查找最佳模型
            best_models = [f for f in model_files if 'best_model' in f.name]
            if best_models:
                latest_best = max(best_models, key=lambda x: x.stat().st_mtime)
                print(f"  最新最佳模型: {latest_best.name}")
            
            # 查找检查点
            checkpoints = [f for f in model_files if 'checkpoint' in f.name]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: x.stat().st_mtime)
                print(f"  最新检查点: {latest_checkpoint.name}")
            
            results[port] = {
                'model_count': len(model_files),
                'has_best_model': len(best_models) > 0,
                'has_checkpoints': len(checkpoints) > 0
            }
        else:
            print(f"\n{port.upper()} 港口: 未训练")
            results[port] = {
                'model_count': 0,
                'has_best_model': False,
                'has_checkpoints': False
            }
    
    print("\n" + "="*60)
    print("训练状态总结:")
    print("="*60)
    
    trained_ports = [port for port, result in results.items() if result['model_count'] > 0]
    print(f"已训练港口数量: {len(trained_ports)}/{len(ports)}")
    print(f"已训练港口: {', '.join(trained_ports)}")
    
    if trained_ports:
        print(f"\n下一步建议:")
        print("1. 运行更多轮次的训练以提高性能")
        print("2. 实现联邦学习聚合算法")
        print("3. 与基线方法进行详细对比")
        print("4. 生成可视化结果")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    summarize_results()