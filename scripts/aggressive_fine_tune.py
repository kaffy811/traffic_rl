#!/usr/bin/env python3
"""
激进的Gulfport标准阶段微调脚本
使用更小的学习率和更多的训练轮数
"""

import torch
from pathlib import Path
import sys

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent / "src" / "federated"))

from curriculum_trainer import CurriculumTrainer, build_agent

def main():
    port = 'gulfport'
    print(f"🚀 开始激进微调 {port} 标准阶段")
    
    # 创建训练器
    trainer = CurriculumTrainer(port)
    
    # 找到标准阶段
    stage = next(s for s in trainer.curriculum_stages if s.name == '标准阶段')
    print(f"📋 目标阶段: {stage.name}")
    print(f"   当前阈值: {stage.success_threshold}")
    print(f"   当前episodes: {stage.episodes}")
    
    # 大幅增加训练轮数
    stage.episodes = max(stage.episodes, 50)  # 从20增加到50
    print(f"   调整后episodes: {stage.episodes}")
    
    # 创建智能体并加载现有权重
    print(f"\n🎯 创建智能体并加载现有权重...")
    
    # 使用独立的build_agent函数
    agent = build_agent(port)
    
    # 加载现有权重
    ckpt_path = Path(f"models/curriculum_v2/{port}/stage_标准阶段_best.pt")
    if ckpt_path.exists():
        print(f"📁 加载现有权重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            agent.actor_critic.load_state_dict(ckpt["model_state_dict"], strict=False)
            print("✅ 权重加载成功")
        else:
            print("⚠️ 权重格式不匹配，使用随机初始化")
    else:
        print("❌ 未找到现有权重，使用随机初始化")
    
    # 使用更小的学习率
    original_lr = 3e-4
    new_lr = original_lr * 0.5  # 降低50%
    print(f"   学习率调整: {original_lr} → {new_lr}")
    
    # 降低学习率
    if hasattr(agent, 'optimizer'):
        for param_group in agent.optimizer.param_groups:
            param_group['lr'] = new_lr
        print(f"✅ 学习率已调整为: {new_lr}")
    
    # 开始微调
    print(f"\n🎯 开始激进微调训练...")
    result = trainer.train_stage(agent, stage)
    
    print(f"\n🎉 激进微调完成!")
    
    # train_stage返回(agent, result_dict)
    agent, result_dict = result
    print(f"📊 最终性能: {result_dict}")
    
    # 显示胜率
    if 'final_performance' in result_dict:
        win_rate = result_dict['final_performance'].get('win_rate', 0)
        print(f"🏆 最终胜率: {win_rate*100:.1f}%")
        if win_rate >= 0.50:
            print("✅ 目标达成！胜率 ≥ 50%")
        else:
            print("⚠️ 胜率仍需提升")
    else:
        print("⚠️ 无法获取胜率信息")
    
    # 检查模型是否已保存
    model_path = Path(f"models/curriculum_v2/{port}/stage_标准阶段_best.pt")
    if model_path.exists():
        print(f"💾 模型已保存到: {model_path}")
    else:
        print("❌ 模型保存失败")

if __name__ == "__main__":
    main() 