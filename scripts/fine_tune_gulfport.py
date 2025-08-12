#!/usr/bin/env python3
"""
Gulfport标准阶段微调脚本
目标：提升胜率从0.462到0.50+，使夜测变绿
"""

import sys
import torch
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent / "src" / "federated"))

from curriculum_trainer import CurriculumTrainer
from gat_ppo_agent import GATPPOAgent

def main():
    port = 'gulfport'
    print(f"🚢 开始微调 {port} 标准阶段")
    
    # 创建训练器
    trainer = CurriculumTrainer(port)
    
    # 找到标准阶段
    stage = next(s for s in trainer.curriculum_stages if s.name == '标准阶段')
    print(f"📋 目标阶段: {stage.name}")
    print(f"   当前阈值: {stage.success_threshold}")
    print(f"   当前episodes: {stage.episodes}")
    
    # 增加训练轮数
    stage.episodes = max(stage.episodes, 35)  # 从25增加到35
    print(f"   调整后episodes: {stage.episodes}")
    
    # 直接使用现有的模型架构，不重新创建
    print("🔧 使用现有模型架构进行微调...")
    
    # 加载现有权重来获取模型架构
    ckpt_path = Path(f"models/curriculum_v2/{port}/stage_标准阶段_best.pt")
    if ckpt_path.exists():
        print(f"📁 加载现有权重: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu")
        
        # 从现有权重推断配置
        model_sd = ckpt["model_state_dict"]
        if "feature_fusion.0.weight" in model_sd:
            state_dim = model_sd["feature_fusion.0.weight"].shape[1]
            print(f"   推断的state_dim: {state_dim}")
        else:
            state_dim = 120  # 默认值
        
        # 创建配置
        config = {
            'state_dim': state_dim,
            'action_dim': 15,
            'hidden_dim': 256,
            'num_heads': 4,
            'dropout': 0.1,
            'learning_rate': 2.1e-4,  # 降低30%
            'gamma': 0.99,
            'gae_lambda': 0.95,
            'clip_ratio': 0.2,
            'ppo_epochs': 4,
            'batch_size': 64,
            'buffer_size': 10000,
            'port_name': port
        }
        
        agent = GATPPOAgent(port, config)
        print(f"🤖 智能体配置: hidden_dim={config['hidden_dim']}, lr={config['learning_rate']}")
        
        # 加载权重
        agent.actor_critic.load_state_dict(ckpt["model_state_dict"], strict=False)
        print("✅ 权重加载成功")
    else:
        print("❌ 未找到现有权重")
        return
    
    # 开始微调
    print(f"\n🎯 开始微调训练...")
    result = trainer.train_stage(agent, stage)
    
    # 保存微调后的模型
    output_path = Path(f"models/curriculum_v2/{port}/stage_标准阶段_best.pt")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    torch.save({
        "model_state_dict": agent.actor_critic.state_dict(),
        "config": config,
        "performance": result
    }, output_path)
    
    print(f"\n🎉 微调完成!")
    print(f"📊 最终性能: {result.get('final_performance', {})}")
    print(f"💾 模型已保存到: {output_path}")
    
    # 显示胜率
    if 'final_performance' in result:
        win_rate = result['final_performance'].get('win_rate', 0)
        print(f"🏆 最终胜率: {win_rate*100:.1f}%")
        if win_rate >= 0.50:
            print("✅ 目标达成！胜率 ≥ 50%")
        else:
            print("⚠️ 胜率仍需提升")

if __name__ == "__main__":
    main() 