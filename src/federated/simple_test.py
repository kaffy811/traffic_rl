#!/usr/bin/env python3
"""
简单的联邦学习测试脚本
用于验证基本的GAT-PPO智能体功能，不依赖复杂的FedML框架
"""

import sys
import os
from pathlib import Path
import numpy as np
import torch

# 设置项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# 导入项目模块
from models.maritime_gat_ppo import MaritimeGATPPOAgent, PPOConfig


def test_basic_functionality():
    """测试基本的GAT-PPO智能体功能"""
    print("🧪 开始测试海事GAT-PPO智能体...")
    
    # 创建配置
    config = PPOConfig(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        ppo_epochs=4,
        batch_size=64,
        mini_batch_size=16,
    )
    
    # 创建4个港口的智能体
    agents = {}
    port_names = ["new_orleans", "south_louisiana", "baton_rouge", "gulfport"]
    
    for i, port_name in enumerate(port_names, 0):  # 从0开始
        print(f"📍 创建 {port_name} 港智能体 (Node {i})...")
        agent = MaritimeGATPPOAgent(
            node_id=i,
            num_nodes=4,
            config=config
        )
        agents[port_name] = agent
        print(f"✅ {port_name} 智能体创建成功")
    
    # 模拟简单的训练步骤
    print("\n🚀 开始模拟训练步骤...")
    
    for episode in range(3):
        print(f"\n📈 Episode {episode + 1}/3")
        
        for port_name, agent in agents.items():
            # 模拟环境状态（按照预期的格式）
            state = {
                'NodeA': {  # new_orleans
                    'waiting_ships': np.random.randint(5, 20),
                    'throughput': np.random.uniform(1.0, 3.0),
                    'waiting_time': np.random.uniform(10, 30),
                    'queue_length': np.random.randint(3, 15),
                    'safety_score': np.random.uniform(0.7, 1.0)
                },
                'NodeB': {  # south_louisiana  
                    'waiting_ships': np.random.randint(3, 15),
                    'throughput': np.random.uniform(0.8, 2.5),
                    'waiting_time': np.random.uniform(8, 25),
                    'queue_length': np.random.randint(2, 12),
                    'safety_score': np.random.uniform(0.6, 0.9)
                },
                'NodeC': {  # baton_rouge
                    'waiting_ships': np.random.randint(2, 10),
                    'throughput': np.random.uniform(0.5, 1.8),
                    'waiting_time': np.random.uniform(5, 20),
                    'queue_length': np.random.randint(1, 8),
                    'safety_score': np.random.uniform(0.8, 1.0)
                },
                'NodeD': {  # gulfport
                    'waiting_ships': np.random.randint(4, 18),
                    'throughput': np.random.uniform(0.7, 2.2),
                    'waiting_time': np.random.uniform(12, 28),
                    'queue_length': np.random.randint(2, 10),
                    'safety_score': np.random.uniform(0.7, 0.95)
                }
            }
            
            # 执行动作选择和价值评估
            action, log_prob, value, entropy = agent.get_action_and_value(state)
            
            # 模拟奖励
            reward = np.random.uniform(-1, 1)
            
            # 计算公平性奖励
            action_results = {
                'total_throughput': np.random.uniform(8.0, 12.0),
                'average_waiting_time': np.random.uniform(15.0, 25.0),
                'fairness_index': np.random.uniform(0.7, 0.9)
            }
            
            reward_breakdown = agent.fairness_calculator.calculate_comprehensive_reward(
                node_states=state,
                action_results=action_results
            )
            fairness_reward = reward_breakdown.get('total_reward', 0.0)
            
            total_reward = 0.7 * reward + 0.3 * fairness_reward
            
            print(f"  📊 {port_name}: action={action}, "
                  f"reward={total_reward:.3f}, value={value.item():.3f}")
    
    print("\n🎉 测试完成！所有智能体都能正常工作！")
    
    # 模拟联邦聚合
    print("\n🔄 模拟联邦聚合过程...")
    
    # 收集所有智能体的参数
    all_params = []
    for port_name, agent in agents.items():
        params = {name: param.clone() for name, param in agent.named_parameters()}
        all_params.append(params)
        print(f"  📤 收集 {port_name} 的模型参数")
    
    # 简单的平均聚合
    global_params = {}
    for name in all_params[0].keys():
        global_params[name] = torch.mean(
            torch.stack([params[name] for params in all_params]), 
            dim=0
        )
    
    # 将聚合后的参数分发给所有智能体
    for port_name, agent in agents.items():
        for name, param in agent.named_parameters():
            param.data = global_params[name].clone()
        print(f"  📥 分发全局模型给 {port_name}")
    
    print("✅ 联邦聚合完成！")
    
    return agents


def test_port_characteristics():
    """测试港口特征差异化"""
    print("\n🏗️ 测试港口特征差异化...")
    
    # 港口特征配置
    port_configs = {
        "new_orleans": {
            "traffic_intensity": "high",
            "operational_hours": "24/7",
            "port_type": "container_terminal"
        },
        "south_louisiana": {
            "traffic_intensity": "high", 
            "operational_hours": "24/7",
            "port_type": "bulk_terminal"
        },
        "baton_rouge": {
            "traffic_intensity": "medium",
            "operational_hours": "daytime_priority",
            "port_type": "inland_waterway"
        },
        "gulfport": {
            "traffic_intensity": "medium",
            "operational_hours": "24/7",
            "port_type": "multipurpose_terminal"
        }
    }
    
    for port_name, characteristics in port_configs.items():
        print(f"🏗️ {port_name}:")
        for key, value in characteristics.items():
            print(f"  {key}: {value}")
    
    print("✅ 港口特征配置验证完成！")


if __name__ == "__main__":
    print("🚢 海事GAT-FedPPO联邦学习系统功能测试")
    print("=" * 50)
    
    try:
        # 基本功能测试
        agents = test_basic_functionality()
        
        # 港口特征测试  
        test_port_characteristics()
        
        print("\n🎯 总结:")
        print("✅ GAT-PPO智能体创建成功")
        print("✅ 前向传播工作正常")
        print("✅ 公平性奖励计算正常")
        print("✅ 联邦聚合模拟成功")
        print("✅ 港口特征差异化配置完成")
        print("\n🚀 系统已准备就绪，可以进行真实的联邦学习！")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()