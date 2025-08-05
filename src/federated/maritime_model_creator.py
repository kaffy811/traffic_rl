#!/usr/bin/env python3
"""
海事GAT-PPO模型创建器
为FedML框架提供模型创建功能
"""

import torch
import torch.nn as nn
import logging
from typing import Any
import sys
from pathlib import Path

# 设置项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.maritime_gat_ppo import MaritimeGATPPOAgent, PPOConfig

logger = logging.getLogger(__name__)


class MaritimeFedMLWrapper(nn.Module):
    """
    海事GAT-PPO的FedML包装器
    将我们的强化学习智能体包装成FedML兼容的模型
    """
    
    def __init__(self, gat_ppo_agent: MaritimeGATPPOAgent, output_dim: int = 4):
        super().__init__()
        self.gat_ppo_agent = gat_ppo_agent
        self.output_dim = output_dim
        
        # 使用智能体的actor网络作为主要的前向传播网络
        self.actor = gat_ppo_agent.actor
        self.critic = gat_ppo_agent.critic
        
        logger.info(f"✅ 创建海事FedML包装器 - 节点ID: {gat_ppo_agent.node_id}")
    
    def forward(self, x):
        """
        前向传播（FedML兼容）
        
        Args:
            x: 输入特征张量 [batch_size, feature_dim]
        
        Returns:
            输出logits [batch_size, output_dim]
        """
        # x的形状应该是 [batch_size, 20] (4个节点 × 5个特征)
        batch_size = x.shape[0]
        
        # 将平坦的特征重新组织成海事观测格式
        observations = []
        for b in range(batch_size):
            # 将20维特征分解为4个节点的观测
            features = x[b].view(4, 5)  # [4 nodes, 5 features]
            
            # 构建海事观测字典格式
            obs = {}
            node_names = ['NodeA', 'NodeB', 'NodeC', 'NodeD']
            for i, node_name in enumerate(node_names):
                node_features = features[i]
                obs[node_name] = {
                    'waiting_ships': float(node_features[0]),
                    'throughput': float(node_features[1]),
                    'waiting_time': float(node_features[2]),
                    'queue_length': float(node_features[3]),
                    'safety_score': float(node_features[4])
                }
            observations.append(obs)
        
        # 批量处理观测
        logits = []
        for obs in observations:
            # 使用GAT-PPO智能体的前向传播
            action_logits, _ = self.gat_ppo_agent.forward(obs)
            logits.append(action_logits)
        
        # 堆叠结果
        output = torch.stack(logits)  # [batch_size, output_dim]
        
        return output
    
    def get_maritime_action_and_value(self, maritime_observations):
        """
        获取海事动作和价值（保持原有接口）
        """
        return self.gat_ppo_agent.get_action_and_value(maritime_observations)
    
    def calculate_comprehensive_reward(self, *args, **kwargs):
        """
        计算综合奖励（保持原有接口）
        """
        return self.gat_ppo_agent.calculate_comprehensive_reward(*args, **kwargs)


def create_maritime_model(args, output_dim: int = 4) -> MaritimeFedMLWrapper:
    """
    创建海事GAT-PPO模型（FedML兼容接口）
    
    Args:
        args: FedML参数
        output_dim: 输出维度（动作空间大小）
    
    Returns:
        包装后的模型
    """
    
    # 获取客户端rank
    client_rank = getattr(args, 'rank', 0)
    
    # 从配置创建PPO配置
    ppo_config = PPOConfig(
        learning_rate=getattr(args, 'learning_rate', 3e-4),
        gamma=getattr(args, 'gamma', 0.99),
        gae_lambda=getattr(args, 'gae_lambda', 0.95),
        clip_ratio=getattr(args, 'clip_ratio', 0.2),
        ppo_epochs=getattr(args, 'ppo_epochs', 4),
        batch_size=getattr(args, 'batch_size', 64),
        mini_batch_size=getattr(args, 'mini_batch_size', 16),
        action_dim=output_dim
    )
    
    # 创建GAT-PPO智能体
    gat_ppo_agent = MaritimeGATPPOAgent(
        node_id=client_rank,
        num_nodes=4,
        config=ppo_config
    )
    
    # 包装为FedML兼容模型
    model = MaritimeFedMLWrapper(gat_ppo_agent, output_dim)
    
    # 获取港口名称
    port_mapping = {
        0: "new_orleans",
        1: "south_louisiana", 
        2: "baton_rouge",
        3: "gulfport"
    }
    port_name = port_mapping.get(client_rank, f"port_{client_rank}")
    
    logger.info(f"✅ 海事模型创建完成 - 港口: {port_name}, 输出维度: {output_dim}")
    
    return model


# FedML模型创建注册
def create_model(args, output_dim: int = 4):
    """FedML模型创建接口"""
    return create_maritime_model(args, output_dim)


if __name__ == "__main__":
    # 测试模型创建
    import argparse
    
    args = argparse.Namespace()
    args.rank = 1
    args.learning_rate = 3e-4
    args.batch_size = 10
    
    model = create_maritime_model(args, output_dim=4)
    
    # 测试前向传播
    batch_size = 5
    feature_dim = 20  # 4个节点 × 5个特征
    
    x = torch.randn(batch_size, feature_dim)
    output = model(x)
    
    print(f"模型创建成功")
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")
    print(f"参数数量: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")