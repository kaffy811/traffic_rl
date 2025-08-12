"""
GAT-PPO智能体实现
基于图注意力网络的PPO强化学习智能体，用于海事港口调度
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import deque
import random

from improved_gat_structure import MaritimeGraphBuilder, ImprovedGATLayer
from maritime_domain_knowledge import MaritimeStateBuilder, MaritimeRewardCalculator

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GATActorCritic(nn.Module):
    """GAT-PPO Actor-Critic网络"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # 网络参数
        self.state_dim = config.get('state_dim', 32)
        self.action_dim = config.get('action_dim', 10)  # 10个泊位的分配概率
        self.hidden_dim = config.get('hidden_dim', 128)
        self.num_heads = config.get('num_heads', 8)
        self.dropout = config.get('dropout', 0.1)
        
        # GAT层
        self.gat_layer1 = ImprovedGATLayer(
            in_features=8,  # 统一的节点特征维度
            out_features=self.hidden_dim // 2,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        self.gat_layer2 = ImprovedGATLayer(
            in_features=self.hidden_dim // 2,
            out_features=self.hidden_dim // 4,
            num_heads=self.num_heads,
            dropout=self.dropout
        )
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(self.state_dim + self.hidden_dim // 4, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.LayerNorm(self.hidden_dim)
        )
        
        # Actor网络 (策略网络)
        self.actor = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, self.action_dim),
            nn.Softmax(dim=-1)  # 输出动作概率分布
        )
        
        # Critic网络 (价值网络)
        self.critic = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, 1)  # 输出状态价值
        )
        
    def forward(self, state: torch.Tensor, node_features: torch.Tensor, 
                adj_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播
        Args:
            state: [batch_size, state_dim] 环境状态
            node_features: [batch_size, num_nodes, node_feature_dim] 节点特征
            adj_matrix: [num_nodes, num_nodes] 邻接矩阵
        Returns:
            action_probs: [batch_size, action_dim] 动作概率分布
            state_value: [batch_size, 1] 状态价值
        """
        # GAT特征提取
        gat_features = self.gat_layer1(node_features, adj_matrix)
        gat_features = self.gat_layer2(gat_features, adj_matrix)
        
        # 图特征聚合 (取平均)
        graph_features = torch.mean(gat_features, dim=1)  # [batch_size, hidden_dim//4]
        
        # 特征融合
        combined_features = torch.cat([state, graph_features], dim=-1)
        fused_features = self.feature_fusion(combined_features)
        
        # Actor和Critic输出
        action_probs = self.actor(fused_features)
        state_value = self.critic(fused_features)
        
        return action_probs, state_value

class PPOBuffer:
    """PPO经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        
    def push(self, state: np.ndarray, node_features: np.ndarray, adj_matrix: np.ndarray,
             action: int, reward: float, next_state: np.ndarray, done: bool,
             log_prob: float, value: float):
        """添加经验"""
        experience = {
            'state': state,
            'node_features': node_features,
            'adj_matrix': adj_matrix,
            'action': action,
            'reward': reward,
            'next_state': next_state,
            'done': done,
            'log_prob': log_prob,
            'value': value
        }
        self.buffer.append(experience)
    
    def sample(self, batch_size: int) -> Dict:
        """采样批次数据"""
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        
        # 使用np.stack避免tensor创建警告
        return {
            'states': torch.from_numpy(np.stack([exp['state'] for exp in batch])).float(),
            'node_features': torch.from_numpy(np.stack([exp['node_features'] for exp in batch])).float(),
            'adj_matrices': torch.from_numpy(np.stack([exp['adj_matrix'] for exp in batch])).float(),
            'actions': torch.LongTensor([exp['action'] for exp in batch]),
            'rewards': torch.FloatTensor([exp['reward'] for exp in batch]),
            'next_states': torch.from_numpy(np.stack([exp['next_state'] for exp in batch])).float(),
            'dones': torch.BoolTensor([exp['done'] for exp in batch]),
            'log_probs': torch.from_numpy(np.stack([exp['log_prob'] for exp in batch])).float(),
            'values': torch.FloatTensor([exp['value'] for exp in batch])
        }
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)

class GATPPOAgent(nn.Module):
    """GAT-PPO智能体"""
    
    def __init__(self, port_name: str, config: Dict):
        super().__init__()
        self.port_name = port_name
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 网络参数
        self.lr = config.get('learning_rate', 3e-4)
        self.gamma = config.get('gamma', 0.99)
        self.gae_lambda = config.get('gae_lambda', 0.95)
        self.clip_ratio = config.get('clip_ratio', 0.2)
        self.ppo_epochs = config.get('ppo_epochs', 4)
        self.batch_size = config.get('batch_size', 64)
        self.buffer_size = config.get('buffer_size', 10000)
        
        # 构建网络
        self.actor_critic = GATActorCritic(config).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor_critic.parameters(), lr=self.lr)
        
        # 经验缓冲区
        self.buffer = PPOBuffer(self.buffer_size)
        
        # 图结构构建器
        self.graph_builder = MaritimeGraphBuilder(port_name)
        self.state_builder = MaritimeStateBuilder(port_name)
        self.reward_calculator = MaritimeRewardCalculator(port_name)
        
        # 训练统计
        self.episode_rewards = []
        self.episode_lengths = []
        self.training_losses = []
        
        logger.info(f"初始化GAT-PPO智能体 - 港口: {port_name}, 设备: {self.device}")
    
    def get_action(self, state: np.ndarray, node_features: np.ndarray, 
                   adj_matrix: np.ndarray, training: bool = True) -> Tuple[int, float, float]:
        """
        获取动作
        Args:
            state: 环境状态
            node_features: 节点特征
            adj_matrix: 邻接矩阵
            training: 是否训练模式
        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
            value: 状态价值
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            # 确保节点特征有正确的batch维度
            if len(node_features.shape) == 2:
                node_features_tensor = torch.FloatTensor(node_features).unsqueeze(0).to(self.device)
            else:
                node_features_tensor = torch.FloatTensor(node_features).to(self.device)
            # 确保邻接矩阵是torch.Tensor
            if isinstance(adj_matrix, np.ndarray):
                adj_matrix_tensor = torch.FloatTensor(adj_matrix).to(self.device)
            else:
                adj_matrix_tensor = adj_matrix.to(self.device)
            
            action_probs, state_value = self.actor_critic(
                state_tensor, node_features_tensor, adj_matrix_tensor
            )
            
            if training:
                # 训练时从概率分布中采样
                dist = torch.distributions.Categorical(action_probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
            else:
                # 评估时选择概率最大的动作
                action = torch.argmax(action_probs, dim=-1)
                log_prob = torch.log(action_probs.gather(1, action.unsqueeze(1))).squeeze(1)
            
            return action.item(), log_prob.item(), state_value.item()
    
    def store_experience(self, state: np.ndarray, node_features: np.ndarray, 
                        adj_matrix: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool, log_prob: float, value: float):
        """存储经验"""
        self.buffer.push(state, node_features, adj_matrix, action, reward, 
                        next_state, done, log_prob, value)
    
    def update(self) -> Dict:
        """PPO更新"""
        if len(self.buffer) < self.batch_size:
            return {'loss': 0.0, 'policy_loss': 0.0, 'value_loss': 0.0}
        
        # 计算GAE优势
        advantages, returns = self._compute_gae()
        
        total_loss = 0.0
        total_policy_loss = 0.0
        total_value_loss = 0.0
        
        # PPO多轮更新
        for _ in range(self.ppo_epochs):
            batch = self.buffer.sample(self.batch_size)
            
            # 移动到设备
            states = batch['states'].to(self.device)
            node_features = batch['node_features'].to(self.device)
            # 处理邻接矩阵 - 使用第一个样本的邻接矩阵（假设批次内相同）
            adj_matrices = batch['adj_matrices'].to(self.device)
            adj_matrix = adj_matrices[0]  # 使用第一个样本的邻接矩阵
            actions = batch['actions'].to(self.device)
            old_log_probs = batch['log_probs'].to(self.device)
            
            # 前向传播
            action_probs, state_values = self.actor_critic(states, node_features, adj_matrix)
            
            # 计算新的log概率
            dist = torch.distributions.Categorical(action_probs)
            new_log_probs = dist.log_prob(actions)
            entropy = dist.entropy().mean()
            
            # PPO损失
            ratio = torch.exp(new_log_probs - old_log_probs)
            surr1 = ratio * advantages[:len(ratio)]
            surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * advantages[:len(ratio)]
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 价值损失
            value_loss = F.mse_loss(state_values.squeeze(), returns[:len(state_values)])
            
            # 总损失
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 0.5)
            self.optimizer.step()
            
            total_loss += loss.item()
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        # 清空缓冲区
        self.buffer.clear()
        
        loss_info = {
            'loss': total_loss / self.ppo_epochs,
            'policy_loss': total_policy_loss / self.ppo_epochs,
            'value_loss': total_value_loss / self.ppo_epochs
        }
        
        self.training_losses.append(loss_info)
        return loss_info
    
    def _compute_gae(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """计算GAE优势和回报"""
        experiences = list(self.buffer.buffer)
        
        advantages = []
        returns = []
        gae = 0
        
        # 从后往前计算GAE
        for i in reversed(range(len(experiences))):
            exp = experiences[i]
            
            if i == len(experiences) - 1:
                next_value = 0 if exp['done'] else exp['value']
            else:
                next_value = experiences[i + 1]['value']
            
            delta = exp['reward'] + self.gamma * next_value - exp['value']
            gae = delta + self.gamma * self.gae_lambda * gae
            
            advantages.insert(0, gae)
            returns.insert(0, gae + exp['value'])
        
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return advantages, returns
    
    def save_model(self, filepath: str):
        """保存模型"""
        torch.save({
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config,
            'port_name': self.port_name,
            'episode_rewards': self.episode_rewards,
            'training_losses': self.training_losses
        }, filepath)
        logger.info(f"模型已保存到: {filepath}")
    
    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.episode_rewards = checkpoint.get('episode_rewards', [])
        self.training_losses = checkpoint.get('training_losses', [])
        logger.info(f"模型已从 {filepath} 加载")
    
    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        if not self.episode_rewards:
            return {}
        
        return {
            'total_episodes': len(self.episode_rewards),
            'avg_reward': np.mean(self.episode_rewards[-100:]),  # 最近100轮平均奖励
            'max_reward': np.max(self.episode_rewards),
            'min_reward': np.min(self.episode_rewards),
            'avg_episode_length': np.mean(self.episode_lengths[-100:]) if self.episode_lengths else 0,
            'recent_loss': self.training_losses[-1] if self.training_losses else {}
        }
    
    def state_dict(self):
        """获取模型状态字典"""
        return self.actor_critic.state_dict()
    
    def load_state_dict(self, state_dict):
        """加载模型状态字典"""
        return self.actor_critic.load_state_dict(state_dict)

def create_default_config(port_name: str) -> Dict:
    """创建默认配置"""
    return {
        'state_dim': 56,  # 统一为56维状态，与模型首层in_features一致
        'action_dim': 10,  # 10个泊位
        'hidden_dim': 128,
        'num_heads': 8,
        'dropout': 0.1,
        'learning_rate': 3e-4,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'clip_ratio': 0.2,
        'ppo_epochs': 4,
        'batch_size': 64,
        'buffer_size': 10000,
        'port_name': port_name
    }

def main():
    """测试GAT-PPO智能体"""
    logger.info("测试GAT-PPO智能体")
    
    # 创建智能体
    config = create_default_config('gulfport')
    agent = GATPPOAgent('gulfport', config)
    
    # 模拟测试
    state = np.random.randn(32)
    node_features = np.random.randn(33, 12)  # 33个节点，每个12维特征
    adj_matrix = np.random.randint(0, 2, (33, 33))
    
    action, log_prob, value = agent.get_action(state, node_features, adj_matrix)
    
    print(f"测试结果:")
    print(f"  动作: {action}")
    print(f"  对数概率: {log_prob:.4f}")
    print(f"  状态价值: {value:.4f}")
    
    # 存储经验
    agent.store_experience(state, node_features, adj_matrix, action, 1.0, 
                          state, False, log_prob, value)
    
    print(f"  缓冲区大小: {len(agent.buffer)}")
    print("GAT-PPO智能体测试完成!")

if __name__ == "__main__":
    main()