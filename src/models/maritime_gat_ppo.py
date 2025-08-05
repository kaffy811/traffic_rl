#!/usr/bin/env python3
"""
海事GAT-PPO智能体：完整的强化学习智能体
完整的PPO实现，包含GAE、完整损失函数、批量处理等
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
from typing import Dict, List, Tuple, Optional, Union
from collections import namedtuple, deque
import logging
from dataclasses import dataclass
import pickle
import os
import sys
from pathlib import Path


# 解决相对导入问题
def setup_imports():
    """设置导入路径"""
    current_dir = Path(__file__).parent
    project_root = current_dir.parent.parent  # 回到项目根目录

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # 添加src目录到路径
    src_dir = project_root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


setup_imports()

# 现在可以进行导入
try:
    # 尝试相对导入（当作为包使用时）
    from .gat_wrapper import MaritimeGATEncoder, MaritimeStateBuilder
    from .fairness_reward import ComprehensiveFairnessRewardCalculator, FairnessConfig
except ImportError:
    # 回退到绝对导入（当直接运行时）
    try:
        from src.models.gat_wrapper import MaritimeGATEncoder, MaritimeStateBuilder
        from src.models.fairness_reward import ComprehensiveFairnessRewardCalculator, FairnessConfig
    except ImportError:
        # 最后的回退选项
        from models.gat_wrapper import MaritimeGATEncoder, MaritimeStateBuilder
        from models.fairness_reward import ComprehensiveFairnessRewardCalculator, FairnessConfig


@dataclass
class PPOConfig:
    """PPO配置参数"""
    # 网络架构
    node_feature_dim: int = 5
    gat_hidden_dim: int = 64
    gat_output_dim: int = 32
    ppo_hidden_dim: int = 128
    action_dim: int = 4

    # PPO超参数
    learning_rate: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5

    # 训练参数
    ppo_epochs: int = 4
    batch_size: int = 64
    mini_batch_size: int = 16
    buffer_size: int = 2048
    update_frequency: int = 2048

    # 网络更新
    target_kl: float = 0.01
    adaptive_lr: bool = True
    lr_decay: float = 0.99

    # 探索参数
    initial_noise: float = 0.1
    noise_decay: float = 0.995
    min_noise: float = 0.01


# 经验数据结构
Experience = namedtuple('Experience', [
    'state', 'action', 'reward', 'next_state', 'done',
    'log_prob', 'value', 'advantage', 'return_'
])


class AdvancedPPOMemory:
    """
    高级PPO经验缓冲区
    支持GAE计算、批量采样、优先级回放等
    """

    def __init__(self,
                 max_size: int = 2048,
                 gamma: float = 0.99,
                 gae_lambda: float = 0.95):
        self.max_size = max_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda

        # 存储缓冲区
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []

        # 计算后的值
        self.advantages = []
        self.returns = []

        # 轨迹分割标记
        self.trajectory_starts = []

        self.ptr = 0
        self.trajectory_start = 0

    def store(self,
              state: Dict[str, Dict],
              action: int,
              reward: float,
              next_state: Dict[str, Dict],
              done: bool,
              log_prob: float,
              value: float):
        """存储单步经验"""

        if len(self.states) < self.max_size:
            self.states.append(state)
            self.actions.append(action)
            self.rewards.append(reward)
            self.next_states.append(next_state)
            self.dones.append(done)
            self.log_probs.append(log_prob)
            self.values.append(value)
        else:
            self.states[self.ptr] = state
            self.actions[self.ptr] = action
            self.rewards[self.ptr] = reward
            self.next_states[self.ptr] = next_state
            self.dones[self.ptr] = done
            self.log_probs[self.ptr] = log_prob
            self.values[self.ptr] = value

        self.ptr = (self.ptr + 1) % self.max_size

        # 如果episode结束，标记轨迹分割点
        if done:
            self.trajectory_starts.append(self.trajectory_start)
            self.trajectory_start = len(self.states) if len(self.states) < self.max_size else self.ptr

    def finish_trajectory(self, next_value: float = 0.0):
        """
        完成轨迹并计算GAE优势和回报
        """
        if len(self.rewards) == 0:
            return

        # 计算GAE优势
        self._compute_gae_advantages(next_value)

        # 计算折扣回报
        self._compute_discounted_returns()

    def _compute_gae_advantages(self, next_value: float):
        """计算GAE（Generalized Advantage Estimation）优势"""

        rewards = np.array(self.rewards)
        # 安全地转换tensor到numpy
        values = np.array([v.detach().cpu().numpy() if torch.is_tensor(v) else v for v in self.values])
        dones = np.array(self.dones)

        # 添加下一个状态的价值
        values_with_next = np.append(values, next_value)

        # 计算时间差分误差
        deltas = rewards + self.gamma * values_with_next[1:] * (1 - dones) - values_with_next[:-1]

        # 计算GAE优势
        advantages = np.zeros_like(rewards)
        advantage = 0

        for t in reversed(range(len(rewards))):
            advantage = deltas[t] + self.gamma * self.gae_lambda * advantage * (1 - dones[t])
            advantages[t] = advantage

        self.advantages = advantages.tolist()

    def _compute_discounted_returns(self):
        """计算折扣回报"""
        returns = np.zeros_like(self.rewards)
        returns[-1] = self.rewards[-1] if not torch.is_tensor(self.rewards[-1]) else self.rewards[-1].detach().cpu().numpy()

        for t in reversed(range(len(self.rewards) - 1)):
            returns[t] = self.rewards[t] + self.gamma * returns[t + 1] * (1 - self.dones[t])

        self.returns = returns.tolist()

    def get_all_data(self) -> Tuple[List, List, List, List, List, List, List, List, List]:
        """获取所有数据"""
        return (self.states, self.actions, self.rewards, self.next_states,
                self.dones, self.log_probs, self.values, self.advantages, self.returns)

    def sample_batch(self, batch_size: int) -> Tuple[List, ...]:
        """随机采样批量数据"""
        indices = random.sample(range(len(self.states)), min(batch_size, len(self.states)))

        batch_states = [self.states[i] for i in indices]
        batch_actions = [self.actions[i] for i in indices]
        batch_rewards = [self.rewards[i] for i in indices]
        batch_next_states = [self.next_states[i] for i in indices]
        batch_dones = [self.dones[i] for i in indices]
        batch_log_probs = [self.log_probs[i] for i in indices]
        batch_values = [self.values[i] for i in indices]
        batch_advantages = [self.advantages[i] for i in indices]
        batch_returns = [self.returns[i] for i in indices]

        return (batch_states, batch_actions, batch_rewards, batch_next_states,
                batch_dones, batch_log_probs, batch_values, batch_advantages, batch_returns)

    def clear(self):
        """清空缓冲区"""
        self.states.clear()
        self.actions.clear()
        self.rewards.clear()
        self.next_states.clear()
        self.dones.clear()
        self.log_probs.clear()
        self.values.clear()
        self.advantages.clear()
        self.returns.clear()
        self.trajectory_starts.clear()
        self.ptr = 0
        self.trajectory_start = 0

    def __len__(self):
        return len(self.states)


class MaritimeGATPPOAgent(nn.Module):
    """
    完整的海事GAT-PPO智能体
    集成了图注意力网络的完整PPO强化学习智能体
    """

    def __init__(self,
                 node_id: int,
                 num_nodes: int = 4,
                 config: PPOConfig = None):
        super().__init__()

        self.config = config or PPOConfig()
        self.node_id = node_id
        self.num_nodes = num_nodes

        # 网络组件
        self._build_networks()

        # 优化器
        self.optimizer = optim.Adam(self.parameters(), lr=self.config.learning_rate)
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(
            self.optimizer, gamma=self.config.lr_decay
        ) if self.config.adaptive_lr else None

        # 经验缓冲区
        self.memory = AdvancedPPOMemory(
            max_size=self.config.buffer_size,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda
        )

        # 公平性奖励计算器
        fairness_config = FairnessConfig(
            efficiency_weight=0.6,
            fairness_weight=0.4,
            adaptive_weights=True
        )
        self.fairness_calculator = ComprehensiveFairnessRewardCalculator(fairness_config)

        # 训练统计
        self.training_stats = {
            'episode': 0,
            'total_steps': 0,
            'policy_loss': [],
            'value_loss': [],
            'entropy_loss': [],
            'kl_divergence': [],
            'explained_variance': [],
            'learning_rate': []
        }

        # 探索噪声
        self.exploration_noise = self.config.initial_noise

        # 日志
        self.logger = logging.getLogger(f"Agent_{node_id}")

    def _build_networks(self):
        """构建神经网络"""

        # GAT图编码器
        self.gat_encoder = MaritimeGATEncoder(
            node_feature_dim=self.config.node_feature_dim,
            hidden_dim=self.config.gat_hidden_dim,
            output_dim=self.config.gat_output_dim
        )

        # 状态构建器
        self.state_builder = MaritimeStateBuilder(self.num_nodes)

        # 本地状态编码器
        self.local_encoder = nn.Sequential(
            nn.Linear(self.config.node_feature_dim, self.config.gat_output_dim),
            nn.ReLU(),
            nn.LayerNorm(self.config.gat_output_dim),
            nn.Linear(self.config.gat_output_dim, self.config.gat_output_dim)
        )

        # 状态融合层
        fusion_input_dim = self.config.gat_output_dim * 3  # 本地 + 节点GAT + 图GAT
        self.state_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.config.ppo_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.config.ppo_hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.config.ppo_hidden_dim, self.config.ppo_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.config.ppo_hidden_dim)
        )

        # PPO Actor（策略网络）
        self.actor = nn.Sequential(
            nn.Linear(self.config.ppo_hidden_dim, self.config.ppo_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.config.ppo_hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.config.ppo_hidden_dim, self.config.ppo_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.ppo_hidden_dim // 2, self.config.action_dim)
        )

        # PPO Critic（价值网络）
        self.critic = nn.Sequential(
            nn.Linear(self.config.ppo_hidden_dim, self.config.ppo_hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(self.config.ppo_hidden_dim),
            nn.Dropout(0.1),
            nn.Linear(self.config.ppo_hidden_dim, self.config.ppo_hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.config.ppo_hidden_dim // 2, 1)
        )

        # 权重初始化
        self._initialize_weights()

    def _initialize_weights(self):
        """初始化网络权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def _encode_state(self, maritime_observations: Dict[str, Dict]) -> torch.Tensor:
        """编码状态"""
        # 1. 构建图状态
        node_features, adjacency_matrix = self.state_builder.build_state(maritime_observations)

        # 2. GAT编码
        node_embeddings, graph_embedding = self.gat_encoder(node_features, adjacency_matrix)

        # 3. 本地状态编码
        local_state = node_features[self.node_id]
        local_embedding = self.local_encoder(local_state)

        # 4. 当前节点的GAT嵌入
        current_node_embedding = node_embeddings[self.node_id]

        # 5. 状态融合
        fused_state = torch.cat([
            local_embedding,
            current_node_embedding,
            graph_embedding
        ])

        return self.state_fusion(fused_state)

    def forward(self, maritime_observations: Dict[str, Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        前向传播

        Args:
            maritime_observations: 海事观测数据
        Returns:
            action_logits: 动作logits
            state_value: 状态价值
        """
        encoded_state = self._encode_state(maritime_observations)

        action_logits = self.actor(encoded_state)
        state_value = self.critic(encoded_state)

        return action_logits, state_value

    def get_action_and_value(self,
                             maritime_observations: Dict[str, Dict],
                             action: Optional[int] = None,
                             deterministic: bool = False) -> Tuple[int, float, torch.Tensor, torch.Tensor]:
        """
        获取动作和价值

        Returns:
            action: 选择的动作
            log_prob: 动作的对数概率
            state_value: 状态价值
            entropy: 策略熵
        """
        action_logits, state_value = self.forward(maritime_observations)

        # 添加探索噪声
        if not deterministic and self.training:
            noise = torch.randn_like(action_logits) * self.exploration_noise
            action_logits = action_logits + noise

        # 计算动作概率分布
        action_probs = F.softmax(action_logits, dim=-1)
        action_dist = torch.distributions.Categorical(action_probs)

        # 选择动作
        if action is None:
            if deterministic:
                action = torch.argmax(action_probs).item()
            else:
                action = action_dist.sample().item()

        # 计算对数概率和熵
        log_prob = action_dist.log_prob(torch.tensor(action))
        entropy = action_dist.entropy()

        return action, log_prob.item(), state_value.squeeze(), entropy

    def calculate_comprehensive_reward(self,
                                       maritime_observations: Dict[str, Dict],
                                       action: int,
                                       next_observations: Dict[str, Dict],
                                       previous_observations: Optional[Dict[str, Dict]] = None) -> float:
        """
        计算综合奖励（包含公平性）
        """
        # 提取统计信息
        current_stats = self._extract_comprehensive_statistics(maritime_observations)
        next_stats = self._extract_comprehensive_statistics(next_observations)

        action_results = {
            'total_throughput': next_stats['total_throughput'],
            'avg_waiting_time': next_stats['avg_waiting_time'],
            'avg_queue_length': next_stats['avg_queue_length']
        }

        # 使用综合公平性奖励计算器
        reward_breakdown = self.fairness_calculator.calculate_comprehensive_reward(
            node_states=next_observations,
            action_results=action_results,
            previous_states=previous_observations
        )

        return reward_breakdown['total_reward']

    def _extract_comprehensive_statistics(self, observations: Dict[str, Dict]) -> Dict[str, float]:
        """提取综合统计信息"""
        total_throughput = sum(obs.get('throughput', 0) for obs in observations.values())
        waiting_times = [obs.get('waiting_time', 0) for obs in observations.values()]
        queue_lengths = [obs.get('waiting_ships', 0) for obs in observations.values()]

        return {
            'total_throughput': total_throughput,
            'avg_waiting_time': np.mean(waiting_times) if waiting_times else 0,
            'avg_queue_length': np.mean(queue_lengths) if queue_lengths else 0,
            'max_waiting_time': max(waiting_times) if waiting_times else 0,
            'min_throughput': min(obs.get('throughput', 0) for obs in observations.values())
        }

    def store_transition(self,
                         observations: Dict[str, Dict],
                         action: int,
                         reward: float,
                         next_observations: Dict[str, Dict],
                         done: bool,
                         log_prob: float,
                         value: float):
        """存储经验"""
        self.memory.store(observations, action, reward, next_observations, done, log_prob, value)
        self.training_stats['total_steps'] += 1

    def update_policy(self) -> Dict[str, float]:
        """
        完整的PPO策略更新

        Returns:
            训练统计信息
        """
        if len(self.memory) < self.config.batch_size:
            return {}

        # 完成轨迹并计算GAE
        self.memory.finish_trajectory()

        # 获取所有数据
        (states, actions, rewards, next_states, dones,
         old_log_probs, old_values, advantages, returns) = self.memory.get_all_data()

        if len(states) < self.config.mini_batch_size:
            return {}

        # 标准化优势
        advantages = torch.tensor(advantages, dtype=torch.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        returns = torch.tensor(returns, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float32)
        old_values = torch.tensor(old_values, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)

        # 训练统计
        policy_losses = []
        value_losses = []
        entropy_losses = []
        kl_divergences = []

        # PPO多轮更新
        for epoch in range(self.config.ppo_epochs):
            # 随机打乱数据
            indices = torch.randperm(len(states))

            # 小批量更新
            for start_idx in range(0, len(states), self.config.mini_batch_size):
                end_idx = start_idx + self.config.mini_batch_size
                batch_indices = indices[start_idx:end_idx]

                # 小批量数据
                batch_states = [states[i] for i in batch_indices]
                batch_actions = actions[batch_indices]
                batch_advantages = advantages[batch_indices]
                batch_returns = returns[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_old_values = old_values[batch_indices]

                # 前向传播
                batch_action_logits, batch_values = self._forward_batch(batch_states)

                # 计算新的动作概率和熵
                batch_action_probs = F.softmax(batch_action_logits, dim=-1)
                batch_action_dist = torch.distributions.Categorical(batch_action_probs)
                batch_new_log_probs = batch_action_dist.log_prob(batch_actions)
                batch_entropy = batch_action_dist.entropy()

                # 计算比率
                ratio = torch.exp(batch_new_log_probs - batch_old_log_probs)

                # PPO策略损失
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.clip_ratio, 1 + self.config.clip_ratio) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # 价值函数损失（Clipped Value Loss）
                if self.config.value_coef > 0:
                    value_pred_clipped = batch_old_values + torch.clamp(
                        batch_values.squeeze() - batch_old_values,
                        -self.config.clip_ratio,
                        self.config.clip_ratio
                    )
                    value_loss1 = F.mse_loss(batch_values.squeeze(), batch_returns)
                    value_loss2 = F.mse_loss(value_pred_clipped, batch_returns)
                    value_loss = torch.max(value_loss1, value_loss2)
                else:
                    value_loss = F.mse_loss(batch_values.squeeze(), batch_returns)

                # 熵损失
                entropy_loss = -batch_entropy.mean()

                # 总损失
                total_loss = (policy_loss +
                              self.config.value_coef * value_loss +
                              self.config.entropy_coef * entropy_loss)

                # 反向传播
                self.optimizer.zero_grad()
                total_loss.backward()

                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)

                self.optimizer.step()

                # 记录统计信息
                policy_losses.append(policy_loss.item())
                value_losses.append(value_loss.item())
                entropy_losses.append(entropy_loss.item())

                # KL散度（用于早停）
                with torch.no_grad():
                    kl_div = (batch_old_log_probs - batch_new_log_probs).mean()
                    kl_divergences.append(kl_div.item())

            # 早停检查
            avg_kl = np.mean(kl_divergences[-len(range(0, len(states), self.config.mini_batch_size)):])
            if avg_kl > self.config.target_kl:
                self.logger.info(f"Early stopping at epoch {epoch} due to KL divergence {avg_kl:.4f}")
                break

        # 更新学习率
        if self.lr_scheduler:
            self.lr_scheduler.step()

        # 更新探索噪声
        self.exploration_noise = max(
            self.config.min_noise,
            self.exploration_noise * self.config.noise_decay
        )

        # 清空缓冲区
        self.memory.clear()

        # 计算解释方差
        explained_var = self._compute_explained_variance(old_values.numpy(), returns.numpy())

        # 更新训练统计
        train_stats = {
            'policy_loss': np.mean(policy_losses),
            'value_loss': np.mean(value_losses),
            'entropy_loss': np.mean(entropy_losses),
            'kl_divergence': np.mean(kl_divergences),
            'explained_variance': explained_var,
            'learning_rate': self.optimizer.param_groups[0]['lr'],
            'exploration_noise': self.exploration_noise
        }

        # 记录到历史
        for key, value in train_stats.items():
            if key in self.training_stats:
                self.training_stats[key].append(value)

        return train_stats

    def _forward_batch(self, batch_states: List[Dict[str, Dict]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """批量前向传播"""
        batch_action_logits = []
        batch_values = []

        for state in batch_states:
            action_logits, value = self.forward(state)
            batch_action_logits.append(action_logits)
            batch_values.append(value)

        return torch.stack(batch_action_logits), torch.stack(batch_values)

    def _compute_explained_variance(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """计算解释方差"""
        var_y = np.var(y_true)
        return 1 - np.var(y_true - y_pred) / (var_y + 1e-8)

    def save_model(self, filepath: str):
        """保存模型"""
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config.__dict__,
            'training_stats': self.training_stats,
            'node_id': self.node_id,
            'exploration_noise': self.exploration_noise
        }

        if self.lr_scheduler:
            checkpoint['lr_scheduler_state_dict'] = self.lr_scheduler.state_dict()

        torch.save(checkpoint, filepath)
        self.logger.info(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """加载模型"""
        checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)

        self.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.training_stats = checkpoint.get('training_stats', self.training_stats)
        self.exploration_noise = checkpoint.get('exploration_noise', self.config.initial_noise)

        if 'lr_scheduler_state_dict' in checkpoint and self.lr_scheduler:
            self.lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

        self.logger.info(f"Model loaded from {filepath}")

    def get_training_stats(self) -> Dict:
        """获取训练统计信息"""
        return {
            'episode': self.training_stats['episode'],
            'total_steps': self.training_stats['total_steps'],
            'recent_policy_loss': self.training_stats['policy_loss'][-10:] if self.training_stats[
                'policy_loss'] else [],
            'recent_value_loss': self.training_stats['value_loss'][-10:] if self.training_stats['value_loss'] else [],
            'recent_kl_div': self.training_stats['kl_divergence'][-10:] if self.training_stats['kl_divergence'] else [],
            'current_lr': self.optimizer.param_groups[0]['lr'],
            'exploration_noise': self.exploration_noise
        }


def comprehensive_test():
    """修复后的完整智能体测试"""
    print("🤖 修复后的海事GAT-PPO智能体测试")
    print("=" * 60)

    # 创建配置 - 🔧 优化batch_size使训练能够触发
    config = PPOConfig(
        learning_rate=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_ratio=0.2,
        ppo_epochs=4,
        batch_size=20,      # 从64降低到20
        mini_batch_size=10  # 从16降低到10
    )

    # 创建智能体（NodeA的智能体）
    agent = MaritimeGATPPOAgent(node_id=0, config=config)
    agent.train()

    print(f"🚀 智能体配置:")
    print(f"  节点ID: {agent.node_id}")
    print(f"  动作维度: {config.action_dim}")
    print(f"  学习率: {config.learning_rate}")
    print(f"  批量大小: {config.batch_size} (已优化)")
    print(f"  小批量大小: {config.mini_batch_size} (已优化)")

    # 模拟训练episode
    print(f"\n模拟训练episode:")

    for episode in range(3):
        print(f"\nEpisode {episode + 1}:")
        
        # 🔧 修复：更新episode计数器
        agent.training_stats['episode'] = episode + 1

        # 模拟observation
        observations = {
            'NodeA': {'waiting_ships': 5, 'throughput': 3, 'waiting_time': 12, 'signal_phase': 1,
                      'weather_condition': 0.8},
            'NodeB': {'waiting_ships': 8, 'throughput': 2, 'waiting_time': 18, 'signal_phase': 0,
                      'weather_condition': 0.8},
            'NodeC': {'waiting_ships': 3, 'throughput': 1, 'waiting_time': 8, 'signal_phase': 1,
                      'weather_condition': 0.8},
            'NodeD': {'waiting_ships': 6, 'throughput': 4, 'waiting_time': 15, 'signal_phase': 0,
                      'weather_condition': 0.8},
        }

        episode_rewards = []

        # 模拟episode步骤
        for step in range(10):
            # 获取动作
            action, log_prob, value, entropy = agent.get_action_and_value(observations)

            # 模拟环境步骤
            next_observations = observations.copy()
            # 简单模拟：动作影响等待时间
            if action == 0:  # 短信号
                next_observations['NodeA']['waiting_time'] = max(0, observations['NodeA']['waiting_time'] - 1)
            elif action == 1:  # 中信号
                next_observations['NodeA']['waiting_time'] = max(0, observations['NodeA']['waiting_time'] - 2)
            elif action == 2:  # 长信号
                next_observations['NodeA']['waiting_time'] = max(0, observations['NodeA']['waiting_time'] - 3)
            else:  # 特长信号
                next_observations['NodeA']['waiting_time'] = max(0, observations['NodeA']['waiting_time'] - 4)

            # 计算奖励
            reward = agent.calculate_comprehensive_reward(observations, action, next_observations)
            episode_rewards.append(reward)

            # 存储经验
            done = (step == 9)  # 最后一步
            agent.store_transition(observations, action, reward, next_observations, done, log_prob, value.item())

            # 更新observations
            observations = next_observations

            if step % 5 == 0:
                print(f"  步骤 {step}: 动作={action}, 奖励={reward:.2f}, 价值={value.item():.2f}")

        print(f"  Episode总奖励: {sum(episode_rewards):.2f}")
        print(f"  缓冲区大小: {len(agent.memory)}")

        # 🔧 修复：每个episode后尝试PPO更新（降低门槛）
        if len(agent.memory) >= config.mini_batch_size:  # 从batch_size(20)改为mini_batch_size(10)
            print(f"  🔄 开始PPO更新...")
            train_stats = agent.update_policy()
            if train_stats:
                print(f"  ✅ 训练统计:")
                for key, value in train_stats.items():
                    print(f"    {key}: {value:.6f}")
            else:
                print(f"  ⚠️ 更新返回空统计")
        else:
            print(f"  ⏸️ 经验不足，跳过更新 (需要{config.mini_batch_size}步，当前{len(agent.memory)}步)")

    # 获取训练统计
    print(f"\n📊 最终训练统计:")
    stats = agent.get_training_stats()
    for key, value in stats.items():
        if key == 'episode':
            print(f"  ✅ {key}: {value}")
        elif 'loss' in key and isinstance(value, list) and len(value) > 0:
            print(f"  ✅ {key}: {value} (有训练数据)")
        else:
            print(f"  {key}: {value}")

    # 测试模型保存和加载
    print(f"\n💾 模型保存/加载测试:")
    save_path = "test_maritime_gat_ppo_agent.pth"
    agent.save_model(save_path)

    # 创建新智能体并加载
    new_agent = MaritimeGATPPOAgent(node_id=0, config=config)
    new_agent.load_model(save_path)

    print(f"  模型保存和加载成功")
    print(f"  加载的总步数: {new_agent.training_stats['total_steps']}")

    # 清理文件
    if os.path.exists(save_path):
        os.remove(save_path)

    print(f"\n完整海事GAT-PPO智能体测试完成！")


if __name__ == "__main__":
    comprehensive_test()