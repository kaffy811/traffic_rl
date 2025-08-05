#!/usr/bin/env python3
"""
海事GAT-PPO联邦训练器
集成FedML框架，实现多节点分布式联邦学习
"""

import logging
import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple
from fedml.core import ClientTrainer

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.maritime_gat_ppo import MaritimeGATPPOAgent, PPOConfig
from models.fairness_reward import ComprehensiveFairnessRewardCalculator


class MaritimeFedTrainer(ClientTrainer):
    """
    海事GAT-PPO联邦训练器
    
    封装MaritimeGATPPOAgent，使其能够参与FedML联邦学习
    """
    
    def __init__(self, model, args, node_id: int = 0, device='cpu'):
        """
        初始化联邦训练器
        
        Args:
            model: 基础PyTorch模型（实际使用内部的GAT-PPO智能体）
            args: FedML参数
            node_id: 节点ID（对应港口/航道节点）
            device: 计算设备
        """
        super().__init__(model, args)
        
        self.node_id = node_id
        self.device = device
        self.args = args
        
        # 创建PPO配置
        self.ppo_config = PPOConfig(
            learning_rate=getattr(args, 'learning_rate', 3e-4),
            gamma=getattr(args, 'gamma', 0.99),
            gae_lambda=getattr(args, 'gae_lambda', 0.95),
            clip_ratio=getattr(args, 'clip_ratio', 0.2),
            ppo_epochs=getattr(args, 'ppo_epochs', 4),
            batch_size=getattr(args, 'ppo_batch_size', 20),
            mini_batch_size=getattr(args, 'ppo_mini_batch_size', 10)
        )
        
        # 创建海事GAT-PPO智能体
        self.maritime_agent = MaritimeGATPPOAgent(
            node_id=node_id, 
            config=self.ppo_config
        )
        
        # 创建公平性奖励计算器
        self.fairness_calculator = ComprehensiveFairnessRewardCalculator()
        
        # 训练统计
        self.training_stats = {
            'local_episodes': 0,
            'total_reward': 0.0,
            'policy_losses': [],
            'value_losses': [],
            'fairness_scores': []
        }
        
        logging.info(f"✅ 海事联邦训练器初始化完成 - 节点ID: {node_id}")

    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """
        获取模型参数（用于联邦聚合）
        
        Returns:
            模型状态字典
        """
        # 返回GAT-PPO智能体的完整状态
        agent_state = self.maritime_agent.state_dict()
        
        # 包含额外的训练统计信息
        agent_state['training_stats'] = {
            'node_id': self.node_id,
            'local_episodes': self.training_stats['local_episodes'],
            'total_reward': self.training_stats['total_reward'],
            'avg_policy_loss': np.mean(self.training_stats['policy_losses']) if self.training_stats['policy_losses'] else 0.0,
            'avg_value_loss': np.mean(self.training_stats['value_losses']) if self.training_stats['value_losses'] else 0.0
        }
        
        return agent_state

    def set_model_params(self, model_parameters: Dict[str, torch.Tensor]):
        """
        设置模型参数（来自联邦聚合）
        
        Args:
            model_parameters: 聚合后的模型参数
        """
        # 提取训练统计信息
        if 'training_stats' in model_parameters:
            stats = model_parameters.pop('training_stats')
            logging.info(f"📊 接收全局模型 - 来源节点: {stats.get('node_id', 'unknown')}")
        
        # 加载模型参数到GAT-PPO智能体
        self.maritime_agent.load_state_dict(model_parameters, strict=False)
        
        logging.info("✅ 全局模型参数已更新")

    def train(self, train_data, device, args) -> Dict[str, float]:
        """
        执行本地训练
        
        Args:
            train_data: 训练数据（海事环境数据）
            device: 计算设备
            args: 训练参数
            
        Returns:
            训练统计信息
        """
        logging.info(f"🚀 开始本地训练 - 节点ID: {self.node_id}")
        
        self.maritime_agent.train()
        
        # 如果传入的是环境数据，直接使用；否则生成模拟数据
        if hasattr(train_data, '__iter__'):
            training_episodes = list(train_data)
        else:
            # 生成模拟的海事环境交互数据
            training_episodes = self._generate_maritime_episodes(args.epochs)
        
        episode_rewards = []
        training_losses = []
        
        for episode_idx, episode_data in enumerate(training_episodes):
            # 执行一个episode的训练
            episode_reward, training_stats = self._run_episode(episode_data)
            
            episode_rewards.append(episode_reward)
            if training_stats:
                training_losses.append(training_stats)
            
            # 更新本地统计
            self.training_stats['local_episodes'] += 1
            self.training_stats['total_reward'] += episode_reward
            
            if (episode_idx + 1) % 5 == 0:
                logging.info(f"  Episode {episode_idx + 1}: 奖励={episode_reward:.2f}")
        
        # 计算平均性能
        avg_reward = np.mean(episode_rewards)
        avg_policy_loss = np.mean([stats.get('policy_loss', 0) for stats in training_losses])
        avg_value_loss = np.mean([stats.get('value_loss', 0) for stats in training_losses])
        
        # 更新训练统计
        self.training_stats['policy_losses'].extend([stats.get('policy_loss', 0) for stats in training_losses])
        self.training_stats['value_losses'].extend([stats.get('value_loss', 0) for stats in training_losses])
        
        train_results = {
            'avg_reward': avg_reward,
            'avg_policy_loss': avg_policy_loss,
            'avg_value_loss': avg_value_loss,
            'total_episodes': len(training_episodes),
            'node_id': self.node_id
        }
        
        logging.info(f"✅ 本地训练完成 - 平均奖励: {avg_reward:.2f}, 策略损失: {avg_policy_loss:.6f}")
        
        return train_results

    def _generate_maritime_episodes(self, num_episodes: int = 10) -> list:
        """
        生成模拟的海事环境episode数据
        
        Args:
            num_episodes: episode数量
            
        Returns:
            episode数据列表
        """
        episodes = []
        
        for _ in range(num_episodes):
            # 模拟一个完整的海事交通场景
            episode = {
                'initial_state': self._generate_maritime_state(),
                'steps': 10,  # 每个episode 10步
                'target_fairness': np.random.uniform(0.7, 0.9)  # 目标公平性分数
            }
            episodes.append(episode)
        
        return episodes

    def _generate_maritime_state(self) -> Dict[str, Dict]:
        """
        生成模拟的海事状态观测 - 基于不同港口特征的差异化数据
        
        Returns:
            包含所有节点状态的字典
        """
        # 定义不同港口/节点的特征差异
        port_characteristics = {
            'NodeA': {  # 新奥尔良港主入口 - 高流量集装箱港
                'base_traffic': (8, 20),     # 等待船舶数量范围
                'throughput_range': (4, 10), # 吞吐量范围
                'wait_time_range': (10, 30), # 等待时间范围
                'traffic_pattern': 'peak_hours',  # 流量模式
                'ship_mix': 'container_heavy'     # 船舶类型组合
            },
            'NodeB': {  # 密西西比河口 - 中等流量散货港
                'base_traffic': (5, 15),
                'throughput_range': (2, 7),
                'wait_time_range': (8, 20),
                'traffic_pattern': 'steady',
                'ship_mix': 'bulk_carrier'
            },
            'NodeC': {  # 河道中段 - 低流量内河港
                'base_traffic': (2, 10),
                'throughput_range': (1, 5),
                'wait_time_range': (5, 15),
                'traffic_pattern': 'seasonal',
                'ship_mix': 'inland_vessels'
            },
            'NodeD': {  # 近海锚地 - 变动流量临时停泊
                'base_traffic': (3, 18),
                'throughput_range': (2, 8),
                'wait_time_range': (15, 40),
                'traffic_pattern': 'irregular',
                'ship_mix': 'mixed_fleet'
            }
        }
        
        state = {}
        current_hour = np.random.randint(0, 24)  # 模拟时间影响
        
        for node_name, characteristics in port_characteristics.items():
            # 基于时间和港口特征调整参数
            traffic_multiplier = self._get_traffic_multiplier(
                characteristics['traffic_pattern'], current_hour, self.node_id
            )
            
            # 生成该节点的状态
            base_range = characteristics['base_traffic']
            waiting_ships = max(1, int(np.random.randint(*base_range) * traffic_multiplier))
            
            throughput_range = characteristics['throughput_range']
            throughput = max(1, int(np.random.randint(*throughput_range) * traffic_multiplier))
            
            wait_range = characteristics['wait_time_range']
            waiting_time = max(1, int(np.random.randint(*wait_range) / traffic_multiplier))
            
            state[node_name] = {
                'waiting_ships': waiting_ships,
                'throughput': throughput,
                'waiting_time': waiting_time,
                'signal_phase': np.random.randint(0, 2),
                'weather_condition': np.random.uniform(0.6, 1.0),
                # 新增港口特征字段
                'port_type': characteristics['ship_mix'],
                'traffic_pattern': characteristics['traffic_pattern'],
                'node_id': self.node_id
            }
        
        return state
    
    def _get_traffic_multiplier(self, pattern: str, hour: int, node_id: int) -> float:
        """
        根据流量模式和时间获取流量乘数
        
        Args:
            pattern: 流量模式
            hour: 当前小时
            node_id: 节点ID（引入节点间差异）
            
        Returns:
            流量乘数
        """
        base_multiplier = 1.0
        
        if pattern == 'peak_hours':
            # 模拟工作时间高峰期
            if 8 <= hour <= 17:
                base_multiplier = 1.5 + 0.3 * np.sin((hour - 8) * np.pi / 9)
            else:
                base_multiplier = 0.7
        elif pattern == 'steady':
            # 相对稳定的流量
            base_multiplier = 1.0 + 0.2 * np.sin(hour * np.pi / 12)
        elif pattern == 'seasonal':
            # 季节性变化（简化为日内变化）
            base_multiplier = 0.8 + 0.4 * np.sin(hour * np.pi / 24)
        elif pattern == 'irregular':
            # 不规则变化
            base_multiplier = 0.6 + 0.8 * np.random.random()
        
        # 加入节点间的差异性
        node_variation = 0.8 + 0.4 * (node_id / 4.0)
        
        return base_multiplier * node_variation

    def _run_episode(self, episode_data: Dict) -> Tuple[float, Optional[Dict]]:
        """
        运行一个完整的episode
        
        Args:
            episode_data: episode配置数据
            
        Returns:
            (episode总奖励, 训练统计)
        """
        observations = episode_data['initial_state']
        episode_reward = 0.0
        training_stats = None
        
        # 执行episode步骤
        for step in range(episode_data['steps']):
            # 获取智能体动作
            action, log_prob, value, entropy = self.maritime_agent.get_action_and_value(observations)
            
            # 模拟环境反应
            next_observations = self._simulate_environment_step(observations, action)
            
            # 计算奖励（包含公平性）
            reward = self.maritime_agent.calculate_comprehensive_reward(
                observations, action, next_observations
            )
            episode_reward += reward
            
            # 存储经验
            done = (step == episode_data['steps'] - 1)
            self.maritime_agent.store_transition(
                observations, action, reward, next_observations, done, log_prob, value.item()
            )
            
            observations = next_observations
        
        # 尝试PPO更新
        if len(self.maritime_agent.memory) >= self.ppo_config.mini_batch_size:
            training_stats = self.maritime_agent.update_policy()
        
        return episode_reward, training_stats

    def _simulate_environment_step(self, current_state: Dict, action: int) -> Dict:
        """
        模拟环境步骤
        
        Args:
            current_state: 当前状态
            action: 智能体动作
            
        Returns:
            下一个状态
        """
        next_state = {node: state.copy() for node, state in current_state.items()}
        
        # 模拟动作对主节点的影响
        main_node = f'Node{chr(65 + self.node_id)}'  # NodeA, NodeB, etc.
        if main_node in next_state:
            # 动作影响等待时间
            time_reduction = [1, 2, 3, 4][action]  # 对应不同信号长度
            next_state[main_node]['waiting_time'] = max(
                0, current_state[main_node]['waiting_time'] - time_reduction
            )
            
            # 吞吐量微调
            next_state[main_node]['throughput'] = min(
                10, current_state[main_node]['throughput'] + time_reduction // 2
            )
        
        return next_state

    def test(self, test_data, device, args) -> Dict[str, float]:
        """
        执行测试评估
        
        Args:
            test_data: 测试数据
            device: 计算设备
            args: 测试参数
            
        Returns:
            测试结果
        """
        self.maritime_agent.eval()
        
        test_episodes = 5  # 测试episode数量
        test_rewards = []
        fairness_scores = []
        
        with torch.no_grad():
            for _ in range(test_episodes):
                observations = self._generate_maritime_state()
                episode_reward = 0.0
                
                for step in range(10):
                    action, _, _, _ = self.maritime_agent.get_action_and_value(observations)
                    next_observations = self._simulate_environment_step(observations, action)
                    
                    reward = self.maritime_agent.calculate_comprehensive_reward(
                        observations, action, next_observations
                    )
                    episode_reward += reward
                    
                    # 计算公平性分数
                    fairness_result = self.fairness_calculator.calculate_comprehensive_reward(
                        observations, {f'Node{chr(65 + self.node_id)}': reward}, next_observations
                    )
                    fairness_scores.append(fairness_result.get('fairness_reward', 0.0))
                    
                    observations = next_observations
                
                test_rewards.append(episode_reward)
        
        test_results = {
            'test_avg_reward': np.mean(test_rewards),
            'test_fairness_score': np.mean(fairness_scores),
            'test_reward_std': np.std(test_rewards),
            'node_id': self.node_id
        }
        
        logging.info(f"🧪 测试完成 - 平均奖励: {test_results['test_avg_reward']:.2f}, "
                    f"公平性分数: {test_results['test_fairness_score']:.2f}")
        
        return test_results

    def get_training_stats(self) -> Dict[str, Any]:
        """
        获取详细的训练统计信息
        
        Returns:
            训练统计字典
        """
        return {
            **self.training_stats,
            'agent_stats': self.maritime_agent.get_training_stats(),
            'node_id': self.node_id,
            'device': str(self.device)
        }


# 创建工厂函数，便于FedML调用
def create_maritime_trainer(model, args, node_id=0, device='cpu'):
    """
    工厂函数：创建海事联邦训练器
    
    Args:
        model: 基础模型
        args: FedML参数
        node_id: 节点ID
        device: 计算设备
        
    Returns:
        MaritimeFedTrainer实例
    """
    return MaritimeFedTrainer(model, args, node_id, device)


if __name__ == "__main__":
    # 简单测试
    class MockArgs:
        learning_rate = 3e-4
        epochs = 3
    
    import torch.nn as nn
    mock_model = nn.Linear(10, 4)  # 占位符模型
    mock_args = MockArgs()
    
    trainer = MaritimeFedTrainer(mock_model, mock_args, node_id=0)
    print("✅ 海事联邦训练器创建成功")
    
    # 测试训练
    train_results = trainer.train(None, 'cpu', mock_args)
    print(f"📊 训练结果: {train_results}") 