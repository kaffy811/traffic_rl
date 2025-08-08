"""
单港GAT-PPO训练器
在单个港口环境下训练GAT-PPO智能体
"""

import os
import json
import numpy as np
import torch
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

from gat_ppo_agent import GATPPOAgent, create_default_config
from improved_gat_structure import MaritimeGraphBuilder
from maritime_domain_knowledge import MaritimeStateBuilder, MaritimeRewardCalculator

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaritimeEnvironment:
    """海事港口仿真环境"""
    
    def __init__(self, port_name: str, training_data: List[Dict]):
        self.port_name = port_name
        self.training_data = training_data
        self.current_step = 0
        self.max_steps = len(training_data)
        
        # 港口状态
        self.berth_status = self._initialize_berths()
        self.vessel_queue = []
        self.completed_vessels = []
        self.current_time = 0
        
        # 构建器
        self.graph_builder = MaritimeGraphBuilder(port_name)
        self.state_builder = MaritimeStateBuilder(port_name)
        self.reward_calculator = MaritimeRewardCalculator(port_name)
        
        # 统计信息
        self.episode_stats = {
            'total_vessels': 0,
            'completed_vessels': 0,
            'total_waiting_time': 0,
            'total_service_time': 0,
            'berth_utilization': []
        }
        
    def _initialize_berths(self) -> Dict:
        """初始化泊位状态"""
        # 根据港口设置泊位数量
        berth_counts = {
            'new_orleans': 40,
            'baton_rouge': 64, 
            'south_louisiana': 25,
            'gulfport': 10
        }
        
        num_berths = berth_counts.get(self.port_name, 10)
        
        berth_status = {}
        for i in range(num_berths):
            berth_status[f'berth_{i}'] = {
                'occupied': False,
                'vessel_id': None,
                'start_time': None,
                'estimated_end_time': None,
                'queue': []
            }
        
        return berth_status
    
    def reset(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """重置环境"""
        self.current_step = 0
        self.current_time = 0
        self.vessel_queue = []
        self.completed_vessels = []
        
        # 重置泊位状态
        for berth in self.berth_status.values():
            berth['occupied'] = False
            berth['vessel_id'] = None
            berth['start_time'] = None
            berth['estimated_end_time'] = None
            berth['queue'] = []
        
        # 重置统计
        self.episode_stats = {
            'total_vessels': 0,
            'completed_vessels': 0,
            'total_waiting_time': 0,
            'total_service_time': 0,
            'berth_utilization': []
        }
        
        return self._get_observation()
    
    def step(self, action: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float, bool, Dict]:
        """执行一步"""
        if self.current_step >= self.max_steps:
            return self._get_observation() + (0.0, True, self.episode_stats)
        
        # 获取当前船舶
        current_vessel = self.training_data[self.current_step]
        self.current_time = current_vessel['timestamp']
        
        # 更新泊位状态
        self._update_berth_status()
        
        # 执行动作 (分配船舶到泊位)
        reward = self._assign_vessel_to_berth(current_vessel, action)
        
        # 更新统计
        self.episode_stats['total_vessels'] += 1
        
        # 计算泊位利用率
        occupied_berths = sum(1 for berth in self.berth_status.values() if berth['occupied'])
        utilization = occupied_berths / len(self.berth_status)
        self.episode_stats['berth_utilization'].append(utilization)
        
        self.current_step += 1
        done = self.current_step >= self.max_steps
        
        return self._get_observation() + (reward, done, self.episode_stats)
    
    def _update_berth_status(self):
        """更新泊位状态"""
        for berth_id, berth in self.berth_status.items():
            if berth['occupied'] and berth['estimated_end_time']:
                if self.current_time >= berth['estimated_end_time']:
                    # 船舶完成服务
                    vessel_id = berth['vessel_id']
                    service_time = self.current_time - berth['start_time']
                    
                    self.completed_vessels.append({
                        'vessel_id': vessel_id,
                        'berth_id': berth_id,
                        'service_time': service_time,
                        'completion_time': self.current_time
                    })
                    
                    self.episode_stats['completed_vessels'] += 1
                    self.episode_stats['total_service_time'] += service_time
                    
                    # 释放泊位
                    berth['occupied'] = False
                    berth['vessel_id'] = None
                    berth['start_time'] = None
                    berth['estimated_end_time'] = None
                    
                    # 处理队列中的下一艘船
                    if berth['queue']:
                        next_vessel = berth['queue'].pop(0)
                        self._assign_vessel_to_berth_directly(next_vessel, berth_id)
    
    def _assign_vessel_to_berth(self, vessel: Dict, action: int) -> float:
        """分配船舶到泊位"""
        berth_id = f'berth_{action}'
        
        if berth_id not in self.berth_status:
            # 无效泊位，给予负奖励
            return -1.0
        
        berth = self.berth_status[berth_id]
        
        if not berth['occupied']:
            # 泊位空闲，直接分配
            return self._assign_vessel_to_berth_directly(vessel, berth_id)
        else:
            # 泊位占用，加入队列
            berth['queue'].append(vessel)
            waiting_time = self._estimate_waiting_time(berth_id)
            self.episode_stats['total_waiting_time'] += waiting_time
            
            # 根据等待时间给予奖励 - 简化版本
            return self._calculate_queue_reward(waiting_time, len(berth['queue']))
    
    def _assign_vessel_to_berth_directly(self, vessel: Dict, berth_id: str) -> float:
        """直接分配船舶到泊位"""
        berth = self.berth_status[berth_id]
        
        # 估算服务时间
        service_time = vessel.get('estimated_service_time', 3600)  # 默认1小时
        
        # 更新泊位状态
        berth['occupied'] = True
        berth['vessel_id'] = vessel['mmsi']
        berth['start_time'] = self.current_time
        berth['estimated_end_time'] = self.current_time + service_time
        
        # 计算奖励 - 简化版本
        reward = self._calculate_simple_reward(vessel, berth_id, service_time)
        
        return reward
    
    def _estimate_waiting_time(self, berth_id: str) -> float:
        """估算等待时间"""
        berth = self.berth_status[berth_id]
        if berth['estimated_end_time']:
            return max(0, berth['estimated_end_time'] - self.current_time)
        return 0
    
    def _get_observation(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """获取观测"""
        # 构建状态
        if self.current_step < len(self.training_data):
            current_vessel = self.training_data[self.current_step]
            state = self.state_builder.build_vessel_state(
                current_vessel, 
                {'current_time': self.current_time},
                {'queue_length': len(self.vessel_queue)},
                self.berth_status
            )
        else:
            state = np.zeros(24)  # MaritimeStateBuilder使用24维状态
        
        # 构建节点特征 - 简化版本，统一维度
        node_features = self._build_unified_node_features()
        
        # 获取邻接矩阵
        adj_matrix = self.graph_builder.adjacency_matrix
        
        # 检查邻接矩阵
        if torch.sum(adj_matrix) == 0:
            print("警告: 邻接矩阵全为0，添加自环")
            # 添加自环避免全0邻接矩阵
            adj_matrix = adj_matrix + torch.eye(adj_matrix.shape[0])
        
        return state, node_features, adj_matrix
    
    def _build_port_status(self) -> Dict:
        """构建港口状态信息"""
        return {
            'berths': {f'berth_{i}': {
                'occupied': berth['occupied'],
                'length': 200 + i * 50,  # 模拟泊位长度
                'utilization': 0.7 if berth['occupied'] else 0.0
            } for i, (berth_id, berth) in enumerate(self.berth_status.items())},
            
            'anchorages': {f'anchor_{i}': {
                'occupancy': min(i + 1, 5),
                'capacity': 15,
                'avg_wait_time': i * 2
            } for i in range(3)},
            
            'channels': {f'channel_{i}': {
                'depth': 20 + i * 5,
                'width': 150 + i * 20,
                'traffic_density': 0.1 + i * 0.1
            } for i in range(4)},
            
            'terminals': {f'terminal_{i}': {
                'berth_count': 3 + i,
                'utilization': 0.6 + i * 0.05,
                'throughput': 20 + i * 10
            } for i in range(6)}
        }
    
    def _build_unified_node_features(self) -> np.ndarray:
        """构建统一维度的节点特征"""
        total_nodes = self.graph_builder.num_nodes
        feature_dim = 8  # 统一为8维特征
        node_features = np.zeros((total_nodes, feature_dim))
        
        node_idx = 0
        
        # 1. 泊位节点 (10个)
        for i, (berth_id, berth) in enumerate(self.berth_status.items()):
            if node_idx < total_nodes:
                node_features[node_idx] = [
                    0.0,  # 节点类型：泊位
                    1.0 if berth['occupied'] else 0.0,  # 占用状态
                    len(berth['queue']) / 10.0,          # 队列长度
                    1.0 if berth['queue'] else 0.0,     # 是否有队列
                    0.5,  # 利用率
                    0.8,  # 效率
                    0.0,  # 预留
                    0.0   # 预留
                ]
                node_idx += 1
        
        # 2. 锚地节点 (3个)
        for i in range(3):
            if node_idx < total_nodes:
                node_features[node_idx] = [
                    0.25,  # 节点类型：锚地
                    min(i + 1, 5) / 10.0,  # 占用数
                    15 / 20.0,             # 容量
                    i * 2 / 24.0,          # 等待时间
                    0.0, 0.0, 0.0, 0.0     # 预留
                ]
                node_idx += 1
        
        # 3. 航道节点 (4个)
        for i in range(4):
            if node_idx < total_nodes:
                node_features[node_idx] = [
                    0.5,  # 节点类型：航道
                    (20 + i * 5) / 50.0,   # 深度
                    (150 + i * 20) / 300.0, # 宽度
                    0.1 + i * 0.1,         # 交通密度
                    0.0, 0.0, 0.0, 0.0     # 预留
                ]
                node_idx += 1
        
        # 4. 码头节点 (6个)
        for i in range(6):
            if node_idx < total_nodes:
                node_features[node_idx] = [
                    0.75,  # 节点类型：码头
                    (3 + i) / 10.0,        # 泊位数
                    0.6 + i * 0.05,        # 利用率
                    (20 + i * 10) / 100.0, # 吞吐量
                    0.0, 0.0, 0.0, 0.0     # 预留
                ]
                node_idx += 1
        
        # 5. 船舶节点 (10个)
        for i in range(10):
            if node_idx < total_nodes:
                if self.current_step < len(self.training_data):
                    vessel = self.training_data[self.current_step]
                    node_features[node_idx] = [
                        1.0,  # 节点类型：船舶
                        vessel.get('length', 150) / 400.0,
                        vessel.get('beam', 25) / 60.0,
                        vessel.get('draught', 8) / 20.0,
                        vessel.get('sog', 0) / 25.0,
                        vessel.get('vessel_type', 70) / 100.0,
                        1.0,  # 激活状态
                        0.0   # 预留
                    ]
                else:
                    node_features[node_idx] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                node_idx += 1
        
        return node_features
    
    def _build_node_features(self) -> np.ndarray:
        """构建节点特征"""
        total_nodes = self.graph_builder.num_nodes
        node_features = np.zeros((total_nodes, 12))  # 12维节点特征
        
        # 泊位节点特征
        for i, (berth_id, berth) in enumerate(self.berth_status.items()):
            if i < total_nodes:
                node_features[i] = [
                    1.0 if berth['occupied'] else 0.0,  # 占用状态
                    len(berth['queue']) / 10.0,          # 队列长度 (归一化)
                    1.0 if berth['queue'] else 0.0,     # 是否有队列
                    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0  # 其他特征
                ]
        
        return node_features
    
    def _calculate_simple_reward(self, vessel: Dict, berth_id: str, service_time: float) -> float:
        """计算简化的分配奖励"""
        reward = 1.0  # 基础奖励
        
        # 根据服务时间调整奖励 (更短的服务时间获得更高奖励)
        if service_time > 0:
            time_factor = max(0.1, min(2.0, 3600 / service_time))  # 1小时为基准
            reward *= time_factor
        
        # 根据船舶优先级调整
        priority_score = vessel.get('vessel_features', {}).get('priority_score', 0.3)
        reward *= (1.0 + priority_score)
        
        return reward
    
    def _calculate_queue_reward(self, waiting_time: float, queue_length: int) -> float:
        """计算队列奖励"""
        # 等待时间越短，奖励越高
        time_penalty = -waiting_time / 3600.0  # 按小时计算惩罚
        
        # 队列长度惩罚
        queue_penalty = -queue_length * 0.1
        
        return max(-2.0, time_penalty + queue_penalty)

class SinglePortTrainer:
    """单港训练器"""
    
    def __init__(self, port_name: str, config: Dict = None):
        self.port_name = port_name
        self.config = config or create_default_config(port_name)
        
        # 创建保存目录
        self.save_dir = Path(f"../../models/single_port/{port_name}")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载训练数据
        self.training_data, self.test_data = self._load_training_data()
        
        # 创建环境和智能体
        self.env = MaritimeEnvironment(port_name, self.training_data)
        self.agent = GATPPOAgent(port_name, self.config)
        
        # 训练统计
        self.training_history = {
            'episode_rewards': [],
            'episode_lengths': [],
            'completion_rates': [],
            'avg_waiting_times': [],
            'berth_utilizations': []
        }
        
        logger.info(f"初始化单港训练器 - 港口: {port_name}")
        logger.info(f"训练数据: {len(self.training_data)} 样本")
        logger.info(f"测试数据: {len(self.test_data)} 样本")
    
    def _load_training_data(self) -> Tuple[List[Dict], List[Dict]]:
        """加载训练数据"""
        data_file = Path(f"../../data/gat_training_data/{self.port_name}_training_data.json")
        
        if not data_file.exists():
            raise FileNotFoundError(f"未找到训练数据文件: {data_file}")
        
        with open(data_file, 'r') as f:
            data = json.load(f)
        
        return data['training_data'], data['test_data']
    
    def train(self, num_episodes: int = 100, save_interval: int = 10) -> Dict:
        """训练智能体"""
        logger.info(f"开始训练 - 总轮数: {num_episodes}")
        
        best_reward = float('-inf')
        
        for episode in range(num_episodes):
            # 重置环境
            state, node_features, adj_matrix = self.env.reset()
            
            episode_reward = 0
            episode_length = 0
            
            while True:
                # 获取动作
                action, log_prob, value = self.agent.get_action(
                    state, node_features, adj_matrix, training=True
                )
                
                # 执行动作
                next_state, next_node_features, next_adj_matrix, reward, done, info = self.env.step(action)
                
                # 存储经验
                self.agent.store_experience(
                    state, node_features, adj_matrix, action, reward,
                    next_state, done, log_prob, value
                )
                
                episode_reward += reward
                episode_length += 1
                
                if done:
                    break
                
                state, node_features, adj_matrix = next_state, next_node_features, next_adj_matrix
            
            # 更新智能体
            loss_info = self.agent.update()
            
            # 记录统计信息
            self.agent.episode_rewards.append(episode_reward)
            self.agent.episode_lengths.append(episode_length)
            
            self.training_history['episode_rewards'].append(episode_reward)
            self.training_history['episode_lengths'].append(episode_length)
            self.training_history['completion_rates'].append(
                info['completed_vessels'] / max(1, info['total_vessels'])
            )
            self.training_history['avg_waiting_times'].append(
                info['total_waiting_time'] / max(1, info['total_vessels'])
            )
            self.training_history['berth_utilizations'].append(
                np.mean(info['berth_utilization']) if info['berth_utilization'] else 0
            )
            
            # 打印进度
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.training_history['episode_rewards'][-10:])
                avg_completion = np.mean(self.training_history['completion_rates'][-10:])
                
                logger.info(f"Episode {episode + 1}/{num_episodes}")
                logger.info(f"  平均奖励: {avg_reward:.2f}")
                logger.info(f"  完成率: {avg_completion:.2%}")
                logger.info(f"  损失: {loss_info.get('loss', 0):.4f}")
            
            # 保存最佳模型
            if episode_reward > best_reward:
                best_reward = episode_reward
                self.save_model(f"best_model_episode_{episode + 1}.pt")
            
            # 定期保存
            if (episode + 1) % save_interval == 0:
                self.save_model(f"checkpoint_episode_{episode + 1}.pt")
                self.save_training_history()
        
        logger.info(f"训练完成 - 最佳奖励: {best_reward:.2f}")
        return self.training_history
    
    def evaluate(self, num_episodes: int = 10) -> Dict:
        """评估智能体"""
        logger.info(f"开始评估 - 测试轮数: {num_episodes}")
        
        # 创建测试环境
        test_env = MaritimeEnvironment(self.port_name, self.test_data)
        
        eval_results = {
            'episode_rewards': [],
            'completion_rates': [],
            'avg_waiting_times': [],
            'berth_utilizations': []
        }
        
        for episode in range(num_episodes):
            state, node_features, adj_matrix = test_env.reset()
            episode_reward = 0
            
            while True:
                # 评估模式下获取动作
                action, _, _ = self.agent.get_action(
                    state, node_features, adj_matrix, training=False
                )
                
                next_state, next_node_features, next_adj_matrix, reward, done, info = test_env.step(action)
                episode_reward += reward
                
                if done:
                    break
                
                state, node_features, adj_matrix = next_state, next_node_features, next_adj_matrix
            
            # 记录评估结果
            eval_results['episode_rewards'].append(episode_reward)
            eval_results['completion_rates'].append(
                info['completed_vessels'] / max(1, info['total_vessels'])
            )
            eval_results['avg_waiting_times'].append(
                info['total_waiting_time'] / max(1, info['total_vessels'])
            )
            eval_results['berth_utilizations'].append(
                np.mean(info['berth_utilization']) if info['berth_utilization'] else 0
            )
        
        # 计算平均结果
        avg_results = {
            'avg_reward': np.mean(eval_results['episode_rewards']),
            'avg_completion_rate': np.mean(eval_results['completion_rates']),
            'avg_waiting_time': np.mean(eval_results['avg_waiting_times']),
            'avg_berth_utilization': np.mean(eval_results['berth_utilizations']),
            'std_reward': np.std(eval_results['episode_rewards']),
            'std_completion_rate': np.std(eval_results['completion_rates'])
        }
        
        logger.info("评估结果:")
        logger.info(f"  平均奖励: {avg_results['avg_reward']:.2f} ± {avg_results['std_reward']:.2f}")
        logger.info(f"  完成率: {avg_results['avg_completion_rate']:.2%} ± {avg_results['std_completion_rate']:.2%}")
        logger.info(f"  平均等待时间: {avg_results['avg_waiting_time']:.1f}秒")
        logger.info(f"  泊位利用率: {avg_results['avg_berth_utilization']:.2%}")
        
        return avg_results
    
    def save_model(self, filename: str):
        """保存模型"""
        filepath = self.save_dir / filename
        self.agent.save_model(str(filepath))
    
    def load_model(self, filename: str):
        """加载模型"""
        filepath = self.save_dir / filename
        self.agent.load_model(str(filepath))
    
    def save_training_history(self):
        """保存训练历史"""
        history_file = self.save_dir / "training_history.json"
        with open(history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)
    
    def plot_training_curves(self):
        """绘制训练曲线"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # 奖励曲线
        axes[0, 0].plot(self.training_history['episode_rewards'])
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        
        # 完成率曲线
        axes[0, 1].plot(self.training_history['completion_rates'])
        axes[0, 1].set_title('Completion Rate')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Completion Rate')
        
        # 等待时间曲线
        axes[1, 0].plot(self.training_history['avg_waiting_times'])
        axes[1, 0].set_title('Average Waiting Time')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Waiting Time (s)')
        
        # 泊位利用率曲线
        axes[1, 1].plot(self.training_history['berth_utilizations'])
        axes[1, 1].set_title('Berth Utilization')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Utilization')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / 'training_curves.png', dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """主函数 - 训练单港GAT-PPO智能体"""
    import argparse
    
    parser = argparse.ArgumentParser(description='单港GAT-PPO训练')
    parser.add_argument('--port', type=str, default='gulfport', 
                       choices=['baton_rouge', 'new_orleans', 'south_louisiana', 'gulfport'],
                       help='港口名称')
    parser.add_argument('--episodes', type=int, default=100, help='训练轮数')
    parser.add_argument('--eval_episodes', type=int, default=10, help='评估轮数')
    
    args = parser.parse_args()
    
    # 创建训练器
    trainer = SinglePortTrainer(args.port)
    
    # 训练
    logger.info(f"开始训练 {args.port} 港口的GAT-PPO智能体")
    training_history = trainer.train(num_episodes=args.episodes)
    
    # 评估
    logger.info("开始评估训练好的智能体")
    eval_results = trainer.evaluate(num_episodes=args.eval_episodes)
    
    # 绘制训练曲线
    trainer.plot_training_curves()
    
    # 保存最终结果
    final_results = {
        'port_name': args.port,
        'training_episodes': args.episodes,
        'eval_episodes': args.eval_episodes,
        'training_history': training_history,
        'eval_results': eval_results,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = trainer.save_dir / "final_results.json"
    with open(results_file, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    logger.info(f"训练完成！结果保存到: {trainer.save_dir}")

if __name__ == "__main__":
    main()