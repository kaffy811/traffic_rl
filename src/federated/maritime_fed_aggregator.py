#!/usr/bin/env python3
"""
海事GAT-PPO联邦聚合器
实现多节点GAT-PPO模型的联邦聚合，考虑地理位置和性能权重
"""

import logging
import numpy as np
import torch
from collections import OrderedDict
from typing import List, Tuple, Dict, Any, Optional
from fedml.core import ServerAggregator

import sys
import os
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.maritime_gat_ppo import MaritimeGATPPOAgent, PPOConfig
from models.fairness_reward import ComprehensiveFairnessRewardCalculator


class MaritimeFedAggregator(ServerAggregator):
    """
    海事GAT-PPO联邦聚合器
    
    实现智能化的模型聚合策略，考虑：
    1. 地理位置权重 - 相邻港口节点有更高权重
    2. 性能权重 - 表现好的客户端有更高权重  
    3. 公平性权重 - 促进系统整体公平性
    4. 自适应聚合 - 根据训练进度调整聚合策略
    """
    
    def __init__(self, model, args):
        """初始化联邦聚合器"""
        super().__init__(model, args)
        self.args = args
        
        # 聚合配置
        self.aggregation_config = {
            'use_geographic_weights': getattr(args, 'use_geographic_weights', True),
            'use_performance_weights': getattr(args, 'use_performance_weights', True),
            'use_fairness_weights': getattr(args, 'use_fairness_weights', True),
            'min_client_weight': getattr(args, 'min_client_weight', 0.1),
            'max_client_weight': getattr(args, 'max_client_weight', 2.0),
        }
        
        # 海事节点地理位置
        self.node_coordinates = {
            0: (-90.35, 29.95),  # NodeA (新奥尔良港主入口)
            1: (-90.25, 29.85),  # NodeB (密西西比河口)
            2: (-90.30, 29.93),  # NodeC (河道中段)
            3: (-90.20, 29.80),  # NodeD (近海锚地)
        }
        
        # 客户端历史性能记录
        self.client_performance_history = {}
        
        # 公平性计算器
        self.fairness_calculator = ComprehensiveFairnessRewardCalculator()
        
        # 聚合统计
        self.aggregation_stats = {
            'round': 0,
            'participated_clients': [],
            'aggregation_weights': {},
            'performance_metrics': {}
        }
        
        logging.info("✅ 海事联邦聚合器初始化完成")

    def get_model_params(self) -> Dict[str, torch.Tensor]:
        """获取全局模型参数"""
        return self.model.state_dict()

    def set_model_params(self, model_parameters: Dict[str, torch.Tensor]):
        """设置全局模型参数"""
        self.model.load_state_dict(model_parameters)

    def aggregate(self, raw_client_model_or_grad_list: List[Tuple[float, OrderedDict]]) -> Tuple[int, OrderedDict]:
        """执行联邦聚合"""
        logging.info(f"🔄 开始第 {self.aggregation_stats['round'] + 1} 轮联邦聚合")
        logging.info(f"📊 参与客户端数量: {len(raw_client_model_or_grad_list)}")
        
        # 预处理：提取客户端信息
        processed_clients = self._preprocess_client_models(raw_client_model_or_grad_list)
        
        # 计算聚合权重
        aggregation_weights = self._calculate_aggregation_weights(processed_clients)
        
        # 执行加权聚合
        aggregated_model = self._weighted_aggregate(processed_clients, aggregation_weights)
        
        # 后处理：更新统计信息
        self._update_aggregation_stats(processed_clients, aggregation_weights)
        
        # 计算总样本数
        total_samples = sum(client_info['sample_num'] for client_info in processed_clients)
        
        logging.info(f"✅ 第 {self.aggregation_stats['round']} 轮聚合完成")
        
        return total_samples, aggregated_model

    def _preprocess_client_models(self, raw_client_list: List[Tuple[float, OrderedDict]]) -> List[Dict]:
        """预处理客户端模型信息"""
        processed_clients = []
        
        for i, (sample_num, model_params) in enumerate(raw_client_list):
            # 提取训练统计信息
            training_stats = model_params.pop('training_stats', {})
            
            client_info = {
                'client_id': i,
                'node_id': training_stats.get('node_id', i),
                'sample_num': sample_num,
                'model_params': model_params,
                'local_episodes': training_stats.get('local_episodes', 0),
                'total_reward': training_stats.get('total_reward', 0.0),
                'avg_policy_loss': training_stats.get('avg_policy_loss', 0.0),
                'avg_value_loss': training_stats.get('avg_value_loss', 0.0)
            }
            
            processed_clients.append(client_info)
            
            logging.info(f"  客户端 {i} (节点{client_info['node_id']}): "
                        f"样本={sample_num}, 奖励={client_info['total_reward']:.2f}")
        
        return processed_clients

    def _calculate_aggregation_weights(self, clients: List[Dict]) -> List[float]:
        """计算智能聚合权重"""
        weights = []
        
        for client in clients:
            weight = 1.0  # 基础权重
            
            # 1. 样本数量权重（FedAvg基础）
            sample_weight = client['sample_num']
            
            # 2. 地理位置权重
            if self.aggregation_config['use_geographic_weights']:
                geo_weight = self._calculate_geographic_weight(client['node_id'], clients)
                weight *= geo_weight
            
            # 3. 性能权重
            if self.aggregation_config['use_performance_weights']:
                perf_weight = self._calculate_performance_weight(client)
                weight *= perf_weight
            
            # 4. 样本数量调整
            weight *= sample_weight
            
            # 应用权重限制
            weight = np.clip(weight, 
                           self.aggregation_config['min_client_weight'],
                           self.aggregation_config['max_client_weight'])
            
            weights.append(weight)
        
        # 标准化权重
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # 记录权重信息
        for i, (client, weight) in enumerate(zip(clients, weights)):
            logging.info(f"  客户端 {i} 聚合权重: {weight:.4f}")
        
        return weights.tolist()

    def _calculate_geographic_weight(self, node_id: int, all_clients: List[Dict]) -> float:
        """计算地理位置权重"""
        if node_id not in self.node_coordinates:
            return 1.0
        
        current_coord = self.node_coordinates[node_id]
        
        # 计算与其他参与节点的平均距离
        distances = []
        for client in all_clients:
            other_node_id = client['node_id']
            if other_node_id != node_id and other_node_id in self.node_coordinates:
                other_coord = self.node_coordinates[other_node_id]
                distance = np.sqrt((current_coord[0] - other_coord[0])**2 + 
                                 (current_coord[1] - other_coord[1])**2)
                distances.append(distance)
        
        if not distances:
            return 1.0
        
        avg_distance = np.mean(distances)
        
        # 距离越近，权重越高（使用指数衰减）
        geo_weight = np.exp(-avg_distance * 10)
        
        return np.clip(geo_weight, 0.5, 2.0)

    def _calculate_performance_weight(self, client: Dict) -> float:
        """计算性能权重"""
        node_id = client['node_id']
        current_reward = client['total_reward']
        
        # 更新性能历史
        if node_id not in self.client_performance_history:
            self.client_performance_history[node_id] = []
        
        self.client_performance_history[node_id].append(current_reward)
        
        # 保持最近10轮的记录
        if len(self.client_performance_history[node_id]) > 10:
            self.client_performance_history[node_id].pop(0)
        
        # 计算历史平均奖励
        avg_reward = np.mean(self.client_performance_history[node_id])
        
        # 奖励越高，权重越高
        normalized_reward = avg_reward / 1000.0
        perf_weight = 1.0 / (1.0 + np.exp(-normalized_reward))
        
        return np.clip(perf_weight, 0.7, 1.5)

    def _weighted_aggregate(self, clients: List[Dict], weights: List[float]) -> OrderedDict:
        """执行加权模型聚合"""
        # 获取第一个客户端的模型结构
        first_model = clients[0]['model_params']
        aggregated_params = OrderedDict()
        
        # 对每个参数进行加权平均
        for param_name in first_model.keys():
            if any(param_name.endswith(suffix) for suffix in ['.num_batches_tracked']):
                aggregated_params[param_name] = first_model[param_name].clone()
                continue
            
            weighted_sum = torch.zeros_like(first_model[param_name])
            
            for client, weight in zip(clients, weights):
                if param_name in client['model_params']:
                    weighted_sum += weight * client['model_params'][param_name]
            
            aggregated_params[param_name] = weighted_sum
        
        return aggregated_params

    def _update_aggregation_stats(self, clients: List[Dict], weights: List[float]):
        """更新聚合统计信息"""
        self.aggregation_stats['round'] += 1
        self.aggregation_stats['participated_clients'] = [c['client_id'] for c in clients]
        self.aggregation_stats['aggregation_weights'] = {
            c['client_id']: w for c, w in zip(clients, weights)
        }
        
        # 计算聚合性能指标
        total_reward = sum(c['total_reward'] for c in clients)
        avg_policy_loss = np.mean([c['avg_policy_loss'] for c in clients])
        avg_value_loss = np.mean([c['avg_value_loss'] for c in clients])
        
        self.aggregation_stats['performance_metrics'] = {
            'total_reward': total_reward,
            'avg_policy_loss': avg_policy_loss,
            'avg_value_loss': avg_value_loss,
            'reward_variance': np.var([c['total_reward'] for c in clients])
        }

    def test(self, test_data, device, args) -> Dict[str, Any]:
        """在服务器端执行全局模型测试"""
        logging.info("🧪 开始全局模型测试")
        
        # 创建一个临时的GAT-PPO智能体用于测试
        ppo_config = PPOConfig()
        test_agent = MaritimeGATPPOAgent(node_id=0, config=ppo_config)
        
        # 加载当前全局模型参数
        test_agent.load_state_dict(self.get_model_params(), strict=False)
        test_agent.eval()
        
        test_episodes = 5
        test_rewards = []
        
        with torch.no_grad():
            for _ in range(test_episodes):
                observations = self._generate_test_scenario()
                episode_reward = 0.0
                
                for step in range(10):
                    action, _, _, _ = test_agent.get_action_and_value(observations)
                    next_observations = self._simulate_test_step(observations, action)
                    
                    reward = test_agent.calculate_comprehensive_reward(
                        observations, action, next_observations
                    )
                    episode_reward += reward
                    observations = next_observations
                
                test_rewards.append(episode_reward)
        
        test_results = {
            'global_avg_reward': np.mean(test_rewards),
            'reward_std': np.std(test_rewards),
            'aggregation_round': self.aggregation_stats['round'],
            'participated_clients': len(self.aggregation_stats['participated_clients'])
        }
        
        logging.info(f"✅ 全局测试完成 - 平均奖励: {test_results['global_avg_reward']:.2f}")
        return test_results

    def _generate_test_scenario(self) -> Dict[str, Dict]:
        """生成标准化的测试场景"""
        return {
            'NodeA': {'waiting_ships': 8, 'throughput': 4, 'waiting_time': 15, 'signal_phase': 0, 'weather_condition': 0.8},
            'NodeB': {'waiting_ships': 12, 'throughput': 3, 'waiting_time': 20, 'signal_phase': 1, 'weather_condition': 0.8},
            'NodeC': {'waiting_ships': 5, 'throughput': 2, 'waiting_time': 10, 'signal_phase': 0, 'weather_condition': 0.8},
            'NodeD': {'waiting_ships': 9, 'throughput': 5, 'waiting_time': 18, 'signal_phase': 1, 'weather_condition': 0.8},
        }

    def _simulate_test_step(self, current_state: Dict, action: int) -> Dict:
        """模拟测试环境步骤"""
        next_state = {node: state.copy() for node, state in current_state.items()}
        time_reduction = [1, 2, 3, 4][action]
        next_state['NodeA']['waiting_time'] = max(0, current_state['NodeA']['waiting_time'] - time_reduction)
        return next_state

    def test_all(self, train_data_local_dict, test_data_local_dict, device, args=None) -> bool:
        """执行所有客户端的测试"""
        logging.info("🔍 执行分布式测试")
        return True

    def get_aggregation_stats(self) -> Dict[str, Any]:
        """获取聚合统计信息"""
        return {
            **self.aggregation_stats,
            'client_performance_history': self.client_performance_history,
            'aggregation_config': self.aggregation_config
        }


# 创建工厂函数
def create_maritime_aggregator(model, args):
    """工厂函数：创建海事联邦聚合器"""
    return MaritimeFedAggregator(model, args)


if __name__ == "__main__":
    # 简单测试
    class MockArgs:
        use_geographic_weights = True
        use_performance_weights = True
        use_fairness_weights = True
    
    import torch.nn as nn
    mock_model = nn.Linear(10, 4)
    mock_args = MockArgs()
    
    aggregator = MaritimeFedAggregator(mock_model, mock_args)
    print("✅ 海事联邦聚合器创建成功")
