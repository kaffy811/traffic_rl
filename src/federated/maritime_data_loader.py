#!/usr/bin/env python3
"""
海事GAT-PPO数据加载器
为FedML框架提供海事数据加载功能
"""

import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Any, Tuple
import logging
from pathlib import Path
import json
import pandas as pd

logger = logging.getLogger(__name__)


class MaritimeEnvironmentDataset(Dataset):
    """海事环境数据集"""
    
    def __init__(self, data_source: str = "simulated", client_rank: int = 0, episodes: int = 100):
        """
        初始化海事数据集
        
        Args:
            data_source: 数据源类型 ("simulated" 或 "real")
            client_rank: 客户端rank (0-3)
            episodes: 生成的episode数量
        """
        self.data_source = data_source
        self.client_rank = client_rank
        self.episodes = episodes
        
        # 港口映射
        self.port_mapping = {
            0: "new_orleans",
            1: "south_louisiana", 
            2: "baton_rouge",
            3: "gulfport"
        }
        
        self.port_name = self.port_mapping.get(client_rank, f"port_{client_rank}")
        
        # 生成模拟数据
        self.data = self._generate_maritime_data()
        
        logger.info(f"📊 创建海事数据集 - 港口: {self.port_name}, Episodes: {len(self.data)}")
    
    def _generate_maritime_data(self):
        """生成模拟的海事环境数据"""
        data = []
        
        # 港口特征差异化
        port_configs = {
            "new_orleans": {"base_traffic": 15, "variance": 5, "peak_hours": [8, 17]},
            "south_louisiana": {"base_traffic": 12, "variance": 4, "peak_hours": [9, 16]},
            "baton_rouge": {"base_traffic": 8, "variance": 3, "peak_hours": [10, 15]},
            "gulfport": {"base_traffic": 10, "variance": 4, "peak_hours": [7, 18]}
        }
        
        config = port_configs.get(self.port_name, port_configs["new_orleans"])
        
        for episode in range(self.episodes):
            # 模拟一个episode的数据
            episode_data = {
                'observations': [],
                'actions': [],
                'rewards': [],
                'episode_id': episode,
                'port_name': self.port_name,
                'client_rank': self.client_rank
            }
            
            # 每个episode包含多个时间步
            for step in range(20):  # 20个时间步
                # 生成海事观测数据
                observation = self._generate_maritime_observation(config, step)
                
                # 生成动作（0-3的离散动作）
                action = np.random.randint(0, 4)
                
                # 计算奖励
                reward = self._calculate_step_reward(observation, action)
                
                episode_data['observations'].append(observation)
                episode_data['actions'].append(action)
                episode_data['rewards'].append(reward)
            
            data.append(episode_data)
        
        return data
    
    def _generate_maritime_observation(self, config: Dict, step: int) -> Dict:
        """生成海事观测数据"""
        # 模拟四个节点的观测数据
        observation = {}
        
        node_names = ['NodeA', 'NodeB', 'NodeC', 'NodeD']
        for i, node_name in enumerate(node_names):
            # 根据港口配置和时间步生成不同的数据
            base_traffic = config['base_traffic']
            if i == self.client_rank:
                # 当前客户端的节点数据稍有不同
                base_traffic *= 1.2
            
            observation[node_name] = {
                'waiting_ships': max(1, int(np.random.normal(base_traffic, config['variance']))),
                'throughput': np.random.uniform(0.5, 3.0),
                'waiting_time': np.random.uniform(5, 30),
                'queue_length': max(0, int(np.random.normal(8, 3))),
                'safety_score': np.random.uniform(0.6, 1.0)
            }
        
        return observation
    
    def _calculate_step_reward(self, observation: Dict, action: int) -> float:
        """计算步骤奖励"""
        # 基于观测和动作计算奖励
        total_waiting = sum(obs['waiting_ships'] for obs in observation.values())
        avg_throughput = np.mean([obs['throughput'] for obs in observation.values()])
        avg_safety = np.mean([obs['safety_score'] for obs in observation.values()])
        
        # 简单的奖励函数
        reward = (avg_throughput * 10) - (total_waiting * 0.1) + (avg_safety * 5)
        
        # 动作奖励调整
        if action == 0:  # 保持当前策略
            reward *= 1.0
        elif action == 1:  # 增加通行量
            reward *= 1.2
        elif action == 2:  # 减少等待时间
            reward *= 1.1
        else:  # 平衡策略
            reward *= 1.05
        
        return float(reward)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """返回一个episode的数据"""
        episode = self.data[idx]
        
        # 转换为张量格式（适配FedML期望的格式）
        # 这里我们返回一个简化的格式，实际训练时会使用observations
        
        # 将observations转换为特征向量
        obs = episode['observations'][0]  # 使用第一个观测作为特征
        features = []
        for node_name in ['NodeA', 'NodeB', 'NodeC', 'NodeD']:
            node_obs = obs[node_name]
            features.extend([
                node_obs['waiting_ships'],
                node_obs['throughput'], 
                node_obs['waiting_time'],
                node_obs['queue_length'],
                node_obs['safety_score']
            ])
        
        # 转换为张量
        x = torch.tensor(features, dtype=torch.float32)
        y = torch.tensor(episode['actions'][0], dtype=torch.long)  # 第一个动作作为标签
        
        return x, y


def load_maritime_data(args):
    """
    加载海事数据（FedML兼容接口）
    
    Returns:
        FedML期望的8元组格式数据
    """
    
    # 获取配置参数
    client_num_in_total = getattr(args, 'client_num_in_total', 4)
    batch_size = getattr(args, 'batch_size', 10)
    
    # 为所有客户端创建数据集
    train_data_local_dict = {}
    test_data_local_dict = {}
    train_data_local_num_dict = {}
    
    total_train_num = 0
    total_test_num = 0
    
    # 创建所有客户端的数据 (索引从0开始)
    for client_idx in range(client_num_in_total):
        logger.info(f"📊 为客户端 {client_idx} 创建数据集...")
        
        # 创建训练数据集
        train_dataset = MaritimeEnvironmentDataset(
            data_source="simulated",
            client_rank=client_idx,
            episodes=100  # 训练episodes
        )
        
        # 创建测试数据集
        test_dataset = MaritimeEnvironmentDataset(
            data_source="simulated", 
            client_rank=client_idx,
            episodes=20   # 测试episodes
        )
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False
        )
        
        # 存储到字典中 (使用客户端索引: 0, 1, 2, 3)
        train_data_local_dict[client_idx] = train_loader
        test_data_local_dict[client_idx] = test_loader
        train_data_local_num_dict[client_idx] = len(train_dataset)
        
        total_train_num += len(train_dataset)
        total_test_num += len(test_dataset)
    
    logger.info(f"✅ 海事数据加载完成 - 客户端数: {client_num_in_total}, 总训练: {total_train_num}, 总测试: {total_test_num}")
    logger.info(f"📋 数据索引范围: {list(train_data_local_dict.keys())}")
    
    # 全局数据集（用于服务器端评估，使用第一个客户端的数据）
    train_data_global = train_data_local_dict[0]
    test_data_global = test_data_local_dict[0]
    
    # 返回FedML期望的8元组格式
    return [
        total_train_num,              # train_data_num
        total_test_num,               # test_data_num  
        train_data_global,            # train_data_global
        test_data_global,             # test_data_global
        train_data_local_num_dict,    # train_data_local_num_dict
        train_data_local_dict,        # train_data_local_dict
        test_data_local_dict,         # test_data_local_dict
        4                             # class_num (动作空间维度)
    ]


# FedML数据加载注册
def create_maritime_dataset(args):
    """FedML数据集创建接口"""
    return load_maritime_data(args)


if __name__ == "__main__":
    # 测试数据加载器
    import argparse
    
    args = argparse.Namespace()
    args.rank = 1
    args.batch_size = 5
    
    dataset, output_dim = load_maritime_data(args)
    print(f"数据集创建成功，输出维度: {output_dim}")
    
    # 测试数据加载
    for i, (x, y) in enumerate(dataset['train']):
        print(f"批次 {i}: 输入形状 {x.shape}, 标签形状 {y.shape}")
        if i >= 2:  # 只测试前3个批次
            break