#!/usr/bin/env python3
"""
多端口CityFlow联邦学习系统
基于真实CityFlow仿真的多端口海事交通控制联邦学习框架
每个端口运行独立的CityFlow仿真环境
"""

import os
import sys
import json
import numpy as np
import torch
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import logging
import subprocess
import multiprocessing as mp

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "FedML" / "python"))

# CityFlow导入
try:
    import cityflow
    CITYFLOW_AVAILABLE = True
    print("✅ CityFlow 可用")
except ImportError:
    CITYFLOW_AVAILABLE = False
    print("⚠️ CityFlow 不可用，将使用模拟环境")

# 自定义模块导入
from src.models.maritime_gat_ppo import MaritimeGATPPOAgent
from src.models.fairness_reward import AlphaFairRewardCalculator
from src.federated.real_data_collector import RealDataCollector, initialize_data_collector

class PortEnvironment:
    """单个港口的CityFlow环境"""
    
    def __init__(self, port_id: int, port_name: str, topology_size: str = "3x3"):
        self.port_id = port_id
        self.port_name = port_name
        self.topology_size = topology_size
        
        # 配置文件路径
        self.topology_dir = project_root / "topologies"
        self.config_file = self.topology_dir / f"maritime_{topology_size}_config.json"
        
        # CityFlow引擎
        self.cityflow_engine = None
        self.current_step = 0
        self.max_steps = 3600  # 1小时仿真
        
        # 状态和观测
        self.last_state = None
        self.action_space_size = 4  # 港口控制动作
        
        # 性能指标
        self.metrics = {
            'total_waiting_time': 0.0,
            'average_speed': 0.0,
            'throughput': 0.0,
            'queue_length': 0.0,
            'safety_score': 1.0
        }
        
        self._initialize_environment()
        
    def _initialize_environment(self):
        """初始化港口环境"""
        try:
            if CITYFLOW_AVAILABLE and self.config_file.exists():
                # 为每个端口创建独立的配置
                port_config = self._create_port_specific_config()
                self.cityflow_engine = cityflow.Engine(port_config, thread_num=1)
                print(f"✅ 端口 {self.port_name} CityFlow环境初始化成功")
            else:
                print(f"⚠️ 端口 {self.port_name} 使用模拟环境")
                self._init_mock_environment()
                
        except Exception as e:
            print(f"❌ 端口 {self.port_name} 环境初始化失败: {e}")
            self._init_mock_environment()
    
    def _create_port_specific_config(self) -> str:
        """为每个端口创建特定的配置文件"""
        # 读取基础配置
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        
        # 修改配置以适应特定端口
        port_config_file = self.topology_dir / f"maritime_{self.topology_size}_{self.port_name}_config.json"
        
        # 修改种子以确保不同端口有不同的随机性
        config["seed"] = 42 + self.port_id * 100
        
        # 修改输出文件路径
        config["roadnetLogFile"] = f"maritime_{self.topology_size}_{self.port_name}_replay_roadnet.json"
        config["replayLogFile"] = f"maritime_{self.topology_size}_{self.port_name}_replay.txt"
        
        # 保存端口特定配置
        with open(port_config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return str(port_config_file)
    
    def _init_mock_environment(self):
        """初始化模拟环境"""
        self.cityflow_engine = None
        # 初始化模拟状态
        self.mock_state = {
            'vehicles': np.random.randint(10, 50),
            'waiting_vehicles': np.random.randint(5, 20),
            'average_speed': np.random.uniform(5, 15),
            'queue_lengths': np.random.randint(0, 10, size=4).tolist()
        }
    
    def reset(self) -> np.ndarray:
        """重置环境"""
        self.current_step = 0
        
        if self.cityflow_engine:
            self.cityflow_engine.reset()
        else:
            self._init_mock_environment()
        
        return self.get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """执行一步动作"""
        if self.cityflow_engine:
            return self._cityflow_step(action)
        else:
            return self._mock_step(action)
    
    def _cityflow_step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """CityFlow环境步进"""
        # 应用动作到信号灯控制
        self._apply_action_to_signals(action)
        
        # 执行一步仿真
        self.cityflow_engine.next_step()
        self.current_step += 1
        
        # 获取新状态
        state = self._get_cityflow_state()
        
        # 计算奖励
        reward = self._calculate_reward(state, action)
        
        # 检查是否结束
        done = self.current_step >= self.max_steps
        
        # 更新指标
        self._update_metrics(state)
        
        info = {
            'step': self.current_step,
            'metrics': self.metrics.copy(),
            'port': self.port_name
        }
        
        return state, reward, done, info
    
    def _mock_step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """模拟环境步进"""
        self.current_step += 1
        
        # 模拟动作效果
        action_effects = {
            0: 0.95,  # 保持当前
            1: 1.1,   # 增加通量
            2: 0.9,   # 减少等待
            3: 1.05   # 平衡策略
        }
        
        effect = action_effects.get(action, 1.0)
        
        # 更新模拟状态
        self.mock_state['vehicles'] = max(5, int(self.mock_state['vehicles'] * np.random.uniform(0.9, 1.1)))
        self.mock_state['waiting_vehicles'] = max(0, int(self.mock_state['waiting_vehicles'] * effect))
        self.mock_state['average_speed'] = np.clip(
            self.mock_state['average_speed'] * np.random.uniform(0.95, 1.05) * (2-effect), 
            2, 20
        )
        
        # 构建状态向量
        state = np.array([
            self.mock_state['vehicles'] / 100.0,
            self.mock_state['waiting_vehicles'] / 50.0,
            self.mock_state['average_speed'] / 20.0,
            np.mean(self.mock_state['queue_lengths']) / 10.0
        ], dtype=np.float32)
        
        # 计算奖励
        reward = self._calculate_mock_reward(state, action)
        
        done = self.current_step >= self.max_steps
        
        # 更新指标
        self.metrics.update({
            'total_waiting_time': self.mock_state['waiting_vehicles'] * 2.0,
            'average_speed': self.mock_state['average_speed'],
            'throughput': max(0, 50 - self.mock_state['waiting_vehicles']),
            'queue_length': np.mean(self.mock_state['queue_lengths']),
            'safety_score': np.random.uniform(0.8, 1.0)
        })
        
        info = {
            'step': self.current_step,
            'metrics': self.metrics.copy(),
            'port': self.port_name
        }
        
        return state, reward, done, info
    
    def _apply_action_to_signals(self, action: int):
        """将动作应用到信号灯控制"""
        if not self.cityflow_engine:
            return
            
        # 获取所有信号灯
        signals = self.cityflow_engine.get_lane_vehicle_count()
        
        # 根据动作调整信号灯相位
        action_to_phase = {
            0: 0,  # 保持当前相位
            1: 1,  # 切换到高通量相位
            2: 2,  # 切换到减少等待相位
            3: 3   # 平衡相位
        }
        
        target_phase = action_to_phase.get(action, 0)
        
        # 设置信号灯相位（如果有多个信号灯，可以设置不同策略）
        try:
            intersections = self.cityflow_engine.get_intersection_ids()
            for intersection_id in intersections[:1]:  # 只控制第一个交叉口作为示例
                self.cityflow_engine.set_tl_phase(intersection_id, target_phase)
        except:
            pass  # 忽略设置错误
    
    def _get_cityflow_state(self) -> np.ndarray:
        """从CityFlow获取状态"""
        try:
            # 获取基础状态信息
            lane_count = self.cityflow_engine.get_lane_vehicle_count()
            lane_waiting = self.cityflow_engine.get_lane_waiting_vehicle_count()
            vehicle_speed = self.cityflow_engine.get_vehicle_speed()
            
            # 计算聚合指标
            total_vehicles = sum(lane_count.values()) if lane_count else 0
            total_waiting = sum(lane_waiting.values()) if lane_waiting else 0
            avg_speed = np.mean(list(vehicle_speed.values())) if vehicle_speed else 0
            
            # 构建状态向量
            state = np.array([
                total_vehicles / 100.0,      # 归一化总车辆数
                total_waiting / 50.0,        # 归一化等待车辆数
                avg_speed / 20.0,            # 归一化平均速度
                len(lane_count) / 20.0       # 归一化车道数
            ], dtype=np.float32)
            
            return state
            
        except Exception as e:
            print(f"获取CityFlow状态失败: {e}")
            # 返回默认状态
            return np.array([0.1, 0.1, 0.5, 0.2], dtype=np.float32)
    
    def _calculate_reward(self, state: np.ndarray, action: int) -> float:
        """计算奖励"""
        # 基础效率奖励
        efficiency_reward = (state[2] * 10) - (state[1] * 5)  # 速度奖励 - 等待惩罚
        
        # 吞吐量奖励
        throughput_reward = (state[0] * 2) if state[1] < 0.3 else 0
        
        # 动作奖励
        action_rewards = {0: 0, 1: 2, 2: 1, 3: 1.5}
        action_reward = action_rewards.get(action, 0)
        
        # 稳定性奖励（避免极端状态）
        stability_reward = 0
        if 0.2 < state[0] < 0.8 and state[1] < 0.4:
            stability_reward = 2
        
        total_reward = efficiency_reward + throughput_reward + action_reward + stability_reward
        return float(total_reward)
    
    def _calculate_mock_reward(self, state: np.ndarray, action: int) -> float:
        """计算模拟奖励"""
        return self._calculate_reward(state, action)
    
    def _update_metrics(self, state: np.ndarray):
        """更新性能指标"""
        self.metrics.update({
            'total_waiting_time': state[1] * 50 * 2,  # 等待车辆 * 平均等待时间
            'average_speed': state[2] * 20,           # 反归一化速度
            'throughput': max(0, state[0] * 100 - state[1] * 50),  # 通过量
            'queue_length': state[1] * 10,            # 队列长度
            'safety_score': min(1.0, 1.2 - state[1]) # 安全分数
        })
    
    def get_state(self) -> np.ndarray:
        """获取当前状态"""
        if self.cityflow_engine:
            return self._get_cityflow_state()
        else:
            return np.array([
                self.mock_state['vehicles'] / 100.0,
                self.mock_state['waiting_vehicles'] / 50.0,
                self.mock_state['average_speed'] / 20.0,
                np.mean(self.mock_state['queue_lengths']) / 10.0
            ], dtype=np.float32)
    
    def get_metrics(self) -> Dict[str, float]:
        """获取当前指标"""
        return self.metrics.copy()
    
    def close(self):
        """关闭环境"""
        if self.cityflow_engine:
            try:
                # CityFlow没有显式的close方法，直接设为None
                self.cityflow_engine = None
            except:
                pass

class MultiPortFederatedAgent:
    """多端口联邦学习智能体"""
    
    def __init__(self, port_id: int, port_name: str, topology_size: str = "3x3"):
        self.port_id = port_id
        self.port_name = port_name
        self.topology_size = topology_size
        
        # 端口环境
        self.env = PortEnvironment(port_id, port_name, topology_size)
        
        # GAT-PPO智能体
        self.state_dim = 4  # 状态维度
        self.action_dim = 4  # 动作维度
        self.node_num = 9 if topology_size == "3x3" else 16  # 根据拓扑确定节点数
        
        self.gat_ppo_agent = MaritimeGATPPOAgent(
            node_num=self.node_num,
            node_dim=self.state_dim,
            action_dim=self.action_dim
        )
        
        # 公平性奖励计算器
        self.fairness_calculator = AlphaFairRewardCalculator(alpha=0.5)
        
        # 训练历史
        self.training_history = []
        self.episode_rewards = []
        
        print(f"✅ 端口 {port_name} 智能体初始化完成")
    
    def train_episode(self, max_steps: int = 1000) -> Dict[str, float]:
        """训练一个episode"""
        state = self.env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_metrics = []
        
        # 构建节点特征（GAT需要）
        node_features = self._build_node_features(state)
        
        for step in range(max_steps):
            # GAT-PPO决策
            action, action_prob = self.gat_ppo_agent.select_action(node_features)
            
            # 环境交互
            next_state, reward, done, info = self.env.step(action)
            
            # 计算公平性增强奖励
            fairness_reward = self.fairness_calculator.calculate_reward(
                base_reward=reward,
                current_state=state,
                action=action,
                agent_id=self.port_id,
                other_agents_states={}  # 在联邦设置中，这里不包含其他智能体状态
            )
            
            total_reward = reward + fairness_reward
            
            # 构建下一状态的节点特征
            next_node_features = self._build_node_features(next_state)
            
            # 存储经验
            self.gat_ppo_agent.store_experience(
                state=node_features,
                action=action,
                reward=total_reward,
                next_state=next_node_features,
                done=done,
                action_prob=action_prob
            )
            
            # 更新状态
            state = next_state
            node_features = next_node_features
            episode_reward += total_reward
            episode_steps += 1
            
            # 记录指标
            episode_metrics.append(info['metrics'])
            
            if done:
                break
        
        # 执行PPO更新
        training_stats = self.gat_ppo_agent.update()
        
        # 计算episode统计
        avg_metrics = {}
        if episode_metrics:
            for key in episode_metrics[0].keys():
                avg_metrics[f'avg_{key}'] = np.mean([m[key] for m in episode_metrics])
        
        episode_result = {
            'episode_reward': episode_reward,
            'episode_steps': episode_steps,
            'avg_reward_per_step': episode_reward / max(episode_steps, 1),
            **avg_metrics,
            **training_stats,
            'port_name': self.port_name,
            'port_id': self.port_id
        }
        
        self.training_history.append(episode_result)
        self.episode_rewards.append(episode_reward)
        
        return episode_result
    
    def _build_node_features(self, state: np.ndarray) -> torch.Tensor:
        """构建GAT需要的节点特征"""
        # 将状态扩展到所有节点
        # 简化实现：每个节点使用相同的基础状态，加上位置编码
        node_features = []
        
        for i in range(self.node_num):
            # 基础状态特征
            base_features = state.copy()
            
            # 添加节点位置编码
            row = i // int(np.sqrt(self.node_num))
            col = i % int(np.sqrt(self.node_num))
            position_encoding = np.array([row / int(np.sqrt(self.node_num)), 
                                        col / int(np.sqrt(self.node_num))])
            
            # 组合特征
            combined_features = np.concatenate([base_features, position_encoding])
            node_features.append(combined_features)
        
        return torch.FloatTensor(node_features).unsqueeze(0)  # 添加batch维度
    
    def get_model_parameters(self) -> Dict:
        """获取模型参数（用于联邦学习）"""
        return self.gat_ppo_agent.get_model_parameters()
    
    def set_model_parameters(self, parameters: Dict):
        """设置模型参数（用于联邦学习）"""
        self.gat_ppo_agent.set_model_parameters(parameters)
    
    def get_training_statistics(self) -> Dict:
        """获取训练统计信息"""
        if not self.episode_rewards:
            return {}
        
        recent_rewards = self.episode_rewards[-10:]  # 最近10个episode
        
        return {
            'total_episodes': len(self.episode_rewards),
            'avg_episode_reward': np.mean(self.episode_rewards),
            'recent_avg_reward': np.mean(recent_rewards),
            'best_reward': max(self.episode_rewards),
            'reward_std': np.std(self.episode_rewards),
            'port_name': self.port_name,
            'port_id': self.port_id
        }
    
    def close(self):
        """关闭智能体"""
        self.env.close()

class MultiPortFederatedSystem:
    """多端口联邦学习系统"""
    
    def __init__(self, num_ports: int = 4, topology_size: str = "3x3"):
        self.num_ports = num_ports
        self.topology_size = topology_size
        
        # 端口名称映射
        self.port_names = {
            0: "new_orleans",
            1: "south_louisiana", 
            2: "baton_rouge",
            3: "gulfport"
        }
        
        # 创建端口智能体
        self.port_agents = {}
        for i in range(num_ports):
            port_name = self.port_names.get(i, f"port_{i}")
            self.port_agents[i] = MultiPortFederatedAgent(i, port_name, topology_size)
        
        # 联邦学习参数
        self.global_model_params = None
        
        # 数据收集器
        self.data_collector = initialize_data_collector("multi_port_cityflow_experiment")
        
        print(f"✅ 四港口联邦学习系统初始化完成 - {num_ports}个端口互相学习")
        print(f"   📍 参与港口: {[agent.port_name for agent in self.port_agents.values()]}")
        print(f"   🤝 联邦学习模式: 知识共享，数据隐私保护")
    
    def federated_training_round(self, episodes_per_agent: int = 5) -> Dict[str, Any]:
        """执行一轮四港口联邦学习 - 所有港口互相学习"""
        print(f"🔄 开始四港口联邦学习轮次 - 每个港口训练 {episodes_per_agent} episodes")
        print(f"   🤝 联邦学习模式: {len(self.port_agents)}个港口共享知识，隐私保护")
        
        round_results = {}
        client_models = {}
        
        # 1. 各端口本地训练
        for port_id, agent in self.port_agents.items():
            print(f"   📍 端口 {agent.port_name} 开始本地训练...")
            
            port_results = []
            for episode in range(episodes_per_agent):
                episode_result = agent.train_episode()
                port_results.append(episode_result)
                
                # 收集训练数据
                if self.data_collector:
                    self.data_collector.collect_training_data(
                        str(port_id), 
                        {
                            'avg_reward': episode_result['episode_reward'],
                            'avg_policy_loss': episode_result.get('policy_loss', 0),
                            'avg_value_loss': episode_result.get('value_loss', 0),
                            'total_episodes': 1
                        }
                    )
            
            # 获取本地模型参数
            client_models[port_id] = agent.get_model_parameters()
            
            # 计算端口平均结果
            avg_reward = np.mean([r['episode_reward'] for r in port_results])
            round_results[port_id] = {
                'port_name': agent.port_name,
                'episodes_trained': len(port_results),
                'avg_episode_reward': avg_reward,
                'training_results': port_results
            }
            
            print(f"   ✅ 端口 {agent.port_name} 训练完成 - 平均奖励: {avg_reward:.2f}")
        
        # 2. 联邦聚合
        print("   🔄 执行联邦模型聚合...")
        aggregated_params = self._federated_averaging(client_models)
        
        # 3. 分发全局模型
        for agent in self.port_agents.values():
            agent.set_model_parameters(aggregated_params)
        
        self.global_model_params = aggregated_params
        
        # 收集聚合数据
        if self.data_collector:
            avg_client_reward = np.mean([r['avg_episode_reward'] for r in round_results.values()])
            self.data_collector.collect_aggregation_data({
                'participating_clients': len(self.port_agents),
                'total_samples': sum(r['episodes_trained'] for r in round_results.values()),
                'aggregation_weights': {str(i): 1.0/len(self.port_agents) for i in range(len(self.port_agents))},
                'avg_client_reward': avg_client_reward
            })
        
        return round_results
    
    def _federated_averaging(self, client_models: Dict[int, Dict]) -> Dict:
        """四港口联邦平均聚合 - 所有港口知识融合"""
        if not client_models:
            return {}
        
        print(f"   🔄 聚合{len(client_models)}个港口的模型参数...")
        
        # 联邦平均聚合
        aggregated_params = {}
        num_clients = len(client_models)
        
        # 获取第一个客户端的参数结构
        first_client_params = next(iter(client_models.values()))
        
        # 为每个港口计算权重（可以基于性能动态调整）
        port_weights = {}
        for port_id in client_models.keys():
            # 简单平均权重，也可以基于端口性能调整
            port_weights[port_id] = 1.0 / num_clients
        
        print(f"   ⚖️ 港口权重: {port_weights}")
        
        for param_name in first_client_params.keys():
            # 计算加权平均
            param_sum = None
            total_weight = 0
            
            for port_id, client_params in client_models.items():
                if param_name in client_params:
                    weight = port_weights[port_id]
                    if param_sum is None:
                        param_sum = client_params[param_name].clone() * weight
                    else:
                        param_sum += client_params[param_name] * weight
                    total_weight += weight
            
            if param_sum is not None and total_weight > 0:
                aggregated_params[param_name] = param_sum / total_weight
        
        print(f"   ✅ 联邦聚合完成 - 融合了{num_clients}个港口的知识")
        return aggregated_params
    
    def run_federated_experiment(self, num_rounds: int = 10, episodes_per_round: int = 5):
        """运行完整的四港口联邦学习实验"""
        print(f"🚀 开始四港口联邦学习实验 - 港口间知识共享")
        print(f"   📊 联邦轮次: {num_rounds}")
        print(f"   🔄 每轮episode数: {episodes_per_round}")
        print(f"   🏭 参与港口: {[agent.port_name for agent in self.port_agents.values()]}")
        print(f"   🤝 学习模式: 每轮所有港口互相学习，共享最优策略")
        
        # 启动数据收集
        if self.data_collector:
            port_names = [agent.port_name for agent in self.port_agents.values()]
            self.data_collector.start_experiment(num_rounds, "Multi-Port-CityFlow-GAT-FedPPO")
        
        experiment_results = []
        
        for round_num in range(1, num_rounds + 1):
            print(f"\n📍 联邦学习轮次 {round_num}/{num_rounds}")
            
            # 启动轮次
            if self.data_collector:
                self.data_collector.start_round(round_num)
            
            # 执行联邦训练轮次
            round_result = self.federated_training_round(episodes_per_round)
            
            # 添加轮次信息
            round_result['round'] = round_num
            round_result['timestamp'] = datetime.now().isoformat()
            
            experiment_results.append(round_result)
            
            # 打印轮次总结
            avg_rewards = [r['avg_episode_reward'] for r in round_result.values() if isinstance(r, dict) and 'avg_episode_reward' in r]
            if avg_rewards:
                print(f"   📊 轮次 {round_num} 平均奖励: {np.mean(avg_rewards):.2f}")
        
        # 完成实验
        if self.data_collector:
            timestamp = self.data_collector.finish_experiment()
            print(f"✅ 实验数据已保存: {timestamp}")
        
        # 生成实验总结
        self._generate_experiment_summary(experiment_results)
        
        return experiment_results
    
    def _generate_experiment_summary(self, results: List[Dict]):
        """生成实验总结"""
        print(f"\n🎉 多端口联邦学习实验完成!")
        print("=" * 60)
        
        # 计算总体统计
        all_rewards = []
        for round_result in results:
            for port_result in round_result.values():
                if isinstance(port_result, dict) and 'avg_episode_reward' in port_result:
                    all_rewards.append(port_result['avg_episode_reward'])
        
        if all_rewards:
            print(f"📊 实验统计:")
            print(f"   总轮次: {len(results)}")
            print(f"   平均奖励: {np.mean(all_rewards):.2f}")
            print(f"   最佳奖励: {max(all_rewards):.2f}")
            print(f"   奖励标准差: {np.std(all_rewards):.2f}")
        
        # 各端口表现
        print(f"\n🏭 各端口表现:")
        for agent in self.port_agents.values():
            stats = agent.get_training_statistics()
            if stats:
                print(f"   {stats['port_name']}: 平均奖励 {stats['avg_episode_reward']:.2f}, "
                      f"最佳奖励 {stats['best_reward']:.2f}")
    
    def close(self):
        """关闭系统"""
        for agent in self.port_agents.values():
            agent.close()

def main():
    """主函数 - 演示四港口联邦学习"""
    print("🚀 启动四港口CityFlow联邦学习系统")
    
    # 创建四港口系统
    system = MultiPortFederatedSystem(num_ports=4, topology_size="3x3")
    
    try:
        # 运行实验
        results = system.run_federated_experiment(num_rounds=5, episodes_per_round=3)
        
        print("\n🎯 实验完成，可以使用以下命令生成可视化:")
        print("python src/federated/visualization_generator.py")
        
    except KeyboardInterrupt:
        print("\n⚠️ 实验被用户中断")
    except Exception as e:
        print(f"\n❌ 实验出错: {e}")
        import traceback
        traceback.print_exc()
    finally:
        system.close()

if __name__ == "__main__":
    main()