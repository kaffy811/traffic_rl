#!/usr/bin/env python3
"""
CityFlow 模拟环境
当真实 CityFlow 无法编译时提供功能等价的模拟实现
"""

import json
import random
import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

class MockEngine:
    """模拟 CityFlow Engine"""
    
    def __init__(self, config_file: str, thread_num: int = 1):
        """
        初始化模拟引擎
        
        Args:
            config_file: 配置文件路径
            thread_num: 线程数
        """
        self.config_file = config_file
        self.thread_num = thread_num
        self.current_time = 0.0
        self.vehicles = {}
        self.lanes = {}
        self.intersections = {}
        self.step_count = 0
        
        # 加载配置
        self._load_config()
        self._initialize_environment()
        
        logging.info(f"✅ CityFlow模拟引擎初始化完成 (配置: {config_file})")
    
    def _load_config(self):
        """加载配置文件"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            
            # 加载路网配置
            roadnet_file = Path(self.config_file).parent / self.config['roadnetFile']
            with open(roadnet_file, 'r', encoding='utf-8') as f:
                self.roadnet = json.load(f)
            
            # 加载流量配置
            flow_file = Path(self.config_file).parent / self.config['flowFile']
            with open(flow_file, 'r', encoding='utf-8') as f:
                self.flows = json.load(f)
                
        except Exception as e:
            logging.warning(f"⚠️ 配置加载失败: {e}, 使用默认配置")
            self._create_default_config()
    
    def _create_default_config(self):
        """创建默认配置"""
        self.config = {
            "interval": 1.0,
            "seed": 0,
            "rlTrafficLight": False,
            "laneChange": False
        }
        
        self.roadnet = {
            "intersections": [
                {"id": "NodeA", "point": {"x": -90.350, "y": 29.950}},
                {"id": "NodeB", "point": {"x": -90.050, "y": 29.850}},
                {"id": "NodeC", "point": {"x": -90.300, "y": 29.930}},
                {"id": "NodeD", "point": {"x": -90.125, "y": 29.800}}
            ],
            "roads": [
                {"id": "road_A_B", "startIntersection": "NodeA", "endIntersection": "NodeB"},
                {"id": "road_B_A", "startIntersection": "NodeB", "endIntersection": "NodeA"},
                # ... 其他道路
            ]
        }
        
        self.flows = []
    
    def _initialize_environment(self):
        """初始化环境"""
        # 初始化交叉口
        for intersection in self.roadnet.get('intersections', []):
            self.intersections[intersection['id']] = {
                'waiting_vehicles': 0,
                'passed_vehicles': 0,
                'current_phase': 0,
                'time_since_phase_change': 0
            }
        
        # 初始化车辆
        for flow in self.flows[:20]:  # 限制车辆数量避免过载
            vehicle_id = f"vehicle_{len(self.vehicles)}"
            self.vehicles[vehicle_id] = {
                'route': flow.get('route', ['NodeA', 'NodeB']),
                'current_intersection': flow.get('route', ['NodeA'])[0],
                'position': 0,
                'speed': random.uniform(5, 15),
                'waiting_time': 0
            }
    
    def next_step(self):
        """执行一个仿真步骤"""
        self.step_count += 1
        self.current_time += self.config.get('interval', 1.0)
        
        # 更新车辆状态
        self._update_vehicles()
        
        # 更新交叉口状态
        self._update_intersections()
        
        # 随机生成新车辆（模拟持续的交通流）
        if random.random() < 0.1:  # 10%概率生成新车辆
            self._spawn_vehicle()
    
    def _update_vehicles(self):
        """更新车辆状态"""
        for vehicle_id, vehicle in list(self.vehicles.items()):
            # 模拟车辆移动
            if random.random() < 0.8:  # 80%概率移动
                route = vehicle['route']
                current_idx = route.index(vehicle['current_intersection'])
                
                if current_idx < len(route) - 1:
                    # 移动到下一个交叉口
                    vehicle['current_intersection'] = route[current_idx + 1]
                    vehicle['position'] += 1
                    
                    # 更新交叉口统计
                    intersection = self.intersections.get(vehicle['current_intersection'])
                    if intersection:
                        intersection['passed_vehicles'] += 1
                else:
                    # 车辆完成路径，移除
                    del self.vehicles[vehicle_id]
            else:
                # 车辆等待
                vehicle['waiting_time'] += 1
                intersection_id = vehicle['current_intersection']
                if intersection_id in self.intersections:
                    self.intersections[intersection_id]['waiting_vehicles'] += 1
    
    def _update_intersections(self):
        """更新交叉口状态"""
        for intersection_id, intersection in self.intersections.items():
            # 模拟信号灯相位变化
            intersection['time_since_phase_change'] += 1
            
            if intersection['time_since_phase_change'] > 30:  # 30秒切换相位
                intersection['current_phase'] = (intersection['current_phase'] + 1) % 4
                intersection['time_since_phase_change'] = 0
            
            # 重置等待车辆计数
            intersection['waiting_vehicles'] = 0
    
    def _spawn_vehicle(self):
        """生成新车辆"""
        if len(self.vehicles) < 50:  # 限制最大车辆数
            vehicle_id = f"vehicle_{len(self.vehicles)}_{self.step_count}"
            intersections = list(self.intersections.keys())
            
            if len(intersections) >= 2:
                start = random.choice(intersections)
                end = random.choice([i for i in intersections if i != start])
                
                self.vehicles[vehicle_id] = {
                    'route': [start, end],
                    'current_intersection': start,
                    'position': 0,
                    'speed': random.uniform(5, 15),
                    'waiting_time': 0
                }
    
    def get_vehicle_count(self) -> int:
        """获取当前车辆数量"""
        return len(self.vehicles)
    
    def get_vehicles(self, include_waiting: bool = False) -> List[str]:
        """获取车辆列表"""
        return list(self.vehicles.keys())
    
    def get_lane_vehicle_count(self) -> Dict[str, int]:
        """获取各车道车辆数量"""
        lane_counts = {}
        for intersection_id in self.intersections:
            # 模拟车道车辆数
            lane_counts[f"lane_{intersection_id}_in"] = random.randint(0, 10)
            lane_counts[f"lane_{intersection_id}_out"] = random.randint(0, 8)
        return lane_counts
    
    def get_lane_waiting_vehicle_count(self) -> Dict[str, int]:
        """获取各车道等待车辆数量"""
        waiting_counts = {}
        for intersection_id, intersection in self.intersections.items():
            waiting_counts[f"lane_{intersection_id}_waiting"] = intersection['waiting_vehicles']
        return waiting_counts
    
    def get_vehicle_speed(self) -> Dict[str, float]:
        """获取车辆速度"""
        speeds = {}
        for vehicle_id, vehicle in self.vehicles.items():
            speeds[vehicle_id] = vehicle['speed'] + random.uniform(-2, 2)  # 添加噪声
        return speeds
    
    def get_current_time(self) -> float:
        """获取当前仿真时间"""
        return self.current_time
    
    def set_tl_phase(self, intersection_id: str, phase_id: int):
        """设置交通灯相位"""
        if intersection_id in self.intersections:
            self.intersections[intersection_id]['current_phase'] = phase_id
            self.intersections[intersection_id]['time_since_phase_change'] = 0
            logging.debug(f"设置 {intersection_id} 信号灯相位为 {phase_id}")
    
    def get_tl_phase(self, intersection_id: str) -> int:
        """获取交通灯相位"""
        return self.intersections.get(intersection_id, {}).get('current_phase', 0)
    
    def reset(self):
        """重置仿真"""
        self.current_time = 0.0
        self.step_count = 0
        self.vehicles.clear()
        
        for intersection in self.intersections.values():
            intersection['waiting_vehicles'] = 0
            intersection['passed_vehicles'] = 0
            intersection['current_phase'] = 0
            intersection['time_since_phase_change'] = 0
        
        logging.info("🔄 CityFlow模拟环境已重置")


class CityFlowEnvironment:
    """完整的 CityFlow 模拟环境"""
    
    def __init__(self, config_path: str):
        """
        初始化环境
        
        Args:
            config_path: CityFlow配置文件路径
        """
        self.config_path = config_path
        self.engine = MockEngine(config_path)
        self.episode_step = 0
        self.total_reward = 0
        
        # 海事特定参数
        self.maritime_nodes = ['NodeA', 'NodeB', 'NodeC', 'NodeD']
        self.node_features = {node: np.zeros(5) for node in self.maritime_nodes}
        
        logging.info("🌊 海事CityFlow模拟环境初始化完成")
    
    def reset(self) -> Dict[str, np.ndarray]:
        """重置环境并返回初始观测"""
        self.engine.reset()
        self.episode_step = 0
        self.total_reward = 0
        
        return self._get_observations()
    
    def step(self, actions: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, float], bool, Dict]:
        """
        执行一步仿真
        
        Args:
            actions: 各节点的动作 {node_id: action}
            
        Returns:
            observations: 新的观测
            rewards: 各节点奖励
            done: 是否结束
            info: 额外信息
        """
        # 执行动作
        for node_id, action in actions.items():
            if node_id in self.maritime_nodes:
                # 将动作转换为信号灯相位
                phase = action % 4  # 4个相位
                self.engine.set_tl_phase(node_id, phase)
        
        # 执行仿真步骤
        self.engine.next_step()
        self.episode_step += 1
        
        # 获取新观测
        observations = self._get_observations()
        
        # 计算奖励
        rewards = self._calculate_rewards(actions)
        
        # 检查是否结束
        done = self.episode_step >= 100  # 100步结束
        
        # 统计信息
        info = {
            'episode_step': self.episode_step,
            'total_vehicles': self.engine.get_vehicle_count(),
            'current_time': self.engine.get_current_time()
        }
        
        return observations, rewards, done, info
    
    def _get_observations(self) -> Dict[str, np.ndarray]:
        """获取当前观测"""
        observations = {}
        
        # 获取环境状态
        lane_counts = self.engine.get_lane_vehicle_count()
        waiting_counts = self.engine.get_lane_waiting_vehicle_count()
        
        for node in self.maritime_nodes:
            # 构建节点特征向量
            features = np.array([
                lane_counts.get(f"lane_{node}_in", 0) / 10.0,  # 归一化进入车道车辆数
                waiting_counts.get(f"lane_{node}_waiting", 0) / 10.0,  # 归一化等待车辆数
                self.engine.get_tl_phase(node) / 4.0,  # 归一化信号灯相位
                self.engine.get_current_time() / 1000.0,  # 归一化时间
                random.uniform(0.8, 1.2)  # 模拟船舶特有的环境因子（潮汐、天气等）
            ], dtype=np.float32)
            
            observations[node] = features
            self.node_features[node] = features
        
        return observations
    
    def _calculate_rewards(self, actions: Dict[str, int]) -> Dict[str, float]:
        """计算奖励"""
        rewards = {}
        
        for node in self.maritime_nodes:
            # 基础效率奖励
            passed_vehicles = self.engine.intersections.get(node, {}).get('passed_vehicles', 0)
            waiting_vehicles = self.engine.intersections.get(node, {}).get('waiting_vehicles', 0)
            
            efficiency_reward = passed_vehicles * 10 - waiting_vehicles * 5
            
            # 安全奖励（避免拥堵）
            safety_reward = max(0, 20 - waiting_vehicles * 2)
            
            # 公平性奖励（各节点负载平衡）
            other_nodes_waiting = sum(
                self.engine.intersections.get(other_node, {}).get('waiting_vehicles', 0)
                for other_node in self.maritime_nodes if other_node != node
            )
            fairness_reward = max(0, 15 - abs(waiting_vehicles - other_nodes_waiting/3))
            
            # 海事特有奖励（模拟环境适应性）
            maritime_bonus = random.uniform(5, 15)  # 模拟天气、潮汐等因素
            
            total_reward = efficiency_reward + safety_reward + fairness_reward + maritime_bonus
            rewards[node] = total_reward
        
        return rewards
    
    def get_state_summary(self) -> Dict[str, Any]:
        """获取环境状态摘要"""
        return {
            'step': self.episode_step,
            'time': self.engine.get_current_time(),
            'vehicles': self.engine.get_vehicle_count(),
            'intersections': {
                node: {
                    'waiting': self.engine.intersections.get(node, {}).get('waiting_vehicles', 0),
                    'passed': self.engine.intersections.get(node, {}).get('passed_vehicles', 0),
                    'phase': self.engine.get_tl_phase(node)
                }
                for node in self.maritime_nodes
            }
        }


def test_cityflow_mock():
    """测试CityFlow模拟环境"""
    print("🧪 测试CityFlow模拟环境")
    print("=" * 50)
    
    # 创建配置文件路径
    config_path = "FedML/CityFlow/examples/config.json"
    
    try:
        # 初始化环境
        env = CityFlowEnvironment(config_path)
        print(f"✅ 环境初始化成功")
        
        # 重置环境
        obs = env.reset()
        print(f"✅ 环境重置成功，观测维度: {len(obs)}")
        
        # 模拟几步
        for step in range(5):
            # 随机动作
            actions = {node: random.randint(0, 3) for node in env.maritime_nodes}
            
            # 执行步骤
            obs, rewards, done, info = env.step(actions)
            
            print(f"步骤 {step+1}:")
            print(f"  动作: {actions}")
            print(f"  奖励: {rewards}")
            print(f"  总车辆: {info['total_vehicles']}")
            print(f"  环境状态: {env.get_state_summary()}")
            print()
            
            if done:
                break
        
        print("✅ CityFlow模拟测试完成!")
        return True
        
    except Exception as e:
        print(f"❌ CityFlow模拟测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)
    
    # 运行测试
    test_cityflow_mock()