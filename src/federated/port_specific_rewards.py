#!/usr/bin/env python3
"""
港口特定奖励函数 - 为不同港口设计差异化奖励机制
"""

import numpy as np
from typing import Dict, Any, List
from abc import ABC, abstractmethod

class BaseRewardFunction(ABC):
    """基础奖励函数抽象类"""
    
    def __init__(self, port_name: str):
        self.port_name = port_name
        self.reward_weights = self._get_default_weights()
    
    @abstractmethod
    def _get_default_weights(self) -> Dict[str, float]:
        """获取默认权重配置"""
        pass
    
    @abstractmethod
    def calculate_reward(self, state: Dict[str, Any], action: int, next_state: Dict[str, Any]) -> float:
        """计算奖励值"""
        pass
    
    def _calculate_waiting_penalty(self, waiting_times: List[float]) -> float:
        """计算等待时间惩罚"""
        if not waiting_times:
            return 0.0
        
        avg_waiting = np.mean(waiting_times)
        max_waiting = np.max(waiting_times)
        
        # 基础等待惩罚
        base_penalty = -avg_waiting / 3600.0  # 转换为小时
        
        # 极长等待额外惩罚
        extreme_penalty = -max(0, max_waiting - 24*3600) / 3600.0  # 超过24小时的额外惩罚
        
        return base_penalty + extreme_penalty
    
    def _calculate_utilization_reward(self, berth_utilization: float, target_utilization: float = 0.8) -> float:
        """计算泊位利用率奖励"""
        if berth_utilization <= target_utilization:
            return berth_utilization * 10  # 鼓励提高利用率
        else:
            # 过度利用的惩罚
            return target_utilization * 10 - (berth_utilization - target_utilization) * 5
    
    def _calculate_completion_reward(self, completed_tasks: int, total_tasks: int) -> float:
        """计算任务完成奖励"""
        if total_tasks == 0:
            return 0.0
        
        completion_rate = completed_tasks / total_tasks
        return completion_rate * 50  # 基础完成奖励
    
    def _calculate_smoothness_reward(self, action_sequence: List[int]) -> float:
        """计算调度平滑性奖励"""
        if len(action_sequence) < 2:
            return 0.0
        
        # 计算动作变化频率
        changes = sum(1 for i in range(1, len(action_sequence)) 
                     if action_sequence[i] != action_sequence[i-1])
        
        # 鼓励稳定的调度策略
        stability_reward = max(0, 10 - changes)
        return stability_reward

class GulfportRewardFunction(BaseRewardFunction):
    """Gulfport港口奖励函数 - 表现最好，保持现有策略"""
    
    def _get_default_weights(self) -> Dict[str, float]:
        return {
            'completion': 1.0,      # 完成任务权重
            'waiting': 0.5,         # 等待时间权重
            'utilization': 0.8,     # 利用率权重
            'smoothness': 0.3       # 平滑性权重
        }
    
    def calculate_reward(self, state: Dict[str, Any], action: int, next_state: Dict[str, Any]) -> float:
        """Gulfport港口奖励计算 - 平衡策略"""
        
        # 基础奖励组件
        completion_reward = self._calculate_completion_reward(
            next_state.get('completed_tasks', 0),
            next_state.get('total_tasks', 1)
        )
        
        waiting_penalty = self._calculate_waiting_penalty(
            next_state.get('waiting_times', [])
        )
        
        utilization_reward = self._calculate_utilization_reward(
            next_state.get('berth_utilization', 0.0)
        )
        
        smoothness_reward = self._calculate_smoothness_reward(
            state.get('recent_actions', [])
        )
        
        # 加权总奖励
        total_reward = (
            self.reward_weights['completion'] * completion_reward +
            self.reward_weights['waiting'] * waiting_penalty +
            self.reward_weights['utilization'] * utilization_reward +
            self.reward_weights['smoothness'] * smoothness_reward
        )
        
        return total_reward

class NewOrleansRewardFunction(BaseRewardFunction):
    """New Orleans港口奖励函数 - 加大等待惩罚，鼓励激进调度"""
    
    def _get_default_weights(self) -> Dict[str, float]:
        return {
            'completion': 1.5,      # 更高的完成奖励
            'waiting': 2.0,         # 大幅增加等待惩罚
            'utilization': 1.2,     # 更重视利用率
            'smoothness': 0.1,      # 降低平滑性要求
            'congestion': 1.0       # 新增拥堵惩罚
        }
    
    def calculate_reward(self, state: Dict[str, Any], action: int, next_state: Dict[str, Any]) -> float:
        """New Orleans港口奖励计算 - 激进调度策略"""
        
        # 基础奖励组件
        completion_reward = self._calculate_completion_reward(
            next_state.get('completed_tasks', 0),
            next_state.get('total_tasks', 1)
        )
        
        # 加强的等待惩罚
        waiting_penalty = self._calculate_enhanced_waiting_penalty(
            next_state.get('waiting_times', [])
        )
        
        utilization_reward = self._calculate_utilization_reward(
            next_state.get('berth_utilization', 0.0),
            target_utilization=0.9  # 更高的目标利用率
        )
        
        # 降低平滑性要求
        smoothness_reward = self._calculate_smoothness_reward(
            state.get('recent_actions', [])
        )
        
        # 新增拥堵惩罚
        congestion_penalty = self._calculate_congestion_penalty(
            next_state.get('queue_length', 0),
            next_state.get('max_queue_capacity', 10)
        )
        
        # 加权总奖励
        total_reward = (
            self.reward_weights['completion'] * completion_reward +
            self.reward_weights['waiting'] * waiting_penalty +
            self.reward_weights['utilization'] * utilization_reward +
            self.reward_weights['smoothness'] * smoothness_reward +
            self.reward_weights['congestion'] * congestion_penalty
        )
        
        return total_reward
    
    def _calculate_enhanced_waiting_penalty(self, waiting_times: List[float]) -> float:
        """增强的等待时间惩罚"""
        if not waiting_times:
            return 0.0
        
        # 基础惩罚
        base_penalty = self._calculate_waiting_penalty(waiting_times)
        
        # 对长等待时间的指数惩罚
        long_waits = [t for t in waiting_times if t > 12*3600]  # 超过12小时
        if long_waits:
            exponential_penalty = -sum(np.exp((t - 12*3600) / 3600) for t in long_waits)
            return base_penalty + exponential_penalty
        
        return base_penalty
    
    def _calculate_congestion_penalty(self, queue_length: int, max_capacity: int) -> float:
        """计算拥堵惩罚"""
        if max_capacity == 0:
            return 0.0
        
        congestion_ratio = queue_length / max_capacity
        
        if congestion_ratio > 0.8:
            return -(congestion_ratio - 0.8) * 20  # 高拥堵惩罚
        else:
            return 0.0

class BatonRougeRewardFunction(BaseRewardFunction):
    """Baton Rouge港口奖励函数 - 强调平滑调度"""
    
    def _get_default_weights(self) -> Dict[str, float]:
        return {
            'completion': 1.0,      # 标准完成奖励
            'waiting': 0.8,         # 中等等待惩罚
            'utilization': 0.9,     # 重视利用率
            'smoothness': 1.0,      # 大幅提高平滑性权重
            'efficiency': 0.7       # 新增效率奖励
        }
    
    def calculate_reward(self, state: Dict[str, Any], action: int, next_state: Dict[str, Any]) -> float:
        """Baton Rouge港口奖励计算 - 平滑调度策略"""
        
        # 基础奖励组件
        completion_reward = self._calculate_completion_reward(
            next_state.get('completed_tasks', 0),
            next_state.get('total_tasks', 1)
        )
        
        waiting_penalty = self._calculate_waiting_penalty(
            next_state.get('waiting_times', [])
        )
        
        utilization_reward = self._calculate_utilization_reward(
            next_state.get('berth_utilization', 0.0)
        )
        
        # 强化的平滑性奖励
        smoothness_reward = self._calculate_enhanced_smoothness_reward(
            state.get('recent_actions', []),
            next_state.get('action_consistency', 0.0)
        )
        
        # 新增效率奖励
        efficiency_reward = self._calculate_efficiency_reward(
            next_state.get('throughput', 0),
            next_state.get('resource_usage', 1)
        )
        
        # 加权总奖励
        total_reward = (
            self.reward_weights['completion'] * completion_reward +
            self.reward_weights['waiting'] * waiting_penalty +
            self.reward_weights['utilization'] * utilization_reward +
            self.reward_weights['smoothness'] * smoothness_reward +
            self.reward_weights['efficiency'] * efficiency_reward
        )
        
        return total_reward
    
    def _calculate_enhanced_smoothness_reward(self, action_sequence: List[int], consistency: float) -> float:
        """增强的平滑性奖励"""
        base_smoothness = self._calculate_smoothness_reward(action_sequence)
        
        # 额外的一致性奖励
        consistency_bonus = consistency * 5
        
        return base_smoothness + consistency_bonus
    
    def _calculate_efficiency_reward(self, throughput: float, resource_usage: float) -> float:
        """计算效率奖励"""
        if resource_usage == 0:
            return 0.0
        
        efficiency = throughput / resource_usage
        return efficiency * 10

class SouthLouisianaRewardFunction(BaseRewardFunction):
    """South Louisiana港口奖励函数 - 平衡策略"""
    
    def _get_default_weights(self) -> Dict[str, float]:
        return {
            'completion': 1.2,      # 较高完成奖励
            'waiting': 0.9,         # 中等等待惩罚
            'utilization': 1.0,     # 标准利用率权重
            'smoothness': 0.6,      # 中等平滑性权重
            'adaptability': 0.5     # 新增适应性奖励
        }
    
    def calculate_reward(self, state: Dict[str, Any], action: int, next_state: Dict[str, Any]) -> float:
        """South Louisiana港口奖励计算 - 平衡策略"""
        
        # 基础奖励组件
        completion_reward = self._calculate_completion_reward(
            next_state.get('completed_tasks', 0),
            next_state.get('total_tasks', 1)
        )
        
        waiting_penalty = self._calculate_waiting_penalty(
            next_state.get('waiting_times', [])
        )
        
        utilization_reward = self._calculate_utilization_reward(
            next_state.get('berth_utilization', 0.0)
        )
        
        smoothness_reward = self._calculate_smoothness_reward(
            state.get('recent_actions', [])
        )
        
        # 新增适应性奖励
        adaptability_reward = self._calculate_adaptability_reward(
            state.get('traffic_pattern', 'normal'),
            action,
            next_state.get('performance_improvement', 0.0)
        )
        
        # 加权总奖励
        total_reward = (
            self.reward_weights['completion'] * completion_reward +
            self.reward_weights['waiting'] * waiting_penalty +
            self.reward_weights['utilization'] * utilization_reward +
            self.reward_weights['smoothness'] * smoothness_reward +
            self.reward_weights['adaptability'] * adaptability_reward
        )
        
        return total_reward
    
    def _calculate_adaptability_reward(self, traffic_pattern: str, action: int, improvement: float) -> float:
        """计算适应性奖励"""
        # 根据交通模式调整奖励
        pattern_multiplier = {
            'low': 0.8,
            'normal': 1.0,
            'high': 1.2,
            'peak': 1.5
        }.get(traffic_pattern, 1.0)
        
        # 基于性能改善的奖励
        base_reward = improvement * 10
        
        return base_reward * pattern_multiplier

class PortRewardFactory:
    """港口奖励函数工厂"""
    
    _reward_functions = {
        'gulfport': GulfportRewardFunction,
        'new_orleans': NewOrleansRewardFunction,
        'baton_rouge': BatonRougeRewardFunction,
        'south_louisiana': SouthLouisianaRewardFunction
    }
    
    @classmethod
    def create_reward_function(cls, port_name: str) -> BaseRewardFunction:
        """创建指定港口的奖励函数"""
        if port_name not in cls._reward_functions:
            raise ValueError(f"不支持的港口: {port_name}")
        
        return cls._reward_functions[port_name](port_name)
    
    @classmethod
    def get_supported_ports(cls) -> List[str]:
        """获取支持的港口列表"""
        return list(cls._reward_functions.keys())

def test_reward_functions():
    """测试不同港口的奖励函数"""
    print("测试港口特定奖励函数...")
    
    # 模拟状态和动作
    test_state = {
        'recent_actions': [1, 1, 2, 1],
        'traffic_pattern': 'high'
    }
    
    test_next_state = {
        'completed_tasks': 8,
        'total_tasks': 10,
        'waiting_times': [3600, 7200, 1800],  # 1h, 2h, 0.5h
        'berth_utilization': 0.75,
        'queue_length': 5,
        'max_queue_capacity': 10
    }
    
    test_action = 2
    
    for port in PortRewardFactory.get_supported_ports():
        reward_func = PortRewardFactory.create_reward_function(port)
        reward = reward_func.calculate_reward(test_state, test_action, test_next_state)
        print(f"{port.upper()}: 奖励 = {reward:.2f}")

if __name__ == "__main__":
    test_reward_functions()