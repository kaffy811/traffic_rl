#!/usr/bin/env python3
"""
完整的公平性奖励函数实现
支持多种公平性度量和动态权重调整
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from enum import Enum
import json
import pickle
from pathlib import Path


class FairnessMetricType(Enum):
    """公平性度量类型"""
    GINI_COEFFICIENT = "gini_coefficient"
    JAIN_INDEX = "jain_index"
    MAX_MIN_RATIO = "max_min_ratio"
    VARIANCE_COEFFICIENT = "variance_coefficient"
    ENTROPY_BASED = "entropy_based"
    THEIL_INDEX = "theil_index"


@dataclass
class FairnessConfig:
    """公平性配置"""
    efficiency_weight: float = 0.7
    fairness_weight: float = 0.3
    min_service_level: float = 0.3
    max_service_level: float = 5.0
    fairness_metrics: List[FairnessMetricType] = None
    metric_weights: Dict[FairnessMetricType, float] = None
    temporal_decay: float = 0.95
    adaptive_weights: bool = True
    penalty_threshold: float = 0.1
    reward_scaling: float = 100.0
    
    # α-公平效用函数参数
    use_alpha_fairness: bool = True
    alpha: float = 1.0  # α参数：α=0(最大化), α=1(比例公平), α→∞(最大最小公平)
    history_window: int = 100  # 历史奖励窗口大小
    min_historical_reward: float = 0.1  # 最小历史奖励，避免除零

    def __post_init__(self):
        if self.fairness_metrics is None:
            self.fairness_metrics = [
                FairnessMetricType.GINI_COEFFICIENT,
                FairnessMetricType.JAIN_INDEX,
                FairnessMetricType.MAX_MIN_RATIO
            ]

        if self.metric_weights is None:
            equal_weight = 1.0 / len(self.fairness_metrics)
            self.metric_weights = {
                metric: equal_weight for metric in self.fairness_metrics
            }


class ServiceLevelCalculator:
    """服务水平计算器"""

    def __init__(self, alpha: float = 0.6, beta: float = 0.4):
        """
        Args:
            alpha: 吞吐量权重
            beta: 响应时间权重
        """
        self.alpha = alpha
        self.beta = beta

    def calculate_service_level(self,
                                throughput: float,
                                waiting_time: float,
                                queue_length: Optional[float] = None) -> float:
        """
        计算服务水平
        服务水平 = α × (吞吐量分量) + β × (响应时间分量)
        """
        # 吞吐量分量（正向）
        throughput_component = np.log(max(1.0, throughput + 1))

        # 响应时间分量（负向，转为正向）
        response_component = 1.0 / (1.0 + waiting_time)

        # 如果有队列长度信息，加入队列分量
        if queue_length is not None:
            queue_component = 1.0 / (1.0 + queue_length)
            # 重新分配权重
            alpha, beta, gamma = 0.4, 0.3, 0.3
            service_level = (alpha * throughput_component +
                             beta * response_component +
                             gamma * queue_component)
        else:
            service_level = self.alpha * throughput_component + self.beta * response_component

        return max(0.0, service_level)


class FairnessMetrics:
    """公平性度量指标集合"""

    @staticmethod
    def gini_coefficient(values: List[float]) -> float:
        """
        基尼系数 (Gini Coefficient)
        0表示完全公平，1表示完全不公平
        """
        if len(values) <= 1:
            return 0.0

        # 过滤非正值
        values = [max(0.001, v) for v in values]
        sorted_values = sorted(values)
        n = len(sorted_values)

        # 基尼系数公式
        cumsum = np.cumsum(sorted_values)
        gini = (2 * np.sum([(i + 1) * val for i, val in enumerate(sorted_values)])) / (n * cumsum[-1]) - (n + 1) / n

        return max(0.0, min(1.0, gini))

    @staticmethod
    def jain_fairness_index(values: List[float]) -> float:
        """
        Jain公平性指数
        1表示完全公平，1/n表示完全不公平
        """
        if len(values) <= 1:
            return 1.0

        values = [max(0.001, v) for v in values]
        sum_values = sum(values)
        sum_squares = sum(v * v for v in values)

        if sum_squares == 0:
            return 1.0

        jain_index = (sum_values * sum_values) / (len(values) * sum_squares)
        return max(0.0, min(1.0, jain_index))

    @staticmethod
    def max_min_ratio(values: List[float]) -> float:
        """
        最大最小比率
        1表示完全公平，比率越大越不公平
        """
        if len(values) <= 1:
            return 1.0

        values = [max(0.001, v) for v in values]
        max_val = max(values)
        min_val = min(values)

        return max_val / min_val

    @staticmethod
    def variance_coefficient(values: List[float]) -> float:
        """
        变异系数（标准差/均值）
        0表示完全公平，值越大越不公平
        """
        if len(values) <= 1:
            return 0.0

        values = [max(0.001, v) for v in values]
        mean_val = np.mean(values)
        std_val = np.std(values)

        if mean_val == 0:
            return 0.0

        return std_val / mean_val

    @staticmethod
    def entropy_based_fairness(values: List[float]) -> float:
        """
        基于熵的公平性度量
        最大熵表示完全公平
        """
        if len(values) <= 1:
            return 1.0

        values = [max(0.001, v) for v in values]
        total = sum(values)

        if total == 0:
            return 1.0

        # 计算概率分布
        probabilities = [v / total for v in values]

        # 计算熵
        entropy = -sum(p * np.log(p + 1e-10) for p in probabilities)
        max_entropy = np.log(len(values))

        # 归一化熵（0-1之间，1表示最公平）
        if max_entropy == 0:
            return 1.0

        return entropy / max_entropy

    @staticmethod
    def theil_index(values: List[float]) -> float:
        """
        泰尔指数 (Theil Index)
        0表示完全公平，值越大越不公平
        """
        if len(values) <= 1:
            return 0.0

        values = [max(0.001, v) for v in values]
        mean_val = np.mean(values)

        if mean_val == 0:
            return 0.0

        # 泰尔指数公式
        theil = np.mean([v * np.log(v / mean_val) for v in values]) / mean_val

        return max(0.0, theil)


class AlphaFairnessUtility:
    """α-公平效用函数实现"""
    
    def __init__(self, alpha: float = 1.0, history_window: int = 100, min_reward: float = 0.1):
        """
        初始化α-公平效用函数
        
        Args:
            alpha: 公平性参数
                - α = 0: 最大化总效用 (utilitarian)
                - α = 1: 比例公平 (proportional fairness)
                - α → ∞: 最大最小公平 (max-min fairness)
            history_window: 历史奖励窗口大小
            min_reward: 最小历史奖励，避免数值问题
        """
        self.alpha = alpha
        self.history_window = history_window
        self.min_reward = min_reward
        
        # 每个agent的历史奖励记录
        self.agent_reward_history: Dict[str, List[float]] = {}
        self.agent_cumulative_rewards: Dict[str, float] = {}
        
    def update_agent_history(self, agent_id: str, reward: float):
        """更新agent的历史奖励"""
        if agent_id not in self.agent_reward_history:
            self.agent_reward_history[agent_id] = []
            self.agent_cumulative_rewards[agent_id] = 0.0
        
        # 添加当前奖励到历史
        self.agent_reward_history[agent_id].append(reward)
        self.agent_cumulative_rewards[agent_id] += reward
        
        # 维护窗口大小
        if len(self.agent_reward_history[agent_id]) > self.history_window:
            removed_reward = self.agent_reward_history[agent_id].pop(0)
            self.agent_cumulative_rewards[agent_id] -= removed_reward
    
    def calculate_fairness_weight(self, agent_id: str) -> float:
        """
        计算agent的公平性权重
        
        基于α-公平效用函数：U'(x) = x^(-α)
        
        Returns:
            公平性权重，用于调整当前奖励
        """
        if agent_id not in self.agent_cumulative_rewards:
            return 1.0  # 新agent，给予正常权重
        
        # 获取历史累积奖励
        cumulative_reward = max(self.min_reward, self.agent_cumulative_rewards[agent_id])
        
        # 计算α-公平效用函数的一阶导数
        if self.alpha == 0:
            # α = 0: utilitarian，所有agent权重相等
            fairness_weight = 1.0
        elif self.alpha == 1:
            # α = 1: 比例公平，权重与累积奖励成反比
            fairness_weight = 1.0 / cumulative_reward
        else:
            # 一般情况: U'(x) = x^(-α)
            fairness_weight = pow(cumulative_reward, -self.alpha)
        
        return fairness_weight
    
    def apply_fairness_adjustment(self, agent_id: str, raw_reward: float) -> float:
        """
        应用公平性调整到原始奖励
        
        r̂_{k,t} = r_{k,t} × U'(∑_{τ=1}^{t-1} r_{k,τ})
        
        Args:
            agent_id: agent标识
            raw_reward: 原始奖励
            
        Returns:
            调整后的公平奖励
        """
        fairness_weight = self.calculate_fairness_weight(agent_id)
        adjusted_reward = raw_reward * fairness_weight
        
        # 更新历史（使用原始奖励）
        self.update_agent_history(agent_id, raw_reward)
        
        return adjusted_reward
    
    def get_fairness_statistics(self) -> Dict:
        """获取公平性统计信息"""
        if not self.agent_cumulative_rewards:
            return {}
        
        cumulative_rewards = list(self.agent_cumulative_rewards.values())
        fairness_weights = [self.calculate_fairness_weight(agent_id) 
                          for agent_id in self.agent_cumulative_rewards.keys()]
        
        return {
            "alpha": self.alpha,
            "num_agents": len(cumulative_rewards),
            "cumulative_rewards": {
                "mean": np.mean(cumulative_rewards),
                "std": np.std(cumulative_rewards),
                "min": np.min(cumulative_rewards),
                "max": np.max(cumulative_rewards),
                "gini": FairnessMetrics.gini_coefficient(cumulative_rewards)
            },
            "fairness_weights": {
                "mean": np.mean(fairness_weights),
                "std": np.std(fairness_weights),
                "min": np.min(fairness_weights),
                "max": np.max(fairness_weights)
            },
            "theoretical_fairness": self._calculate_theoretical_fairness()
        }
    
    def _calculate_theoretical_fairness(self) -> float:
        """计算理论公平性度量"""
        if not self.agent_cumulative_rewards:
            return 1.0
            
        cumulative_rewards = list(self.agent_cumulative_rewards.values())
        
        if self.alpha == 0:
            # utilitarian: 关注总和
            return 1.0  # 总是"公平"的
        elif self.alpha == 1:
            # 比例公平: 使用调和平均数
            harmonic_mean = len(cumulative_rewards) / sum(1.0/max(self.min_reward, r) for r in cumulative_rewards)
            arithmetic_mean = np.mean(cumulative_rewards)
            return harmonic_mean / arithmetic_mean if arithmetic_mean > 0 else 0
        else:
            # 一般α-公平: 使用广义平均数
            if self.alpha == float('inf'):
                # max-min fairness
                return np.min(cumulative_rewards) / np.max(cumulative_rewards) if np.max(cumulative_rewards) > 0 else 0
            else:
                # α-mean
                alpha_mean = pow(np.mean([pow(max(self.min_reward, r), 1-self.alpha) for r in cumulative_rewards]), 
                               1/(1-self.alpha))
                arithmetic_mean = np.mean(cumulative_rewards)
                return alpha_mean / arithmetic_mean if arithmetic_mean > 0 else 0
    
    def reset_history(self, agent_id: Optional[str] = None):
        """重置历史记录"""
        if agent_id is None:
            # 重置所有agent
            self.agent_reward_history.clear()
            self.agent_cumulative_rewards.clear()
        else:
            # 重置特定agent
            if agent_id in self.agent_reward_history:
                del self.agent_reward_history[agent_id]
            if agent_id in self.agent_cumulative_rewards:
                del self.agent_cumulative_rewards[agent_id]


class AdaptiveWeightManager:
    """自适应权重管理器"""

    def __init__(self,
                 initial_efficiency_weight: float = 0.7,
                 initial_fairness_weight: float = 0.3,
                 adaptation_rate: float = 0.01,
                 min_weight: float = 0.1,
                 max_weight: float = 0.9):

        self.efficiency_weight = initial_efficiency_weight
        self.fairness_weight = initial_fairness_weight
        self.adaptation_rate = adaptation_rate
        self.min_weight = min_weight
        self.max_weight = max_weight

        # 历史性能记录
        self.efficiency_history = []
        self.fairness_history = []
        self.performance_window = 100

    def update_weights(self,
                       current_efficiency: float,
                       current_fairness: float,
                       target_efficiency: float = 0.8,
                       target_fairness: float = 0.8) -> Tuple[float, float]:
        """
        基于当前性能动态调整权重
        """
        # 记录历史
        self.efficiency_history.append(current_efficiency)
        self.fairness_history.append(current_fairness)

        # 保持窗口大小
        if len(self.efficiency_history) > self.performance_window:
            self.efficiency_history.pop(0)
            self.fairness_history.pop(0)

        # 计算性能差距
        efficiency_gap = target_efficiency - current_efficiency
        fairness_gap = target_fairness - current_fairness

        # 自适应调整
        if efficiency_gap > 0.1:  # 效率不足
            self.efficiency_weight = min(self.max_weight,
                                         self.efficiency_weight + self.adaptation_rate)
            self.fairness_weight = max(self.min_weight,
                                       1.0 - self.efficiency_weight)
        elif fairness_gap > 0.1:  # 公平性不足
            self.fairness_weight = min(self.max_weight,
                                       self.fairness_weight + self.adaptation_rate)
            self.efficiency_weight = max(self.min_weight,
                                         1.0 - self.fairness_weight)

        return self.efficiency_weight, self.fairness_weight


class ComprehensiveFairnessRewardCalculator:
    """
    综合公平性奖励计算器
    支持多种公平性度量、自适应权重、时序分析等完整功能
    """

    def __init__(self, config: FairnessConfig = None):
        self.config = config or FairnessConfig()
        self.service_calculator = ServiceLevelCalculator()
        self.weight_manager = AdaptiveWeightManager(
            self.config.efficiency_weight,
            self.config.fairness_weight
        )

        # α-公平效用函数
        if self.config.use_alpha_fairness:
            self.alpha_fairness = AlphaFairnessUtility(
                alpha=self.config.alpha,
                history_window=self.config.history_window,
                min_reward=self.config.min_historical_reward
            )
        else:
            self.alpha_fairness = None

        # 历史状态记录
        self.state_history = []
        self.reward_history = []
        self.fairness_scores_history = []

        # 性能统计
        self.performance_stats = {
            'total_episodes': 0,
            'avg_efficiency': 0.0,
            'avg_fairness': 0.0,
            'best_fairness': 0.0,
            'worst_fairness': 1.0
        }

        # 设置日志
        self.logger = logging.getLogger(__name__)

    def calculate_alpha_fair_reward(self, 
                                   agent_id: str, 
                                   raw_reward: float) -> Dict[str, float]:
        """
        计算单个agent的α-公平调整奖励
        
        Args:
            agent_id: agent标识
            raw_reward: 原始奖励
            
        Returns:
            包含原始奖励、调整后奖励和公平性信息的字典
        """
        if not self.config.use_alpha_fairness or self.alpha_fairness is None:
            return {
                'raw_reward': raw_reward,
                'adjusted_reward': raw_reward,
                'fairness_weight': 1.0,
                'cumulative_reward': raw_reward
            }
        
        # 计算公平性权重
        fairness_weight = self.alpha_fairness.calculate_fairness_weight(agent_id)
        
        # 应用公平性调整
        adjusted_reward = self.alpha_fairness.apply_fairness_adjustment(agent_id, raw_reward)
        
        # 获取累积奖励
        cumulative_reward = self.alpha_fairness.agent_cumulative_rewards.get(agent_id, 0.0)
        
        return {
            'raw_reward': raw_reward,
            'adjusted_reward': adjusted_reward,
            'fairness_weight': fairness_weight,
            'cumulative_reward': cumulative_reward,
            'alpha': self.config.alpha
        }

    def calculate_comprehensive_reward(self,
                                       node_states: Dict[str, Dict],
                                       action_results: Dict[str, float],
                                       previous_states: Optional[Dict[str, Dict]] = None) -> Dict[str, float]:
        """
        计算综合奖励

        Returns:
            完整的奖励分解字典
        """

        # 1. 效率奖励
        efficiency_reward = self._calculate_efficiency_reward(action_results)

        # 2. 公平性奖励
        if self.config.use_alpha_fairness:
            # 使用α-公平效用函数计算公平性度量
            fairness_reward = self._calculate_alpha_fairness_metric(node_states)
        else:
            # 使用传统多指标公平性计算
            fairness_reward = self._calculate_comprehensive_fairness_reward(node_states)

        # 3. 时序奖励（如果有历史状态）
        temporal_reward = 0.0
        if previous_states is not None:
            temporal_reward = self._calculate_temporal_reward(previous_states, node_states)

        # 4. 稳定性奖励
        stability_reward = self._calculate_stability_reward(node_states)

        # 5. 自适应权重更新
        if self.config.adaptive_weights:
            eff_weight, fair_weight = self.weight_manager.update_weights(
                efficiency_reward, fairness_reward
            )
        else:
            eff_weight = self.config.efficiency_weight
            fair_weight = self.config.fairness_weight

        # 6. 综合奖励计算
        total_reward = (
                eff_weight * efficiency_reward +
                fair_weight * fairness_reward +
                0.1 * temporal_reward +
                0.1 * stability_reward
        )

        # 7. 奖励缩放
        total_reward *= self.config.reward_scaling

        # 8. 记录历史和统计
        self._update_history_and_stats(node_states, total_reward, fairness_reward)

        return {
            'total_reward': total_reward,
            'efficiency_reward': efficiency_reward,
            'fairness_reward': fairness_reward,
            'temporal_reward': temporal_reward,
            'stability_reward': stability_reward,
            'efficiency_weight': eff_weight,
            'fairness_weight': fair_weight,
            'fairness_metrics': self._get_detailed_fairness_metrics(node_states)
        }

    def _calculate_efficiency_reward(self, action_results: Dict[str, float]) -> float:
        """计算效率奖励"""
        throughput = action_results.get('total_throughput', 0)
        avg_waiting = action_results.get('avg_waiting_time', 100)
        avg_queue_length = action_results.get('avg_queue_length', 0)

        # 多维度效率指标
        throughput_score = np.log(max(1.0, throughput + 1))
        waiting_score = 1.0 / (1.0 + avg_waiting)
        queue_score = 1.0 / (1.0 + avg_queue_length)

        # 加权平均
        efficiency_reward = 0.5 * throughput_score + 0.3 * waiting_score + 0.2 * queue_score

        return efficiency_reward

    def _calculate_comprehensive_fairness_reward(self, node_states: Dict[str, Dict]) -> float:
        """计算综合公平性奖励"""
        if len(node_states) < 2:
            return 1.0  # 单节点情况返回最高公平性

        # 计算各节点的服务水平
        service_levels = []
        for node_id, state in node_states.items():
            waiting_time = state.get('waiting_time', 0)
            throughput = state.get('throughput', 0)
            queue_length = state.get('waiting_ships', 0)

            service_level = self.service_calculator.calculate_service_level(
                throughput, waiting_time, queue_length
            )
            service_levels.append(service_level)

        # 计算各种公平性度量
        fairness_scores = {}

        for metric_type in self.config.fairness_metrics:
            if metric_type == FairnessMetricType.GINI_COEFFICIENT:
                score = 1.0 - FairnessMetrics.gini_coefficient(service_levels)
            elif metric_type == FairnessMetricType.JAIN_INDEX:
                score = FairnessMetrics.jain_fairness_index(service_levels)
            elif metric_type == FairnessMetricType.MAX_MIN_RATIO:
                ratio = FairnessMetrics.max_min_ratio(service_levels)
                score = 1.0 / ratio  # 转换为越大越好
            elif metric_type == FairnessMetricType.VARIANCE_COEFFICIENT:
                cv = FairnessMetrics.variance_coefficient(service_levels)
                score = 1.0 / (1.0 + cv)  # 转换为越大越好
            elif metric_type == FairnessMetricType.ENTROPY_BASED:
                score = FairnessMetrics.entropy_based_fairness(service_levels)
            elif metric_type == FairnessMetricType.THEIL_INDEX:
                theil = FairnessMetrics.theil_index(service_levels)
                score = 1.0 / (1.0 + theil)  # 转换为越大越好
            else:
                score = 0.5  # 默认值

            fairness_scores[metric_type] = score

        # 加权平均各项公平性度量
        weighted_fairness = sum(
            self.config.metric_weights[metric] * score
            for metric, score in fairness_scores.items()
        )

        # 最小服务水平保障
        min_service = min(service_levels)
        if min_service < self.config.min_service_level:
            penalty = (self.config.min_service_level - min_service) / self.config.min_service_level
            weighted_fairness *= (1.0 - penalty)

        return weighted_fairness

    def _calculate_alpha_fairness_metric(self, node_states: Dict[str, Dict]) -> float:
        """
        基于α-公平效用函数计算全局公平性度量
        
        这个方法计算所有节点的理论公平性度量，用于替代传统的多指标公平性
        注意：这与单个agent的α-公平奖励调整不同
        """
        if not self.alpha_fairness or len(node_states) < 2:
            return 1.0
        
        # 获取当前所有节点的"虚拟累积奖励"（基于服务水平）
        node_service_levels = []
        for node_id, state in node_states.items():
            waiting_time = state.get('waiting_time', 0)
            throughput = state.get('throughput', 0)
            queue_length = state.get('waiting_ships', 0)
            
            service_level = self.service_calculator.calculate_service_level(
                throughput, waiting_time, queue_length
            )
            node_service_levels.append(service_level)
        
        # 使用α-公平效用理论计算全局公平性
        if self.config.alpha == 0:
            # utilitarian: 最大化总和
            fairness_score = 1.0  # 总是"公平"
        elif self.config.alpha == 1:
            # 比例公平: 最大化几何平均数的对数
            if all(level > 0 for level in node_service_levels):
                geometric_mean = np.exp(np.mean(np.log(node_service_levels)))
                arithmetic_mean = np.mean(node_service_levels)
                fairness_score = geometric_mean / arithmetic_mean
            else:
                fairness_score = 0.5
        elif self.config.alpha == float('inf'):
            # max-min公平: 最大化最小值
            min_level = min(node_service_levels)
            max_level = max(node_service_levels)
            fairness_score = min_level / max_level if max_level > 0 else 0
        else:
            # 一般α-公平: 最大化α-平均数
            try:
                if self.config.alpha == 1:
                    # 避免数值问题，使用对数
                    alpha_mean = np.exp(np.mean(np.log([max(0.001, level) for level in node_service_levels])))
                else:
                    alpha_values = [pow(max(0.001, level), 1 - self.config.alpha) for level in node_service_levels]
                    alpha_mean = pow(np.mean(alpha_values), 1 / (1 - self.config.alpha))
                
                arithmetic_mean = np.mean(node_service_levels)
                fairness_score = alpha_mean / arithmetic_mean if arithmetic_mean > 0 else 0
            except (OverflowError, ZeroDivisionError):
                # 数值溢出时的后备方案
                fairness_score = FairnessMetrics.jain_fairness_index(node_service_levels)
        
        return max(0.0, min(1.0, fairness_score))

    def _calculate_temporal_reward(self,
                                   previous_states: Dict[str, Dict],
                                   current_states: Dict[str, Dict]) -> float:
        """计算时序奖励（改进趋势）"""

        # 计算前后状态的服务水平变化
        prev_service_levels = []
        curr_service_levels = []

        for node_id in current_states.keys():
            if node_id in previous_states:
                # 前一状态的服务水平
                prev_state = previous_states[node_id]
                prev_service = self.service_calculator.calculate_service_level(
                    prev_state.get('throughput', 0),
                    prev_state.get('waiting_time', 0),
                    prev_state.get('waiting_ships', 0)
                )
                prev_service_levels.append(prev_service)

                # 当前状态的服务水平
                curr_state = current_states[node_id]
                curr_service = self.service_calculator.calculate_service_level(
                    curr_state.get('throughput', 0),
                    curr_state.get('waiting_time', 0),
                    curr_state.get('waiting_ships', 0)
                )
                curr_service_levels.append(curr_service)

        if not prev_service_levels:
            return 0.0

        # 计算改进程度
        prev_fairness = FairnessMetrics.jain_fairness_index(prev_service_levels)
        curr_fairness = FairnessMetrics.jain_fairness_index(curr_service_levels)

        improvement = curr_fairness - prev_fairness

        # 时序衰减
        temporal_reward = improvement * self.config.temporal_decay

        return temporal_reward

    def _calculate_stability_reward(self, node_states: Dict[str, Dict]) -> float:
        """计算稳定性奖励"""
        if len(self.state_history) < 5:
            return 0.0

        # 分析最近几个状态的波动性
        recent_fairness = self.fairness_scores_history[-5:]
        fairness_variance = np.var(recent_fairness)

        # 稳定性奖励：波动小的奖励更高
        stability_reward = 1.0 / (1.0 + fairness_variance)

        return stability_reward

    def _get_detailed_fairness_metrics(self, node_states: Dict[str, Dict]) -> Dict[str, float]:
        """获取详细的公平性度量"""
        if len(node_states) < 2:
            return {}

        service_levels = []
        for state in node_states.values():
            service_level = self.service_calculator.calculate_service_level(
                state.get('throughput', 0),
                state.get('waiting_time', 0),
                state.get('waiting_ships', 0)
            )
            service_levels.append(service_level)

        return {
            'gini_coefficient': FairnessMetrics.gini_coefficient(service_levels),
            'jain_index': FairnessMetrics.jain_fairness_index(service_levels),
            'max_min_ratio': FairnessMetrics.max_min_ratio(service_levels),
            'variance_coefficient': FairnessMetrics.variance_coefficient(service_levels),
            'entropy_fairness': FairnessMetrics.entropy_based_fairness(service_levels),
            'theil_index': FairnessMetrics.theil_index(service_levels),
            'min_service_level': min(service_levels),
            'max_service_level': max(service_levels),
            'avg_service_level': np.mean(service_levels)
        }

    def _update_history_and_stats(self,
                                  node_states: Dict[str, Dict],
                                  total_reward: float,
                                  fairness_reward: float):
        """更新历史记录和统计信息"""

        # 更新历史
        self.state_history.append(node_states.copy())
        self.reward_history.append(total_reward)
        self.fairness_scores_history.append(fairness_reward)

        # 保持历史窗口大小
        max_history = 1000
        if len(self.state_history) > max_history:
            self.state_history.pop(0)
            self.reward_history.pop(0)
            self.fairness_scores_history.pop(0)

        # 更新性能统计
        self.performance_stats['total_episodes'] += 1
        self.performance_stats['avg_fairness'] = np.mean(self.fairness_scores_history)
        self.performance_stats['best_fairness'] = max(self.fairness_scores_history)
        self.performance_stats['worst_fairness'] = min(self.fairness_scores_history)

    def get_performance_report(self) -> Dict:
        """获取性能报告"""
        report = {
            'performance_stats': self.performance_stats.copy(),
            'current_weights': {
                'efficiency': self.weight_manager.efficiency_weight,
                'fairness': self.weight_manager.fairness_weight
            },
            'recent_fairness_trend': self.fairness_scores_history[-10:] if len(
                self.fairness_scores_history) >= 10 else self.fairness_scores_history,
            'config': self.config.__dict__
        }
        
        # 如果使用α-公平效用函数，添加相关统计信息
        if self.config.use_alpha_fairness and self.alpha_fairness:
            report['alpha_fairness_stats'] = self.alpha_fairness.get_fairness_statistics()
        
        return report

    def save_state(self, filepath: str):
        """保存计算器状态"""
        state = {
            'config': self.config.__dict__,
            'weight_manager': {
                'efficiency_weight': self.weight_manager.efficiency_weight,
                'fairness_weight': self.weight_manager.fairness_weight,
                'efficiency_history': self.weight_manager.efficiency_history,
                'fairness_history': self.weight_manager.fairness_history
            },
            'state_history': self.state_history,
            'reward_history': self.reward_history,
            'fairness_scores_history': self.fairness_scores_history,
            'performance_stats': self.performance_stats
        }

        with open(filepath, 'wb') as f:
            pickle.dump(state, f)

    def load_state(self, filepath: str):
        """加载计算器状态"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)

        # 恢复配置
        self.config = FairnessConfig(**state['config'])

        # 恢复权重管理器
        wm_state = state['weight_manager']
        self.weight_manager.efficiency_weight = wm_state['efficiency_weight']
        self.weight_manager.fairness_weight = wm_state['fairness_weight']
        self.weight_manager.efficiency_history = wm_state['efficiency_history']
        self.weight_manager.fairness_history = wm_state['fairness_history']

        # 恢复历史数据
        self.state_history = state['state_history']
        self.reward_history = state['reward_history']
        self.fairness_scores_history = state['fairness_scores_history']
        self.performance_stats = state['performance_stats']


def comprehensive_test():
    """完整的测试函数"""

    print("完整公平性奖励计算器测试")
    print("=" * 60)

    # 创建配置
    config = FairnessConfig(
        efficiency_weight=0.6,
        fairness_weight=0.4,
        fairness_metrics=[
            FairnessMetricType.GINI_COEFFICIENT,
            FairnessMetricType.JAIN_INDEX,
            FairnessMetricType.MAX_MIN_RATIO,
            FairnessMetricType.ENTROPY_BASED
        ],
        adaptive_weights=True,
        reward_scaling=100.0
    )

    # 创建计算器
    calculator = ComprehensiveFairnessRewardCalculator(config)

    # 测试场景
    test_scenarios = [
        {
            "name": "理想平衡状态",
            "node_states": {
                'NodeA': {'waiting_ships': 5, 'throughput': 10, 'waiting_time': 5},
                'NodeB': {'waiting_ships': 5, 'throughput': 10, 'waiting_time': 5},
                'NodeC': {'waiting_ships': 4, 'throughput': 9, 'waiting_time': 6},
                'NodeD': {'waiting_ships': 6, 'throughput': 11, 'waiting_time': 4},
            },
            "action_results": {'total_throughput': 40, 'avg_waiting_time': 5, 'avg_queue_length': 5}
        },
        {
            "name": "严重不平衡状态",
            "node_states": {
                'NodeA': {'waiting_ships': 2, 'throughput': 20, 'waiting_time': 2},
                'NodeB': {'waiting_ships': 3, 'throughput': 18, 'waiting_time': 3},
                'NodeC': {'waiting_ships': 25, 'throughput': 1, 'waiting_time': 30},
                'NodeD': {'waiting_ships': 20, 'throughput': 2, 'waiting_time': 25},
            },
            "action_results": {'total_throughput': 41, 'avg_waiting_time': 15, 'avg_queue_length': 12.5}
        },
        {
            "name": "中等不平衡状态",
            "node_states": {
                'NodeA': {'waiting_ships': 8, 'throughput': 12, 'waiting_time': 6},
                'NodeB': {'waiting_ships': 6, 'throughput': 10, 'waiting_time': 8},
                'NodeC': {'waiting_ships': 10, 'throughput': 6, 'waiting_time': 12},
                'NodeD': {'waiting_ships': 4, 'throughput': 14, 'waiting_time': 4},
            },
            "action_results": {'total_throughput': 42, 'avg_waiting_time': 7.5, 'avg_queue_length': 7}
        }
    ]

    print("\n场景测试结果:")
    print("-" * 60)

    previous_states = None

    for i, scenario in enumerate(test_scenarios):
        print(f"\n{i + 1}. {scenario['name']}")

        reward_breakdown = calculator.calculate_comprehensive_reward(
            scenario['node_states'],
            scenario['action_results'],
            previous_states
        )

        print(f"  总奖励: {reward_breakdown['total_reward']:.2f}")
        print(f"  效率奖励: {reward_breakdown['efficiency_reward']:.4f}")
        print(f"  公平性奖励: {reward_breakdown['fairness_reward']:.4f}")
        print(f"  时序奖励: {reward_breakdown['temporal_reward']:.4f}")
        print(f"  稳定性奖励: {reward_breakdown['stability_reward']:.4f}")
        print(f"  效率权重: {reward_breakdown['efficiency_weight']:.3f}")
        print(f"  公平性权重: {reward_breakdown['fairness_weight']:.3f}")

        metrics = reward_breakdown['fairness_metrics']
        print(f"  详细公平性度量:")
        print(f"    基尼系数: {metrics['gini_coefficient']:.4f}")
        print(f"    Jain指数: {metrics['jain_index']:.4f}")
        print(f"    最大最小比: {metrics['max_min_ratio']:.4f}")
        print(f"    熵公平性: {metrics['entropy_fairness']:.4f}")
        print(f"    服务水平范围: {metrics['min_service_level']:.4f} - {metrics['max_service_level']:.4f}")

        previous_states = scenario['node_states']

    # 性能报告
    print(f"\n性能报告:")
    print("-" * 60)
    report = calculator.get_performance_report()

    for key, value in report['performance_stats'].items():
        print(f"  {key}: {value}")

    # 保存状态测试
    print(f"\n状态保存测试:")
    save_path = "test_fairness_calculator_state.pkl"
    calculator.save_state(save_path)
    print(f"  状态已保存到: {save_path}")

    # 加载状态测试
    new_calculator = ComprehensiveFairnessRewardCalculator()
    new_calculator.load_state(save_path)
    print(f"  状态加载成功，历史记录数: {len(new_calculator.state_history)}")

    print(f"\n完整公平性奖励计算器测试完成！")


# 向后兼容的简化版本
class AlphaFairRewardCalculator:
    """
    简化的α-公平奖励计算器
    为了向后兼容而创建的包装类
    """
    
    def __init__(self, alpha: float = 1.0, history_window: int = 100, min_reward: float = 0.1):
        """
        初始化α-公平奖励计算器
        
        Args:
            alpha: 公平性参数
            history_window: 历史窗口大小
            min_reward: 最小奖励值
        """
        self.alpha = alpha
        
        # 创建配置
        config = FairnessConfig(
            use_alpha_fairness=True,
            alpha=alpha,
            history_window=history_window,
            min_historical_reward=min_reward,
            efficiency_weight=0.7,
            fairness_weight=0.3
        )
        
        # 使用综合计算器
        self.comprehensive_calculator = ComprehensiveFairnessRewardCalculator(config)
        
        # 直接访问α-公平工具
        self.alpha_fairness = self.comprehensive_calculator.alpha_fairness
    
    def calculate_fairness_reward(self, agent_id: str, raw_reward: float) -> float:
        """
        计算单个agent的公平性调整奖励
        
        Args:
            agent_id: agent标识
            raw_reward: 原始奖励
            
        Returns:
            调整后的奖励
        """
        if self.alpha_fairness:
            result = self.comprehensive_calculator.calculate_alpha_fair_reward(agent_id, raw_reward)
            return result['adjusted_reward']
        else:
            return raw_reward
    
    def update_agent_reward(self, agent_id: str, reward: float):
        """
        更新agent的奖励历史
        
        Args:
            agent_id: agent标识
            reward: 奖励值
        """
        if self.alpha_fairness:
            self.alpha_fairness.update_agent_history(agent_id, reward)
    
    def get_fairness_weight(self, agent_id: str) -> float:
        """
        获取agent的公平性权重
        
        Args:
            agent_id: agent标识
            
        Returns:
            公平性权重
        """
        if self.alpha_fairness:
            return self.alpha_fairness.calculate_fairness_weight(agent_id)
        else:
            return 1.0
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if self.alpha_fairness:
            return self.alpha_fairness.get_fairness_statistics()
        else:
            return {}
    
    def reset(self, agent_id: str = None):
        """重置历史记录"""
        if self.alpha_fairness:
            self.alpha_fairness.reset_history(agent_id)


if __name__ == "__main__":
    comprehensive_test()