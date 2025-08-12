"""
Maritime Domain Knowledge Configuration
基于实际港口业务的专业知识配置
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

@dataclass
class PortSpecification:
    """港口规格和业务特征"""
    name: str
    lat: float
    lon: float
    num_berths: int
    berth_lengths: List[float]  # 各泊位长度(英尺)
    channel_depth: float        # 航道深度(英尺)
    max_draft: float           # 最大吃水(英尺)
    anchorage_capacity: int    # 锚地容量
    terminal_types: List[str]  # 码头类型
    tidal_range: float         # 潮差(英尺)

# 四个港口的实际规格
PORT_SPECIFICATIONS = {
    "new_orleans": PortSpecification(
        name="Port of New Orleans",
        lat=29.9511, lon=-90.0715,
        num_berths=40,
        berth_lengths=[2000, 4345, 1500, 1200] * 10,  # 主要码头长度
        channel_depth=45.0,
        max_draft=45.0,
        anchorage_capacity=15,
        terminal_types=["container", "bulk", "breakbulk", "cruise"],
        tidal_range=1.5
    ),
    "south_louisiana": PortSpecification(
        name="Port of South Louisiana", 
        lat=30.0543, lon=-90.4070,
        num_berths=25,
        berth_lengths=[1800, 2200, 1600] * 8,
        channel_depth=45.0,
        max_draft=45.0,
        anchorage_capacity=12,
        terminal_types=["bulk", "liquid", "container"],
        tidal_range=2.0
    ),
    "baton_rouge": PortSpecification(
        name="Port of Greater Baton Rouge",
        lat=30.4333, lon=-91.2000,
        num_berths=64,
        berth_lengths=[1000, 1500, 800] * 21,
        channel_depth=45.0,
        max_draft=45.0,
        anchorage_capacity=20,
        terminal_types=["grain", "liquid", "dry_bulk", "barge"],
        tidal_range=3.0
    ),
    "gulfport": PortSpecification(
        name="Port of Gulfport",
        lat=30.3674, lon=-89.0928,
        num_berths=10,
        berth_lengths=[525, 750, 600, 675, 700, 550, 625, 725, 575, 650],
        channel_depth=36.0,
        max_draft=36.0,
        anchorage_capacity=8,
        terminal_types=["container", "roro", "project_cargo"],
        tidal_range=2.5
    )
}

class MaritimeStateBuilder:
    """基于海事领域知识的状态构建器"""
    
    def __init__(self, port_name: str):
        self.port_spec = PORT_SPECIFICATIONS[port_name]
        self.state_dim = 56  # 统一为56维状态向量
        
    def build_vessel_state(self, vessel_data: Dict, port_status: Dict, 
                          queue_status: Dict, berth_status: Dict) -> np.ndarray:
        """
        构建24维状态向量
        基于实际港口业务需求
        """
        
        # 1. 船舶基本信息 (6维)
        vessel_features = self._extract_vessel_features(vessel_data)
        
        # 2. 空间位置信息 (4维) 
        spatial_features = self._extract_spatial_features(vessel_data)
        
        # 3. 队列管理信息 (6维)
        queue_features = self._extract_queue_features(queue_status)
        
        # 4. 泊位状态信息 (4维)
        berth_features = self._extract_berth_features(berth_status)
        
        # 5. 港口运营信息 (4维)
        port_features = self._extract_port_features(port_status)
        
        # 组合所有特征
        state = np.concatenate([
            vessel_features,    # 6维
            spatial_features,   # 4维  
            queue_features,     # 6维
            berth_features,     # 4维
            port_features       # 4维
        ]).astype(np.float32)
        
        # 数据验证和清理
        state = self._validate_and_clean_state(state)
        
        return state
    
    def _extract_vessel_features(self, vessel_data: Dict) -> np.ndarray:
        """提取船舶特征"""
        try:
            # 确保所有数值都转换为float类型，避免类型比较错误
            length = float(vessel_data.get('length', 150))
            length = min(length, 400.0) / 400.0  # 最大400米
            
            width = float(vessel_data.get('width', 25))
            width = min(width, 60.0) / 60.0       # 最大60米
            
            draft = float(vessel_data.get('draught', 8))
            draft = min(draft, 20.0) / 20.0      # 最大20米吃水
            
            # 运动特征
            sog = float(vessel_data.get('sog', 0))
            sog = min(sog, 25.0) / 25.0            # 最大25节
            
            cog = float(vessel_data.get('cog', 0)) / 360.0                    # 航向归一化
            
            # 船舶类型 (简化编码)
            vessel_type = float(vessel_data.get('vessel_type', 70))
            vessel_type = min(vessel_type, 100.0) / 100.0
            
            return np.array([length, width, draft, sog, cog, vessel_type])
            
        except Exception as e:
            print(f"Warning: Error extracting vessel features: {e}")
            return np.zeros(6)
    
    def _extract_spatial_features(self, vessel_data: Dict) -> np.ndarray:
        """提取空间位置特征"""
        try:
            lat = vessel_data.get('lat', self.port_spec.lat)
            lon = vessel_data.get('lon', self.port_spec.lon)
            
            # 相对港口位置 (标准化到[-1, 1])
            lat_diff = (lat - self.port_spec.lat) / 0.5  # ±0.5度范围
            lon_diff = (lon - self.port_spec.lon) / 0.5
            
            # 距离港口中心的距离
            distance = np.sqrt(lat_diff**2 + lon_diff**2)
            distance_norm = min(distance, 2.0) / 2.0  # 最大2.0标准化距离
            
            # 是否在港区内 (简化判断)
            in_port = 1.0 if distance < 0.1 else 0.0
            
            return np.array([lat_diff, lon_diff, distance_norm, in_port])
            
        except Exception as e:
            print(f"Warning: Error extracting spatial features: {e}")
            return np.zeros(4)
    
    def _extract_queue_features(self, queue_status: Dict) -> np.ndarray:
        """提取队列管理特征"""
        try:
            # 当前排队长度
            queue_length = min(queue_status.get('current_queue', 0), 20) / 20.0
            
            # 平均等待时间 (小时)
            avg_wait_time = min(queue_status.get('avg_wait_time', 0), 24) / 24.0
            
            # 锚地利用率
            anchorage_util = min(queue_status.get('anchorage_occupied', 0), 
                               self.port_spec.anchorage_capacity) / self.port_spec.anchorage_capacity
            
            # 队列变化趋势 (-1: 减少, 0: 稳定, 1: 增加)
            queue_trend = max(-1, min(1, queue_status.get('queue_trend', 0)))
            queue_trend_norm = (queue_trend + 1) / 2.0  # 归一化到[0,1]
            
            # 高优先级船舶数量
            priority_vessels = min(queue_status.get('priority_count', 0), 10) / 10.0
            
            # 预计处理时间
            est_processing = min(queue_status.get('est_processing_hours', 0), 48) / 48.0
            
            return np.array([queue_length, avg_wait_time, anchorage_util, 
                           queue_trend_norm, priority_vessels, est_processing])
            
        except Exception as e:
            print(f"Warning: Error extracting queue features: {e}")
            return np.zeros(6)
    
    def _extract_berth_features(self, berth_status: Dict) -> np.ndarray:
        """提取泊位状态特征"""
        try:
            # 泊位可用率
            available_berths = berth_status.get('available_count', 0)
            berth_availability = available_berths / self.port_spec.num_berths
            
            # 适合当前船舶的泊位数
            suitable_berths = min(berth_status.get('suitable_count', 0), 
                                self.port_spec.num_berths) / self.port_spec.num_berths
            
            # 平均泊位利用率
            berth_utilization = min(berth_status.get('utilization_rate', 0), 1.0)
            
            # 下个可用泊位的预计时间 (小时)
            next_available = min(berth_status.get('next_available_hours', 0), 72) / 72.0
            
            return np.array([berth_availability, suitable_berths, 
                           berth_utilization, next_available])
            
        except Exception as e:
            print(f"Warning: Error extracting berth features: {e}")
            return np.zeros(4)
    
    def _extract_port_features(self, port_status: Dict) -> np.ndarray:
        """提取港口运营特征"""
        try:
            # 当前吞吐量负载 (相对于日常平均值)
            throughput_load = min(port_status.get('throughput_ratio', 0.5), 2.0) / 2.0
            
            # 天气影响因子 (0: 无影响, 1: 严重影响)
            weather_impact = min(port_status.get('weather_factor', 0), 1.0)
            
            # 潮汐因子 (影响大船进出港)
            tide_factor = (port_status.get('current_tide', 0) + self.port_spec.tidal_range) / (2 * self.port_spec.tidal_range)
            tide_factor = max(0, min(1, tide_factor))
            
            # 港口拥堵指数 (0-1)
            congestion_index = min(port_status.get('congestion_level', 0), 1.0)
            
            return np.array([throughput_load, weather_impact, tide_factor, congestion_index])
            
        except Exception as e:
            print(f"Warning: Error extracting port features: {e}")
            return np.zeros(4)
    
    def _validate_and_clean_state(self, state: np.ndarray) -> np.ndarray:
        """验证和清理状态向量"""
        # 检查NaN和inf
        if np.any(np.isnan(state)) or np.any(np.isinf(state)):
            print(f"Warning: Found NaN or inf in state vector: {state}")
            state = np.nan_to_num(state, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 确保在[0,1]范围内 (除了相对位置可以为负)
        state = np.clip(state, -2.0, 2.0)
        
        # 确保维度正确
        if len(state) != self.state_dim:
            print(f"Warning: State dimension mismatch. Expected {self.state_dim}, got {len(state)}")
            if len(state) < self.state_dim:
                state = np.pad(state, (0, self.state_dim - len(state)), 'constant')
            else:
                state = state[:self.state_dim]
        
        return state.astype(np.float32)

class MaritimeRewardCalculator:
    """基于海事优化目标的奖励函数"""
    
    def __init__(self, port_name: str):
        self.port_spec = PORT_SPECIFICATIONS[port_name]
        self.reward_weights = {
            'transit_time': -0.3,      # 通行时间 (负奖励)
            'queue_reduction': 0.2,     # 减少排队 (正奖励) 
            'throughput': 0.25,         # 提高吞吐量
            'berth_utilization': 0.15,  # 泊位利用率
            'fairness': 0.1,           # α-公平性
            'safety': -0.2             # 安全性 (负奖励惩罚风险)
        }
    
    def calculate_reward(self, vessel_state: Dict, action: np.ndarray, 
                        port_state: Dict, queue_state: Dict) -> Tuple[float, Dict]:
        """
        计算综合奖励
        返回: (总奖励, 奖励分解字典)
        """
        
        reward_breakdown = {}
        
        # 1. 通行时间奖励
        transit_reward = self._calculate_transit_reward(vessel_state, port_state)
        reward_breakdown['transit_time'] = transit_reward
        
        # 2. 队列管理奖励
        queue_reward = self._calculate_queue_reward(queue_state)
        reward_breakdown['queue_reduction'] = queue_reward
        
        # 3. 吞吐量奖励
        throughput_reward = self._calculate_throughput_reward(port_state)
        reward_breakdown['throughput'] = throughput_reward
        
        # 4. 泊位利用率奖励
        berth_reward = self._calculate_berth_reward(port_state)
        reward_breakdown['berth_utilization'] = berth_reward
        
        # 5. 公平性奖励
        fairness_reward = self._calculate_fairness_reward(vessel_state, queue_state)
        reward_breakdown['fairness'] = fairness_reward
        
        # 6. 安全性奖励
        safety_reward = self._calculate_safety_reward(vessel_state, action)
        reward_breakdown['safety'] = safety_reward
        
        # 加权求和
        total_reward = sum(
            self.reward_weights[key] * value 
            for key, value in reward_breakdown.items()
        )
        
        return total_reward, reward_breakdown
    
    def _calculate_transit_reward(self, vessel_state: Dict, port_state: Dict) -> float:
        """计算通行时间奖励 (越短越好)"""
        try:
            transit_time = float(vessel_state.get('current_transit_time', 0))
            baseline_time = float(port_state.get('avg_transit_time', 24))  # 24小时基准
            
            if baseline_time <= 0:
                return 0.0
            
            # 相对改进 (负值表示减少了时间)
            improvement = (baseline_time - transit_time) / baseline_time
            return max(-1.0, min(1.0, improvement))
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0
    
    def _calculate_queue_reward(self, queue_state: Dict) -> float:
        """计算队列管理奖励"""
        try:
            current_queue = float(queue_state.get('current_queue', 0))
            baseline_queue = float(queue_state.get('baseline_queue', 10))
            
            if baseline_queue > 0:
                reduction_ratio = (baseline_queue - current_queue) / baseline_queue
                return max(-1.0, min(1.0, reduction_ratio))
            return 0.0
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0
    
    def _calculate_throughput_reward(self, port_state: Dict) -> float:
        """计算吞吐量奖励"""
        try:
            current_throughput = float(port_state.get('hourly_throughput', 0))
            target_throughput = float(port_state.get('target_throughput', 2))
            
            if target_throughput > 0:
                ratio = current_throughput / target_throughput
                return max(0.0, min(1.0, ratio))
            return 0.0
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0
    
    def _calculate_berth_reward(self, port_state: Dict) -> float:
        """计算泊位利用率奖励"""
        try:
            utilization = float(port_state.get('berth_utilization', 0.5))
            optimal_utilization = 0.85  # 85%为最优
            
            # 避免过度利用或利用不足
            if utilization <= optimal_utilization:
                return utilization / optimal_utilization
            else:
                return max(0.0, 2.0 - utilization / optimal_utilization)
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.5
    
    def _calculate_fairness_reward(self, vessel_state: Dict, queue_state: Dict) -> float:
        """计算α-公平性奖励"""
        waiting_times = queue_state.get('vessel_waiting_times', [])
        if len(waiting_times) < 2:
            return 0.0
        
        # 简化的α-公平性计算 (α=1为比例公平)
        alpha = 1.0
        mean_wait = np.mean(waiting_times)
        if mean_wait > 0:
            fairness_score = np.sum([t**(-alpha) for t in waiting_times]) / len(waiting_times)
            return min(1.0, fairness_score / (mean_wait**(-alpha)))
        return 0.0
    
    def _calculate_safety_reward(self, vessel_state: Dict, action: np.ndarray) -> float:
        """计算安全性奖励 (惩罚危险行为)"""
        try:
            penalty = 0.0
            
            # 速度安全检查
            sog = float(vessel_state.get('sog', 0))
            if sog > 15:  # 超过15节认为过快
                penalty += (sog - 15) / 10.0
            
            # 航向变化检查
            if len(action) > 1:
                heading_change = abs(float(action[1]))  # 假设action[1]是航向变化
                if heading_change > 30:  # 超过30度认为急转
                    penalty += (heading_change - 30) / 60.0
            
            return -min(1.0, penalty)
        except (ValueError, TypeError, ZeroDivisionError):
            return 0.0

def print_state_sample(state_builder: MaritimeStateBuilder, vessel_data: Dict, 
                      port_status: Dict, queue_status: Dict, berth_status: Dict):
    """打印状态样本用于调试"""
    state = state_builder.build_vessel_state(vessel_data, port_status, queue_status, berth_status)
    
    print("\n" + "="*60)
    print(f"状态向量样本 - {state_builder.port_spec.name}")
    print("="*60)
    print(f"船舶特征 (0-5):   {state[:6]}")
    print(f"空间特征 (6-9):   {state[6:10]}")
    print(f"队列特征 (10-15): {state[10:16]}")
    print(f"泊位特征 (16-19): {state[16:20]}")
    print(f"港口特征 (20-23): {state[20:24]}")
    print(f"状态统计: min={state.min():.3f}, max={state.max():.3f}, mean={state.mean():.3f}")
    print(f"NaN检查: {np.any(np.isnan(state))}, Inf检查: {np.any(np.isinf(state))}")
    print("="*60)