"""
基线策略仿真系统
实现随机策略和规则策略的海事交通仿真
"""

import json
import numpy as np
import pandas as pd
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
import logging
from pathlib import Path
import random
from collections import defaultdict, deque
import statistics

# 添加CityFlow路径
sys.path.append('../../FedML/CityFlow')

try:
    import cityflow
    CITYFLOW_AVAILABLE = True
except ImportError:
    CITYFLOW_AVAILABLE = False
    print("警告: CityFlow未安装，将使用模拟器")

from maritime_domain_knowledge import PORT_SPECIFICATIONS

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineStrategy:
    """基线策略基类"""
    
    def __init__(self, strategy_name: str):
        self.strategy_name = strategy_name
        self.decision_history = []
        
    def make_decision(self, vessel_info: Dict, port_status: Dict, queue_status: Dict) -> Dict:
        """做出决策"""
        raise NotImplementedError
    
    def reset(self):
        """重置策略状态"""
        self.decision_history = []

class RandomStrategy(BaselineStrategy):
    """随机策略 - 随机选择可用泊位/通道"""
    
    def __init__(self, seed: int = None):
        super().__init__("Random")
        self.rng = random.Random(seed)
        
    def make_decision(self, vessel_info: Dict, port_status: Dict, queue_status: Dict) -> Dict:
        """随机选择可用泊位"""
        available_berths = port_status.get('available_berths', [])
        available_channels = port_status.get('available_channels', [])
        
        if not available_berths and not available_channels:
            # 没有可用资源，选择等待
            decision = {
                'action': 'wait',
                'target': None,
                'priority': 0,
                'reasoning': 'No available berths or channels'
            }
        else:
            # 随机选择泊位或通道
            all_options = []
            if available_berths:
                all_options.extend([('berth', b) for b in available_berths])
            if available_channels:
                all_options.extend([('channel', c) for c in available_channels])
            
            if all_options:
                resource_type, target = self.rng.choice(all_options)
                decision = {
                    'action': 'assign',
                    'resource_type': resource_type,
                    'target': target,
                    'priority': self.rng.uniform(0, 1),
                    'reasoning': f'Random selection: {resource_type} {target}'
                }
            else:
                decision = {
                    'action': 'wait',
                    'target': None,
                    'priority': 0,
                    'reasoning': 'No valid options available'
                }
        
        self.decision_history.append(decision)
        return decision

class FCFSStrategy(BaselineStrategy):
    """先到先服务策略 (First Come First Served)"""
    
    def __init__(self):
        super().__init__("FCFS")
        self.arrival_times = {}
        
    def make_decision(self, vessel_info: Dict, port_status: Dict, queue_status: Dict) -> Dict:
        """基于到达时间的先到先服务"""
        vessel_id = vessel_info.get('mmsi', 'unknown')
        current_time = vessel_info.get('timestamp', time.time())
        
        # 记录到达时间
        if vessel_id not in self.arrival_times:
            self.arrival_times[vessel_id] = current_time
        
        available_berths = port_status.get('available_berths', [])
        queue_length = queue_status.get('queue_length', 0)
        
        if available_berths:
            # 有可用泊位，分配给最早到达的船舶
            decision = {
                'action': 'assign',
                'resource_type': 'berth',
                'target': available_berths[0],  # 选择第一个可用泊位
                'priority': -self.arrival_times[vessel_id],  # 负值表示越早到达优先级越高
                'reasoning': f'FCFS: Arrived at {self.arrival_times[vessel_id]}'
            }
        else:
            # 没有可用泊位，加入队列
            decision = {
                'action': 'queue',
                'target': None,
                'priority': -self.arrival_times[vessel_id],
                'reasoning': f'FCFS: Queuing, arrived at {self.arrival_times[vessel_id]}'
            }
        
        self.decision_history.append(decision)
        return decision

class ShortestQueueStrategy(BaselineStrategy):
    """最短队列优先策略"""
    
    def __init__(self):
        super().__init__("Shortest_Queue")
        
    def make_decision(self, vessel_info: Dict, port_status: Dict, queue_status: Dict) -> Dict:
        """选择队列最短的泊位/通道"""
        berth_queues = port_status.get('berth_queues', {})
        channel_queues = port_status.get('channel_queues', {})
        
        # 找到队列最短的资源
        min_queue_length = float('inf')
        best_option = None
        
        # 检查泊位队列
        for berth_id, queue_length in berth_queues.items():
            if queue_length < min_queue_length:
                min_queue_length = queue_length
                best_option = ('berth', berth_id)
        
        # 检查通道队列
        for channel_id, queue_length in channel_queues.items():
            if queue_length < min_queue_length:
                min_queue_length = queue_length
                best_option = ('channel', channel_id)
        
        if best_option:
            resource_type, target = best_option
            decision = {
                'action': 'assign',
                'resource_type': resource_type,
                'target': target,
                'priority': -min_queue_length,  # 队列越短优先级越高
                'reasoning': f'Shortest queue: {resource_type} {target} (queue: {min_queue_length})'
            }
        else:
            decision = {
                'action': 'wait',
                'target': None,
                'priority': 0,
                'reasoning': 'No available resources'
            }
        
        self.decision_history.append(decision)
        return decision

class MaritimeSimulator:
    """海事交通仿真器"""
    
    def __init__(self, port_name: str, config_path: str = None):
        self.port_name = port_name
        self.port_spec = PORT_SPECIFICATIONS[port_name]
        self.config_path = config_path
        
        # 仿真状态
        self.current_time = 0
        self.vessels = {}
        self.berth_status = {}
        self.channel_status = {}
        self.queue_status = defaultdict(list)
        
        # 性能指标
        self.metrics = {
            'waiting_times': [],
            'throughput': 0,
            'queue_peaks': [],
            'berth_utilization': [],
            'total_vessels': 0,
            'completed_vessels': 0
        }
        
        # CityFlow引擎（如果可用）
        self.cityflow_engine = None
        self.use_cityflow = CITYFLOW_AVAILABLE and config_path is not None
        
        if self.use_cityflow and config_path:
            try:
                self.cityflow_engine = cityflow.Engine(config_path, thread_num=1)
                logger.info(f"CityFlow引擎初始化成功: {config_path}")
            except Exception as e:
                logger.warning(f"CityFlow初始化失败: {e}，使用内置仿真器")
                self.use_cityflow = False
        
        self._initialize_port_resources()
        
    def _initialize_port_resources(self):
        """初始化港口资源"""
        # 初始化泊位
        for i in range(self.port_spec.num_berths):
            self.berth_status[f'berth_{i}'] = {
                'occupied': False,
                'vessel_id': None,
                'start_time': None,
                'queue': deque()
            }
        
        # 初始化通道
        num_channels = max(3, self.port_spec.num_berths // 2)
        for i in range(num_channels):
            self.channel_status[f'channel_{i}'] = {
                'occupied': False,
                'vessel_id': None,
                'queue': deque()
            }
    
    def load_vessel_data(self, flow_file: str):
        """加载船舶数据"""
        try:
            logger.info(f"📂 [数据加载] 开始加载文件: {flow_file}")
            
            with open(flow_file, 'r') as f:
                flows = json.load(f)
            
            logger.info(f"📊 [数据解析] 解析到 {len(flows)} 条flow记录")
            
            # 统计时间范围
            start_times = []
            
            for flow in flows:
                vessel_id = flow['_metadata']['mmsi']
                start_time = flow['startTime']
                start_times.append(start_time)
                
                self.vessels[vessel_id] = {
                    'mmsi': vessel_id,
                    'start_time': start_time,
                    'end_time': flow['endTime'],
                    'origin_region': flow['_metadata']['origin_region'],
                    'dest_region': flow['_metadata']['dest_region'],
                    'vessel_type': flow['_metadata']['vessel_type'],
                    'avg_speed': flow['_metadata']['avg_speed_knots'],
                    'status': 'pending',
                    'arrival_time': None,
                    'service_start_time': None,
                    'departure_time': None,
                    'waiting_time': 0
                }
            
            # 时间范围分析
            if start_times:
                min_time = min(start_times)
                max_time = max(start_times)
                logger.info(f"⏰ [时间范围] 最早到达: {min_time}, 最晚到达: {max_time}")
                logger.info(f"⏰ [时间跨度] {max_time - min_time} 秒 ({(max_time - min_time)/3600:.1f} 小时)")
            
            self.metrics['total_vessels'] = len(self.vessels)
            logger.info(f"✅ [数据加载完成] 成功加载 {len(self.vessels)} 艘船舶的数据")
            
        except Exception as e:
            logger.error(f"❌ [数据加载失败] {e}")
    
    def run_simulation(self, strategy: BaselineStrategy, duration: int = 86400) -> Dict:
        """运行仿真"""
        logger.info(f"开始仿真 - 策略: {strategy.strategy_name}, 时长: {duration}秒")
        
        start_time = time.time()
        
        # 设置仿真时间基准：使用最早船舶到达时间作为起点
        if self.vessels:
            min_start_time = min(vessel['start_time'] for vessel in self.vessels.values())
            self.simulation_start_time = min_start_time
            self.current_time = 0  # 相对于simulation_start_time的偏移
            logger.info(f"⏰ [时间基准] 仿真起始时间: {min_start_time} (Unix时间戳)")
        else:
            self.simulation_start_time = 0
            self.current_time = 0
            logger.warning("⚠️ [时间基准] 没有船舶数据，使用默认时间基准")
        
        strategy.reset()
        
        # 重置指标
        self.metrics = {
            'waiting_times': [],
            'throughput': 0,
            'queue_peaks': [],
            'berth_utilization': [],
            'total_vessels': len(self.vessels),
            'completed_vessels': 0
        }
        
        # 仿真主循环
        time_step = 60  # 1分钟时间步长
        
        while self.current_time < duration:
            self._simulation_step(strategy, time_step)
            self.current_time += time_step
            
            # 记录队列峰值
            current_queue_size = sum(len(berth['queue']) for berth in self.berth_status.values())
            self.metrics['queue_peaks'].append(current_queue_size)
            
            # 记录泊位利用率
            occupied_berths = sum(1 for berth in self.berth_status.values() if berth['occupied'])
            utilization = occupied_berths / len(self.berth_status)
            self.metrics['berth_utilization'].append(utilization)
        
        # 计算最终指标
        simulation_results = self._calculate_final_metrics()
        simulation_time = time.time() - start_time
        
        logger.info(f"仿真完成 - 用时: {simulation_time:.2f}秒")
        
        return simulation_results
    
    def _simulation_step(self, strategy: BaselineStrategy, time_step: int):
        """单步仿真"""
        # 处理新到达的船舶
        self._process_arrivals()
        
        # 处理正在服务的船舶
        self._process_services()
        
        # 为等待中的船舶做决策
        self._process_decisions(strategy)
        
        # 更新CityFlow（如果使用）
        if self.use_cityflow and self.cityflow_engine:
            try:
                self.cityflow_engine.next_step()
            except Exception as e:
                logger.warning(f"CityFlow步进失败: {e}")
    
    def _process_arrivals(self):
        """处理船舶到达"""
        arrivals_count = 0
        pending_vessels = [v for v in self.vessels.values() if v['status'] == 'pending']
        
        logger.debug(f"[到达检查] 当前时间: {self.current_time}, 待处理船舶: {len(pending_vessels)}")
        
        for vessel_id, vessel in self.vessels.items():
            if vessel['status'] == 'pending':
                # 将绝对时间转换为相对时间
                relative_start_time = vessel['start_time'] - self.simulation_start_time
                logger.debug(f"[到达检查] 船舶 {vessel_id}: relative_start_time={relative_start_time}, current_time={self.current_time}")
                
                if relative_start_time <= self.current_time:
                    vessel['status'] = 'arrived'
                    vessel['arrival_time'] = self.current_time
                    arrivals_count += 1
                    logger.info(f"✅ [船舶到达] 船舶 {vessel_id} 在时间 {self.current_time} 到达")
        
        if arrivals_count > 0:
            logger.info(f"[到达汇总] 本轮新到达船舶: {arrivals_count} 艘")
    
    def _process_services(self):
        """处理正在服务的船舶"""
        completed_count = 0
        in_service_count = 0
        
        for berth_id, berth in self.berth_status.items():
            if berth['occupied'] and berth['vessel_id']:
                in_service_count += 1
                vessel_id = berth['vessel_id']
                vessel = self.vessels[vessel_id]
                
                # 检查是否完成服务
                service_duration = self.current_time - berth['start_time']
                expected_duration = self._calculate_service_duration(vessel)
                
                logger.debug(f"[服务检查] 泊位 {berth_id}, 船舶 {vessel_id}: 已服务 {service_duration}s, 需要 {expected_duration}s")
                
                if service_duration >= expected_duration:
                    # 完成服务
                    vessel['status'] = 'completed'
                    vessel['departure_time'] = self.current_time
                    vessel['waiting_time'] = berth['start_time'] - vessel['arrival_time']
                    
                    self.metrics['waiting_times'].append(vessel['waiting_time'])
                    self.metrics['completed_vessels'] += 1
                    completed_count += 1
                    
                    logger.info(f"🚢 [服务完成] 船舶 {vessel_id} 完成服务，等待时间: {vessel['waiting_time']}s")
                    
                    # 释放泊位
                    berth['occupied'] = False
                    berth['vessel_id'] = None
                    berth['start_time'] = None
                    
                    logger.info(f"🔓 [泊位释放] 泊位 {berth_id} 已释放")
                    
                    # 处理队列中的下一艘船
                    if berth['queue']:
                        next_vessel_id = berth['queue'].popleft()
                        logger.info(f"📋 [队列处理] 从队列分配船舶 {next_vessel_id} 到泊位 {berth_id}")
                        self._assign_berth(next_vessel_id, berth_id)
        
        if in_service_count > 0:
            logger.debug(f"[服务状态] 正在服务: {in_service_count} 艘, 本轮完成: {completed_count} 艘")
    
    def _process_decisions(self, strategy: BaselineStrategy):
        """处理决策"""
        waiting_vessels = [
            vessel_id for vessel_id, vessel in self.vessels.items()
            if vessel['status'] == 'arrived'
        ]
        
        logger.debug(f"[决策处理] 等待决策的船舶: {len(waiting_vessels)} 艘")
        
        if waiting_vessels:
            logger.info(f"🤔 [决策开始] 为 {len(waiting_vessels)} 艘船舶制定决策")
        
        for vessel_id in waiting_vessels:
            vessel = self.vessels[vessel_id]
            
            # 构建状态信息
            port_status = self._get_port_status()
            queue_status = self._get_queue_status()
            
            logger.debug(f"[港口状态] 可用泊位: {port_status['available_berths']}, 队列长度: {queue_status['queue_length']}")
            
            # 策略决策
            decision = strategy.make_decision(vessel, port_status, queue_status)
            
            logger.info(f"💭 [策略决策] 船舶 {vessel_id} -> {decision['action']}: {decision.get('reasoning', 'N/A')}")
            
            # 执行决策
            self._execute_decision(vessel_id, decision)
    
    def _get_port_status(self) -> Dict:
        """获取港口状态"""
        available_berths = [
            berth_id for berth_id, berth in self.berth_status.items()
            if not berth['occupied']
        ]
        
        berth_queues = {
            berth_id: len(berth['queue'])
            for berth_id, berth in self.berth_status.items()
        }
        
        channel_queues = {
            channel_id: len(channel['queue'])
            for channel_id, channel in self.channel_status.items()
        }
        
        return {
            'available_berths': available_berths,
            'berth_queues': berth_queues,
            'channel_queues': channel_queues,
            'total_berths': len(self.berth_status),
            'occupied_berths': sum(1 for berth in self.berth_status.values() if berth['occupied'])
        }
    
    def _get_queue_status(self) -> Dict:
        """获取队列状态"""
        total_queue_length = sum(len(berth['queue']) for berth in self.berth_status.values())
        
        return {
            'queue_length': total_queue_length,
            'avg_waiting_time': np.mean(self.metrics['waiting_times']) if self.metrics['waiting_times'] else 0
        }
    
    def _execute_decision(self, vessel_id: str, decision: Dict):
        """执行决策"""
        action = decision.get('action', 'wait')
        
        logger.debug(f"[决策执行] 船舶 {vessel_id} 执行动作: {action}")
        
        if action == 'assign':
            resource_type = decision.get('resource_type')
            target = decision.get('target')
            
            if resource_type == 'berth' and target in self.berth_status:
                if not self.berth_status[target]['occupied']:
                    logger.info(f"🎯 [直接分配] 船舶 {vessel_id} 分配到泊位 {target}")
                    self._assign_berth(vessel_id, target)
                else:
                    # 泊位被占用，加入队列
                    self.berth_status[target]['queue'].append(vessel_id)
                    self.vessels[vessel_id]['status'] = 'queued'
                    logger.info(f"📝 [加入队列] 船舶 {vessel_id} 加入泊位 {target} 队列")
            
        elif action == 'queue':
            # 选择队列最短的泊位
            min_queue_berth = min(
                self.berth_status.keys(),
                key=lambda b: len(self.berth_status[b]['queue'])
            )
            self.berth_status[min_queue_berth]['queue'].append(vessel_id)
            self.vessels[vessel_id]['status'] = 'queued'
            logger.info(f"📝 [智能排队] 船舶 {vessel_id} 加入最短队列泊位 {min_queue_berth}")
        
        else:
            # 其他情况保持等待状态
            logger.debug(f"⏳ [等待] 船舶 {vessel_id} 继续等待")
    
    def _assign_berth(self, vessel_id: str, berth_id: str):
        """分配泊位"""
        self.berth_status[berth_id]['occupied'] = True
        self.berth_status[berth_id]['vessel_id'] = vessel_id
        self.berth_status[berth_id]['start_time'] = self.current_time
        
        self.vessels[vessel_id]['status'] = 'in_service'
        self.vessels[vessel_id]['service_start_time'] = self.current_time
        
        # 计算预期服务时长
        expected_duration = self._calculate_service_duration(self.vessels[vessel_id])
        
        logger.info(f"🔒 [泊位分配] 船舶 {vessel_id} 分配到泊位 {berth_id}, 预期服务时长: {expected_duration}s")
    
    def _calculate_service_duration(self, vessel: Dict) -> float:
        """计算服务时长"""
        # 基于船舶类型和大小的简化服务时长模型
        # 调整为更合理的基础时长：30分钟
        base_duration = 1800  # 30分钟基础时长
        
        vessel_type = vessel.get('vessel_type', 70)
        if vessel_type == 80:  # 油轮 - 需要更长时间
            multiplier = 1.3
        elif vessel_type == 70:  # 货船 - 标准时间
            multiplier = 1.0
        elif vessel_type == 31:  # 拖船 - 较短时间
            multiplier = 0.7
        else:  # 其他
            multiplier = 0.9
        
        # 添加适度的随机变化
        variation = np.random.uniform(0.8, 1.2)
        
        duration = base_duration * multiplier * variation
        
        # 确保最小服务时长为10分钟，最大为2小时
        duration = max(600, min(7200, duration))
        
        return duration
    
    def _calculate_final_metrics(self) -> Dict:
        """计算最终指标"""
        results = {
            'strategy': 'Unknown',
            'port': self.port_name,
            'simulation_time': self.current_time,
            'total_vessels': self.metrics['total_vessels'],
            'completed_vessels': self.metrics['completed_vessels'],
            'completion_rate': self.metrics['completed_vessels'] / max(1, self.metrics['total_vessels']),
            'avg_waiting_time': np.mean(self.metrics['waiting_times']) if self.metrics['waiting_times'] else 0,
            'std_waiting_time': np.std(self.metrics['waiting_times']) if self.metrics['waiting_times'] else 0,
            'max_waiting_time': max(self.metrics['waiting_times']) if self.metrics['waiting_times'] else 0,
            'throughput': self.metrics['completed_vessels'] / (self.current_time / 3600),  # 每小时
            'avg_queue_peak': np.mean(self.metrics['queue_peaks']) if self.metrics['queue_peaks'] else 0,
            'max_queue_peak': max(self.metrics['queue_peaks']) if self.metrics['queue_peaks'] else 0,
            'avg_berth_utilization': np.mean(self.metrics['berth_utilization']) if self.metrics['berth_utilization'] else 0,
            'max_berth_utilization': max(self.metrics['berth_utilization']) if self.metrics['berth_utilization'] else 0
        }
        
        return results

class BaselineExperimentRunner:
    """基线实验运行器"""
    
    def __init__(self, ports: List[str] = None):
        self.ports = ports or ['baton_rouge', 'new_orleans', 'south_louisiana', 'gulfport']
        self.strategies = {
            'Random': RandomStrategy,
            'FCFS': FCFSStrategy,
            'Shortest_Queue': ShortestQueueStrategy
        }
        self.results = defaultdict(list)
        
    def run_baseline_experiments(self, num_rounds: int = 5, num_days: int = 7) -> Dict:
        """运行5×7基线实验"""
        logger.info(f"开始基线实验 - {num_rounds}轮 × {num_days}天")
        
        experiment_results = {
            'experiment_config': {
                'num_rounds': num_rounds,
                'num_days': num_days,
                'ports': self.ports,
                'strategies': list(self.strategies.keys())
            },
            'detailed_results': defaultdict(lambda: defaultdict(list)),
            'summary_statistics': {}
        }
        
        total_experiments = len(self.ports) * len(self.strategies) * num_rounds * num_days
        completed_experiments = 0
        
        for port in self.ports:
            logger.info(f"处理港口: {port}")
            
            for strategy_name, strategy_class in self.strategies.items():
                logger.info(f"  策略: {strategy_name}")
                
                for round_num in range(num_rounds):
                    logger.info(f"    轮次: {round_num + 1}/{num_rounds}")
                    
                    for day in range(num_days):
                        logger.info(f"      天数: {day + 1}/{num_days}")
                        
                        try:
                            # 运行单次实验
                            result = self._run_single_experiment(
                                port, strategy_name, strategy_class, round_num, day
                            )
                            
                            experiment_results['detailed_results'][port][strategy_name].append(result)
                            completed_experiments += 1
                            
                            progress = (completed_experiments / total_experiments) * 100
                            logger.info(f"        完成 ({progress:.1f}%)")
                            
                        except Exception as e:
                            logger.error(f"实验失败 - {port}/{strategy_name}/R{round_num}/D{day}: {e}")
        
        # 计算汇总统计
        experiment_results['summary_statistics'] = self._calculate_summary_statistics(
            experiment_results['detailed_results']
        )
        
        # 保存结果
        self._save_experiment_results(experiment_results)
        
        return experiment_results
    
    def _run_single_experiment(self, port: str, strategy_name: str, strategy_class, 
                              round_num: int, day: int) -> Dict:
        """运行单次实验"""
        # 创建策略实例
        if strategy_name == 'Random':
            strategy = strategy_class(seed=round_num * 100 + day)
        else:
            strategy = strategy_class()
        
        # 查找配置文件
        config_file = f"../../topologies/maritime_3x3_{port}_config.json"
        if not os.path.exists(config_file):
            config_file = "../../topologies/maritime_3x3_config.json"
        
        # 创建仿真器
        simulator = MaritimeSimulator(port, config_file if os.path.exists(config_file) else None)
        
        # 加载船舶数据
        flow_file = f"../../data/processed_flows/{port}_week1/flow_2024070{day+1}.json"
        if os.path.exists(flow_file):
            simulator.load_vessel_data(flow_file)
        else:
            # 使用默认流量文件
            default_flow = "../../topologies/maritime_3x3_flows.json"
            if os.path.exists(default_flow):
                simulator.load_vessel_data(default_flow)
            else:
                logger.warning(f"未找到流量文件: {flow_file}")
        
        # 运行仿真
        simulation_duration = 24 * 3600  # 24小时
        result = simulator.run_simulation(strategy, simulation_duration)
        
        # 添加实验元数据
        result.update({
            'strategy': strategy_name,
            'port': port,
            'round': round_num,
            'day': day,
            'timestamp': datetime.now().isoformat()
        })
        
        return result
    
    def _calculate_summary_statistics(self, detailed_results: Dict) -> Dict:
        """计算汇总统计"""
        summary = {}
        
        for port, port_results in detailed_results.items():
            summary[port] = {}
            
            for strategy, strategy_results in port_results.items():
                if not strategy_results:
                    continue
                
                # 提取关键指标
                waiting_times = [r['avg_waiting_time'] for r in strategy_results]
                throughputs = [r['throughput'] for r in strategy_results]
                queue_peaks = [r['max_queue_peak'] for r in strategy_results]
                utilizations = [r['avg_berth_utilization'] for r in strategy_results]
                
                summary[port][strategy] = {
                    'avg_waiting_time': {
                        'mean': np.mean(waiting_times),
                        'std': np.std(waiting_times),
                        'min': np.min(waiting_times),
                        'max': np.max(waiting_times)
                    },
                    'throughput': {
                        'mean': np.mean(throughputs),
                        'std': np.std(throughputs),
                        'min': np.min(throughputs),
                        'max': np.max(throughputs)
                    },
                    'queue_peak': {
                        'mean': np.mean(queue_peaks),
                        'std': np.std(queue_peaks),
                        'min': np.min(queue_peaks),
                        'max': np.max(queue_peaks)
                    },
                    'berth_utilization': {
                        'mean': np.mean(utilizations),
                        'std': np.std(utilizations),
                        'min': np.min(utilizations),
                        'max': np.max(utilizations)
                    },
                    'num_experiments': len(strategy_results)
                }
        
        return summary
    
    def _save_experiment_results(self, results: Dict):
        """保存实验结果"""
        output_dir = Path("../../data/baseline_experiments")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 保存详细结果
        detailed_file = output_dir / f"baseline_detailed_{timestamp}.json"
        with open(detailed_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # 保存汇总结果
        summary_file = output_dir / f"baseline_summary_{timestamp}.json"
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(results['summary_statistics'], f, indent=2, ensure_ascii=False, default=str)
        
        # 生成CSV报告
        self._generate_csv_report(results, output_dir / f"baseline_report_{timestamp}.csv")
        
        logger.info(f"实验结果已保存到: {output_dir}")
    
    def _generate_csv_report(self, results: Dict, csv_file: Path):
        """生成CSV报告"""
        rows = []
        
        for port, port_results in results['summary_statistics'].items():
            for strategy, metrics in port_results.items():
                row = {
                    'Port': port,
                    'Strategy': strategy,
                    'Avg_Waiting_Time_Mean': metrics['avg_waiting_time']['mean'],
                    'Avg_Waiting_Time_Std': metrics['avg_waiting_time']['std'],
                    'Throughput_Mean': metrics['throughput']['mean'],
                    'Throughput_Std': metrics['throughput']['std'],
                    'Queue_Peak_Mean': metrics['queue_peak']['mean'],
                    'Queue_Peak_Std': metrics['queue_peak']['std'],
                    'Berth_Utilization_Mean': metrics['berth_utilization']['mean'],
                    'Berth_Utilization_Std': metrics['berth_utilization']['std'],
                    'Num_Experiments': metrics['num_experiments']
                }
                rows.append(row)
        
        df = pd.DataFrame(rows)
        df.to_csv(csv_file, index=False)
        logger.info(f"CSV报告已保存: {csv_file}")

def main():
    """主函数"""
    print("启动基线策略仿真系统...")
    
    # 创建实验运行器
    runner = BaselineExperimentRunner()
    
    # 运行基线实验
    results = runner.run_baseline_experiments(num_rounds=5, num_days=7)
    
    # 打印汇总结果
    print("\n" + "="*80)
    print("基线实验结果汇总")
    print("="*80)
    
    for port, port_results in results['summary_statistics'].items():
        print(f"\n港口: {port.upper()}")
        print("-" * 40)
        
        for strategy, metrics in port_results.items():
            print(f"\n策略: {strategy}")
            print(f"  平均等待时间: {metrics['avg_waiting_time']['mean']:.2f} ± {metrics['avg_waiting_time']['std']:.2f} 秒")
            print(f"  吞吐量: {metrics['throughput']['mean']:.2f} ± {metrics['throughput']['std']:.2f} 船/小时")
            print(f"  队列峰值: {metrics['queue_peak']['mean']:.2f} ± {metrics['queue_peak']['std']:.2f}")
            print(f"  泊位利用率: {metrics['berth_utilization']['mean']:.2%} ± {metrics['berth_utilization']['std']:.2%}")
    
    print("\n" + "="*80)
    print("基线实验完成！")

if __name__ == "__main__":
    main()