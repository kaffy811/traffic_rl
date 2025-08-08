"""
GAT-PPO训练数据处理器
负责准备训练数据、时间归一化、训练测试集切分
"""

import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GATTrainingDataProcessor:
    """GAT-PPO训练数据处理器"""
    
    def __init__(self, ports: List[str] = None):
        self.ports = ports or ['baton_rouge', 'new_orleans', 'south_louisiana', 'gulfport']
        self.data_dir = Path("../../data")
        self.processed_flows_dir = self.data_dir / "processed_flows"
        self.training_data_dir = self.data_dir / "gat_training_data"
        self.training_data_dir.mkdir(exist_ok=True)
        
        # 时间归一化参数
        self.time_window_hours = 24  # 24小时窗口
        self.time_step_minutes = 60  # 1小时时间步长
        
    def prepare_all_ports_data(self, num_days: int = 7) -> Dict:
        """为所有港口准备训练数据"""
        logger.info(f"开始为 {len(self.ports)} 个港口准备训练数据")
        
        all_ports_data = {}
        
        for port in self.ports:
            logger.info(f"处理港口: {port}")
            port_data = self.prepare_single_port_data(port, num_days)
            all_ports_data[port] = port_data
            
        # 保存汇总数据
        summary_file = self.training_data_dir / "training_data_summary.json"
        with open(summary_file, 'w') as f:
            json.dump({
                'ports': self.ports,
                'num_days': num_days,
                'data_summary': {
                    port: {
                        'training_samples': len(data['training_data']),
                        'test_samples': len(data['test_data']),
                        'total_vessels': data['metadata']['total_vessels'],
                        'time_range': data['metadata']['time_range']
                    }
                    for port, data in all_ports_data.items()
                },
                'created_at': datetime.now().isoformat()
            }, f, indent=2)
            
        logger.info(f"所有港口数据准备完成，保存到: {self.training_data_dir}")
        return all_ports_data
    
    def prepare_single_port_data(self, port: str, num_days: int = 7) -> Dict:
        """为单个港口准备训练数据"""
        logger.info(f"准备 {port} 港口的训练数据")
        
        # 1. 加载原始流量数据
        raw_data = self._load_port_flow_data(port, num_days)
        
        # 2. 时间归一化
        normalized_data = self._normalize_time_series(raw_data, port)
        
        # 3. 特征工程
        feature_data = self._extract_training_features(normalized_data, port)
        
        # 4. 训练测试集切分 (前5天训练，后2天测试)
        training_data, test_data = self._split_train_test(feature_data, train_days=5)
        
        # 5. 构建图结构数据
        graph_data = self._build_graph_structure_data(training_data, test_data, port)
        
        port_data = {
            'port_name': port,
            'training_data': training_data,
            'test_data': test_data,
            'graph_structure': graph_data,
            'normalization_stats': self._calculate_normalization_stats(feature_data),
            'metadata': {
                'total_vessels': len(set(item['mmsi'] for item in feature_data)),
                'time_range': {
                    'start': min(item['timestamp'] for item in feature_data),
                    'end': max(item['timestamp'] for item in feature_data)
                },
                'num_days': num_days,
                'train_test_split': f"{5}:{num_days-5}"
            }
        }
        
        # 保存单港口数据
        port_file = self.training_data_dir / f"{port}_training_data.json"
        with open(port_file, 'w') as f:
            json.dump(port_data, f, indent=2, default=str)
            
        logger.info(f"{port} 数据准备完成: 训练样本 {len(training_data)}, 测试样本 {len(test_data)}")
        return port_data
    
    def _load_port_flow_data(self, port: str, num_days: int) -> List[Dict]:
        """加载港口流量数据"""
        port_flow_dir = self.processed_flows_dir / f"{port}_week1"
        
        if not port_flow_dir.exists():
            logger.warning(f"未找到 {port} 的流量数据目录: {port_flow_dir}")
            return self._generate_mock_flow_data(port, num_days)
        
        all_flows = []
        
        for day in range(1, num_days + 1):
            flow_file = port_flow_dir / f"flow_2024070{day}.json"
            
            if flow_file.exists():
                with open(flow_file, 'r') as f:
                    daily_flows = json.load(f)
                    
                # 添加日期标识和唯一ID
                for idx, flow in enumerate(daily_flows):
                    flow['day'] = day
                    flow['port'] = port
                    # 为每个流量记录生成唯一ID
                    flow['mmsi'] = f"{port}_{day}_{idx:04d}"
                    
                all_flows.extend(daily_flows)
                logger.info(f"加载 {port} Day{day}: {len(daily_flows)} 条流量记录")
            else:
                logger.warning(f"未找到 {port} Day{day} 的流量文件: {flow_file}")
        
        logger.info(f"{port} 总共加载 {len(all_flows)} 条流量记录")
        return all_flows
    
    def _generate_mock_flow_data(self, port: str, num_days: int) -> List[Dict]:
        """生成模拟流量数据"""
        logger.info(f"为 {port} 生成模拟流量数据")
        
        mock_flows = []
        base_time = datetime(2024, 7, 1)
        
        # 根据港口规模设置船舶数量
        port_vessel_counts = {
            'new_orleans': 15,
            'baton_rouge': 12,
            'south_louisiana': 10,
            'gulfport': 8
        }
        
        vessels_per_day = port_vessel_counts.get(port, 10)
        
        for day in range(1, num_days + 1):
            day_start = base_time + timedelta(days=day-1)
            
            for vessel_idx in range(vessels_per_day):
                # 随机到达时间
                arrival_offset = np.random.uniform(0, 24 * 3600)  # 0-24小时内随机到达
                arrival_time = day_start + timedelta(seconds=arrival_offset)
                
                mock_flow = {
                    'mmsi': f"{port}_{day}_{vessel_idx:03d}",
                    'start_time': arrival_time.timestamp(),
                    'vessel_type': np.random.choice([70, 80, 31, 60, 50]),  # 不同船型
                    'length': np.random.uniform(50, 300),
                    'width': np.random.uniform(10, 50),
                    'draught': np.random.uniform(3, 15),
                    'destination_berth': f"berth_{np.random.randint(0, 10)}",
                    'estimated_service_time': np.random.uniform(1800, 7200),  # 0.5-2小时
                    'priority': np.random.choice(['normal', 'high', 'urgent'], p=[0.7, 0.2, 0.1]),
                    'day': day,
                    'port': port
                }
                
                mock_flows.append(mock_flow)
        
        logger.info(f"生成 {port} 模拟数据: {len(mock_flows)} 条流量记录")
        return mock_flows
    
    def _normalize_time_series(self, raw_data: List[Dict], port: str) -> List[Dict]:
        """时间序列归一化"""
        if not raw_data:
            return []
        
        # 找到时间范围 - 适配实际数据格式
        timestamps = [item.get('startTime', item.get('start_time', 0)) for item in raw_data]
        min_time = min(timestamps)
        max_time = max(timestamps)
        
        logger.info(f"{port} 时间范围: {datetime.fromtimestamp(min_time)} - {datetime.fromtimestamp(max_time)}")
        
        normalized_data = []
        
        for item in raw_data:
            normalized_item = item.copy()
            
            # 获取开始时间 - 适配不同格式
            start_time = item.get('startTime', item.get('start_time', 0))
            normalized_item['start_time'] = start_time  # 统一字段名
            
            # 时间归一化到 [0, 1] 范围
            normalized_item['normalized_time'] = (start_time - min_time) / (max_time - min_time) if max_time > min_time else 0
            
            # 计算相对于一天开始的时间 (小时)
            day_start = datetime.fromtimestamp(start_time).replace(hour=0, minute=0, second=0)
            normalized_item['hour_of_day'] = (start_time - day_start.timestamp()) / 3600
            
            # 船舶特征归一化 - 适配实际数据格式
            vehicle_info = item.get('vehicle', {})
            normalized_item['normalized_length'] = min(vehicle_info.get('length', 100) / 400, 1.0)  # 最大400米
            normalized_item['normalized_width'] = min(vehicle_info.get('width', 20) / 60, 1.0)     # 最大60米
            normalized_item['normalized_draught'] = min(item.get('draught', 8) / 20, 1.0)  # 最大20米
            
            # 提取车辆类型
            normalized_item['vehicle_type'] = vehicle_info.get('vehicleType', 'cargo')
            normalized_item['max_speed'] = vehicle_info.get('maxSpeed', 5.0)
            
            normalized_data.append(normalized_item)
        
        return normalized_data
    
    def _extract_training_features(self, normalized_data: List[Dict], port: str) -> List[Dict]:
        """提取训练特征"""
        feature_data = []
        
        for item in normalized_data:
            features = {
                # 基础信息
                'timestamp': item['start_time'],
                'mmsi': item['mmsi'],
                'port': port,
                'day': item['day'],
                
                # 船舶状态特征
                'vessel_features': {
                    'type': item.get('vehicle_type', 'cargo'),
                    'normalized_length': item['normalized_length'],
                    'normalized_width': item['normalized_width'],
                    'normalized_draught': item['normalized_draught'],
                    'max_speed': item.get('max_speed', 5.0),
                    'priority_score': self._get_priority_score(item.get('priority', 'normal'))
                },
                
                # 时间特征
                'temporal_features': {
                    'normalized_time': item['normalized_time'],
                    'hour_of_day': item['hour_of_day'],
                    'day_of_week': item['day'] % 7,  # 简化的星期计算
                    'is_peak_hour': 1 if 8 <= item['hour_of_day'] <= 18 else 0
                },
                
                # 港口状态特征 (需要根据当前时间计算)
                'port_features': self._calculate_port_features(item, normalized_data),
                
                # 目标泊位
                'target_berth': item.get('destination_berth', 'berth_0'),
                'estimated_service_time': item.get('estimated_service_time', 3600)
            }
            
            feature_data.append(features)
        
        return feature_data
    
    def _get_priority_score(self, priority: str) -> float:
        """获取优先级分数"""
        priority_scores = {
            'normal': 0.3,
            'high': 0.7,
            'urgent': 1.0
        }
        return priority_scores.get(priority, 0.3)
    
    def _calculate_port_features(self, current_item: Dict, all_data: List[Dict]) -> Dict:
        """计算港口状态特征"""
        current_time = current_item['start_time']
        
        # 计算当前时间窗口内的港口状态
        window_start = current_time - 3600  # 前1小时
        window_end = current_time + 3600    # 后1小时
        
        window_vessels = [
            item for item in all_data 
            if window_start <= item['start_time'] <= window_end
        ]
        
        return {
            'current_load': len(window_vessels) / 20,  # 归一化负载
            'avg_vessel_size': np.mean([item.get('length', 100) for item in window_vessels]) / 400 if window_vessels else 0.25,
            'congestion_level': min(len(window_vessels) / 10, 1.0),  # 拥堵程度
            'utilization_rate': min(len(window_vessels) / 15, 1.0)   # 利用率
        }
    
    def _split_train_test(self, feature_data: List[Dict], train_days: int = 5) -> Tuple[List[Dict], List[Dict]]:
        """切分训练测试集"""
        # 按天分组
        daily_data = {}
        for item in feature_data:
            day = item['day']
            if day not in daily_data:
                daily_data[day] = []
            daily_data[day].append(item)
        
        training_data = []
        test_data = []
        
        for day, day_data in daily_data.items():
            if day <= train_days:
                training_data.extend(day_data)
            else:
                test_data.extend(day_data)
        
        logger.info(f"训练测试集切分完成: 训练 {len(training_data)} 样本, 测试 {len(test_data)} 样本")
        return training_data, test_data
    
    def _build_graph_structure_data(self, training_data: List[Dict], test_data: List[Dict], port: str) -> Dict:
        """构建图结构数据"""
        # 定义节点类型和数量
        port_node_configs = {
            'new_orleans': {'berths': 40, 'anchorages': 6, 'channels': 8, 'terminals': 12},
            'baton_rouge': {'berths': 64, 'anchorages': 8, 'channels': 10, 'terminals': 16},
            'south_louisiana': {'berths': 25, 'anchorages': 4, 'channels': 6, 'terminals': 8},
            'gulfport': {'berths': 10, 'anchorages': 3, 'channels': 4, 'terminals': 6}
        }
        
        config = port_node_configs.get(port, port_node_configs['gulfport'])
        
        # 计算最大船舶数量
        max_vessels_train = len(set(item['mmsi'] for item in training_data))
        max_vessels_test = len(set(item['mmsi'] for item in test_data))
        max_vessels = max(max_vessels_train, max_vessels_test, 10)
        
        graph_structure = {
            'node_config': {
                **config,
                'max_vessels': max_vessels
            },
            'total_nodes': sum(config.values()) + max_vessels,
            'node_type_mapping': {
                'berth': 0,
                'anchorage': 1, 
                'channel': 2,
                'terminal': 3,
                'vessel': 4
            },
            'adjacency_template': self._create_adjacency_template(config, max_vessels)
        }
        
        return graph_structure
    
    def _create_adjacency_template(self, config: Dict, max_vessels: int) -> List[List[int]]:
        """创建邻接矩阵模板"""
        total_nodes = sum(config.values()) + max_vessels
        adj_matrix = [[0] * total_nodes for _ in range(total_nodes)]
        
        # 简化的连接规则
        berth_end = config['berths']
        anchor_end = berth_end + config['anchorages']
        channel_end = anchor_end + config['channels']
        terminal_end = channel_end + config['terminals']
        
        # 泊位-码头连接
        for i in range(config['berths']):
            for j in range(channel_end, terminal_end):
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1
        
        # 锚地-航道连接
        for i in range(berth_end, anchor_end):
            for j in range(anchor_end, channel_end):
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1
        
        # 航道-码头连接
        for i in range(anchor_end, channel_end):
            for j in range(channel_end, terminal_end):
                adj_matrix[i][j] = 1
                adj_matrix[j][i] = 1
        
        return adj_matrix
    
    def _calculate_normalization_stats(self, feature_data: List[Dict]) -> Dict:
        """计算归一化统计信息"""
        if not feature_data:
            return {}
        
        # 提取数值特征
        vessel_lengths = [item['vessel_features']['normalized_length'] for item in feature_data]
        vessel_widths = [item['vessel_features']['normalized_width'] for item in feature_data]
        service_times = [item['estimated_service_time'] for item in feature_data]
        
        stats = {
            'vessel_length': {'mean': np.mean(vessel_lengths), 'std': np.std(vessel_lengths)},
            'vessel_width': {'mean': np.mean(vessel_widths), 'std': np.std(vessel_widths)},
            'service_time': {'mean': np.mean(service_times), 'std': np.std(service_times)},
            'total_samples': len(feature_data),
            'unique_vessels': len(set(item['mmsi'] for item in feature_data))
        }
        
        return stats

def main():
    """主函数 - 准备所有港口的训练数据"""
    processor = GATTrainingDataProcessor()
    
    logger.info("开始准备GAT-PPO训练数据")
    all_data = processor.prepare_all_ports_data(num_days=7)
    
    # 打印汇总信息
    print("\n" + "="*60)
    print("GAT-PPO训练数据准备完成")
    print("="*60)
    
    for port, data in all_data.items():
        print(f"\n港口: {port.upper()}")
        print(f"  训练样本: {len(data['training_data'])}")
        print(f"  测试样本: {len(data['test_data'])}")
        print(f"  总船舶数: {data['metadata']['total_vessels']}")
        print(f"  图节点数: {data['graph_structure']['total_nodes']}")
    
    print(f"\n数据保存位置: {processor.training_data_dir}")
    print("="*60)

if __name__ == "__main__":
    main()