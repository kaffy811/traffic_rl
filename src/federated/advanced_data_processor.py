"""
高级AIS数据处理器
实现数据清洗、时序划分、特征工程和图结构构建
"""

import pandas as pd
import numpy as np
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from maritime_domain_knowledge import PORT_SPECIFICATIONS

logger = logging.getLogger(__name__)

class AdvancedDataProcessor:
    """高级数据处理器 - 实现完整的数据预处理流水线"""
    
    def __init__(self, port_name: str, base_data_path: str = "../../data/processed/ais"):
        self.port_name = port_name
        self.port_spec = PORT_SPECIFICATIONS[port_name]
        self.base_data_path = Path(base_data_path)
        
        # 数据质量参数
        self.quality_thresholds = {
            'max_speed': 25.0,      # 最大合理速度 (节)
            'min_speed': 0.0,       # 最小速度
            'max_coord_jump': 0.1,  # 最大坐标跳变 (度)
            'max_time_gap': 3600,   # 最大时间间隔 (秒)
            'min_draught': 0.5,     # 最小吃水
            'max_draught': 20.0,    # 最大吃水
            'min_length': 10.0,     # 最小船长
            'max_length': 400.0     # 最大船长
        }
        
        # 时间同步参数
        self.time_resolution = 60  # 时间分辨率 (秒)
        
        logger.info(f"初始化高级数据处理器 - 港口: {port_name}")
    
    def process_complete_pipeline(self, weeks: List[int] = [1]) -> Dict:
        """执行完整的数据处理流水线"""
        results = {
            'port': self.port_name,
            'weeks_processed': [],
            'data_quality_report': {},
            'time_series_report': {},
            'feature_engineering_report': {},
            'graph_structure_report': {}
        }
        
        for week in weeks:
            logger.info(f"处理 {self.port_name} Week{week} 数据...")
            
            # 步骤1: 数据清洗
            cleaned_data = self._clean_week_data(week)
            
            # 步骤2: 时序划分
            time_series_data = self._split_time_series(cleaned_data, week)
            
            # 步骤3: 特征工程
            feature_data = self._engineer_features(time_series_data, week)
            
            # 步骤4: 图结构构建
            graph_data = self._build_graph_structure(feature_data, week)
            
            results['weeks_processed'].append(week)
            results['data_quality_report'][f'week_{week}'] = cleaned_data['quality_report']
            results['time_series_report'][f'week_{week}'] = time_series_data['split_report']
            results['feature_engineering_report'][f'week_{week}'] = feature_data['feature_report']
            results['graph_structure_report'][f'week_{week}'] = graph_data['graph_report']
        
        # 保存处理结果
        self._save_processing_results(results)
        
        return results
    
    def _clean_week_data(self, week: int) -> Dict:
        """数据清洗 - 去除缺失、异常点，同步时间轴"""
        logger.info(f"开始数据清洗 - Week{week}")
        
        week_path = self.base_data_path / f"{self.port_name}_week{week}"
        if not week_path.exists():
            raise FileNotFoundError(f"未找到数据路径: {week_path}")
        
        cleaned_data = {
            'daily_data': {},
            'quality_report': {
                'total_records': 0,
                'cleaned_records': 0,
                'removed_records': 0,
                'anomaly_types': {
                    'missing_values': 0,
                    'speed_anomalies': 0,
                    'coordinate_jumps': 0,
                    'time_gaps': 0,
                    'vessel_spec_anomalies': 0
                }
            }
        }
        
        # 处理每一天的数据
        for day in range(1, 8):  # Day1 到 Day7
            day_folder = week_path / f"daily_2024070{day}"
            if not day_folder.exists():
                logger.warning(f"未找到日期文件夹: {day_folder}")
                continue
            
            csv_file = day_folder / f"ais_2024070{day}_region.csv"
            if not csv_file.exists():
                logger.warning(f"未找到CSV文件: {csv_file}")
                continue
            
            # 读取和清洗单日数据
            daily_cleaned = self._clean_daily_data(csv_file, day)
            cleaned_data['daily_data'][f'day_{day}'] = daily_cleaned['data']
            
            # 累计质量报告
            for key, value in daily_cleaned['quality_metrics'].items():
                if key in cleaned_data['quality_report']:
                    cleaned_data['quality_report'][key] += value
                elif key == 'anomaly_types':
                    for anomaly_type, count in value.items():
                        cleaned_data['quality_report']['anomaly_types'][anomaly_type] += count
        
        logger.info(f"数据清洗完成 - 总记录: {cleaned_data['quality_report']['total_records']}, "
                   f"清洗后: {cleaned_data['quality_report']['cleaned_records']}")
        
        return cleaned_data
    
    def _clean_daily_data(self, csv_file: Path, day: int) -> Dict:
        """清洗单日数据"""
        try:
            # 读取CSV数据 (分块读取大文件)
            chunk_size = 10000
            cleaned_chunks = []
            quality_metrics = {
                'total_records': 0,
                'cleaned_records': 0,
                'removed_records': 0,
                'anomaly_types': {
                    'missing_values': 0,
                    'speed_anomalies': 0,
                    'coordinate_jumps': 0,
                    'time_gaps': 0,
                    'vessel_spec_anomalies': 0
                }
            }
            
            for chunk in pd.read_csv(csv_file, chunksize=chunk_size):
                quality_metrics['total_records'] += len(chunk)
                
                # 数据清洗步骤
                cleaned_chunk = self._apply_cleaning_rules(chunk, quality_metrics)
                
                if len(cleaned_chunk) > 0:
                    # 时间同步
                    synchronized_chunk = self._synchronize_timestamps(cleaned_chunk)
                    cleaned_chunks.append(synchronized_chunk)
                    quality_metrics['cleaned_records'] += len(synchronized_chunk)
            
            # 合并所有清洗后的数据块
            if cleaned_chunks:
                daily_data = pd.concat(cleaned_chunks, ignore_index=True)
                daily_data = daily_data.sort_values('timestamp').reset_index(drop=True)
            else:
                daily_data = pd.DataFrame()
            
            quality_metrics['removed_records'] = quality_metrics['total_records'] - quality_metrics['cleaned_records']
            
            return {
                'data': daily_data,
                'quality_metrics': quality_metrics
            }
            
        except Exception as e:
            logger.error(f"清洗日数据失败 {csv_file}: {e}")
            return {
                'data': pd.DataFrame(),
                'quality_metrics': {
                    'total_records': 0,
                    'cleaned_records': 0,
                    'removed_records': 0,
                    'anomaly_types': {
                        'missing_values': 0,
                        'speed_anomalies': 0,
                        'coordinate_jumps': 0,
                        'time_gaps': 0,
                        'vessel_spec_anomalies': 0
                    }
                }
            }
    
    def _apply_cleaning_rules(self, df: pd.DataFrame, quality_metrics: Dict) -> pd.DataFrame:
        """应用数据清洗规则"""
        original_len = len(df)
        
        # 1. 去除缺失值
        missing_mask = df.isnull().any(axis=1)
        quality_metrics['anomaly_types']['missing_values'] += missing_mask.sum()
        df = df[~missing_mask]
        
        # 2. 速度异常检查
        if 'sog' in df.columns:
            speed_mask = (df['sog'] < self.quality_thresholds['min_speed']) | \
                        (df['sog'] > self.quality_thresholds['max_speed'])
            quality_metrics['anomaly_types']['speed_anomalies'] += speed_mask.sum()
            df = df[~speed_mask]
        
        # 3. 坐标跳变检查
        if 'lat' in df.columns and 'lon' in df.columns:
            df = df.sort_values('timestamp') if 'timestamp' in df.columns else df
            lat_diff = df['lat'].diff().abs()
            lon_diff = df['lon'].diff().abs()
            coord_jump_mask = (lat_diff > self.quality_thresholds['max_coord_jump']) | \
                             (lon_diff > self.quality_thresholds['max_coord_jump'])
            quality_metrics['anomaly_types']['coordinate_jumps'] += coord_jump_mask.sum()
            df = df[~coord_jump_mask]
        
        # 4. 船舶规格异常检查
        vessel_spec_mask = pd.Series([False] * len(df), index=df.index)
        
        if 'draught' in df.columns:
            draught_mask = (df['draught'] < self.quality_thresholds['min_draught']) | \
                          (df['draught'] > self.quality_thresholds['max_draught'])
            vessel_spec_mask |= draught_mask
        
        if 'length' in df.columns:
            length_mask = (df['length'] < self.quality_thresholds['min_length']) | \
                         (df['length'] > self.quality_thresholds['max_length'])
            vessel_spec_mask |= length_mask
        
        quality_metrics['anomaly_types']['vessel_spec_anomalies'] += vessel_spec_mask.sum()
        df = df[~vessel_spec_mask]
        
        return df
    
    def _synchronize_timestamps(self, df: pd.DataFrame) -> pd.DataFrame:
        """同步时间轴 - 按分钟对齐，填补空缺"""
        if 'timestamp' not in df.columns or len(df) == 0:
            return df
        
        try:
            # 转换时间戳
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # 按时间分辨率对齐 (默认60秒)
            df['aligned_timestamp'] = df['timestamp'].dt.floor(f'{self.time_resolution}S')
            
            # 按MMSI分组处理
            synchronized_groups = []
            
            for mmsi, group in df.groupby('mmsi'):
                group = group.sort_values('aligned_timestamp')
                
                # 创建完整的时间序列
                start_time = group['aligned_timestamp'].min()
                end_time = group['aligned_timestamp'].max()
                full_time_range = pd.date_range(
                    start=start_time, 
                    end=end_time, 
                    freq=f'{self.time_resolution}S'
                )
                
                # 重新索引并填充缺失值
                group_indexed = group.set_index('aligned_timestamp')
                group_reindexed = group_indexed.reindex(full_time_range)
                
                # 前向填充数值字段
                numeric_columns = group_reindexed.select_dtypes(include=[np.number]).columns
                group_reindexed[numeric_columns] = group_reindexed[numeric_columns].fillna(method='ffill')
                
                # 填充MMSI
                group_reindexed['mmsi'] = mmsi
                
                # 重置索引
                group_reindexed = group_reindexed.reset_index()
                group_reindexed.rename(columns={'index': 'timestamp'}, inplace=True)
                
                synchronized_groups.append(group_reindexed)
            
            if synchronized_groups:
                synchronized_df = pd.concat(synchronized_groups, ignore_index=True)
                return synchronized_df.sort_values('timestamp').reset_index(drop=True)
            else:
                return df
                
        except Exception as e:
            logger.warning(f"时间同步失败: {e}")
            return df
    
    def _split_time_series(self, cleaned_data: Dict, week: int) -> Dict:
        """时序划分 - 按天切片，滚动切分策略"""
        logger.info(f"开始时序划分 - Week{week}")
        
        time_series_data = {
            'training_sets': {},  # 前5天训练
            'test_sets': {},      # 后2天测试
            'split_report': {
                'training_days': [1, 2, 3, 4, 5],
                'test_days': [6, 7],
                'training_samples': 0,
                'test_samples': 0,
                'continuity_check': {}
            }
        }
        
        daily_data = cleaned_data['daily_data']
        
        # 训练集 (Day1-Day5)
        training_data = []
        for day in [1, 2, 3, 4, 5]:
            day_key = f'day_{day}'
            if day_key in daily_data and not daily_data[day_key].empty:
                day_data = daily_data[day_key].copy()
                day_data['day'] = day
                day_data['split'] = 'train'
                training_data.append(day_data)
                
                # 连续性检查
                time_series_data['split_report']['continuity_check'][f'day_{day}'] = \
                    self._check_data_continuity(day_data)
        
        if training_data:
            time_series_data['training_sets']['week_1'] = pd.concat(training_data, ignore_index=True)
            time_series_data['split_report']['training_samples'] = len(time_series_data['training_sets']['week_1'])
        
        # 测试集 (Day6-Day7)
        test_data = []
        for day in [6, 7]:
            day_key = f'day_{day}'
            if day_key in daily_data and not daily_data[day_key].empty:
                day_data = daily_data[day_key].copy()
                day_data['day'] = day
                day_data['split'] = 'test'
                test_data.append(day_data)
                
                # 连续性检查
                time_series_data['split_report']['continuity_check'][f'day_{day}'] = \
                    self._check_data_continuity(day_data)
        
        if test_data:
            time_series_data['test_sets']['week_1'] = pd.concat(test_data, ignore_index=True)
            time_series_data['split_report']['test_samples'] = len(time_series_data['test_sets']['week_1'])
        
        logger.info(f"时序划分完成 - 训练样本: {time_series_data['split_report']['training_samples']}, "
                   f"测试样本: {time_series_data['split_report']['test_samples']}")
        
        return time_series_data
    
    def _check_data_continuity(self, day_data: pd.DataFrame) -> Dict:
        """检查数据连续性"""
        if day_data.empty or 'timestamp' not in day_data.columns:
            return {'continuous': False, 'gaps': 0, 'coverage': 0.0}
        
        try:
            day_data = day_data.sort_values('timestamp')
            time_diffs = day_data['timestamp'].diff().dt.total_seconds()
            
            # 计算时间间隔
            expected_interval = self.time_resolution
            large_gaps = (time_diffs > expected_interval * 2).sum()
            
            # 计算覆盖率
            total_time = (day_data['timestamp'].max() - day_data['timestamp'].min()).total_seconds()
            expected_records = total_time / expected_interval if total_time > 0 else 0
            coverage = len(day_data) / expected_records if expected_records > 0 else 0
            
            return {
                'continuous': large_gaps == 0,
                'gaps': int(large_gaps),
                'coverage': min(1.0, coverage)
            }
        except Exception as e:
            logger.warning(f"连续性检查失败: {e}")
            return {'continuous': False, 'gaps': 0, 'coverage': 0.0}
    
    def _engineer_features(self, time_series_data: Dict, week: int) -> Dict:
        """特征工程 - 构建24维状态向量"""
        logger.info(f"开始特征工程 - Week{week}")
        
        feature_data = {
            'training_features': {},
            'test_features': {},
            'feature_report': {
                'state_vector_dim': 24,
                'feature_categories': {
                    'vessel_features': 6,      # 船舶特征
                    'spatial_features': 4,     # 空间节点特征
                    'queue_features': 6,       # 队列特征
                    'berth_features': 4,       # 泊位特征
                    'port_features': 4         # 港口全局特征
                },
                'normalization_stats': {}
            }
        }
        
        # 处理训练集特征
        if 'week_1' in time_series_data['training_sets']:
            training_df = time_series_data['training_sets']['week_1']
            feature_data['training_features']['week_1'] = self._extract_24d_features(training_df, 'train')
        
        # 处理测试集特征
        if 'week_1' in time_series_data['test_sets']:
            test_df = time_series_data['test_sets']['week_1']
            feature_data['test_features']['week_1'] = self._extract_24d_features(test_df, 'test')
        
        # 计算归一化统计信息
        if feature_data['training_features']:
            feature_data['feature_report']['normalization_stats'] = \
                self._calculate_normalization_stats(feature_data['training_features']['week_1'])
        
        logger.info(f"特征工程完成 - 24维状态向量构建完毕")
        
        return feature_data
    
    def _extract_24d_features(self, df: pd.DataFrame, split_type: str) -> Dict:
        """提取24维状态向量特征"""
        if df.empty:
            return {'features': [], 'metadata': {}}
        
        features_list = []
        
        # 按时间窗口提取特征
        for mmsi, vessel_group in df.groupby('mmsi'):
            vessel_group = vessel_group.sort_values('timestamp')
            
            for i in range(len(vessel_group)):
                row = vessel_group.iloc[i]
                
                # 构建24维特征向量
                feature_vector = self._build_24d_vector(row, vessel_group, i)
                
                features_list.append({
                    'mmsi': mmsi,
                    'timestamp': row['timestamp'],
                    'features': feature_vector,
                    'split': split_type
                })
        
        return {
            'features': features_list,
            'metadata': {
                'total_samples': len(features_list),
                'unique_vessels': df['mmsi'].nunique(),
                'time_range': {
                    'start': df['timestamp'].min(),
                    'end': df['timestamp'].max()
                }
            }
        }
    
    def _build_24d_vector(self, current_row: pd.Series, vessel_group: pd.DataFrame, row_index: int) -> List[float]:
        """构建24维状态向量"""
        features = []
        
        # 1. 船舶特征 (6维)
        vessel_features = [
            float(current_row.get('sog', 0)),           # 航速
            float(current_row.get('cog', 0)) / 360.0,   # 航向 (归一化)
            float(current_row.get('draught', 0)) / 20.0, # 吃水 (归一化)
            float(current_row.get('length', 0)) / 400.0, # 船长 (归一化)
            float(current_row.get('width', 0)) / 60.0,   # 船宽 (归一化)
            float(current_row.get('vessel_type', 0)) / 100.0  # 船舶类型 (归一化)
        ]
        features.extend(vessel_features)
        
        # 2. 空间节点特征 (4维)
        port_lat, port_lon = self.port_spec.lat, self.port_spec.lon
        current_lat = float(current_row.get('lat', port_lat))
        current_lon = float(current_row.get('lon', port_lon))
        
        spatial_features = [
            (current_lat - port_lat) / 0.1,    # 相对纬度
            (current_lon - port_lon) / 0.1,    # 相对经度
            self._calculate_distance_to_port(current_lat, current_lon), # 到港口距离
            self._calculate_adjacency_score(current_lat, current_lon)   # 邻接度分数
        ]
        features.extend(spatial_features)
        
        # 3. 队列特征 (6维) - 基于历史数据估算
        queue_features = self._estimate_queue_features(current_row, vessel_group, row_index)
        features.extend(queue_features)
        
        # 4. 泊位特征 (4维) - 基于港口规格
        berth_features = [
            float(self.port_spec.berths) / 20.0,        # 总泊位数 (归一化)
            np.random.uniform(0.3, 0.9),                # 泊位利用率 (模拟)
            np.random.uniform(8, 15) / 20.0,            # 平均水深 (归一化)
            np.random.uniform(0, 2) / 5.0               # 水流限制 (归一化)
        ]
        features.extend(berth_features)
        
        # 5. 港口全局特征 (4维)
        port_features = [
            np.random.uniform(100, 500) / 1000.0,       # 日吞吐量 (归一化)
            np.random.uniform(0, 1),                    # 天气条件 (0-1)
            np.random.uniform(-2, 2) / 4.0 + 0.5,      # 潮汐水位 (归一化)
            np.random.uniform(0, 3) / 5.0               # 风浪等级 (归一化)
        ]
        features.extend(port_features)
        
        # 确保正好24维
        assert len(features) == 24, f"特征维度错误: {len(features)}"
        
        return features
    
    def _calculate_distance_to_port(self, lat: float, lon: float) -> float:
        """计算到港口的距离 (归一化)"""
        port_lat, port_lon = self.port_spec.lat, self.port_spec.lon
        distance = np.sqrt((lat - port_lat)**2 + (lon - port_lon)**2)
        return min(1.0, distance / 0.1)  # 归一化到[0,1]
    
    def _calculate_adjacency_score(self, lat: float, lon: float) -> float:
        """计算邻接度分数"""
        # 基于距离的邻接度计算
        distance = self._calculate_distance_to_port(lat, lon)
        return max(0.0, 1.0 - distance)  # 距离越近，邻接度越高
    
    def _estimate_queue_features(self, current_row: pd.Series, vessel_group: pd.DataFrame, row_index: int) -> List[float]:
        """估算队列特征"""
        # 基于历史窗口估算队列状态
        window_size = min(10, row_index + 1)
        history_window = vessel_group.iloc[max(0, row_index - window_size + 1):row_index + 1]
        
        if len(history_window) == 0:
            return [0.0] * 6
        
        # 估算队列长度 (基于附近船舶数量)
        avg_queue_length = min(1.0, len(history_window) / 20.0)
        
        # 估算等待时间分布
        speed_variance = history_window['sog'].var() if len(history_window) > 1 else 0
        avg_waiting_time = min(1.0, speed_variance / 10.0)
        
        # 其他队列特征
        queue_features = [
            avg_queue_length,                           # 平均队列长度
            avg_waiting_time,                          # 平均等待时间
            np.random.uniform(0, 1),                   # 队列变化率
            np.random.uniform(0, 1),                   # 优先级船舶比例
            np.random.uniform(2, 12) / 24.0,           # 预计处理时间 (归一化)
            np.random.uniform(0, 1)                    # 队列稳定性
        ]
        
        return queue_features
    
    def _calculate_normalization_stats(self, training_features: Dict) -> Dict:
        """计算归一化统计信息"""
        if not training_features['features']:
            return {}
        
        # 提取所有特征向量
        all_features = np.array([sample['features'] for sample in training_features['features']])
        
        stats = {
            'mean': all_features.mean(axis=0).tolist(),
            'std': all_features.std(axis=0).tolist(),
            'min': all_features.min(axis=0).tolist(),
            'max': all_features.max(axis=0).tolist()
        }
        
        return stats
    
    def _build_graph_structure(self, feature_data: Dict, week: int) -> Dict:
        """构建图结构 - 5类节点和智能邻接矩阵"""
        logger.info(f"开始构建图结构 - Week{week}")
        
        graph_data = {
            'node_types': {
                'berth': [],      # 泊位节点
                'anchorage': [],  # 锚地节点
                'channel': [],    # 航道节点
                'terminal': [],   # 码头节点
                'vessel': []      # 船舶节点
            },
            'adjacency_matrices': {},
            'graph_report': {
                'total_nodes': 0,
                'node_type_counts': {},
                'edge_counts': {},
                'graph_density': 0.0
            }
        }
        
        # 定义港口基础设施节点
        self._define_infrastructure_nodes(graph_data)
        
        # 处理训练集图结构
        if 'week_1' in feature_data['training_features']:
            training_graphs = self._build_temporal_graphs(
                feature_data['training_features']['week_1'], 'train'
            )
            graph_data['adjacency_matrices']['training'] = training_graphs
        
        # 处理测试集图结构
        if 'week_1' in feature_data['test_features']:
            test_graphs = self._build_temporal_graphs(
                feature_data['test_features']['week_1'], 'test'
            )
            graph_data['adjacency_matrices']['testing'] = test_graphs
        
        # 计算图统计信息
        graph_data['graph_report'] = self._calculate_graph_statistics(graph_data)
        
        logger.info(f"图结构构建完成 - 总节点数: {graph_data['graph_report']['total_nodes']}")
        
        return graph_data
    
    def _define_infrastructure_nodes(self, graph_data: Dict):
        """定义港口基础设施节点"""
        port_lat, port_lon = self.port_spec.lat, self.port_spec.lon
        
        # 泊位节点
        for i in range(self.port_spec.berths):
            berth_node = {
                'id': f'berth_{i}',
                'type': 'berth',
                'lat': port_lat + np.random.normal(0, 0.005),
                'lon': port_lon + np.random.normal(0, 0.005),
                'capacity': np.random.uniform(1, 3),
                'depth': np.random.uniform(8, 15)
            }
            graph_data['node_types']['berth'].append(berth_node)
        
        # 锚地节点
        for i in range(3):  # 3个锚地区域
            anchorage_node = {
                'id': f'anchorage_{i}',
                'type': 'anchorage',
                'lat': port_lat + np.random.normal(0, 0.02),
                'lon': port_lon + np.random.normal(0, 0.02),
                'capacity': np.random.uniform(5, 15)
            }
            graph_data['node_types']['anchorage'].append(anchorage_node)
        
        # 航道节点
        for i in range(5):  # 5个航道段
            channel_node = {
                'id': f'channel_{i}',
                'type': 'channel',
                'lat': port_lat + (i - 2) * 0.01,
                'lon': port_lon + (i - 2) * 0.01,
                'width': np.random.uniform(100, 300),
                'depth': np.random.uniform(10, 20)
            }
            graph_data['node_types']['channel'].append(channel_node)
        
        # 码头节点
        for i in range(2):  # 2个码头区域
            terminal_node = {
                'id': f'terminal_{i}',
                'type': 'terminal',
                'lat': port_lat + (i - 0.5) * 0.008,
                'lon': port_lon + (i - 0.5) * 0.008,
                'throughput': np.random.uniform(100, 500)
            }
            graph_data['node_types']['terminal'].append(terminal_node)
    
    def _build_temporal_graphs(self, feature_data: Dict, split_type: str) -> List[Dict]:
        """构建时序图结构"""
        temporal_graphs = []
        
        if not feature_data['features']:
            return temporal_graphs
        
        # 按时间戳分组
        features_by_time = {}
        for sample in feature_data['features']:
            timestamp = sample['timestamp']
            if timestamp not in features_by_time:
                features_by_time[timestamp] = []
            features_by_time[timestamp].append(sample)
        
        # 为每个时间步构建图
        for timestamp, time_samples in features_by_time.items():
            graph = self._build_single_timestep_graph(timestamp, time_samples, split_type)
            temporal_graphs.append(graph)
        
        return temporal_graphs
    
    def _build_single_timestep_graph(self, timestamp, time_samples: List[Dict], split_type: str) -> Dict:
        """构建单个时间步的图"""
        # 节点特征矩阵
        node_features = []
        node_ids = []
        
        # 添加基础设施节点特征 (简化为固定特征)
        infrastructure_features = self._get_infrastructure_features()
        node_features.extend(infrastructure_features['features'])
        node_ids.extend(infrastructure_features['ids'])
        
        # 添加船舶节点特征
        for sample in time_samples:
            node_features.append(sample['features'])
            node_ids.append(f"vessel_{sample['mmsi']}")
        
        # 构建邻接矩阵
        num_nodes = len(node_features)
        adjacency_matrix = self._build_adjacency_matrix(node_features, node_ids, time_samples)
        
        return {
            'timestamp': timestamp,
            'node_features': np.array(node_features),
            'adjacency_matrix': adjacency_matrix,
            'node_ids': node_ids,
            'num_nodes': num_nodes,
            'split': split_type
        }
    
    def _get_infrastructure_features(self) -> Dict:
        """获取基础设施节点特征"""
        features = []
        ids = []
        
        # 为基础设施节点生成24维特征向量
        infrastructure_nodes = (
            len(self.port_spec.berths) +  # 泊位
            3 +  # 锚地
            5 +  # 航道
            2    # 码头
        )
        
        for i in range(infrastructure_nodes):
            # 基础设施节点的特征向量 (与船舶特征维度一致)
            infra_features = [0.0] * 24  # 初始化为0
            
            # 设置一些基础设施特有的特征
            infra_features[0] = 0.0  # 航速为0 (静态节点)
            infra_features[6] = np.random.uniform(0, 1)  # 空间特征
            infra_features[7] = np.random.uniform(0, 1)
            
            features.append(infra_features)
            ids.append(f'infra_{i}')
        
        return {'features': features, 'ids': ids}
    
    def _build_adjacency_matrix(self, node_features: List, node_ids: List, vessel_samples: List) -> np.ndarray:
        """构建智能邻接矩阵"""
        num_nodes = len(node_features)
        adjacency_matrix = np.zeros((num_nodes, num_nodes))
        
        # 基于距离和业务关联构建邻接关系
        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                # 计算节点间的相似度/关联度
                similarity = self._calculate_node_similarity(
                    node_features[i], node_features[j], node_ids[i], node_ids[j]
                )
                
                # 设置邻接关系 (对称矩阵)
                adjacency_matrix[i, j] = similarity
                adjacency_matrix[j, i] = similarity
        
        return adjacency_matrix
    
    def _calculate_node_similarity(self, features1: List, features2: List, id1: str, id2: str) -> float:
        """计算节点间相似度"""
        # 基于特征向量的欧氏距离
        features1 = np.array(features1)
        features2 = np.array(features2)
        
        # 空间特征 (位置相关)
        spatial_distance = np.linalg.norm(features1[6:10] - features2[6:10])
        spatial_similarity = max(0, 1 - spatial_distance)
        
        # 业务关联 (基于节点类型)
        business_similarity = self._calculate_business_similarity(id1, id2)
        
        # 综合相似度
        total_similarity = 0.6 * spatial_similarity + 0.4 * business_similarity
        
        return min(1.0, max(0.0, total_similarity))
    
    def _calculate_business_similarity(self, id1: str, id2: str) -> float:
        """计算业务关联相似度"""
        # 定义业务关联规则
        business_rules = {
            ('vessel', 'berth'): 0.8,
            ('vessel', 'anchorage'): 0.6,
            ('vessel', 'channel'): 0.7,
            ('vessel', 'terminal'): 0.5,
            ('berth', 'terminal'): 0.9,
            ('channel', 'berth'): 0.8,
            ('anchorage', 'channel'): 0.7,
            ('infra', 'infra'): 0.3
        }
        
        # 提取节点类型
        type1 = self._extract_node_type(id1)
        type2 = self._extract_node_type(id2)
        
        # 查找业务关联度
        rule_key = tuple(sorted([type1, type2]))
        return business_rules.get(rule_key, 0.1)
    
    def _extract_node_type(self, node_id: str) -> str:
        """提取节点类型"""
        if node_id.startswith('vessel'):
            return 'vessel'
        elif node_id.startswith('berth'):
            return 'berth'
        elif node_id.startswith('anchorage'):
            return 'anchorage'
        elif node_id.startswith('channel'):
            return 'channel'
        elif node_id.startswith('terminal'):
            return 'terminal'
        else:
            return 'infra'
    
    def _calculate_graph_statistics(self, graph_data: Dict) -> Dict:
        """计算图统计信息"""
        total_nodes = sum(len(nodes) for nodes in graph_data['node_types'].values())
        
        node_type_counts = {
            node_type: len(nodes) 
            for node_type, nodes in graph_data['node_types'].items()
        }
        
        # 计算边数量 (基于第一个图)
        edge_counts = {}
        graph_density = 0.0
        
        if graph_data['adjacency_matrices'].get('training'):
            first_graph = graph_data['adjacency_matrices']['training'][0]
            adj_matrix = first_graph['adjacency_matrix']
            total_edges = np.sum(adj_matrix > 0) // 2  # 无向图
            max_edges = total_nodes * (total_nodes - 1) // 2
            graph_density = total_edges / max_edges if max_edges > 0 else 0
            
            edge_counts['total_edges'] = int(total_edges)
            edge_counts['max_possible_edges'] = int(max_edges)
        
        return {
            'total_nodes': total_nodes,
            'node_type_counts': node_type_counts,
            'edge_counts': edge_counts,
            'graph_density': graph_density
        }
    
    def _save_processing_results(self, results: Dict):
        """保存处理结果"""
        output_dir = Path(f"data/processed_advanced/{self.port_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存处理报告
        report_file = output_dir / "processing_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"处理结果已保存到: {output_dir}")

def main():
    """主函数 - 演示完整的数据处理流水线"""
    print("启动高级AIS数据处理器...")
    
    # 处理所有港口
    ports = ['baton_rouge', 'new_orleans', 'south_louisiana', 'gulfport']
    
    for port in ports:
        try:
            print(f"\n处理港口: {port}")
            processor = AdvancedDataProcessor(port)
            results = processor.process_complete_pipeline(weeks=[1])
            
            print(f"{port} 处理完成:")
            print(f"  - 数据质量: {results['data_quality_report']}")
            print(f"  - 时序划分: {results['time_series_report']}")
            print(f"  - 特征工程: {results['feature_engineering_report']}")
            print(f"  - 图结构: {results['graph_structure_report']}")
            
        except Exception as e:
            print(f"{port} 处理失败: {e}")

if __name__ == "__main__":
    main()