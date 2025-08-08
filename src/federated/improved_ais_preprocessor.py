"""
改进的AIS数据预处理器
基于海事领域知识，识别锚地、航道、泊位等关键区域
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon
from typing import Dict, List, Tuple, Optional
import logging
from datetime import datetime, timedelta
import os

from maritime_domain_knowledge import PORT_SPECIFICATIONS

logger = logging.getLogger(__name__)

class MaritimeZoneDetector:
    """海事区域检测器"""
    
    def __init__(self, port_name: str):
        self.port_spec = PORT_SPECIFICATIONS[port_name]
        self.port_name = port_name
        
        # 定义港口相关区域
        self.zones = self._define_port_zones()
        
    def _define_port_zones(self) -> Dict:
        """定义港口相关的地理区域"""
        port_lat, port_lon = self.port_spec.lat, self.port_spec.lon
        
        zones = {
            # 港口区域 (核心区域)
            'port_area': self._create_polygon_around_point(
                port_lat, port_lon, 0.01  # ±0.01度 (~1.1km)
            ),
            
            # 锚地区域 (港口外围)
            'anchorage_area': self._create_polygon_around_point(
                port_lat, port_lon, 0.05  # ±0.05度 (~5.5km)
            ),
            
            # 航道区域 (进港路径)
            'channel_area': self._create_channel_polygon(port_lat, port_lon),
            
            # 泊位区域 (港内停泊)
            'berth_area': self._create_polygon_around_point(
                port_lat, port_lon, 0.005  # ±0.005度 (~550m)
            ),
            
            # 进港等待区域
            'approach_area': self._create_polygon_around_point(
                port_lat, port_lon, 0.1  # ±0.1度 (~11km)
            )
        }
        
        return zones
    
    def _create_polygon_around_point(self, lat: float, lon: float, radius: float) -> Polygon:
        """在指定点周围创建矩形区域"""
        return Polygon([
            (lon - radius, lat - radius),
            (lon + radius, lat - radius),
            (lon + radius, lat + radius),
            (lon - radius, lat + radius)
        ])
    
    def _create_channel_polygon(self, port_lat: float, port_lon: float) -> Polygon:
        """创建航道区域 (根据港口特点定制)"""
        if self.port_name in ['new_orleans', 'south_louisiana', 'baton_rouge']:
            # 密西西比河航道 (南北向)
            return Polygon([
                (port_lon - 0.02, port_lat - 0.15),  # 河道下游
                (port_lon + 0.02, port_lat - 0.15),
                (port_lon + 0.02, port_lat + 0.15),  # 河道上游
                (port_lon - 0.02, port_lat + 0.15)
            ])
        else:  # gulfport
            # 海岸航道 (东西向)
            return Polygon([
                (port_lon - 0.1, port_lat - 0.02),
                (port_lon + 0.1, port_lat - 0.02),
                (port_lon + 0.1, port_lat + 0.02),
                (port_lon - 0.1, port_lat + 0.02)
            ])
    
    def classify_vessel_location(self, lat: float, lon: float) -> str:
        """分类船舶位置"""
        point = Point(lon, lat)
        
        if self.zones['berth_area'].contains(point):
            return 'at_berth'
        elif self.zones['port_area'].contains(point):
            return 'in_port'
        elif self.zones['anchorage_area'].contains(point):
            return 'at_anchor'
        elif self.zones['channel_area'].contains(point):
            return 'in_channel'
        elif self.zones['approach_area'].contains(point):
            return 'approaching'
        else:
            return 'at_sea'

class ImprovedAISPreprocessor:
    """改进的AIS数据预处理器"""
    
    def __init__(self, port_name: str):
        self.port_name = port_name
        self.port_spec = PORT_SPECIFICATIONS[port_name]
        self.zone_detector = MaritimeZoneDetector(port_name)
        
        # 数据过滤参数
        self.min_length = 50  # 最小船长(米)
        self.max_sog = 30    # 最大航速(节)
        self.min_movement_threshold = 0.001  # 最小移动距离(度)
        self.max_stationary_hours = 24      # 最大静止时间(小时)
        
    def preprocess_ais_data(self, raw_data_path: str, output_dir: str) -> Dict:
        """预处理AIS数据的主函数"""
        logger.info(f"开始预处理 {self.port_name} 的AIS数据...")
        
        # 1. 加载原始数据
        df = self._load_raw_data(raw_data_path)
        logger.info(f"加载原始数据: {len(df)} 条记录")
        
        # 2. 基础数据清理
        df = self._basic_cleaning(df)
        logger.info(f"基础清理后: {len(df)} 条记录")
        
        # 3. 船舶轨迹识别
        df = self._identify_vessel_trajectories(df)
        logger.info(f"轨迹识别后: {len(df)} 条记录")
        
        # 4. 区域分类
        df = self._classify_locations(df)
        logger.info(f"位置分类完成")
        
        # 5. 识别关键事件
        df = self._identify_key_events(df)
        logger.info(f"事件识别完成")
        
        # 6. 计算业务指标
        metrics = self._calculate_maritime_metrics(df)
        logger.info(f"业务指标计算完成")
        
        # 7. 生成训练数据
        training_data = self._generate_training_data(df, metrics)
        
        # 8. 保存处理结果
        self._save_processed_data(df, training_data, metrics, output_dir)
        
        return {
            'processed_records': len(df),
            'training_samples': len(training_data),
            'metrics': metrics,
            'output_dir': output_dir
        }
    
    def _load_raw_data(self, data_path: str) -> pd.DataFrame:
        """加载原始AIS数据"""
        try:
            if data_path.endswith('.csv'):
                df = pd.read_csv(data_path)
            elif data_path.endswith('.parquet'):
                df = pd.read_parquet(data_path)
            else:
                # 尝试从目录加载多个文件
                dfs = []
                for file in os.listdir(data_path):
                    if file.endswith('.csv'):
                        file_path = os.path.join(data_path, file)
                        dfs.append(pd.read_csv(file_path))
                df = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
            
            # 标准化列名
            df = self._standardize_columns(df)
            
            return df
            
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            return pd.DataFrame()
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """标准化列名"""
        column_mapping = {
            'mmsi': 'mmsi',
            'MMSI': 'mmsi',
            'lat': 'lat',
            'latitude': 'lat',
            'LAT': 'lat',
            'lon': 'lon',
            'longitude': 'lon',
            'LON': 'lon',
            'sog': 'sog',
            'SOG': 'sog',
            'cog': 'cog',
            'COG': 'cog',
            'heading': 'heading',
            'timestamp': 'timestamp',
            'time': 'timestamp',
            'datetime': 'timestamp',
            'vessel_type': 'vessel_type',
            'length': 'length',
            'width': 'width',
            'draught': 'draught',
            'cargo': 'cargo'
        }
        
        df = df.rename(columns=column_mapping)
        
        # 确保必要列存在
        required_columns = ['mmsi', 'lat', 'lon', 'timestamp']
        for col in required_columns:
            if col not in df.columns:
                logger.warning(f"缺少必要列: {col}")
                if col == 'timestamp':
                    df[col] = pd.Timestamp.now()
                else:
                    df[col] = 0
        
        return df
    
    def _basic_cleaning(self, df: pd.DataFrame) -> pd.DataFrame:
        """基础数据清理"""
        initial_count = len(df)
        
        # 1. 删除无效位置
        df = df.dropna(subset=['lat', 'lon', 'mmsi'])
        df = df[(df['lat'] != 0) & (df['lon'] != 0)]
        
        # 2. 地理范围过滤 (港口周边区域)
        lat_range = (self.port_spec.lat - 0.2, self.port_spec.lat + 0.2)
        lon_range = (self.port_spec.lon - 0.2, self.port_spec.lon + 0.2)
        
        df = df[
            (df['lat'].between(*lat_range)) &
            (df['lon'].between(*lon_range))
        ]
        
        # 3. 速度过滤
        if 'sog' in df.columns:
            df['sog'] = pd.to_numeric(df['sog'], errors='coerce').fillna(0)
            df = df[df['sog'] <= self.max_sog]
        else:
            df['sog'] = 0
        
        # 4. 船舶尺寸过滤
        if 'length' in df.columns:
            df['length'] = pd.to_numeric(df['length'], errors='coerce').fillna(150)
            df = df[df['length'] >= self.min_length]
        else:
            df['length'] = 150  # 默认长度
        
        # 5. 填充缺失值
        df['cog'] = pd.to_numeric(df.get('cog', 0), errors='coerce').fillna(0)
        df['heading'] = pd.to_numeric(df.get('heading', 0), errors='coerce').fillna(0)
        df['width'] = pd.to_numeric(df.get('width', 25), errors='coerce').fillna(25)
        df['draught'] = pd.to_numeric(df.get('draught', 8), errors='coerce').fillna(8)
        df['vessel_type'] = pd.to_numeric(df.get('vessel_type', 70), errors='coerce').fillna(70)
        
        # 6. 时间戳处理
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
        df = df.dropna(subset=['timestamp'])
        df = df.sort_values(['mmsi', 'timestamp'])
        
        logger.info(f"基础清理: {initial_count} -> {len(df)} 条记录")
        return df
    
    def _identify_vessel_trajectories(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别船舶轨迹和关键路径"""
        trajectory_data = []
        
        for mmsi, vessel_df in df.groupby('mmsi'):
            vessel_df = vessel_df.sort_values('timestamp')
            
            # 计算移动距离
            vessel_df['lat_diff'] = vessel_df['lat'].diff()
            vessel_df['lon_diff'] = vessel_df['lon'].diff()
            vessel_df['distance'] = np.sqrt(
                vessel_df['lat_diff']**2 + vessel_df['lon_diff']**2
            )
            
            # 识别静止点
            vessel_df['is_stationary'] = vessel_df['distance'] < self.min_movement_threshold
            
            # 计算静止时长
            vessel_df['stationary_duration'] = self._calculate_stationary_duration(vessel_df)
            
            # 过滤长时间静止的点
            vessel_df = vessel_df[
                (vessel_df['stationary_duration'] <= self.max_stationary_hours) |
                (~vessel_df['is_stationary'])
            ]
            
            # 识别轨迹段
            vessel_df['trajectory_segment'] = self._identify_trajectory_segments(vessel_df)
            
            trajectory_data.append(vessel_df)
        
        return pd.concat(trajectory_data, ignore_index=True) if trajectory_data else df
    
    def _calculate_stationary_duration(self, vessel_df: pd.DataFrame) -> pd.Series:
        """计算静止时长"""
        durations = []
        current_duration = 0
        
        for i, is_stationary in enumerate(vessel_df['is_stationary']):
            if is_stationary:
                if i > 0:
                    time_diff = (vessel_df.iloc[i]['timestamp'] - 
                               vessel_df.iloc[i-1]['timestamp']).total_seconds() / 3600
                    current_duration += time_diff
            else:
                current_duration = 0
            
            durations.append(current_duration)
        
        return pd.Series(durations)
    
    def _identify_trajectory_segments(self, vessel_df: pd.DataFrame) -> pd.Series:
        """识别轨迹段"""
        segments = []
        current_segment = 0
        
        for i in range(len(vessel_df)):
            if i > 0 and vessel_df.iloc[i]['is_stationary'] != vessel_df.iloc[i-1]['is_stationary']:
                current_segment += 1
            segments.append(current_segment)
        
        return pd.Series(segments)
    
    def _classify_locations(self, df: pd.DataFrame) -> pd.DataFrame:
        """对位置进行分类"""
        df['location_type'] = df.apply(
            lambda row: self.zone_detector.classify_vessel_location(row['lat'], row['lon']),
            axis=1
        )
        
        return df
    
    def _identify_key_events(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别关键事件"""
        event_data = []
        
        for mmsi, vessel_df in df.groupby('mmsi'):
            vessel_df = vessel_df.sort_values('timestamp')
            
            # 识别位置变化事件
            vessel_df['location_change'] = vessel_df['location_type'] != vessel_df['location_type'].shift()
            
            # 识别靠泊/离泊事件
            vessel_df['berth_event'] = self._identify_berth_events(vessel_df)
            
            # 识别排队事件
            vessel_df['queue_event'] = self._identify_queue_events(vessel_df)
            
            # 识别进港/出港事件
            vessel_df['port_event'] = self._identify_port_events(vessel_df)
            
            event_data.append(vessel_df)
        
        return pd.concat(event_data, ignore_index=True) if event_data else df
    
    def _identify_berth_events(self, vessel_df: pd.DataFrame) -> pd.Series:
        """识别靠泊/离泊事件"""
        events = ['none'] * len(vessel_df)
        
        for i in range(1, len(vessel_df)):
            current_loc = vessel_df.iloc[i]['location_type']
            prev_loc = vessel_df.iloc[i-1]['location_type']
            
            if prev_loc != 'at_berth' and current_loc == 'at_berth':
                events[i] = 'berthing'
            elif prev_loc == 'at_berth' and current_loc != 'at_berth':
                events[i] = 'departure'
        
        return pd.Series(events)
    
    def _identify_queue_events(self, vessel_df: pd.DataFrame) -> pd.Series:
        """识别排队事件"""
        events = ['none'] * len(vessel_df)
        
        for i in range(1, len(vessel_df)):
            current_loc = vessel_df.iloc[i]['location_type']
            current_sog = vessel_df.iloc[i]['sog']
            
            # 在锚地且低速/静止视为排队
            if current_loc == 'at_anchor' and current_sog < 2:
                events[i] = 'queuing'
            elif current_loc == 'approaching' and current_sog < 3:
                events[i] = 'waiting'
        
        return pd.Series(events)
    
    def _identify_port_events(self, vessel_df: pd.DataFrame) -> pd.Series:
        """识别进港/出港事件"""
        events = ['none'] * len(vessel_df)
        
        for i in range(1, len(vessel_df)):
            current_loc = vessel_df.iloc[i]['location_type']
            prev_loc = vessel_df.iloc[i-1]['location_type']
            
            if prev_loc == 'at_sea' and current_loc in ['approaching', 'in_channel']:
                events[i] = 'entering'
            elif prev_loc in ['in_port', 'at_berth'] and current_loc == 'at_sea':
                events[i] = 'leaving'
        
        return pd.Series(events)
    
    def _calculate_maritime_metrics(self, df: pd.DataFrame) -> Dict:
        """计算海事业务指标"""
        metrics = {}
        
        # 1. 港口利用率
        port_records = df[df['location_type'].isin(['in_port', 'at_berth'])]
        total_time_hours = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 3600
        port_utilization = len(port_records) / len(df) if len(df) > 0 else 0
        
        # 2. 平均等待时间
        queue_records = df[df['queue_event'].isin(['queuing', 'waiting'])]
        avg_queue_time = self._calculate_average_queue_time(queue_records)
        
        # 3. 吞吐量
        berth_events = df[df['berth_event'] == 'berthing']
        throughput = len(berth_events) / max(1, total_time_hours / 24)  # 船/天
        
        # 4. 锚地利用率
        anchor_records = df[df['location_type'] == 'at_anchor']
        anchor_utilization = len(anchor_records) / len(df) if len(df) > 0 else 0
        
        # 5. 平均通行时间
        avg_transit_time = self._calculate_average_transit_time(df)
        
        metrics = {
            'port_utilization': port_utilization,
            'avg_queue_time_hours': avg_queue_time,
            'daily_throughput': throughput,
            'anchor_utilization': anchor_utilization,
            'avg_transit_time_hours': avg_transit_time,
            'total_vessels': df['mmsi'].nunique(),
            'total_records': len(df),
            'analysis_period_hours': total_time_hours
        }
        
        return metrics
    
    def _calculate_average_queue_time(self, queue_records: pd.DataFrame) -> float:
        """计算平均排队时间"""
        if len(queue_records) == 0:
            return 0.0
        
        queue_times = []
        for mmsi, vessel_df in queue_records.groupby('mmsi'):
            vessel_df = vessel_df.sort_values('timestamp')
            if len(vessel_df) > 1:
                queue_duration = (vessel_df['timestamp'].iloc[-1] - 
                                vessel_df['timestamp'].iloc[0]).total_seconds() / 3600
                queue_times.append(queue_duration)
        
        return np.mean(queue_times) if queue_times else 0.0
    
    def _calculate_average_transit_time(self, df: pd.DataFrame) -> float:
        """计算平均通行时间"""
        transit_times = []
        
        for mmsi, vessel_df in df.groupby('mmsi'):
            vessel_df = vessel_df.sort_values('timestamp')
            
            # 查找完整的进港-离港周期
            enter_events = vessel_df[vessel_df['port_event'] == 'entering']
            leave_events = vessel_df[vessel_df['port_event'] == 'leaving']
            
            for enter_idx in enter_events.index:
                # 查找对应的离港事件
                future_leaves = leave_events[leave_events.index > enter_idx]
                if len(future_leaves) > 0:
                    leave_idx = future_leaves.index[0]
                    transit_time = (vessel_df.loc[leave_idx, 'timestamp'] - 
                                  vessel_df.loc[enter_idx, 'timestamp']).total_seconds() / 3600
                    transit_times.append(transit_time)
        
        return np.mean(transit_times) if transit_times else 24.0
    
    def _generate_training_data(self, df: pd.DataFrame, metrics: Dict) -> List[Dict]:
        """生成训练数据"""
        training_samples = []
        
        for mmsi, vessel_df in df.groupby('mmsi'):
            vessel_df = vessel_df.sort_values('timestamp')
            
            for i in range(len(vessel_df)):
                row = vessel_df.iloc[i]
                
                # 构建训练样本
                sample = {
                    'timestamp': row['timestamp'].isoformat(),
                    'mmsi': mmsi,
                    'vessel_state': {
                        'lat': row['lat'],
                        'lon': row['lon'],
                        'sog': row['sog'],
                        'cog': row['cog'],
                        'heading': row['heading'],
                        'length': row['length'],
                        'width': row['width'],
                        'draught': row['draught'],
                        'vessel_type': row['vessel_type'],
                        'location_type': row['location_type']
                    },
                    'port_status': {
                        'hourly_throughput': metrics['daily_throughput'] / 24,
                        'port_utilization': metrics['port_utilization'],
                        'anchor_utilization': metrics['anchor_utilization'],
                        'avg_transit_time': metrics['avg_transit_time_hours']
                    },
                    'queue_status': {
                        'avg_wait_time': metrics['avg_queue_time_hours'],
                        'current_queue': self._estimate_current_queue(df, row['timestamp']),
                        'queue_event': row['queue_event']
                    },
                    'events': {
                        'berth_event': row['berth_event'],
                        'queue_event': row['queue_event'],
                        'port_event': row['port_event']
                    }
                }
                
                training_samples.append(sample)
        
        return training_samples
    
    def _estimate_current_queue(self, df: pd.DataFrame, timestamp: pd.Timestamp) -> int:
        """估计当前排队长度"""
        # 查找同一时间点附近的排队船舶
        time_window = timedelta(hours=1)
        nearby_records = df[
            (df['timestamp'] >= timestamp - time_window) &
            (df['timestamp'] <= timestamp + time_window) &
            (df['queue_event'].isin(['queuing', 'waiting']))
        ]
        
        return nearby_records['mmsi'].nunique()
    
    def _save_processed_data(self, df: pd.DataFrame, training_data: List[Dict], 
                           metrics: Dict, output_dir: str):
        """保存处理后的数据"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 保存处理后的数据
        processed_file = os.path.join(output_dir, f"{self.port_name}_processed_ais.csv")
        df.to_csv(processed_file, index=False)
        
        # 保存训练数据
        training_file = os.path.join(output_dir, f"{self.port_name}_training_data.json")
        import json
        with open(training_file, 'w') as f:
            json.dump(training_data, f, indent=2, default=str)
        
        # 保存指标
        metrics_file = os.path.join(output_dir, f"{self.port_name}_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        logger.info(f"数据已保存到: {output_dir}")

def test_improved_preprocessor():
    """测试改进的预处理器"""
    print("测试改进的AIS预处理器...")
    
    # 创建模拟数据
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        # 写入模拟AIS数据
        f.write("mmsi,lat,lon,sog,cog,timestamp,length,width,vessel_type\n")
        
        # 新奥尔良港周边的模拟数据
        base_lat, base_lon = 29.9511, -90.0715
        for i in range(100):
            mmsi = 1000 + i % 10
            lat = base_lat + np.random.normal(0, 0.05)
            lon = base_lon + np.random.normal(0, 0.05)
            sog = np.random.uniform(0, 15)
            cog = np.random.uniform(0, 360)
            timestamp = datetime.now() + timedelta(hours=i*0.1)
            length = np.random.uniform(80, 300)
            width = np.random.uniform(15, 50)
            vessel_type = np.random.randint(70, 90)
            
            f.write(f"{mmsi},{lat},{lon},{sog},{cog},{timestamp},{length},{width},{vessel_type}\n")
        
        test_file = f.name
    
    # 测试预处理器
    preprocessor = ImprovedAISPreprocessor("new_orleans")
    result = preprocessor.preprocess_ais_data(test_file, "./test_output")
    
    print(f"预处理结果: {result}")
    
    # 清理测试文件
    os.unlink(test_file)
    
    print("测试完成！")

if __name__ == "__main__":
    test_improved_preprocessor()