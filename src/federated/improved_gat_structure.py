"""
改进的GAT图结构设计
基于港口区域和设施的真实连接关系
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional

class MaritimeGraphBuilder:
    """构建海事港口的图结构"""
    
    def __init__(self, port_name: str):
        self.port_name = port_name
        self.node_types = {
            'berth': 0,          # 泊位节点
            'anchorage': 1,      # 锚地节点  
            'channel': 2,        # 航道节点
            'terminal': 3,       # 码头节点
            'vessel': 4          # 船舶节点
        }
        
        # 根据港口设置节点数量
        self.num_nodes = self._get_port_node_count(port_name)
        self.adjacency_matrix = self._build_adjacency_matrix()
        
    def _get_port_node_count(self, port_name: str) -> int:
        """根据港口设置节点总数"""
        port_configs = {
            'new_orleans': {
                'berths': 40, 'anchorages': 6, 'channels': 8, 
                'terminals': 12, 'max_vessels': 20
            },
            'south_louisiana': {
                'berths': 25, 'anchorages': 4, 'channels': 6,
                'terminals': 8, 'max_vessels': 15  
            },
            'baton_rouge': {
                'berths': 64, 'anchorages': 8, 'channels': 10,
                'terminals': 16, 'max_vessels': 25
            },
            'gulfport': {
                'berths': 10, 'anchorages': 3, 'channels': 4,
                'terminals': 6, 'max_vessels': 10
            }
        }
        
        config = port_configs.get(port_name, port_configs['new_orleans'])
        self.node_config = config
        
        return sum(config.values())
    
    def _build_adjacency_matrix(self) -> torch.Tensor:
        """构建邻接矩阵"""
        adj_matrix = torch.zeros(self.num_nodes, self.num_nodes)
        
        # 节点起始索引
        berth_start = 0
        berth_end = self.node_config['berths']
        
        anchor_start = berth_end
        anchor_end = anchor_start + self.node_config['anchorages']
        
        channel_start = anchor_end
        channel_end = channel_start + self.node_config['channels']
        
        terminal_start = channel_end
        terminal_end = terminal_start + self.node_config['terminals']
        
        vessel_start = terminal_end
        vessel_end = vessel_start + self.node_config['max_vessels']
        
        # 1. 泊位-码头连接 (每个泊位属于某个码头)
        for berth_idx in range(berth_start, berth_end):
            terminal_idx = terminal_start + (berth_idx % self.node_config['terminals'])
            adj_matrix[berth_idx, terminal_idx] = 1
            adj_matrix[terminal_idx, berth_idx] = 1
        
        # 2. 锚地-航道连接 (锚地通过航道到达泊位)
        for anchor_idx in range(anchor_start, anchor_end):
            for channel_idx in range(channel_start, channel_end):
                adj_matrix[anchor_idx, channel_idx] = 1
                adj_matrix[channel_idx, anchor_idx] = 1
        
        # 3. 航道-泊位连接 (航道连接到泊位)
        for channel_idx in range(channel_start, channel_end):
            # 每个航道连接多个泊位
            connected_berths = np.random.choice(
                range(berth_start, berth_end), 
                size=min(5, self.node_config['berths']), 
                replace=False
            )
            for berth_idx in connected_berths:
                adj_matrix[channel_idx, berth_idx] = 1
                adj_matrix[berth_idx, channel_idx] = 1
        
        # 4. 船舶-锚地/泊位动态连接 (在运行时更新)
        # 这里只设置基础连接，具体连接在使用时动态调整
        
        # 5. 添加自环，确保每个节点至少连接到自己
        adj_matrix += torch.eye(self.num_nodes)
        
        return adj_matrix
    
    def get_node_features(self, vessel_states: List[Dict], 
                         port_status: Dict) -> torch.Tensor:
        """获取所有节点的特征矩阵"""
        node_features = []
        
        # 1. 泊位节点特征
        berth_features = self._get_berth_features(port_status)
        node_features.extend(berth_features)
        
        # 2. 锚地节点特征  
        anchor_features = self._get_anchor_features(port_status)
        node_features.extend(anchor_features)
        
        # 3. 航道节点特征
        channel_features = self._get_channel_features(port_status)
        node_features.extend(channel_features)
        
        # 4. 码头节点特征
        terminal_features = self._get_terminal_features(port_status)
        node_features.extend(terminal_features)
        
        # 5. 船舶节点特征
        vessel_features = self._get_vessel_features(vessel_states)
        node_features.extend(vessel_features)
        
        return torch.tensor(node_features, dtype=torch.float32)
    
    def _get_berth_features(self, port_status: Dict) -> List[List[float]]:
        """泊位节点特征 (8维)"""
        features = []
        berth_data = port_status.get('berths', {})
        
        for i in range(self.node_config['berths']):
            berth_info = berth_data.get(f'berth_{i}', {})
            feature = [
                self.node_types['berth'] / 4.0,           # 节点类型
                berth_info.get('occupied', 0),            # 占用状态
                berth_info.get('length', 200) / 800.0,    # 泊位长度归一化
                berth_info.get('depth', 10) / 50.0,       # 水深归一化
                berth_info.get('utilization', 0),         # 利用率
                berth_info.get('maintenance', 0),         # 维护状态
                berth_info.get('cargo_type', 0) / 10.0,   # 货物类型
                berth_info.get('efficiency', 0.8)         # 作业效率
            ]
            features.append(feature)
        
        return features
    
    def _get_anchor_features(self, port_status: Dict) -> List[List[float]]:
        """锚地节点特征 (6维)"""
        features = []
        anchor_data = port_status.get('anchorages', {})
        
        for i in range(self.node_config['anchorages']):
            anchor_info = anchor_data.get(f'anchor_{i}', {})
            feature = [
                self.node_types['anchorage'] / 4.0,       # 节点类型
                anchor_info.get('occupancy', 0) / 10.0,   # 占用船舶数
                anchor_info.get('capacity', 10) / 20.0,   # 容量归一化
                anchor_info.get('avg_wait_time', 0) / 24.0, # 平均等待时间
                anchor_info.get('weather_impact', 0),     # 天气影响
                anchor_info.get('distance_to_berth', 5) / 20.0  # 到泊位距离
            ]
            features.append(feature)
        
        return features
    
    def _get_channel_features(self, port_status: Dict) -> List[List[float]]:
        """航道节点特征 (7维)"""
        features = []
        channel_data = port_status.get('channels', {})
        
        for i in range(self.node_config['channels']):
            channel_info = channel_data.get(f'channel_{i}', {})
            feature = [
                self.node_types['channel'] / 4.0,          # 节点类型
                channel_info.get('depth', 15) / 50.0,      # 航道深度
                channel_info.get('width', 100) / 300.0,    # 航道宽度
                channel_info.get('traffic_density', 0),    # 交通密度
                channel_info.get('tide_effect', 0),        # 潮汐影响
                channel_info.get('current_speed', 0) / 5.0, # 水流速度
                channel_info.get('restriction', 0)         # 通行限制
            ]
            features.append(feature)
        
        return features
    
    def _get_terminal_features(self, port_status: Dict) -> List[List[float]]:
        """码头节点特征 (9维)"""
        features = []
        terminal_data = port_status.get('terminals', {})
        
        for i in range(self.node_config['terminals']):
            terminal_info = terminal_data.get(f'terminal_{i}', {})
            feature = [
                self.node_types['terminal'] / 4.0,         # 节点类型
                terminal_info.get('berth_count', 3) / 10.0, # 泊位数量
                terminal_info.get('utilization', 0.6),     # 利用率
                terminal_info.get('throughput', 0) / 100.0, # 吞吐量
                terminal_info.get('cargo_type', 1) / 5.0,  # 主要货物类型
                terminal_info.get('equipment_status', 0.9), # 设备状态
                terminal_info.get('storage_capacity', 50) / 200.0, # 存储容量
                terminal_info.get('truck_queue', 0) / 50.0, # 卡车排队
                terminal_info.get('rail_connection', 0)     # 铁路连接
            ]
            features.append(feature)
        
        return features
    
    def _get_vessel_features(self, vessel_states: List[Dict]) -> List[List[float]]:
        """船舶节点特征 (12维)"""
        features = []
        
        # 填充到最大船舶数
        actual_vessels = len(vessel_states)
        max_vessels = self.node_config['max_vessels']
        
        for i in range(max_vessels):
            if i < actual_vessels:
                vessel = vessel_states[i]
                feature = [
                    self.node_types['vessel'] / 4.0,        # 节点类型
                    vessel.get('length', 150) / 400.0,      # 船长
                    vessel.get('width', 25) / 60.0,         # 船宽
                    vessel.get('draught', 8) / 20.0,        # 吃水
                    vessel.get('sog', 0) / 25.0,            # 航速
                    vessel.get('cog', 0) / 360.0,           # 航向
                    vessel.get('vessel_type', 70) / 100.0,  # 船舶类型
                    vessel.get('cargo_amount', 0) / 100.0,  # 货物量
                    vessel.get('priority', 0) / 10.0,       # 优先级
                    vessel.get('eta_hours', 24) / 72.0,     # 预计到达时间
                    vessel.get('service_time', 8) / 48.0,   # 预计服务时间
                    1.0                                     # 激活状态
                ]
            else:
                # 空船舶节点
                feature = [self.node_types['vessel'] / 4.0] + [0.0] * 11
            
            features.append(feature)
        
        return features
    
    def update_vessel_connections(self, vessel_states: List[Dict], 
                                port_status: Dict) -> torch.Tensor:
        """动态更新船舶节点的连接"""
        adj_matrix = self.adjacency_matrix.clone()
        
        vessel_start = (self.node_config['berths'] + 
                       self.node_config['anchorages'] + 
                       self.node_config['channels'] + 
                       self.node_config['terminals'])
        
        berth_start = 0
        berth_end = self.node_config['berths']
        
        anchor_start = berth_end
        anchor_end = anchor_start + self.node_config['anchorages']
        
        for i, vessel in enumerate(vessel_states[:self.node_config['max_vessels']]):
            vessel_idx = vessel_start + i
            
            # 根据船舶状态连接到相应节点
            vessel_status = vessel.get('status', 'at_sea')
            
            if vessel_status == 'at_berth':
                # 连接到分配的泊位
                berth_id = vessel.get('berth_id', 0)
                if berth_id < self.node_config['berths']:
                    adj_matrix[vessel_idx, berth_start + berth_id] = 1
                    adj_matrix[berth_start + berth_id, vessel_idx] = 1
                    
            elif vessel_status == 'at_anchor':
                # 连接到锚地
                anchor_id = vessel.get('anchor_id', 0)
                if anchor_id < self.node_config['anchorages']:
                    adj_matrix[vessel_idx, anchor_start + anchor_id] = 1
                    adj_matrix[anchor_start + anchor_id, vessel_idx] = 1
            
            else:  # 'in_transit', 'approaching'
                # 连接到最近的锚地或航道
                # 简化：连接到第一个锚地
                if self.node_config['anchorages'] > 0:
                    adj_matrix[vessel_idx, anchor_start] = 1
                    adj_matrix[anchor_start, vessel_idx] = 1
        
        return adj_matrix

class ImprovedGATLayer(nn.Module):
    """改进的GAT层，处理海事图结构"""
    
    def __init__(self, in_features: int, out_features: int, 
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        
        # 线性映射
        self.W = nn.Linear(in_features, num_heads * out_features, bias=False)
        # 注意力参数：每个头有一个长度 2*out_features 的向量
        self.a = nn.Parameter(torch.randn(num_heads, 2 * out_features))
        
        self.leakyrelu    = nn.LeakyReLU(0.2)
        self.dropout      = nn.Dropout(dropout)
        self.layer_norm   = nn.LayerNorm(out_features)
    
    def forward(self, x: torch.Tensor, adj_matrix: torch.Tensor) -> torch.Tensor:
        """
        x: [batch_size, num_nodes, in_features]
        adj_matrix: [num_nodes, num_nodes]
        """
        B, N, _ = x.shape
        
        # 1. 线性映射并 reshape 为多头
        h = self.W(x)  # [B, N, H*out]
        h = h.view(B, N, self.num_heads, self.out_features)  # [B, N, H, out]
        h = h.permute(0, 2, 1, 3)                            # [B, H, N, out]
        
        # 2. 构造节点对特征
        h_i = h.unsqueeze(3).expand(-1, -1, -1, N, -1)       # [B,H,N,N,out]
        h_j = h.unsqueeze(2).expand(-1, -1, N, -1, -1)       # [B,H,N,N,out]
        h_cat = torch.cat([h_i, h_j], dim=-1)                # [B,H,N,N,2*out]
        
        # 3. 计算注意力打分
        a = self.a.unsqueeze(0).unsqueeze(2).unsqueeze(2)    # [1,H,1,1,2*out]
        e = (h_cat * a).sum(dim=-1)                          # [B,H,N,N]
        e = self.leakyrelu(e)
        
        # 4. 掩码并 softmax
        # 规范化邻接矩阵维度
        if adj_matrix.dim() == 2:  # [N, N]
            mask = adj_matrix.unsqueeze(0).unsqueeze(0).expand(B, self.num_heads, N, N)
        elif adj_matrix.dim() == 3:  # [B, N, N]
            mask = adj_matrix.unsqueeze(1).expand(B, self.num_heads, N, N)
        elif adj_matrix.dim() == 4 and adj_matrix.size(1) == 1:  # [B, 1, N, N]
            mask = adj_matrix.squeeze(1).unsqueeze(1).expand(B, self.num_heads, N, N)
        else:
            raise ValueError(f"Unsupported adj_matrix shape: {adj_matrix.shape}")
        e = e.masked_fill(mask == 0, float('-inf'))
        alpha = torch.softmax(e, dim=-1)
        alpha = self.dropout(alpha)
        
        # 5. 用注意力加权聚合
        h_out = torch.matmul(alpha, h)                      # [B,H,N,out]
        h_out = h_out.permute(0, 2, 1, 3).contiguous()      # [B,N,H,out]
        h_out = h_out.view(B, N, -1)                        # [B,N,H*out]
        
        # 6. 多头平均
        h_out = h_out.view(B, N, self.num_heads, self.out_features).mean(dim=2)
        
        # 7. 残差 + 层归一化
        if x.shape[-1] == self.out_features:
            h_out = self.layer_norm(h_out + x)
        else:
            h_out = self.layer_norm(h_out)
        
        return h_out

def test_graph_structure():
    """测试图结构构建"""
    print("测试GAT图结构构建...")
    
    # 测试新奥尔良港
    graph_builder = MaritimeGraphBuilder("new_orleans")
    print(f"节点总数: {graph_builder.num_nodes}")
    print(f"节点配置: {graph_builder.node_config}")
    print(f"邻接矩阵形状: {graph_builder.adjacency_matrix.shape}")
    print(f"邻接矩阵连接数: {graph_builder.adjacency_matrix.sum().item()}")
    
    # 测试节点特征
    mock_vessel_states = [
        {'length': 200, 'width': 30, 'draught': 10, 'sog': 12, 'cog': 45, 
         'vessel_type': 80, 'status': 'approaching'},
        {'length': 150, 'width': 25, 'draught': 8, 'sog': 8, 'cog': 90, 
         'vessel_type': 70, 'status': 'at_anchor'}
    ]
    
    mock_port_status = {
        'berths': {f'berth_{i}': {'occupied': i % 2, 'length': 200 + i*50, 
                                 'utilization': 0.3 + i*0.1} for i in range(5)},
        'anchorages': {f'anchor_{i}': {'occupancy': i+1, 'capacity': 15, 
                                      'avg_wait_time': i*2} for i in range(3)},
        'channels': {f'channel_{i}': {'depth': 20 + i*5, 'width': 150 + i*20, 
                                     'traffic_density': 0.1 + i*0.1} for i in range(4)},
        'terminals': {f'terminal_{i}': {'berth_count': 3 + i, 'utilization': 0.6 + i*0.05, 
                                       'throughput': 20 + i*10} for i in range(6)}
    }
    
    node_features = graph_builder.get_node_features(mock_vessel_states, mock_port_status)
    print(f"节点特征矩阵形状: {node_features.shape}")
    print(f"特征统计: min={node_features.min():.3f}, max={node_features.max():.3f}")
    
    # 测试动态连接更新
    updated_adj = graph_builder.update_vessel_connections(mock_vessel_states, mock_port_status)
    print(f"更新后连接数: {updated_adj.sum().item()}")
    
    print("GAT图结构测试完成！")

if __name__ == "__main__":
    test_graph_structure()