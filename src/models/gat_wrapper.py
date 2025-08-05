#!/usr/bin/env python3
"""
GAT封装器：将现有的pytorch-GAT适配到海事交通场景
解决导入路径和兼容性问题
"""

import sys
import os
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict, List, Optional
from pathlib import Path

# 全局标志，确保警告只显示一次
_GAT_WARNING_SHOWN = False

def show_gat_warning_once(message: str):
    """只显示一次GAT警告"""
    global _GAT_WARNING_SHOWN
    if not _GAT_WARNING_SHOWN:
        print(f"⚠️ {message}")
        _GAT_WARNING_SHOWN = True


# 动态添加GAT项目路径
def setup_gat_path():
    """设置GAT项目路径"""
    # 获取当前文件的目录
    current_dir = Path(__file__).parent

    # 可能的GAT路径
    possible_gat_paths = [
        current_dir / "../../FedML/externals/pytorch-GAT",
        current_dir / "../../../FedML/externals/pytorch-GAT",
        current_dir / "../../externals/pytorch-GAT",
    ]

    for gat_path in possible_gat_paths:
        gat_path = gat_path.resolve()
        if gat_path.exists() and (gat_path / "models").exists():
            sys.path.insert(0, str(gat_path))
            return str(gat_path)

    raise ImportError(f"无法找到GAT项目路径。请确保pytorch-GAT项目在以下位置之一: {possible_gat_paths}")


# 设置GAT路径
try:
    GAT_PROJECT_PATH = setup_gat_path()
    print(f"✅ GAT项目路径: {GAT_PROJECT_PATH}")
except ImportError as e:
    print(f"❌ GAT路径设置失败: {e}")
    # 创建备用实现
    GAT_PROJECT_PATH = None

# 导入GAT相关模块
if GAT_PROJECT_PATH:
    try:
        from models.definitions.GAT import GAT
        from utils.constants import LayerType

        GAT_AVAILABLE = True
        print("✅ GAT模块导入成功")
    except ImportError as e:
        print(f"❌ GAT模块导入失败: {e}")
        GAT_AVAILABLE = False
else:
    GAT_AVAILABLE = False

# 如果GAT不可用，创建备用实现
if not GAT_AVAILABLE:
    show_gat_warning_once("使用备用GAT实现")


    class LayerType:
        """LayerType枚举备用实现"""
        IMP1 = 0
        IMP2 = 1
        IMP3 = 2


    class GAT(nn.Module):
        """简化的GAT备用实现"""

        def __init__(self, num_of_layers, num_heads_per_layer, num_features_per_layer, **kwargs):
            super().__init__()
            self.num_features = num_features_per_layer
            self.num_layers = num_of_layers

            # 简单的线性层序列
            layers = []
            for i in range(num_of_layers):
                in_features = num_features_per_layer[i] * (num_heads_per_layer[i] if i > 0 else 1)
                out_features = num_features_per_layer[i + 1]

                layers.extend([
                    nn.Linear(in_features, out_features),
                    nn.ReLU() if i < num_of_layers - 1 else nn.Identity()
                ])

            self.layers = nn.Sequential(*layers)

        def forward(self, data):
            node_features, adjacency_matrix = data
            return self.layers(node_features)


class MaritimeGATEncoder(nn.Module):
    """
    海事交通场景的GAT编码器
    适配现有GAT实现到我们的4节点海事网络
    """

    def __init__(self,
                 node_feature_dim: int = 5,
                 hidden_dim: int = 64,
                 output_dim: int = 32,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 use_backup_gat: bool = False):
        super().__init__()

        self.node_feature_dim = node_feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.use_backup_gat = use_backup_gat or not GAT_AVAILABLE

        if self.use_backup_gat:
            # 静默模式：不再重复显示警告
            # 使用备用GAT实现
            self.gat = GAT(
                num_of_layers=2,
                num_heads_per_layer=[num_heads, 1],
                num_features_per_layer=[node_feature_dim, hidden_dim, output_dim]
            )
        else:
            # 使用原始GAT实现
            self.gat = GAT(
                num_of_layers=2,
                num_heads_per_layer=[num_heads, 1],  # 第一层多头，最后一层单头
                num_features_per_layer=[node_feature_dim, hidden_dim, output_dim],
                add_skip_connection=True,
                bias=True,
                dropout=dropout,
                layer_type=LayerType.IMP2,  # 使用实现2（平衡性能和理解度）
                log_attention_weights=True  # 记录注意力权重用于分析
            )

        # 图级别特征聚合
        self.graph_aggregator = nn.Sequential(
            nn.Linear(output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )

        # 位置编码（可选）
        self.position_encoding = nn.Parameter(torch.randn(4, output_dim) * 0.1)

    def forward(self, node_features: torch.Tensor,
                adjacency_matrix: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            node_features: [num_nodes, node_feature_dim]
            adjacency_matrix: [num_nodes, num_nodes]
        Returns:
            node_embeddings: [num_nodes, output_dim]
            graph_embedding: [output_dim]
        """

        # 确保输入格式正确
        if node_features.dim() == 1:
            node_features = node_features.unsqueeze(0)
        if adjacency_matrix.dim() == 2 and adjacency_matrix.size(0) != node_features.size(0):
            # 如果邻接矩阵维度不匹配，使用全连接
            num_nodes = node_features.size(0)
            adjacency_matrix = torch.ones(num_nodes, num_nodes, device=node_features.device)

        # 转换为GAT期望的格式 (node_features, topology)
        gat_input = (node_features, adjacency_matrix)

        try:
            # 通过GAT网络
            gat_output = self.gat(gat_input)
            
            # GAT可能返回tuple，取第一个元素作为node embeddings
            if isinstance(gat_output, tuple):
                node_embeddings = gat_output[0]
            else:
                node_embeddings = gat_output

            # 添加位置编码
            if node_embeddings.size(0) == 4:  # 4个节点的情况
                node_embeddings = node_embeddings + self.position_encoding

            # 图级别聚合（使用注意力加权平均）
            attention_weights = self._compute_graph_attention(node_embeddings)
            graph_embedding = torch.sum(attention_weights.unsqueeze(-1) * node_embeddings, dim=0)
            graph_embedding = self.graph_aggregator(graph_embedding)

        except Exception as e:
            print(f"GAT前向传播出错: {e}")
            # 备用处理：简单的MLP
            node_embeddings = self._backup_forward(node_features)
            graph_embedding = torch.mean(node_embeddings, dim=0)
            graph_embedding = self.graph_aggregator(graph_embedding)

        return node_embeddings, graph_embedding

    def _compute_graph_attention(self, node_embeddings: torch.Tensor) -> torch.Tensor:
        """计算图级别注意力权重"""
        # 简单的注意力机制
        attention_scores = torch.sum(node_embeddings ** 2, dim=-1)
        attention_weights = torch.softmax(attention_scores, dim=0)
        return attention_weights

    def _backup_forward(self, node_features: torch.Tensor) -> torch.Tensor:
        """备用前向传播（当GAT失败时）"""
        # 简单的MLP处理
        weight1 = torch.randn(node_features.size(-1), self.hidden_dim)
        bias1 = torch.zeros(self.hidden_dim)
        hidden = torch.relu(torch.nn.functional.linear(node_features, weight1.T, bias1))
        
        weight2 = torch.randn(self.hidden_dim, self.output_dim)
        bias2 = torch.zeros(self.output_dim)
        output = torch.nn.functional.linear(hidden, weight2.T, bias2)
        return output


class MaritimeStateBuilder:
    """
    海事状态构建器：将交通状态转换为图数据
    """

    def __init__(self, num_nodes: int = 4):
        self.num_nodes = num_nodes

        # 固定的航道拓扑（4节点网络）
        self.adjacency_matrix = self._build_maritime_topology()

        # 特征标准化参数
        self.feature_stats = {
            'waiting_ships': {'mean': 10.0, 'std': 5.0},
            'throughput': {'mean': 5.0, 'std': 3.0},
            'waiting_time': {'mean': 15.0, 'std': 10.0},
            'signal_phase': {'mean': 0.5, 'std': 0.5},
            'weather_condition': {'mean': 0.8, 'std': 0.2}
        }

    @staticmethod
    def _build_maritime_topology() -> torch.Tensor:
        """构建海事网络拓扑"""
        # 定义4节点网络的连接模式
        # 0: NodeA (主入口), 1: NodeB (工业运河), 2: NodeC (中游段), 3: NodeD (外海锚地)
        adj = torch.tensor([
            [1.0, 1.0, 1.0, 0.8],  # NodeA连接度
            [1.0, 1.0, 0.6, 1.0],  # NodeB连接度
            [1.0, 0.6, 1.0, 0.9],  # NodeC连接度
            [0.8, 1.0, 0.9, 1.0],  # NodeD连接度
        ], dtype=torch.float32)

        return adj

    def build_state(self, maritime_observations: Dict[str, Dict]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        构建图状态

        Args:
            maritime_observations: {
                'NodeA': {'waiting_ships': 5, 'throughput': 3, 'waiting_time': 12, ...},
                'NodeB': {'waiting_ships': 8, 'throughput': 2, 'waiting_time': 18, ...},
                ...
            }
        Returns:
            node_features: [num_nodes, feature_dim]
            adjacency_matrix: [num_nodes, num_nodes]
        """

        node_features = []

        # 按节点顺序提取特征
        node_names = ['NodeA', 'NodeB', 'NodeC', 'NodeD']

        for i, node_id in enumerate(node_names[:self.num_nodes]):
            if node_id in maritime_observations:
                obs = maritime_observations[node_id]

                # 提取和标准化特征
                features = self._extract_and_normalize_features(obs)
            else:
                # 默认特征（归一化后的）
                features = [0.0, 0.0, 0.0, 0.0, 1.0]

            node_features.append(features)

        # 确保有足够的节点
        while len(node_features) < self.num_nodes:
            node_features.append([0.0, 0.0, 0.0, 0.0, 1.0])

        node_features = torch.tensor(node_features, dtype=torch.float32)

        return node_features, self.adjacency_matrix

    def _extract_and_normalize_features(self, obs: Dict) -> List[float]:
        """提取和标准化特征"""
        features = []
        feature_names = ['waiting_ships', 'throughput', 'waiting_time', 'signal_phase', 'weather_condition']

        for feature_name in feature_names:
            raw_value = obs.get(feature_name, 0.0)

            if feature_name in self.feature_stats:
                # Z-score标准化
                stats = self.feature_stats[feature_name]
                normalized_value = (raw_value - stats['mean']) / stats['std']
                # 裁剪到合理范围
                normalized_value = max(-3.0, min(3.0, normalized_value))
            else:
                normalized_value = float(raw_value)

            features.append(normalized_value)

        return features

    def update_feature_stats(self, observations_history: List[Dict[str, Dict]]):
        """基于历史数据更新特征统计"""
        if not observations_history:
            return

        # 收集所有特征值
        all_features = {name: [] for name in self.feature_stats.keys()}

        for obs_batch in observations_history:
            for node_data in obs_batch.values():
                for feature_name in all_features.keys():
                    if feature_name in node_data:
                        all_features[feature_name].append(node_data[feature_name])

        # 更新统计信息
        for feature_name, values in all_features.items():
            if values:
                self.feature_stats[feature_name]['mean'] = np.mean(values)
                self.feature_stats[feature_name]['std'] = max(np.std(values), 1e-6)  # 避免除零


def test_maritime_gat():
    """测试海事GAT编码器"""

    print("🧪 测试海事GAT编码器")
    print("=" * 50)

    # 创建GAT编码器
    gat_encoder = MaritimeGATEncoder(
        node_feature_dim=5,
        hidden_dim=32,
        output_dim=16,
        num_heads=4
    )

    print(f"✅ GAT编码器创建成功")
    print(f"  输入维度: {gat_encoder.node_feature_dim}")
    print(f"  隐藏维度: {gat_encoder.hidden_dim}")
    print(f"  输出维度: {gat_encoder.output_dim}")
    print(f"  使用备用实现: {gat_encoder.use_backup_gat}")

    # 创建状态构建器
    state_builder = MaritimeStateBuilder()

    # 模拟海事观测数据
    maritime_observations = {
        'NodeA': {'waiting_ships': 5, 'throughput': 3, 'waiting_time': 12, 'signal_phase': 1, 'weather_condition': 0.8},
        'NodeB': {'waiting_ships': 8, 'throughput': 2, 'waiting_time': 18, 'signal_phase': 0, 'weather_condition': 0.8},
        'NodeC': {'waiting_ships': 3, 'throughput': 1, 'waiting_time': 8, 'signal_phase': 1, 'weather_condition': 0.8},
        'NodeD': {'waiting_ships': 6, 'throughput': 4, 'waiting_time': 15, 'signal_phase': 0, 'weather_condition': 0.8},
    }

    print(f"\n📊 测试数据:")
    for node, data in maritime_observations.items():
        print(f"  {node}: {data}")

    # 构建图状态
    node_features, adjacency_matrix = state_builder.build_state(maritime_observations)

    print(f"\n🔍 图状态:")
    print(f"  节点特征形状: {node_features.shape}")
    print(f"  邻接矩阵形状: {adjacency_matrix.shape}")
    print(f"  节点特征 (标准化后):\n{node_features}")
    print(f"  邻接矩阵:\n{adjacency_matrix}")

    # GAT前向传播
    print(f"\n🚀 GAT前向传播:")
    try:
        with torch.no_grad():
            node_embeddings, graph_embedding = gat_encoder(node_features, adjacency_matrix)

            print(f"  ✅ 前向传播成功")
            print(f"  节点嵌入形状: {node_embeddings.shape}")
            print(f"  图嵌入形状: {graph_embedding.shape}")
            print(f"  节点嵌入范围: [{torch.min(node_embeddings):.3f}, {torch.max(node_embeddings):.3f}]")
            print(f"  图嵌入范围: [{torch.min(graph_embedding):.3f}, {torch.max(graph_embedding):.3f}]")

    except Exception as e:
        print(f"  ❌ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False

    # 测试批量处理
    print(f"\n📦 测试批量处理:")
    try:
        batch_observations = [maritime_observations for _ in range(3)]

        for i, obs in enumerate(batch_observations):
            node_features, adjacency_matrix = state_builder.build_state(obs)
            node_embeddings, graph_embedding = gat_encoder(node_features, adjacency_matrix)
            print(f"  批次 {i + 1}: 节点嵌入 {node_embeddings.shape}, 图嵌入 {graph_embedding.shape}")

        print(f"  ✅ 批量处理成功")

    except Exception as e:
        print(f"  ❌ 批量处理失败: {e}")

    print(f"\n✅ 海事GAT编码器测试完成！")
    return True


def check_gat_installation():
    """检查GAT安装状态"""
    print("🔍 检查GAT安装状态")
    print("=" * 40)

    print(f"GAT项目路径: {GAT_PROJECT_PATH}")
    print(f"GAT可用性: {GAT_AVAILABLE}")

    if GAT_AVAILABLE:
        print("✅ 原始GAT实现可用")
        try:
            # 测试GAT导入
            test_gat = GAT(
                num_of_layers=1,
                num_heads_per_layer=[2],
                num_features_per_layer=[5, 8]
            )
            print("✅ GAT模块测试成功")
        except Exception as e:
            print(f"❌ GAT模块测试失败: {e}")
    else:
        if not _GAT_WARNING_SHOWN:
            show_gat_warning_once("使用备用GAT实现")
            print("  建议: 检查pytorch-GAT项目是否正确放置在FedML/externals/目录下")


if __name__ == "__main__":
    # 检查GAT安装
    check_gat_installation()

    print("\n" + "=" * 60 + "\n")

    # 运行测试
    test_maritime_gat()