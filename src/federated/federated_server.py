#!/usr/bin/env python3
"""
联邦学习服务器 - 集成α-fair权重分配
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional
import argparse
import logging
import json
import csv
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import copy

# 配置matplotlib（必须在pyplot导入前）
import matplotlib
matplotlib.use("Agg")  # 非交互后端
from matplotlib import rcParams
rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC', 'PingFang SC', 'Heiti SC',
    'Hiragino Sans GB', 'Source Han Sans SC', 'Arial Unicode MS'
]
rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt

@dataclass
class ClientMetrics:
    """客户端性能指标"""
    port_name: str
    completion_rate: float
    sample_count: int
    last_updated: int  # 轮次
    is_valid: bool = True

class AlphaFairAggregator:
    """α-fair权重聚合器"""
    
    def __init__(self, alpha: float = 1.0, temp: float = 1.5, 
                 floor: float = 0.03, uniform_mix: float = 0.15,
                 smoothing_beta: float = 0.2, sample_correction: bool = False,
                 logger: Optional[logging.Logger] = None):
        self.alpha = alpha
        self.temp = temp
        self.floor = floor
        self.uniform_mix = uniform_mix
        self.smoothing_beta = smoothing_beta  # 权重平滑强度
        self.sample_correction = sample_correction  # 样本量修正
        self.logger = logger or logging.getLogger(__name__)
        
        # 历史记录
        self.weight_history = []
        self.performance_history = []
        self.prev_weights = {}  # 上一轮权重，用于平滑
        
    def compute_alpha_fair_weights(self, client_metrics: Dict[str, ClientMetrics]) -> Dict[str, float]:
        """
        计算α-fair权重分配
        """
        if not client_metrics:
            return {}
            
        # 提取有效的性能数据
        valid_clients = {name: metrics for name, metrics in client_metrics.items() 
                        if metrics.is_valid and 0 <= metrics.completion_rate <= 1}
        
        if not valid_clients:
            self.logger.warning("没有有效的客户端性能数据，使用均匀权重")
            return {name: 1.0/len(client_metrics) for name in client_metrics.keys()}
        
        ports = list(valid_clients.keys())
        eps = 1e-6
        
        # 转换为"缺口"（弱者更大）
        perf_values = np.array([valid_clients[p].completion_rate for p in ports], dtype=float)
        perf_values = np.clip(perf_values, eps, 0.999999)
        
        # 用"缺口"刻画弱者（越弱越大）
        utilities = 1.0 - perf_values
        
        # 可选：样本量修正（防止小样本的噪声性能放大权重）
        if self.sample_correction:
            sample_counts = np.array([valid_clients[p].sample_count for p in ports])
            max_samples = np.max(sample_counts)
            
            # 样本量修正：小样本的缺口权重会被适当降低
            sample_weights = np.power(sample_counts / max_samples, 0.3)  # gamma=0.3
            utilities = utilities * sample_weights
            
            self.logger.debug(f"样本量修正: counts={dict(zip(ports, sample_counts))}, weights={dict(zip(ports, sample_weights))}")
        
        self.logger.debug(f"α={self.alpha}: 性能值 = {dict(zip(ports, perf_values))}")
        self.logger.debug(f"α={self.alpha}: 修正后缺口值 = {dict(zip(ports, utilities))}")
        
        # α 越大越偏弱者：raw ∝ utilities^alpha
        alpha = max(0.0, float(self.alpha))
        raw_weights = np.power(utilities + eps, alpha)
        raw_weights = np.nan_to_num(raw_weights, nan=eps, posinf=1e6, neginf=eps)
        
        # 归一化
        norm_weights = raw_weights / max(raw_weights.sum(), eps)
        
        # 温度调节（降低权重抖动）
        if self.temp != 1.0:
            norm_weights = np.power(norm_weights, 1.0 / self.temp)
            norm_weights = norm_weights / max(norm_weights.sum(), eps)
        
        # 权重地板（避免某些港口被完全忽略）
        norm_weights = np.maximum(norm_weights, self.floor)
        norm_weights = norm_weights / max(norm_weights.sum(), eps)
        
        # 与均匀分布混合（避免极端分配）
        if self.uniform_mix > 0:
            uniform_weights = np.ones_like(norm_weights) / len(norm_weights)
            norm_weights = (1 - self.uniform_mix) * norm_weights + self.uniform_mix * uniform_weights
            norm_weights = norm_weights / max(norm_weights.sum(), eps)
        
        weights = dict(zip(ports, norm_weights))
        
        # 为无效客户端分配最小权重
        for name in client_metrics:
            if name not in weights:
                weights[name] = self.floor / len(client_metrics)
        
        # 重新归一化
        total_weight = sum(weights.values())
        if total_weight > 0:
            weights = {name: w/total_weight for name, w in weights.items()}
        
        # 权重平滑（动量平滑，避免锯齿）
        if self.prev_weights and self.smoothing_beta > 0:
            smoothed_weights = {}
            for port in weights:
                prev_w = self.prev_weights.get(port, weights[port])
                smoothed_weights[port] = self.smoothing_beta * prev_w + (1 - self.smoothing_beta) * weights[port]
            
            # 重新归一化
            total_smoothed = sum(smoothed_weights.values())
            if total_smoothed > 0:
                smoothed_weights = {name: w/total_smoothed for name, w in smoothed_weights.items()}
            
            self.logger.debug(f"权重平滑: 原始={weights}, 平滑后={smoothed_weights}")
            weights = smoothed_weights
        
        # 更新历史权重
        self.prev_weights = weights.copy()
        
        self.logger.info(f"α={self.alpha}: 最终权重分配 = {weights}")
        
        # 记录历史
        self.weight_history.append(weights.copy())
        self.performance_history.append({name: metrics.completion_rate 
                                       for name, metrics in client_metrics.items()})
        
        return weights
    
    def aggregate_parameters(self, client_params: Dict[str, Dict], 
                           client_metrics: Dict[str, ClientMetrics]) -> Dict:
        """
        使用α-fair权重聚合客户端参数，支持FedProx
        """
        if not client_params:
            return {}
        
        # 计算权重
        weights = self.compute_alpha_fair_weights(client_metrics)
        
        # 选择聚合策略
        aggregation_strategy = self._select_aggregation_strategy(client_metrics)
        
        if aggregation_strategy == "fedprox":
            return self._fedprox_aggregate(client_params, weights, mu=0.01)
        else:
            return self._fedavg_aggregate(client_params, weights)
    
    def _select_aggregation_strategy(self, client_metrics: Dict[str, ClientMetrics]) -> str:
        """根据港口类型选择聚合策略"""
        # 针对窄弯港口使用FedProx抑制跨港漂移
        narrow_bend_ports = ['baton_rouge', 'new_orleans']
        
        for port_name in client_metrics.keys():
            if any(nb_port in port_name.lower() for nb_port in narrow_bend_ports):
                return "fedprox"
        
        return "fedavg"
    
    def _fedavg_aggregate(self, client_params: Dict[str, Dict], weights: Dict[str, float]) -> Dict:
        """标准FedAvg聚合"""
        aggregated = {}
        first_client = next(iter(client_params.values()))
        
        for param_name in first_client:
            weighted_sum = None
            total_weight = 0.0
            
            for client_name, params in client_params.items():
                if client_name in weights and param_name in params:
                    weight = weights[client_name]
                    param_tensor = params[param_name]
                    
                    if weighted_sum is None:
                        weighted_sum = weight * param_tensor.clone()
                    else:
                        weighted_sum += weight * param_tensor
                    
                    total_weight += weight
            
            if weighted_sum is not None and total_weight > 0:
                aggregated[param_name] = weighted_sum / total_weight
            else:
                aggregated[param_name] = first_client[param_name].clone()
                self.logger.warning(f"参数 {param_name} 聚合失败，使用回退值")
        
        return aggregated
    
    def _fedprox_aggregate(self, client_params: Dict[str, Dict], weights: Dict[str, float], mu: float = 0.01) -> Dict:
        """FedProx聚合，抑制跨港漂移"""
        self.logger.info(f"使用FedProx聚合 (μ={mu}) 抑制跨港漂移")
        
        aggregated = {}
        first_client = next(iter(client_params.values()))
        
        for param_name in first_client:
            weighted_sum = None
            total_weight = 0.0
            
            for client_name, params in client_params.items():
                if client_name in weights and param_name in params:
                    weight = weights[client_name]
                    param_tensor = params[param_name]
                    
                    # FedProx: 添加正则化项抑制漂移
                    if hasattr(self, 'global_model') and param_name in self.global_model:
                        global_param = self.global_model[param_name]
                        # 正则化项: -μ * (θ - θ_global)
                        prox_term = -mu * (param_tensor - global_param)
                        param_tensor = param_tensor + prox_term
                    
                    if weighted_sum is None:
                        weighted_sum = weight * param_tensor.clone()
                    else:
                        weighted_sum += weight * param_tensor
                    
                    total_weight += weight
            
            if weighted_sum is not None and total_weight > 0:
                aggregated[param_name] = weighted_sum / total_weight
            else:
                aggregated[param_name] = first_client[param_name].clone()
                self.logger.warning(f"参数 {param_name} 聚合失败，使用回退值")
        
        return aggregated
    
    def check_stability(self, round_num: int, check_interval: int = 10):
        """
        稳定性检查（每N轮检查一次）- 检查异常抖动和逻辑矛盾
        """
        if round_num % check_interval != 0 or len(self.weight_history) < 2:
            return
        
        self.logger.info(f"\n--- 第{round_num}轮稳定性检查 ---")
        
        # 检查最近两轮的变化
        if len(self.weight_history) >= 2 and len(self.performance_history) >= 2:
            prev_weights = self.weight_history[-2]
            curr_weights = self.weight_history[-1]
            prev_perfs = self.performance_history[-2]
            curr_perfs = self.performance_history[-1]
            
            # 稳定性阈值
            ABS_TOL = 0.02      # 绝对变化 >2% 才考虑异常
            REL_TOL = 0.10      # 或相对自身 >10%
            
            def big_drop(prev_w, curr_w):
                return (prev_w - curr_w) > max(ABS_TOL, prev_w * REL_TOL)
            
            def contradictory_change(prev_perf, curr_perf, prev_w, curr_w):
                got_weaker = (curr_perf < prev_perf - 1e-9)    # 更弱
                got_stronger = (curr_perf > prev_perf + 1e-9)  # 更强
                w_down = curr_w < prev_w - 1e-9
                w_up = curr_w > prev_w + 1e-9
                return (got_weaker and w_down) or (got_stronger and w_up)
            
            for port in curr_weights.keys():
                if port in prev_weights and port in prev_perfs and port in curr_perfs:
                    prev_w = prev_weights[port]
                    curr_w = curr_weights[port]
                    prev_perf = prev_perfs[port]
                    curr_perf = curr_perfs[port]
                    
                    # 检查大幅下降且逻辑矛盾
                    if big_drop(prev_w, curr_w) and contradictory_change(prev_perf, curr_perf, prev_w, curr_w):
                        self.logger.warning(f"⚠️ [{port}] 权重变化与性能趋势矛盾: w {prev_w:.3f}→{curr_w:.3f}, perf {prev_perf:.3f}→{curr_perf:.3f}")
                    
                    # 正常的性能-权重变化（仅INFO）
                    elif abs(curr_w - prev_w) > 0.01:
                        direction = "↑" if curr_w > prev_w else "↓"
                        perf_direction = "↑" if curr_perf > prev_perf else "↓"
                        self.logger.info(f"[{port}] 权重{direction} {prev_w:.3f}→{curr_w:.3f}, 性能{perf_direction} {prev_perf:.3f}→{curr_perf:.3f}")
        
        # 检查整体权重分布健康度
        if self.weight_history:
            curr_weights = self.weight_history[-1]
            weight_values = list(curr_weights.values())
            weight_range = max(weight_values) - min(weight_values)
            weight_ratio = max(weight_values) / max(min(weight_values), 1e-6)
            
            if weight_range > 0.4:
                self.logger.warning(f"⚠️ 权重分布过于不均: 范围={weight_range:.3f}")
            if weight_ratio > 20:
                self.logger.warning(f"⚠️ 权重比例过大: 最大/最小={weight_ratio:.1f}")
            
            self.logger.info(f"权重分布: 范围={weight_range:.3f}, 比例={weight_ratio:.1f}")
    
    def save_weight_log(self, output_dir: str, round_num: int):
        """保存权重日志到CSV"""
        if not self.weight_history:
            return
        
        output_path = Path(output_dir) / "federated_weights.csv"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 写入CSV
        with open(output_path, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # 表头
            if self.weight_history:
                ports = list(self.weight_history[0].keys())
                header = ['round'] + ports + ['alpha', 'temp', 'floor', 'uniform_mix']
                writer.writerow(header)
                
                # 数据行
                for i, weights in enumerate(self.weight_history):
                    row = [i] + [weights.get(port, 0) for port in ports]
                    row += [self.alpha, self.temp, self.floor, self.uniform_mix]
                    writer.writerow(row)
        
        self.logger.info(f"权重日志已保存: {output_path}")

class FederatedServer:
    """联邦学习服务器"""
    
    def __init__(self, args):
        self.args = args
        self.logger = logging.getLogger(__name__)
        
        # α-fair聚合器
        self.aggregator = AlphaFairAggregator(
            alpha=args.alpha,
            temp=args.temp,
            floor=args.floor,
            uniform_mix=args.uniform_mix,
            smoothing_beta=args.smoothing_beta,
            logger=self.logger
        )
        
        # 客户端状态
        self.client_metrics = {}
        self.global_model = None
        self.round_num = 0
        
    def update_client_metrics(self, client_name: str, completion_rate: float, 
                            sample_count: int):
        """更新客户端性能指标"""
        # 数据验证
        is_valid = True
        if not (0 <= completion_rate <= 1) or np.isnan(completion_rate):
            self.logger.warning(f"客户端 {client_name} 完成率异常: {completion_rate}")
            is_valid = False
        
        if sample_count <= 0:
            self.logger.warning(f"客户端 {client_name} 样本数异常: {sample_count}")
            is_valid = False
        
        self.client_metrics[client_name] = ClientMetrics(
            port_name=client_name,
            completion_rate=completion_rate,
            sample_count=sample_count,
            last_updated=self.round_num,
            is_valid=is_valid
        )
        
        if not is_valid:
            self.logger.warning(f"客户端 {client_name} 数据异常，将使用回退权重")
    
    def federated_round(self, client_params: Dict[str, Dict]) -> Dict:
        """执行一轮联邦学习"""
        self.round_num += 1
        
        self.logger.info(f"\n=== 联邦学习第 {self.round_num} 轮 ===")
        
        # 打印当前客户端状态
        self.logger.info("客户端性能状态:")
        for name, metrics in self.client_metrics.items():
            status = "✓" if metrics.is_valid else "✗"
            self.logger.info(f"  {name}: {metrics.completion_rate:.3f} ({metrics.sample_count} 样本) {status}")
        
        # 聚合参数
        aggregated_params = self.aggregator.aggregate_parameters(client_params, self.client_metrics)
        
        # 稳定性检查
        self.aggregator.check_stability(self.round_num, check_interval=10)
        
        # 保存权重日志
        if self.round_num % 5 == 0:  # 每5轮保存一次
            self.aggregator.save_weight_log(self.args.output_dir, self.round_num)
        
        return aggregated_params

def parse_federated_args():
    """解析联邦学习参数"""
    parser = argparse.ArgumentParser(description="联邦学习服务器")
    
    # α-fair参数
    parser.add_argument("--alpha", type=float, default=1.2,
                       help="α-fair参数 (越大越偏向弱者)")
    parser.add_argument("--temp", type=float, default=1.5,
                       help="温度参数 (>1拉平，<1拉尖)")
    parser.add_argument("--floor", type=float, default=0.03,
                       help="最小权重地板")
    parser.add_argument("--uniform-mix", type=float, default=0.15,
                       help="与均匀分布的混合比例")
    parser.add_argument("--smoothing-beta", type=float, default=0.2,
                       help="权重平滑强度 (0=无平滑, 0.3=强平滑)")
    
    # 训练参数
    parser.add_argument("--rounds", type=int, default=100,
                       help="联邦学习轮数")
    parser.add_argument("--output-dir", type=str, default="../../results/federated",
                       help="输出目录")
    
    # 日志参数
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    return parser.parse_args()

def setup_federated_logging(log_level: str = "INFO"):
    """设置联邦学习日志"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('federated_server.log')
        ]
    )

def demo_federated_training():
    """演示联邦学习训练流程"""
    args = parse_federated_args()
    setup_federated_logging(args.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("启动联邦学习服务器演示")
    
    # 创建服务器
    server = FederatedServer(args)
    
    # 模拟客户端参数（简化版）
    def create_dummy_params():
        return {
            'layer1.weight': torch.randn(10, 5),
            'layer1.bias': torch.randn(10),
            'layer2.weight': torch.randn(1, 10),
            'layer2.bias': torch.randn(1)
        }
    
    # 模拟训练过程
    for round_num in range(1, args.rounds + 1):
        # 模拟客户端性能更新（基于港口特征）
        if round_num == 1:
            # 初始性能
            server.update_client_metrics("gulfport", 0.95, 1000)
            server.update_client_metrics("baton_rouge", 0.33, 800)
            server.update_client_metrics("new_orleans", 0.14, 600)
            server.update_client_metrics("south_louisiana", 0.44, 900)
        else:
            # 模拟性能变化
            for port in ["gulfport", "baton_rouge", "new_orleans", "south_louisiana"]:
                current = server.client_metrics[port].completion_rate
                # 添加小幅随机变化
                change = np.random.normal(0, 0.01)
                new_rate = np.clip(current + change, 0.05, 0.99)
                server.update_client_metrics(port, new_rate, 
                                           server.client_metrics[port].sample_count)
        
        # 模拟客户端参数
        client_params = {
            "gulfport": create_dummy_params(),
            "baton_rouge": create_dummy_params(),
            "new_orleans": create_dummy_params(),
            "south_louisiana": create_dummy_params()
        }
        
        # 执行联邦聚合
        aggregated = server.federated_round(client_params)
        
        # 每10轮打印详细信息
        if round_num % 10 == 0:
            logger.info(f"第 {round_num} 轮完成，聚合了 {len(aggregated)} 个参数")
    
    # 最终保存
    server.aggregator.save_weight_log(args.output_dir, server.round_num)
    logger.info("联邦学习演示完成")

if __name__ == "__main__":
    demo_federated_training()