#!/usr/bin/env python3
"""
Flower联邦学习服务器
支持FedAvg策略，可配置轮数，等待4个客户端，每轮落盘
"""

import flwr as fl
import torch
import logging
import argparse
from typing import List, Tuple
import os
import sys
from pathlib import Path
from flwr.common import parameters_to_ndarrays
from flwr.server.strategy import FedAvg

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 按你的模型键顺序来（和 client 打印的一致）
PARAM_KEYS = [
    "layer1.weight","layer1.bias",
    "layer2.weight","layer2.bias",
    "output.weight","output.bias",
]

class SaveFedAvg(FedAvg):
    def __init__(self, save_dir="models/flw/flower_run", **kwargs):
        super().__init__(**kwargs)
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def aggregate_fit(self, server_round, results, failures):
        aggregated, metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated is not None:
            arrays = parameters_to_ndarrays(aggregated)
            sd = {k: torch.tensor(a) for k, a in zip(PARAM_KEYS, arrays)}
            out = self.save_dir / f"global_round_{server_round:03d}.pt"
            torch.save({"model_state_dict": sd}, out)
            torch.save({"model_state_dict": sd}, self.save_dir / "global_best.pt")
            logging.info(f"💾 保存第{server_round}轮全局模型到: {out}")
        return aggregated, metrics

def weighted_avg(metrics):
    """加权平均聚合指标"""
    total, s_loss, s_reward = 0, 0.0, 0.0
    for n, m in metrics:
        total += n
        if "loss" in m: s_loss += n * m["loss"]
        if "reward" in m: s_reward += n * m["reward"]
    out = {}
    if total and s_loss: out["loss"] = s_loss/total
    if total and s_reward: out["reward"] = s_reward/total
    return out

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """启动Flower服务器"""
    
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Flower联邦学习服务器')
    parser.add_argument('--rounds', type=int, default=30, help='训练轮数 (默认: 30)')
    parser.add_argument('--min-clients', type=int, default=4, help='最小客户端数量 (默认: 4)')
    parser.add_argument('--save-dir', type=str, default='models/flw/flower_run', help='模型保存目录')
    args = parser.parse_args()
    
    # 配置参数
    num_rounds = args.rounds
    min_available_clients = args.min_clients
    min_fit_clients = args.min_clients
    
    logger.info(f"🚀 启动Flower服务器 - 轮数: {num_rounds}")
    logger.info(f"📊 最小可用客户端: {min_available_clients}")
    logger.info(f"🎯 最小训练客户端: {min_fit_clients}")
    logger.info(f"💾 模型保存目录: {args.save_dir}")
    
    # 创建带保存的FedAvg策略 - 等待4个客户端到齐再开始
    strategy = SaveFedAvg(
        save_dir=args.save_dir,
        fraction_fit=1.0,  # 每轮都训练所有可用客户端
        fraction_evaluate=1.0,  # 每轮都评估所有可用客户端
        min_available_clients=min_available_clients,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_available_clients,
        on_fit_config_fn=lambda _: {"epochs": 1},
        on_evaluate_config_fn=lambda _: {"epochs": 1},
        fit_metrics_aggregation_fn=weighted_avg,
        evaluate_metrics_aggregation_fn=weighted_avg,
    )
    
    # 启动服务器
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main() 