#!/usr/bin/env python3
"""
Flower联邦学习服务器
- FedAvg + 可选公平加权（invsize）
- 每轮保存到 save_dir/global_round_XXX.pt 与 save_dir/global_best.pt
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import List, Tuple

import flwr as fl
import numpy as np
import torch
from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters
from flwr.server.strategy import FedAvg

# 将项目根目录加入 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# 与客户端保持一致的参数顺序
PARAM_KEYS = [
    "layer1.weight","layer1.bias",
    "layer2.weight","layer2.bias",
    "output.weight","output.bias",
]

def weighted_avg(metrics):
    """聚合指标（可选）"""
    total, s_loss, s_reward = 0, 0.0, 0.0
    for n, m in metrics:
        total += n
        if "loss" in m: s_loss += n * m["loss"]
        if "reward" in m: s_reward += n * m["reward"]
    out = {}
    if total and s_loss: out["loss"] = s_loss/total
    if total and s_reward: out["reward"] = s_reward/total
    return out

class SaveFairFedAvg(FedAvg):
    """在FedAvg基础上：支持公平加权并落盘"""
    def __init__(self, save_dir="models/flw/flower_run", mode="fedavg", alpha=0.5, **kwargs):
        super().__init__(**kwargs)
        self.save_dir = Path(save_dir); self.save_dir.mkdir(parents=True, exist_ok=True)
        self.mode = mode
        self.alpha = float(alpha)
        self._eps = 1e-12

    def _fair_weights(self, n_list: List[float]) -> np.ndarray:
        n = np.asarray(n_list, dtype=np.float64)
        if self.mode == "invsize":
            w = (1.0 / (n + self._eps)) ** self.alpha
        elif self.mode == "fedavg":
            w = n.copy()
        else:  # 等权
            w = np.ones_like(n)
        w = w / (w.sum() + self._eps)
        return w

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple["fl.server.client_proxy.ClientProxy", "fl.common.FitRes"]],
        failures: List[BaseException],
    ):
        if not results:
            return None, {}

        # 拿到每个客户端的参数与样本量
        params_list, n_list = [], []
        for _, fit_res in results:
            params_list.append(parameters_to_ndarrays(fit_res.parameters))
            n_list.append(float(getattr(fit_res, "num_examples", 1)))

        # 计算权重并加权聚合
        w = self._fair_weights(n_list)
        agg = [np.zeros_like(t) for t in params_list[0]]
        for wi, plist in zip(w, params_list):
            for j, arr in enumerate(plist):
                agg[j] = agg[j] + wi * arr

        aggregated = ndarrays_to_parameters(agg)

        # 落盘
        arrays = agg
        sd = {k: torch.tensor(a) for k, a in zip(PARAM_KEYS, arrays)}
        out = self.save_dir / f"global_round_{server_round:03d}.pt"
        torch.save({"model_state_dict": sd}, out)
        torch.save({"model_state_dict": sd}, self.save_dir / "global_best.pt")
        logging.info(f"💾 保存第{server_round}轮全局模型到: {out}")

        return aggregated, {"fair_mode": self.mode, "alpha": self.alpha}

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Flower联邦学习服务器')
    parser.add_argument('--rounds', type=int, default=30, help='训练轮数 (默认: 30)')
    parser.add_argument('--min-clients', type=int, default=4, help='最小客户端数量 (默认: 4)')
    parser.add_argument('--save-dir', type=str, default='models/flw/flower_run', help='模型保存目录')
    parser.add_argument('--fair-agg', type=str, default='fedavg', choices=['fedavg','invsize'],
                        help="聚合加权方式：fedavg（默认）或 invsize（反数据量加权）")
    parser.add_argument('--alpha', type=float, default=0.5, help="invsize的指数，越大越偏向少数域 (默认0.5)")
    args = parser.parse_args()

    num_rounds = args.rounds
    min_available_clients = args.min_clients
    min_fit_clients = args.min_clients

    logger.info(f"🚀 启动Flower服务器 - 轮数: {num_rounds}")
    logger.info(f"📊 最小可用客户端: {min_available_clients}")
    logger.info(f"🎯 最小训练客户端: {min_fit_clients}")
    logger.info(f"🤝 公平聚合: {args.fair_agg}, α={args.alpha}")
    logger.info(f"💾 模型保存目录: {args.save_dir}")

    strategy = SaveFairFedAvg(
        save_dir=args.save_dir,
        mode=args.fair_agg,
        alpha=args.alpha,
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_available_clients=min_available_clients,
        min_fit_clients=min_fit_clients,
        min_evaluate_clients=min_available_clients,
        on_fit_config_fn=lambda _: {"epochs": 1},
        on_evaluate_config_fn=lambda _: {"epochs": 1},
        fit_metrics_aggregation_fn=weighted_avg,
        evaluate_metrics_aggregation_fn=weighted_avg,
    )

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
    )

if __name__ == "__main__":
    main()
