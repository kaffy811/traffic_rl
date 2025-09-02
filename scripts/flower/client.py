#!/usr/bin/env python3
"""
Flower联邦学习客户端
- 指定港口，与服务器通信
- 本地训练（占位/真实训练由 FedAvgClient 决定）
- 用 episodes 作为 num_examples 回传，让不均衡生效
"""
import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple

import flwr as fl
import numpy as np

# 将项目根目录加入 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from src.federated.fedavg_client import FedAvgClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, port: str, server_address: str = "localhost:8080", init_weights: str = None,
                 episodes: int = 8, ppo_epochs: int = 4, batch_size: int = 64, entropy_coef: float = 0.01):
        self.port = port
        self.server_address = server_address
        self.init_weights = init_weights
        self.episodes = int(episodes)
        self.ppo_epochs = int(ppo_epochs)
        self.batch_size = int(batch_size)
        self.entropy_coef = float(entropy_coef)
        self.client = FedAvgClient(port=port, init_weights=init_weights)
        self.param_keys = list(self.client.get_parameters().keys())
        logger.info(f"🔑 参数键顺序: {self.param_keys}")
        logger.info(f"⚙️ 训练参数: episodes={self.episodes}, ppo_epochs={self.ppo_epochs}, batch_size={self.batch_size}, entropy_coef={self.entropy_coef}")

    # Flower 回调
    def get_parameters(self, config: Dict[str, str] = None) -> List[np.ndarray]:
        params = self.client.get_parameters()
        return [params[k] for k in self.param_keys]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        pdict = {k: v for k, v in zip(self.param_keys, parameters)}
        self.client.set_parameters(pdict)

    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        self.set_parameters(parameters)
        logger.info(f"🏋️ 开始本地训练 - 港口: {self.port}")
        train_stats = self.client.train(
            episodes=self.episodes,
            ppo_epochs=self.ppo_epochs,
            batch_size=self.batch_size,
            entropy_coef=self.entropy_coef
        )
        new_params = self.get_parameters(config)
        num_examples = int(self.episodes)  # 用 episodes 作为样本权重
        metrics = {
            "port": self.port,
            "episodes": self.episodes,
            "ppo_epochs": self.ppo_epochs,
            "batch_size": self.batch_size,
            "entropy_coef": self.entropy_coef,
            "loss": float(train_stats.get("loss", 0.0)),
            "reward": float(train_stats.get("avg_reward", 0.0)),
            "num_samples": num_examples,
        }
        return new_params, num_examples, metrics

    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        self.set_parameters(parameters)
        logger.info(f"📊 开始本地评估 - 港口: {self.port}")
        ev = self.client.evaluate()
        loss = float(ev.get("loss", 0.0))
        reward = float(ev.get("avg_reward", 0.0))
        num_samples = int(ev.get("num_samples", 800))
        metrics = {"port": self.port, "loss": loss, "reward": reward, "num_samples": num_samples}
        return loss, num_samples, metrics

def main():
    parser = argparse.ArgumentParser(description="Flower联邦学习客户端")
    parser.add_argument("--port", type=str, required=True, help="港口名称")
    parser.add_argument("--server", type=str, default="localhost:8080", help="服务器地址 <IP>:<PORT>")
    parser.add_argument("--init", type=str, help="初始权重")
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--ppo-epochs", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    args = parser.parse_args()

    logger.info(f"🚀 启动Flower客户端 - 港口: {args.port}")
    logger.info(f"🌐 服务器地址: {args.server}")
    logger.info(f"⚙️ 训练参数: episodes={args.episodes}, ppo_epochs={args.ppo_epochs}, batch_size={args.batch_size}, entropy_coef={args.entropy_coef}")

    client = FlowerClient(
        port=args.port,
        server_address=args.server,
        init_weights=args.init,
        episodes=args.episodes,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef
    )

    fl.client.start_numpy_client(
        server_address=args.server,
        client=client,
    )

if __name__ == "__main__":
    main()
