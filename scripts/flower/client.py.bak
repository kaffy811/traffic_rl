#!/usr/bin/env python3
"""
Flower联邦学习客户端
支持指定港口，与服务器通信，真正训练
"""

import flwr as fl
import torch
import logging
import argparse
import os
import sys
import numpy as np
from typing import Dict, List, Tuple, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.federated.fedavg_client import FedAvgClient

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FlowerClient(fl.client.NumPyClient):
    """Flower客户端实现"""
    
    def __init__(self, port: str, server_address: str = "localhost:8080", init_weights: str = None, 
                 episodes: int = 8, ppo_epochs: int = 4, batch_size: int = 64, entropy_coef: float = 0.01):
        self.port = port
        self.server_address = server_address
        self.init_weights = init_weights
        self.episodes = episodes
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.entropy_coef = entropy_coef
        self.client = FedAvgClient(port=port, init_weights=init_weights)
        
        # 获取模型参数键的顺序（固定顺序避免键错位）
        self.param_keys = list(self.client.get_parameters().keys())
        logger.info(f"🔑 参数键顺序: {self.param_keys}")
        logger.info(f"⚙️ 训练参数: episodes={episodes}, ppo_epochs={ppo_epochs}, batch_size={batch_size}, entropy_coef={entropy_coef}")
        
    def get_parameters(self, config: Dict[str, str]) -> List[np.ndarray]:
        """获取模型参数"""
        params = self.client.get_parameters()
        # 按固定顺序返回参数
        return [params[key] for key in self.param_keys]
    
    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """设置模型参数"""
        params_dict = {key: param for key, param in zip(self.param_keys, parameters)}
        self.client.set_parameters(params_dict)
    
    def fit(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[List[np.ndarray], int, Dict]:
        """训练模型"""
        # 设置参数
        self.set_parameters(parameters)
        
        # 本地训练 - 使用传入的训练参数
        logger.info(f"🏋️ 开始本地训练 - 港口: {self.port}")
        logger.info(f"📊 训练配置: episodes={self.episodes}, ppo_epochs={self.ppo_epochs}, batch_size={self.batch_size}")
        
        # 调用客户端的训练方法，传入训练参数
        train_stats = self.client.train(
            episodes=self.episodes,
            ppo_epochs=self.ppo_epochs,
            batch_size=self.batch_size,
            entropy_coef=self.entropy_coef
        )
        
        # 获取训练后的参数
        new_params = self.get_parameters(config)
        
        # 返回参数、样本数、指标（包含训练统计信息）
        metrics = {
            "port": self.port,
            "episodes": self.episodes,
            "ppo_epochs": self.ppo_epochs,
            "batch_size": self.batch_size,
            "entropy_coef": self.entropy_coef,
            "loss": train_stats.get("loss", 0.0),
            "reward": train_stats.get("avg_reward", 0.0),
            "num_samples": train_stats.get("num_samples", 800)
        }
        
        return new_params, metrics["num_samples"], metrics
    
    def evaluate(self, parameters: List[np.ndarray], config: Dict[str, str]) -> Tuple[float, int, Dict]:
        """评估模型"""
        # 设置参数
        self.set_parameters(parameters)
        
        # 本地评估
        logger.info(f"📊 开始本地评估 - 港口: {self.port}")
        eval_metrics = self.client.evaluate()
        
        # 返回损失、样本数、指标
        loss = eval_metrics.get("loss", 0.0)
        reward = eval_metrics.get("avg_reward", 0.0)
        num_samples = eval_metrics.get("num_samples", 800)
        
        metrics = {
            "port": self.port,
            "loss": loss,
            "reward": reward,
            "num_samples": num_samples
        }
        
        return loss, num_samples, metrics

def main():
    """启动Flower客户端"""
    parser = argparse.ArgumentParser(description="Flower联邦学习客户端")
    parser.add_argument("--port", type=str, required=True, help="港口名称")
    parser.add_argument("--server", type=str, default="localhost:8080", help="服务器地址")
    parser.add_argument("--init", type=str, help="初始权重文件路径")
    
    # 训练参数
    parser.add_argument("--episodes", type=int, default=8, help="训练episodes数")
    parser.add_argument("--ppo-epochs", type=int, default=4, help="PPO训练轮数")
    parser.add_argument("--batch-size", type=int, default=64, help="批次大小")
    parser.add_argument("--entropy-coef", type=float, default=0.01, help="熵系数")
    
    args = parser.parse_args()
    
    logger.info(f"🚀 启动Flower客户端 - 港口: {args.port}")
    logger.info(f"🌐 服务器地址: {args.server}")
    logger.info(f"⚙️ 训练参数: episodes={args.episodes}, ppo_epochs={args.ppo_epochs}, batch_size={args.batch_size}, entropy_coef={args.entropy_coef}")
    
    # 创建客户端
    client = FlowerClient(
        port=args.port, 
        server_address=args.server, 
        init_weights=args.init,
        episodes=args.episodes,
        ppo_epochs=args.ppo_epochs,
        batch_size=args.batch_size,
        entropy_coef=args.entropy_coef
    )
    
    # 启动客户端
    fl.client.start_numpy_client(
        server_address=args.server,
        client=client,
    )

if __name__ == "__main__":
    main() 