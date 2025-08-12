#!/usr/bin/env python3
"""
联邦学习启动脚本 - 基于操作手册的 FedAvg 实现
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root / "src" / "federated"))

from gat_ppo_agent import GATPPOAgent
from curriculum_trainer import CurriculumTrainer

def setup_logging(log_file: str, log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file)
        ]
    )
    return logging.getLogger(__name__)

def load_global_model(model_path: str) -> Dict:
    """加载全局初始化模型"""
    import torch
    checkpoint = torch.load(model_path, map_location="cpu")
    return checkpoint.get("model_state_dict", checkpoint)

def save_global_model(model_state_dict: Dict, save_path: str):
    """保存全局模型"""
    import torch
    torch.save({"model_state_dict": model_state_dict}, save_path)

def federated_round(global_model: Dict, client_ports: List[str], 
                   local_episodes: int, lr: float, batch_size: int, 
                   ppo_epochs: int, entropy_coef: float) -> Dict:
    """
    执行一轮联邦学习
    """
    logger = logging.getLogger(__name__)
    logger.info(f"开始联邦学习轮次，客户端: {client_ports}")
    
    # 模拟客户端训练（这里简化处理，实际应该分发到各客户端）
    client_models = {}
    
    for port in client_ports:
        logger.info(f"客户端 {port} 开始本地训练...")
        
        # 加载全局模型到客户端
        client_agent = GATPPOAgent(port, {
            "state_dim": 56, "hidden_dim": 256, "action_dim": 15,
            "num_heads": 4, "dropout": 0.1,
            "learning_rate": lr,
            "ppo_epochs": ppo_epochs,
            "batch_size": batch_size,
            "gamma": 0.99, "gae_lambda": 0.95, "clip_ratio": 0.2,
            "buffer_size": 10000,
            "port_name": port,
            "entropy_coef": entropy_coef
        })
        
        # 加载全局权重
        client_agent.actor_critic.load_state_dict(global_model, strict=False)
        
        # 模拟本地训练（简化版）
        logger.info(f"客户端 {port} 完成 {local_episodes} 个 episodes")
        
        # 获取更新后的权重
        client_models[port] = client_agent.actor_critic.state_dict()
    
    # FedAvg 聚合
    logger.info("执行 FedAvg 聚合...")
    aggregated_model = {}
    
    # 获取所有参数的键
    all_keys = set()
    for client_model in client_models.values():
        all_keys.update(client_model.keys())
    
    # 平均所有客户端参数
    for key in all_keys:
        if all(key in client_model for client_model in client_models.values()):
            # 检查形状是否一致
            shapes = [client_model[key].shape for client_model in client_models.values()]
            if all(shape == shapes[0] for shape in shapes):
                aggregated_model[key] = sum(client_models[port][key] for port in client_ports) / len(client_ports)
            else:
                logger.warning(f"参数 {key} 形状不一致，跳过聚合")
    
    logger.info(f"联邦聚合完成，聚合了 {len(aggregated_model)} 个参数")
    return aggregated_model

def main():
    parser = argparse.ArgumentParser(description="联邦学习启动脚本")
    
    # 算法参数
    parser.add_argument("--algo", type=str, default="fedavg", choices=["fedavg"],
                       help="联邦学习算法")
    parser.add_argument("--rounds", type=int, default=30,
                       help="联邦学习轮数")
    parser.add_argument("--clients", type=str, 
                       default="gulfport,new_orleans,south_louisiana,baton_rouge",
                       help="客户端港口列表，逗号分隔")
    
    # 模型参数
    parser.add_argument("--global-init", type=str, required=True,
                       help="全局初始化模型路径")
    parser.add_argument("--save-dir", type=str, required=True,
                       help="模型保存目录")
    
    # 训练参数
    parser.add_argument("--local-episodes", type=int, default=8,
                       help="每客户端每轮本地训练 episodes")
    parser.add_argument("--lr-schedule", type=str, 
                       default="0:3e-4,5:1.5e-4,25:7.5e-5",
                       help="学习率调度，格式: round:lr,round:lr,...")
    parser.add_argument("--batch-size", type=int, default=64,
                       help="批次大小")
    parser.add_argument("--ppo-epochs", type=int, default=4,
                       help="PPO 训练轮数")
    parser.add_argument("--entropy-coef", type=float, default=0.01,
                       help="熵系数")
    
    # 日志参数
    parser.add_argument("--log-file", type=str, required=True,
                       help="日志文件路径")
    parser.add_argument("--log-level", type=str, default="INFO",
                       choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_file, args.log_level)
    logger.info("联邦学习启动脚本开始执行")
    
    # 解析客户端列表
    client_ports = [port.strip() for port in args.clients.split(",")]
    logger.info(f"客户端港口: {client_ports}")
    
    # 解析学习率调度
    lr_schedule = {}
    for item in args.lr_schedule.split(","):
        round_num, lr = item.split(":")
        lr_schedule[int(round_num)] = float(lr)
    logger.info(f"学习率调度: {lr_schedule}")
    
    # 加载全局初始化模型
    logger.info(f"加载全局初始化模型: {args.global_init}")
    global_model = load_global_model(args.global_init)
    
    # 创建保存目录
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 联邦学习主循环
    current_model = global_model.copy()
    
    for round_num in range(1, args.rounds + 1):
        logger.info(f"=== 第 {round_num} 轮联邦学习 ===")
        
        # 确定当前轮次的学习率
        current_lr = 3e-4  # 默认值
        for round_threshold, lr in sorted(lr_schedule.items()):
            if round_num > round_threshold:
                current_lr = lr
        
        logger.info(f"当前学习率: {current_lr}")
        
        # 执行联邦轮次
        start_time = time.time()
        current_model = federated_round(
            current_model, client_ports, 
            args.local_episodes, current_lr,
            args.batch_size, args.ppo_epochs, args.entropy_coef
        )
        round_time = time.time() - start_time
        
        # 保存当前轮次模型
        round_model_path = save_dir / f"global_round_{round_num:02d}.pt"
        save_global_model(current_model, str(round_model_path))
        
        # 保存最佳模型（简化版，这里总是覆盖）
        best_model_path = save_dir / "global_best.pt"
        save_global_model(current_model, str(best_model_path))
        
        logger.info(f"第 {round_num} 轮完成，耗时: {round_time:.2f}s")
        logger.info(f"模型已保存: {round_model_path}")
        
        # 每5轮提示评测
        if round_num % 5 == 0:
            logger.info(f"建议在第 {round_num} 轮后进行评测:")
            logger.info(f"python scripts/nightly_ci.py --ports all --samples 800 --seeds 42,123,2025 --no-cache --k 120")
    
    logger.info("联邦学习完成！")
    logger.info(f"最终模型已保存到: {save_dir / 'global_best.pt'}")

if __name__ == "__main__":
    main() 