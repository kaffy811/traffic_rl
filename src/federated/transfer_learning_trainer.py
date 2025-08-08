#!/usr/bin/env python3
"""
转移学习训练器 - 使用Gulfport模型作为预训练基础
"""

import os
import sys
import torch
import numpy as np
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gat_ppo_agent import GATPPOAgent
from improved_ais_preprocessor import ImprovedAISPreprocessor
from advanced_data_processor import AdvancedDataProcessor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransferLearningTrainer:
    """转移学习训练器"""
    
    def __init__(self, source_port: str = "gulfport", target_port: str = None):
        self.source_port = source_port
        self.target_port = target_port
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 数据处理器
        self.preprocessor = ImprovedAISPreprocessor()
        self.data_processor = AdvancedDataProcessor()
        
        # 模型保存路径
        self.source_model_dir = Path(f"../../models/single_port/{source_port}")
        self.target_model_dir = Path(f"../../models/transfer_learning/{target_port}")
        self.target_model_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"初始化转移学习训练器 - 源港口: {source_port}, 目标港口: {target_port}")
    
    def load_source_model(self) -> GATPPOAgent:
        """加载源港口的最佳模型"""
        # 查找最佳模型文件
        best_models = list(self.source_model_dir.glob("best_model_*.pt"))
        if not best_models:
            raise FileNotFoundError(f"未找到源港口 {self.source_port} 的最佳模型")
        
        # 选择最新的最佳模型
        latest_model = max(best_models, key=lambda x: x.stat().st_mtime)
        logger.info(f"加载源模型: {latest_model}")
        
        # 创建智能体并加载模型
        agent = GATPPOAgent(
            state_dim=20,  # 基础状态维度
            action_dim=10,  # 动作空间
            hidden_dim=256,
            port_name=self.source_port,
            device=self.device
        )
        
        # 加载预训练权重
        checkpoint = torch.load(latest_model, map_location=self.device)
        agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"成功加载源模型，训练轮数: {checkpoint.get('episode', 'unknown')}")
        return agent
    
    def create_target_agent(self, source_agent: GATPPOAgent) -> GATPPOAgent:
        """创建目标港口智能体并初始化为源模型权重"""
        target_agent = GATPPOAgent(
            state_dim=20,
            action_dim=10,
            hidden_dim=256,
            port_name=self.target_port,
            device=self.device,
            learning_rate=1e-4  # 使用较小的学习率进行微调
        )
        
        # 复制源模型的权重
        target_agent.actor_critic.load_state_dict(
            source_agent.actor_critic.state_dict()
        )
        
        logger.info(f"目标智能体已初始化为源模型权重")
        return target_agent
    
    def load_target_data(self) -> Tuple[List, List]:
        """加载目标港口的训练数据"""
        logger.info(f"加载 {self.target_port} 港口数据...")
        
        # 加载预处理数据
        data_file = f"../../data/processed/{self.target_port}_processed.pkl"
        if not os.path.exists(data_file):
            raise FileNotFoundError(f"未找到目标港口数据文件: {data_file}")
        
        # 使用数据处理器加载和处理数据
        train_data, test_data = self.data_processor.load_port_data(self.target_port)
        
        logger.info(f"目标港口数据加载完成 - 训练: {len(train_data)}, 测试: {len(test_data)}")
        return train_data, test_data
    
    def fine_tune(self, num_episodes: int = 20, eval_episodes: int = 5) -> Dict:
        """执行转移学习微调"""
        logger.info(f"开始转移学习微调 - 目标港口: {self.target_port}")
        
        # 1. 加载源模型
        source_agent = self.load_source_model()
        
        # 2. 创建目标智能体
        target_agent = self.create_target_agent(source_agent)
        
        # 3. 加载目标数据
        train_data, test_data = self.load_target_data()
        
        # 4. 微调训练
        logger.info(f"开始微调训练 - 总轮数: {num_episodes}")
        
        best_reward = float('-inf')
        training_history = []
        
        for episode in range(1, num_episodes + 1):
            # 训练一轮
            episode_reward = 0
            episode_steps = 0
            completed_tasks = 0
            
            # 随机选择训练样本
            episode_data = np.random.choice(train_data, size=min(50, len(train_data)), replace=False)
            
            for data_point in episode_data:
                # 构建状态和图特征
                state = self._extract_state(data_point)
                node_features, adj_matrix = self._extract_graph_features(data_point)
                
                # 获取动作
                action, log_prob, value = target_agent.get_action(state, node_features, adj_matrix)
                
                # 计算奖励（简化版本）
                reward = self._calculate_reward(data_point, action)
                
                # 存储经验
                next_state = self._get_next_state(data_point, action)
                next_node_features, next_adj_matrix = self._extract_graph_features(data_point)
                
                target_agent.store_experience(
                    state, node_features, adj_matrix, action, 
                    reward, next_state, False, log_prob, value
                )
                
                episode_reward += reward
                episode_steps += 1
                
                if reward > 0:  # 简单的完成判断
                    completed_tasks += 1
            
            # 更新模型
            if len(target_agent.buffer.buffer) >= target_agent.batch_size:
                loss_info = target_agent.update()
            else:
                loss_info = {'total_loss': 0}
            
            # 计算完成率
            completion_rate = (completed_tasks / len(episode_data)) * 100 if episode_data.size > 0 else 0
            avg_reward = episode_reward / len(episode_data) if episode_data.size > 0 else 0
            
            # 记录训练历史
            training_history.append({
                'episode': episode,
                'reward': avg_reward,
                'completion_rate': completion_rate,
                'loss': loss_info['total_loss']
            })
            
            # 保存最佳模型
            if avg_reward > best_reward:
                best_reward = avg_reward
                model_path = self.target_model_dir / f"transfer_best_episode_{episode}.pt"
                torch.save({
                    'episode': episode,
                    'model_state_dict': target_agent.actor_critic.state_dict(),
                    'optimizer_state_dict': target_agent.optimizer.state_dict(),
                    'reward': avg_reward,
                    'source_port': self.source_port
                }, model_path)
                logger.info(f"转移学习模型已保存到: {model_path}")
            
            # 定期输出进度
            if episode % 5 == 0:
                logger.info(f"Episode {episode}/{num_episodes}")
                logger.info(f"  平均奖励: {avg_reward:.2f}")
                logger.info(f"  完成率: {completion_rate:.2f}%")
                logger.info(f"  损失: {loss_info['total_loss']:.4f}")
        
        logger.info(f"微调完成 - 最佳奖励: {best_reward:.2f}")
        
        # 5. 评估微调后的模型
        eval_results = self._evaluate_model(target_agent, test_data, eval_episodes)
        
        return {
            'training_history': training_history,
            'best_reward': best_reward,
            'eval_results': eval_results,
            'source_port': self.source_port,
            'target_port': self.target_port
        }
    
    def _extract_state(self, data_point) -> np.ndarray:
        """从数据点提取状态特征"""
        # 简化的状态提取
        return np.random.randn(20)  # 占位符实现
    
    def _extract_graph_features(self, data_point) -> Tuple[np.ndarray, np.ndarray]:
        """从数据点提取图特征"""
        # 简化的图特征提取
        num_nodes = 50
        node_features = np.random.randn(num_nodes, 10)
        adj_matrix = np.eye(num_nodes)  # 简化的邻接矩阵
        return node_features, adj_matrix
    
    def _calculate_reward(self, data_point, action) -> float:
        """计算奖励"""
        # 简化的奖励计算
        return np.random.randn()
    
    def _get_next_state(self, data_point, action) -> np.ndarray:
        """获取下一状态"""
        return self._extract_state(data_point)
    
    def _evaluate_model(self, agent: GATPPOAgent, test_data: List, num_episodes: int) -> Dict:
        """评估模型性能"""
        logger.info(f"开始评估转移学习模型 - 测试轮数: {num_episodes}")
        
        total_rewards = []
        completion_rates = []
        
        for episode in range(num_episodes):
            episode_reward = 0
            completed_tasks = 0
            
            # 随机选择测试样本
            test_samples = np.random.choice(test_data, size=min(30, len(test_data)), replace=False)
            
            for data_point in test_samples:
                state = self._extract_state(data_point)
                node_features, adj_matrix = self._extract_graph_features(data_point)
                
                # 使用贪婪策略评估
                with torch.no_grad():
                    action_probs, _ = agent.actor_critic(
                        torch.FloatTensor(state).unsqueeze(0),
                        torch.FloatTensor(node_features).unsqueeze(0),
                        torch.FloatTensor(adj_matrix)
                    )
                    action = torch.argmax(action_probs, dim=-1).item()
                
                reward = self._calculate_reward(data_point, action)
                episode_reward += reward
                
                if reward > 0:
                    completed_tasks += 1
            
            avg_reward = episode_reward / len(test_samples) if test_samples.size > 0 else 0
            completion_rate = (completed_tasks / len(test_samples)) * 100 if test_samples.size > 0 else 0
            
            total_rewards.append(avg_reward)
            completion_rates.append(completion_rate)
        
        results = {
            'avg_reward': np.mean(total_rewards),
            'std_reward': np.std(total_rewards),
            'avg_completion_rate': np.mean(completion_rates),
            'std_completion_rate': np.std(completion_rates)
        }
        
        logger.info("转移学习评估结果:")
        logger.info(f"  平均奖励: {results['avg_reward']:.2f} ± {results['std_reward']:.2f}")
        logger.info(f"  完成率: {results['avg_completion_rate']:.2f}% ± {results['std_completion_rate']:.2f}%")
        
        return results

def main():
    parser = argparse.ArgumentParser(description="转移学习训练器")
    parser.add_argument("--source", default="gulfport", help="源港口名称")
    parser.add_argument("--target", required=True, help="目标港口名称")
    parser.add_argument("--episodes", type=int, default=20, help="微调轮数")
    parser.add_argument("--eval_episodes", type=int, default=5, help="评估轮数")
    
    args = parser.parse_args()
    
    # 创建转移学习训练器
    trainer = TransferLearningTrainer(
        source_port=args.source,
        target_port=args.target
    )
    
    # 执行转移学习
    results = trainer.fine_tune(
        num_episodes=args.episodes,
        eval_episodes=args.eval_episodes
    )
    
    logger.info("转移学习完成！")
    logger.info(f"结果保存到: ../../models/transfer_learning/{args.target}")

if __name__ == "__main__":
    main()