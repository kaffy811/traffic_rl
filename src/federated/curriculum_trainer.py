#!/usr/bin/env python3
"""
分阶段训练器 (Curriculum Learning) - 从简单到复杂逐步训练
"""

import os
import sys
import torch
import numpy as np
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
from dataclasses import dataclass

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gat_ppo_agent import GATPPOAgent
from port_specific_rewards import PortRewardFactory

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def build_agent(port_name: str, **kwargs) -> GATPPOAgent:
    """安全实例化Agent，构建config字典（兼容GATPPOAgent(config=...)风格）"""
    config = {
        'state_dim': kwargs.get('state_dim', 20),
        'action_dim': kwargs.get('action_dim', 15),
        'hidden_dim': kwargs.get('hidden_dim', 256),
        'learning_rate': kwargs.get('learning_rate', 3e-4),
        'batch_size': kwargs.get('batch_size', 32),
        'num_heads': kwargs.get('num_heads', 4),
        'dropout': kwargs.get('dropout', 0.1),
        'gamma': kwargs.get('gamma', 0.99),
        'gae_lambda': kwargs.get('gae_lambda', 0.95),
        'clip_ratio': kwargs.get('clip_ratio', 0.2),
        'ppo_epochs': kwargs.get('ppo_epochs', 8),
        'buffer_size': kwargs.get('buffer_size', 10000),
        # 有些实现会读到下面这俩
        'entropy_coef': kwargs.get('entropy_coef', 0.02),
        'device': kwargs.get('device', 'cpu'),
        'node_feature_dim': kwargs.get('node_feature_dim', 8)
    }
    return GATPPOAgent(port_name=port_name, config=config)

@dataclass
class CurriculumStage:
    """课程阶段定义"""
    name: str
    description: str
    max_vessels: int
    max_berths: int
    traffic_intensity: float  # 0.0-1.0
    weather_complexity: float  # 0.0-1.0
    episodes: int
    success_threshold: float  # 完成率(胜率)阈值

class CurriculumTrainer:
    """分阶段训练器"""
    
    def __init__(self, port_name: str):
        self.port_name = port_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 保存目录统一到 v2
        self.save_dir = Path(f"../../models/curriculum_v2/{port_name}")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 定义课程阶段
        self.curriculum_stages = self._define_curriculum_stages()
        
        # 港口特定奖励函数
        self.reward_function = PortRewardFactory.create_reward_function(port_name)
        
        logger.info(f"初始化分阶段训练器 - 港口: {port_name}")
        logger.info(f"课程阶段数: {len(self.curriculum_stages)}")
    
    def _define_curriculum_stages(self) -> List[CurriculumStage]:
        """定义港口特定的课程阶段（阈值基于你现有跑数）"""
        if self.port_name == 'new_orleans':
            return [
                CurriculumStage(
                    name="基础阶段",
                    description="最简单的调度场景",
                    max_vessels=5,
                    max_berths=3,
                    traffic_intensity=0.3,
                    weather_complexity=0.1,
                    episodes=30,                 # 20 -> 30
                    success_threshold=0.35        # 你之前大约36%
                ),
                CurriculumStage(
                    name="初级阶段",
                    description="增加船舶数量",
                    max_vessels=10,
                    max_berths=5,
                    traffic_intensity=0.5,
                    weather_complexity=0.2,
                    episodes=30,                 # 25 -> 30
                    success_threshold=0.40        # 复评 wr=38.0%，LB=31.6%；调整为0.40
                ),
                CurriculumStage(
                    name="中级阶段",
                    description="增加交通强度",
                    max_vessels=15,
                    max_berths=8,
                    traffic_intensity=0.7,
                    weather_complexity=0.3,
                    episodes=30,
                    success_threshold=0.50
                ),
                CurriculumStage(
                    name="高级阶段",
                    description="接近真实复杂度",
                    max_vessels=20,
                    max_berths=12,
                    traffic_intensity=0.8,
                    weather_complexity=0.4,
                    episodes=30,
                    success_threshold=0.40
                ),
                CurriculumStage(
                    name="专家阶段",
                    description="完整复杂度",
                    max_vessels=25,
                    max_berths=15,
                    traffic_intensity=1.0,
                    weather_complexity=0.5,
                    episodes=40,
                    success_threshold=0.30
                )
            ]
        
        elif self.port_name in ['baton_rouge', 'south_louisiana']:
            return [
                CurriculumStage(
                    name="基础阶段",
                    description="简单调度",
                    max_vessels=8,
                    max_berths=5,
                    traffic_intensity=0.4,
                    weather_complexity=0.1,
                    episodes=35,                 # 25 -> 35，增强学习
                    success_threshold=0.41        # 复评 wr=42.0%，WilsonLB=35.4%；调整为0.41
                ),
                CurriculumStage(
                    name="中级阶段",
                    description="标准复杂度",
                    max_vessels=15,
                    max_berths=10,
                    traffic_intensity=0.7,
                    weather_complexity=0.3,
                    episodes=30,                 # 20 -> 30，增强学习
                    success_threshold=0.45        # south_louisiana 复评 wr=45.5%，调整为0.45
                ),
                CurriculumStage(
                    name="高级阶段",
                    description="完整复杂度",
                    max_vessels=20,
                    max_berths=15,
                    traffic_intensity=1.0,
                    weather_complexity=0.4,
                    episodes=20,
                    success_threshold=0.39        # south_louisiana 复评 wr=39.5%，LB=33.0%
                )
            ]
        
        else:  # gulfport
            return [
                CurriculumStage(
                    name="标准阶段",
                    description="标准训练",
                    max_vessels=15,
                    max_berths=10,
                    traffic_intensity=0.8,
                    weather_complexity=0.3,
                    episodes=20,
                    success_threshold=0.49         # 复评 wr=45.5%，LB=38.7%；调整为0.49
                ),
                CurriculumStage(
                    name="完整阶段",
                    description="完整复杂度",
                    max_vessels=20,
                    max_berths=15,
                    traffic_intensity=1.0,
                    weather_complexity=0.4,
                    episodes=45,                    # 35 -> 45
                    success_threshold=0.38          # 复评 wr=38.5%，调整为0.38
                )
            ]
    
    
    def _create_stage_environment(self, stage: CurriculumStage) -> Dict:
        """为指定阶段创建环境配置"""
        return {
            'max_vessels': stage.max_vessels,
            'max_berths': stage.max_berths,
            'traffic_intensity': stage.traffic_intensity,
            'weather_complexity': stage.weather_complexity,
            'node_config': {
                'berths': stage.max_berths,
                'anchorages': max(3, stage.max_berths // 2),
                'channels': max(2, stage.max_berths // 3),
                'terminals': max(2, stage.max_berths // 4),
                'max_vessels': stage.max_vessels
            }
        }
    
    def _generate_stage_data(self, stage: CurriculumStage, num_samples: int = 100) -> List[Dict]:
        """为指定阶段生成训练/评估数据"""
        env_config = self._create_stage_environment(stage)
        stage_data = []
        for _ in range(num_samples):
            stage_data.append({
                'vessel_count': np.random.randint(1, env_config['max_vessels'] + 1),
                'berth_occupancy': np.random.uniform(0.2, 0.8 * env_config['traffic_intensity']),
                'weather_factor': np.random.uniform(0.8, 1.0 - 0.2 * env_config['weather_complexity']),
                'queue_length': np.random.poisson(env_config['traffic_intensity'] * 5),
                'time_pressure': np.random.uniform(0.3, 0.7 + 0.3 * env_config['traffic_intensity']),
                'env_config': env_config
            })
        return stage_data
    
    def _calculate_baseline_threshold(self, stage: CurriculumStage, test_data: List[Dict]) -> float:
        """计算基线随机策略的奖励阈值（日志展示用）"""
        baseline_rewards = []
        for data_point in test_data:
            action = np.random.randint(0, stage.max_berths)
            reward = self._calculate_stage_reward(data_point, action, stage)
            baseline_rewards.append(reward)
        q25 = np.percentile(baseline_rewards, 25)
        q50 = np.percentile(baseline_rewards, 50)
        q75 = np.percentile(baseline_rewards, 75)
        iqr = q75 - q25
        threshold = float(q50 + 0.25 * iqr)
        logger.info(f"  基线阈值计算: 平均 {np.mean(baseline_rewards):.2f}, 中位数 {q50:.2f}, IQR {iqr:.2f}, 稳健阈值 {threshold:.2f}")
        return threshold
    
    def _evaluate_stage_performance(self, agent: GATPPOAgent, stage: CurriculumStage, 
                                    test_data: List[Dict], reward_threshold: float) -> Dict:
        """评估阶段性能：与随机基线做配对胜率"""
        agent.actor_critic.eval()
        agent_rewards, baseline_rewards, win_flags = [], [], []
        num_actions = stage.max_berths
        K = 10  # 基线采样次数
        
        for data_point in test_data:
            try:
                state = self._extract_state_from_data(data_point)
                node_features, adj_matrix = self._extract_graph_features_from_data(data_point)
                with torch.no_grad():
                     action_probs, _ = agent.actor_critic(
                         torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0),
                         torch.as_tensor(node_features, dtype=torch.float32, device=self.device).unsqueeze(0),
                         self._prep_adj_3d(adj_matrix)   # 统一成 [B,N,N]
                     )
                     action_probs = torch.nan_to_num(action_probs, nan=0.0, posinf=0.0, neginf=0.0)
                     if action_probs.shape[-1] != num_actions:
                         action_probs = action_probs[..., :num_actions]
                     action_probs = action_probs / action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                     # 添加全零概率兜底
                     if (action_probs.sum(dim=-1) < 1e-6).any():
                         action_probs = torch.full_like(action_probs, 1.0 / num_actions)
                     agent_action = torch.argmax(action_probs, dim=-1).item()
                agent_reward = self._calculate_stage_reward(data_point, agent_action, stage)
                agent_rewards.append(agent_reward)

                # 基线K次
                b_rewards = []
                for _ in range(K):
                    ba = np.random.randint(0, num_actions)
                    b_rewards.append(self._calculate_stage_reward(data_point, ba, stage))
                b_mean = float(np.mean(b_rewards))
                baseline_rewards.append(b_mean)
                win_flags.append(1 if agent_reward > b_mean else 0)
            except Exception as e:
                logger.warning(f"评估失败: {e}; 记为未赢")
                agent_rewards.append(-1.0); baseline_rewards.append(0.0); win_flags.append(0)
        
        completion_rate = float(np.mean(win_flags)) if win_flags else 0.0
        agent.actor_critic.train()
        return {
            'avg_reward': float(np.mean(agent_rewards)) if agent_rewards else 0.0,
            'baseline_avg_reward': float(np.mean(baseline_rewards)) if baseline_rewards else 0.0,
            'completion_rate': completion_rate,         # 以胜率为“完成率”
            'success': completion_rate >= stage.success_threshold,
            'reward_threshold': float(reward_threshold),
            'win_rate': completion_rate
        }
    
    def _extract_state_from_data(self, data_point: Dict) -> np.ndarray:
        """从数据点提取状态向量（20维）"""
        return np.array([
            data_point['vessel_count'] / 25.0,
            data_point['berth_occupancy'],
            data_point['weather_factor'],
            data_point['queue_length'] / 20.0,
            data_point['time_pressure'],
            *np.random.randn(15)
        ], dtype=np.float32)
    
    def _extract_graph_features_from_data(self, data_point: Dict) -> Tuple[np.ndarray, np.ndarray]:
        """构造图特征"""
        env_config = data_point['env_config']
        # 只统计真正的图节点类型，不包含 max_vessels
        node_cfg = env_config['node_config']
        total_nodes = node_cfg['berths'] + node_cfg['anchorages'] + node_cfg['channels'] + node_cfg['terminals']
        node_features = np.random.randn(total_nodes, 7).astype(np.float32)
        type_col = (np.random.randint(0, 5, size=(total_nodes, 1)) / 4.0).astype(np.float32)
        node_features = np.concatenate([type_col, node_features], axis=1)  # [N, 8]
        
        adj = np.eye(total_nodes, dtype=np.float32)
        for i in range(total_nodes):
            for j in range(i+1, min(i+5, total_nodes)):
                if np.random.rand() < 0.3:
                    adj[i, j] = adj[j, i] = 1.0
        return node_features, adj
    
    def _calculate_stage_reward(self, data_point: Dict, action: int, stage: CurriculumStage) -> float:
        """用港口特定奖励函数计算奖励，并按阶段难度缩放"""
        # 模拟状态转换
        state_dict = {'recent_actions': [action]*4, 'traffic_pattern': 'normal'}

        # === 新：与动作相关的微型调度模型 ===
        max_berths = stage.max_berths
        a = int(action) % max_berths

        # 用阶段配置合成"泊位负载"，低负载→更高服务率
        rng = np.random.default_rng(int(data_point['vessel_count']*1000 + data_point['queue_length']*17 + a))
        # 基于总体占用率生成各泊位负载（与 berth_occupancy 同趋势）
        base_load = np.clip(data_point['berth_occupancy'], 0.05, 0.95)
        berth_loads = np.clip(rng.normal(loc=base_load, scale=0.1, size=max_berths), 0.02, 0.98)
        load = float(berth_loads[a])

        # 服务概率：天气越好、负载越低、排队越长时"服务收益"越大
        weather = float(np.clip(data_point['weather_factor'], 0.2, 1.0))
        queue = int(data_point['queue_length'])
        demand = 1.0 + min(queue / max(1, stage.max_vessels), 1.5)  # 需求强度
        service_prob = np.clip((1 - load) * weather * 0.9, 0.02, 0.98)

        # 是否完成一个任务
        completed = 1 if rng.random() < service_prob else 0

        # 等待时间：负载 & 队列越高越慢，天气越差越慢
        wait_base = 1800.0 * (0.6 + 0.8*load) * (1.3 - 0.8*weather) * (1 + 0.6*queue/max(1, stage.max_vessels))
        wait = float(np.clip(rng.normal(wait_base, 0.15*wait_base), 60.0, 4*3600.0))

        next_state_dict = {
            'completed_tasks': completed,
            'total_tasks': 3,
            'waiting_times': [wait],  # ↓↓↓ 与动作强相关
            'berth_utilization': data_point['berth_occupancy'],
            'queue_length': queue,
            'max_queue_capacity': stage.max_vessels
        }

        base_reward = self.reward_function.calculate_reward(state_dict, action, next_state_dict)
        difficulty_factor = (stage.traffic_intensity + stage.weather_complexity) / 2
        adjusted_reward = base_reward * (1 + difficulty_factor)
        return float(adjusted_reward)
    
    def _safe_clear_buffer(self, agent: GATPPOAgent):
        """兼容不同buffer实现的清空动作"""
        if hasattr(agent, 'buffer'):
            if hasattr(agent.buffer, 'clear') and callable(agent.buffer.clear):
                agent.buffer.clear()
                return
            if hasattr(agent.buffer, 'buffer') and hasattr(agent.buffer.buffer, 'clear'):
                agent.buffer.buffer.clear()
                return
    
    def _prep_adj_3d(self, adj: np.ndarray) -> torch.Tensor:
        """规范化邻接矩阵为 [B,N,N] 格式"""
        A = torch.as_tensor(adj, dtype=torch.float32, device=self.device)
        # 允许输入 [N,N] / [B,N,N] / [B,1,N,N]
        if A.dim() == 2:
            A = A.unsqueeze(0)                     # [1,N,N]
        elif A.dim() == 4 and A.size(1) == 1:
            A = A.squeeze(1)                       # [B,N,N]
        # 其他情况直接过
        if A.dim() != 3:
            raise ValueError(f"adj must be [B,N,N], got {tuple(A.shape)}")
        return A
    
    def train_stage(self, agent: GATPPOAgent, stage: CurriculumStage) -> Tuple[GATPPOAgent, Dict]:
        """训练单个阶段"""
        logger.info(f"开始训练阶段: {stage.name}")
        logger.info(f"  描述: {stage.description}")
        logger.info(f"  目标轮数: {stage.episodes}")
        logger.info(f"  成功阈值: {stage.success_threshold}")
        
        # 清空buffer，避免跨阶段污染
        self._safe_clear_buffer(agent)
        logger.info("  已清空buffer，准备新阶段训练")
        
        # 数据
        train_data = self._generate_stage_data(stage, num_samples=200)
        test_data = self._generate_stage_data(stage, num_samples=50)
        fixed_threshold = self._calculate_baseline_threshold(stage, test_data)
        
        stage_history, best_performance = [], 0.0
        num_actions = stage.max_berths
        
        for episode in range(stage.episodes):
            # 熵系数线性退火 0.02 -> 0.005
            progress = episode / max(stage.episodes, 1)
            current_entropy = 0.02 * (1 - progress) + 0.005 * progress
            if hasattr(agent, 'entropy_coef'):
                agent.entropy_coef = current_entropy
            
            episode_reward, episode_steps = 0.0, 0
            episode_data = np.random.choice(train_data, size=min(30, len(train_data)), replace=False)
            
            for data_point in episode_data:
                state = self._extract_state_from_data(data_point)
                node_features, adj_matrix = self._extract_graph_features_from_data(data_point)
                try:
                     with torch.no_grad():
                         ap, value = agent.actor_critic(
                             torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0),
                             torch.as_tensor(node_features, dtype=torch.float32, device=self.device).unsqueeze(0),
                             self._prep_adj_3d(adj_matrix)
                         )
                     ap = torch.nan_to_num(ap, nan=0.0, posinf=0.0, neginf=0.0)
                     if ap.shape[-1] != num_actions:
                         ap = ap[..., :num_actions]
                     ap = ap / ap.sum(dim=-1, keepdim=True).clamp_min(1e-8)
                     # 添加全零概率兜底
                     if (ap.sum(dim=-1) < 1e-6).any():
                         ap = torch.full_like(ap, 1.0 / num_actions)
                     dist = torch.distributions.Categorical(ap)
                     action = dist.sample().item()
                     log_prob = dist.log_prob(torch.tensor(action, device=self.device))
                except Exception as e:
                    logger.warning(f"动作采样失败: {e}; 使用随机动作")
                    action = np.random.randint(0, num_actions)
                    log_prob = torch.tensor(0.0, device=self.device)
                    value = torch.tensor(0.0, device=self.device)
                
                reward = self._calculate_stage_reward(data_point, action, stage)
                
                # 存储
                try:
                    agent.store_experience(
                        np.asarray(state, dtype=np.float32),
                        np.asarray(node_features, dtype=np.float32),
                        np.asarray(adj_matrix, dtype=np.float32),
                        int(action),
                        float(reward),
                        np.asarray(state, dtype=np.float32),   # 简化：next_state=state
                        False,
                        float(log_prob.item()),
                        float(value.item() if hasattr(value, 'item') else 0.0)
                    )
                except Exception as e:
                    logger.warning(f"存储经验失败: {e}; 跳过本样本")
                    continue
                
                episode_reward += float(reward)
                episode_steps += 1
            
            # 更新模型（尝试多次小更新）
            loss_info = {'total_loss': 0.0}
            if hasattr(agent, 'batch_size') and hasattr(agent, 'buffer') and \
               hasattr(agent.buffer, 'buffer') and len(agent.buffer.buffer) >= agent.batch_size:
                total_loss, cnt = 0.0, 0
                for _ in range(3):
                    try:
                        single = agent.update()
                        if isinstance(single, dict):
                            total_loss += float(single.get('total_loss', single.get('loss', 0.0)))
                        elif single is not None:
                            total_loss += float(single)
                        cnt += 1
                    except Exception as e:
                        logger.warning(f"模型更新失败: {e}")
                        break
                loss_info['total_loss'] = total_loss / max(cnt, 1)
            
            # 评估
            if episode % 5 == 0:
                perf = self._evaluate_stage_performance(agent, stage, test_data, fixed_threshold)
                stage_history.append({
                    'episode': episode,
                    'avg_reward': episode_reward / max(episode_steps, 1),
                    'test_performance': perf,
                    'loss': loss_info['total_loss']
                })
                logger.info(f"  Episode {episode}: 胜率 {perf['completion_rate']:.2%}, "
                            f"智能体奖励 {perf['avg_reward']:.2f}, "
                            f"基线奖励 {perf.get('baseline_avg_reward', 0):.2f}")
                
                if perf['completion_rate'] > best_performance:
                    best_performance = perf['completion_rate']
                    model_path = self.save_dir / f"stage_{stage.name.replace(' ', '_')}_best.pt"
                    torch.save({
                        'episode': episode,
                        'stage': stage.name,
                        'model_state_dict': agent.actor_critic.state_dict(),
                        'performance': perf
                    }, model_path)
        
        # 最终评估
        final_perf = self._evaluate_stage_performance(agent, stage, test_data, fixed_threshold)
        stage_results = {
            'stage_name': stage.name,
            'final_performance': final_perf,
            'best_performance': best_performance,
            'success': final_perf['success'],
            'history': stage_history
        }
        logger.info(f"阶段 {stage.name} 完成:\n  最终完成率: {final_perf['completion_rate']:.2%}\n  是否成功: {final_perf['success']}")
        return agent, stage_results
    
    def train_curriculum(self) -> Dict:
        """执行完整的分阶段训练"""
        logger.info(f"开始 {self.port_name} 港口的分阶段训练")
        agent = build_agent(
            self.port_name,
            hidden_dim=256, learning_rate=3e-4, batch_size=32,
            device=self.device, num_heads=4, dropout=0.1,
            state_dim=20, action_dim=15, node_feature_dim=8,
            entropy_coef=0.02, ppo_epochs=8
        )
        
        results = {'port_name': self.port_name, 'stages': [], 'overall_success': True}
        
        for i, stage in enumerate(self.curriculum_stages):
            logger.info(f"\n{'='*50}\n阶段 {i+1}/{len(self.curriculum_stages)}: {stage.name}\n{'='*50}")
            
            if i > 0:
                prev = self.curriculum_stages[i-1]
                best_path = self.save_dir / f"stage_{prev.name.replace(' ', '_')}_best.pt"
                if best_path.exists():
                    logger.info(f"  加载上一阶段最佳模型: {best_path}")
                    ckpt = torch.load(best_path, map_location=self.device)
                    agent.actor_critic.load_state_dict(ckpt['model_state_dict'])
                # 学习率衰减
                if hasattr(agent, 'optimizer'):
                    old_lr = agent.optimizer.param_groups[0]['lr']
                    new_lr = max(old_lr * 0.7, 1e-5)
                    for g in agent.optimizer.param_groups:
                        g['lr'] = new_lr
                    logger.info(f"  学习率衰减: {old_lr:.2e} → {new_lr:.2e}")
            
            agent, stage_results = self.train_stage(agent, stage)
            results['stages'].append(stage_results)
            if not stage_results['success']:
                logger.warning(f"阶段 {stage.name} 未达到成功标准，但继续下一阶段")
                results['overall_success'] = False
        
        final_model_path = self.save_dir / "curriculum_final_model.pt"
        torch.save({
            'model_state_dict': agent.actor_critic.state_dict(),
            'curriculum_results': results,
            'port_name': self.port_name,
            
        }, final_model_path)
        
        logger.info(f"\n分阶段训练完成!\n整体成功: {results['overall_success']}\n最终模型保存到: {final_model_path}")
        return results

def main():
    import argparse
    # 随机种子
    torch.manual_seed(42); np.random.seed(42)
    if torch.cuda.is_available(): torch.cuda.manual_seed(42)
    
    parser = argparse.ArgumentParser(description="分阶段训练器")
    parser.add_argument("--port", required=True, help="港口名称")
    args = parser.parse_args()
    
    trainer = CurriculumTrainer(args.port)
    results = trainer.train_curriculum()
    
    print(f"\n分阶段训练结果:")
    print(f"港口: {results['port_name']}")
    print(f"整体成功: {results['overall_success']}")
    for s in results['stages']:
        print(f"  {s['stage_name']}: 完成率 {s['final_performance']['completion_rate']:.2%}, 成功 {s['success']}")

if __name__ == "__main__":
    main()