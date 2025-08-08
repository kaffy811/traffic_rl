#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段微调脚本 - 针对接近阈值的阶段进行短期补训
基于路线B的训练改进方案，用于提升模型性能
"""
import os
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from curriculum_trainer import CurriculumTrainer, build_agent
from consistency_test_fixed import eval_one_stage

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StageFinetuner:
    """阶段微调器"""
    
    def __init__(self, port_name: str, device: str = "cpu"):
        self.port_name = port_name
        self.device = torch.device(device)
        self.trainer = CurriculumTrainer(port_name)
        self.agent = build_agent(port_name, device=self.device)
        
        logger.info(f"初始化阶段微调器 - 港口: {port_name}, 设备: {device}")
    
    def get_stage_by_name(self, stage_name: str):
        """根据名称获取阶段配置"""
        for stage in self.trainer.curriculum_stages:
            if stage.name == stage_name:
                return stage
        return None
    
    def load_stage_model(self, stage_name: str) -> bool:
        """加载阶段模型"""
        model_path = Path(f"../../models/curriculum_v2/{self.port_name}/stage_{stage_name}_best.pt")
        
        if not model_path.exists():
            logger.error(f"模型文件不存在: {model_path}")
            return False
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    self.agent.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                self.agent.actor_critic.load_state_dict(checkpoint)
            
            logger.info(f"✅ 加载模型: {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 加载模型失败: {e}")
            return False
    
    def evaluate_stage_performance(self, stage_name: str, n_samples: int = 200) -> Dict:
        """评估阶段性能"""
        stage = self.get_stage_by_name(stage_name)
        if stage is None:
            return {'error': f'未找到阶段: {stage_name}'}
        
        try:
            result = eval_one_stage(
                self.trainer, self.agent, stage,
                n_samples=n_samples, device=self.device
            )
            return result
        except Exception as e:
            logger.error(f"评估失败: {e}")
            return {'error': str(e)}
    
    def fine_tune_stage(self, stage_name: str, target_improvement: float = 0.05,
                       max_episodes: int = 40, eval_interval: int = 5) -> Dict:
        """微调阶段模型"""
        logger.info(f"🎯 开始微调阶段: {stage_name}")
        logger.info(f"目标提升: {target_improvement*100:.1f}%, 最大轮数: {max_episodes}")
        
        stage = self.get_stage_by_name(stage_name)
        if stage is None:
            return {'error': f'未找到阶段: {stage_name}'}
        
        # 加载模型
        if not self.load_stage_model(stage_name):
            return {'error': '模型加载失败'}
        
        # 评估初始性能
        logger.info("📊 评估初始性能...")
        initial_perf = self.evaluate_stage_performance(stage_name)
        if 'error' in initial_perf:
            return initial_perf
        
        initial_wr = initial_perf['win_rate']
        target_wr = initial_wr + target_improvement
        threshold = stage.success_threshold
        
        logger.info(f"初始胜率: {initial_wr*100:.1f}%")
        logger.info(f"目标胜率: {target_wr*100:.1f}%")
        logger.info(f"阈值: {threshold*100:.1f}%")
        
        # 微调配置
        fine_tune_config = self._get_fine_tune_config(stage_name)
        
        # 应用微调配置
        self._apply_fine_tune_config(fine_tune_config)
        
        # 训练历史
        training_history = {
            'initial_performance': initial_perf,
            'episodes': [],
            'best_performance': initial_perf,
            'best_episode': 0
        }
        
        best_wr = initial_wr
        no_improvement_count = 0
        early_stop_patience = 5
        
        # 微调训练循环
        for episode in range(1, max_episodes + 1):
            logger.info(f"🔄 微调轮次 {episode}/{max_episodes}")
            
            # 生成训练数据
            train_data = self.trainer._generate_stage_data(stage, num_samples=50)
            
            # 训练一个episode
            episode_metrics = self._train_episode(train_data, fine_tune_config)
            
            # 定期评估
            if episode % eval_interval == 0 or episode == max_episodes:
                logger.info(f"📊 评估轮次 {episode} 性能...")
                current_perf = self.evaluate_stage_performance(stage_name, n_samples=100)
                
                if 'error' not in current_perf:
                    current_wr = current_perf['win_rate']
                    
                    episode_record = {
                        'episode': episode,
                        'win_rate': current_wr,
                        'improvement': current_wr - initial_wr,
                        'metrics': episode_metrics,
                        'performance': current_perf
                    }
                    training_history['episodes'].append(episode_record)
                    
                    logger.info(f"当前胜率: {current_wr*100:.1f}% (提升: {(current_wr-initial_wr)*100:+.1f}%)")
                    
                    # 更新最佳性能
                    if current_wr > best_wr:
                        best_wr = current_wr
                        training_history['best_performance'] = current_perf
                        training_history['best_episode'] = episode
                        no_improvement_count = 0
                        
                        # 保存最佳模型
                        self._save_fine_tuned_model(stage_name, episode, current_perf)
                        logger.info(f"🏆 新的最佳性能! 已保存模型")
                    else:
                        no_improvement_count += 1
                    
                    # 检查是否达到目标
                    if current_wr >= target_wr:
                        logger.info(f"🎉 达到目标胜率! {current_wr*100:.1f}% >= {target_wr*100:.1f}%")
                        break
                    
                    # 早停检查
                    if no_improvement_count >= early_stop_patience:
                        logger.info(f"⏹️ 早停: {early_stop_patience} 轮无改善")
                        break
        
        # 最终评估
        logger.info("📊 最终性能评估...")
        final_perf = self.evaluate_stage_performance(stage_name, n_samples=200)
        
        training_history['final_performance'] = final_perf
        training_history['total_episodes'] = episode
        training_history['target_achieved'] = (
            final_perf.get('win_rate', 0) >= target_wr if 'error' not in final_perf else False
        )
        
        # 保存训练历史
        self._save_training_history(stage_name, training_history)
        
        logger.info(f"✅ 微调完成: {stage_name}")
        if 'error' not in final_perf:
            final_wr = final_perf['win_rate']
            total_improvement = final_wr - initial_wr
            logger.info(f"最终胜率: {final_wr*100:.1f}% (总提升: {total_improvement*100:+.1f}%)")
        
        return training_history
    
    def _get_fine_tune_config(self, stage_name: str) -> Dict:
        """获取微调配置"""
        base_config = {
            'learning_rate_factor': 0.7,  # 降低学习率
            'entropy_coef_end': 0.003,    # 略收探索
            'ppo_epochs': 8,              # 增加PPO轮数
            'freeze_encoder_episodes': 10, # 前10轮冻结编码器
            'advantage_normalization': True,
            'nan_to_num': True,
            'early_stopping_patience': 5
        }
        
        # 根据阶段特点调整配置
        stage_specific = {
            '基础阶段': {'learning_rate_factor': 0.8},
            '初级阶段': {'learning_rate_factor': 0.7},
            '中级阶段': {'learning_rate_factor': 0.6, 'ppo_epochs': 10},
            '高级阶段': {'learning_rate_factor': 0.5, 'ppo_epochs': 12},
            '专家阶段': {'learning_rate_factor': 0.4, 'ppo_epochs': 15}
        }
        
        if stage_name in stage_specific:
            base_config.update(stage_specific[stage_name])
        
        return base_config
    
    def _apply_fine_tune_config(self, config: Dict):
        """应用微调配置"""
        # 调整学习率
        if 'learning_rate_factor' in config:
            for param_group in self.agent.optimizer.param_groups:
                param_group['lr'] *= config['learning_rate_factor']
            logger.info(f"调整学习率: x{config['learning_rate_factor']}")
        
        # 调整其他超参数
        if hasattr(self.agent, 'entropy_coef'):
            self.agent.entropy_coef = config.get('entropy_coef_end', 0.003)
        
        if hasattr(self.agent, 'ppo_epochs'):
            self.agent.ppo_epochs = config.get('ppo_epochs', 8)
    
    def _train_episode(self, train_data: List, config: Dict) -> Dict:
        """训练一个episode"""
        # 这里应该实现具体的训练逻辑
        # 由于原始训练代码比较复杂，这里提供一个简化的框架
        
        metrics = {
            'avg_reward': 0.0,
            'policy_loss': 0.0,
            'value_loss': 0.0,
            'entropy': 0.0
        }
        
        # TODO: 实现具体的训练逻辑
        # 1. 从train_data生成经验
        # 2. 计算优势和回报
        # 3. 执行PPO更新
        # 4. 记录训练指标
        
        return metrics
    
    def _save_fine_tuned_model(self, stage_name: str, episode: int, performance: Dict):
        """保存微调后的模型"""
        output_dir = Path(f"../../models/fine_tuned/{self.port_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_path = output_dir / f"stage_{stage_name}_fine_tuned_ep{episode}.pt"
        
        checkpoint = {
            'model_state_dict': self.agent.actor_critic.state_dict(),
            'optimizer_state_dict': self.agent.optimizer.state_dict(),
            'episode': episode,
            'performance': performance,
            'port_name': self.port_name,
            'stage_name': stage_name
        }
        
        torch.save(checkpoint, model_path)
        logger.info(f"保存微调模型: {model_path}")
    
    def _save_training_history(self, stage_name: str, history: Dict):
        """保存训练历史"""
        output_dir = Path(f"../../models/fine_tuned/{self.port_name}")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        history_path = output_dir / f"stage_{stage_name}_fine_tune_history.json"
        
        with open(history_path, 'w', encoding='utf-8') as f:
            json.dump(history, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"保存训练历史: {history_path}")

def identify_risk_stages(results_dir: str, margin_threshold: float = 0.05) -> List[Tuple[str, str, float]]:
    """识别风险阶段（余量小于阈值的）"""
    risk_stages = []
    
    results_path = Path(results_dir)
    for json_file in results_path.glob("consistency_*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            port_name = data.get('port', 'unknown')
            stages = data.get('stages', [])
            
            for stage in stages:
                if stage.get('pass', False):  # 只考虑通过的阶段
                    win_rate = stage.get('win_rate', 0.0)
                    threshold = stage.get('threshold', 0.0)
                    margin = win_rate - threshold
                    
                    if 0 <= margin <= margin_threshold:  # 余量在0到阈值之间
                        risk_stages.append((port_name, stage.get('stage', ''), margin))
        
        except Exception as e:
            logger.warning(f"无法解析文件 {json_file}: {e}")
    
    # 按余量排序
    risk_stages.sort(key=lambda x: x[2])
    return risk_stages

def main():
    parser = argparse.ArgumentParser(description="阶段微调脚本")
    parser.add_argument("--port", help="港口名称")
    parser.add_argument("--stage", help="阶段名称")
    parser.add_argument("--target-improvement", type=float, default=0.05,
                       help="目标提升幅度 (默认5%%)")
    parser.add_argument("--max-episodes", type=int, default=40,
                       help="最大训练轮数")
    parser.add_argument("--device", default="cpu", help="设备类型")
    parser.add_argument("--auto-identify", action="store_true",
                       help="自动识别风险阶段")
    parser.add_argument("--results-dir", default="../../models/releases/2025-08-07",
                       help="测试结果目录")
    parser.add_argument("--margin-threshold", type=float, default=0.05,
                       help="风险余量阈值")
    
    args = parser.parse_args()
    
    if args.auto_identify:
        logger.info("🔍 自动识别风险阶段...")
        risk_stages = identify_risk_stages(args.results_dir, args.margin_threshold)
        
        if not risk_stages:
            logger.info("✅ 未发现风险阶段")
            return
        
        logger.info(f"⚠️ 发现 {len(risk_stages)} 个风险阶段:")
        for port, stage, margin in risk_stages:
            logger.info(f"  - {port}/{stage}: 余量 {margin*100:.1f}%")
        
        # 微调前几个最高风险的阶段
        for port, stage, margin in risk_stages[:3]:  # 只处理前3个
            logger.info(f"\n🎯 微调风险阶段: {port}/{stage}")
            
            finetuner = StageFinetuner(port, args.device)
            result = finetuner.fine_tune_stage(
                stage, 
                target_improvement=args.target_improvement,
                max_episodes=args.max_episodes
            )
            
            if 'error' in result:
                logger.error(f"❌ 微调失败: {result['error']}")
            else:
                logger.info(f"✅ 微调完成: {port}/{stage}")
    
    elif args.port and args.stage:
        # 微调指定阶段
        logger.info(f"🎯 微调指定阶段: {args.port}/{args.stage}")
        
        finetuner = StageFinetuner(args.port, args.device)
        result = finetuner.fine_tune_stage(
            args.stage,
            target_improvement=args.target_improvement,
            max_episodes=args.max_episodes
        )
        
        if 'error' in result:
            logger.error(f"❌ 微调失败: {result['error']}")
        else:
            logger.info(f"✅ 微调完成")
            
            # 显示结果摘要
            if 'final_performance' in result and 'error' not in result['final_performance']:
                final_wr = result['final_performance']['win_rate']
                initial_wr = result['initial_performance']['win_rate']
                improvement = final_wr - initial_wr
                
                print(f"\n📊 微调结果摘要:")
                print(f"初始胜率: {initial_wr*100:.1f}%")
                print(f"最终胜率: {final_wr*100:.1f}%")
                print(f"提升幅度: {improvement*100:+.1f}%")
                print(f"训练轮数: {result['total_episodes']}")
                print(f"目标达成: {'✅' if result['target_achieved'] else '❌'}")
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()