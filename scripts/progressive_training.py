#!/usr/bin/env python3
"""
递进式训练脚本 - 针对Baton Rouge和New Orleans
实现"宽→窄→急弯→急弯+潮"四段递进训练
"""

import os
import sys
import logging
import torch
from pathlib import Path
from typing import Dict, List, Tuple

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + "/../src/federated")

from curriculum_trainer import CurriculumTrainer, CurriculumStage
from gat_ppo_agent import GATPPOAgent, create_default_config

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
log = logging.getLogger(__name__)

class ProgressiveTrainer:
    """递进式训练器"""
    
    def __init__(self, port_name: str):
        self.port_name = port_name
        self.base_trainer = CurriculumTrainer(port_name)
        
        # 定义递进阶段
        self.progressive_stages = self._define_progressive_stages()
        
        log.info(f"初始化递进式训练器 - 港口: {port_name}")
        log.info(f"递进阶段数: {len(self.progressive_stages)}")
    
    def _define_progressive_stages(self) -> List[Dict]:
        """定义四段递进训练配置"""
        if self.port_name == 'baton_rouge':
            return [
                {
                    "name": "宽航道",
                    "description": "宽直航道基础训练",
                    "max_vessels": 8,
                    "max_berths": 5,
                    "traffic_intensity": 0.4,
                    "weather_complexity": 0.1,
                    "episodes": 25,
                    "success_threshold": 0.45,
                    "features": ["basic"]  # 基础特征
                },
                {
                    "name": "窄航道", 
                    "description": "增加航道约束",
                    "max_vessels": 12,
                    "max_berths": 8,
                    "traffic_intensity": 0.6,
                    "weather_complexity": 0.2,
                    "episodes": 30,
                    "success_threshold": 0.42,
                    "features": ["basic", "curvature"]  # 加入曲率特征
                },
                {
                    "name": "急弯",
                    "description": "窄弯道训练",
                    "max_vessels": 15,
                    "max_berths": 10,
                    "traffic_intensity": 0.7,
                    "weather_complexity": 0.3,
                    "episodes": 35,
                    "success_threshold": 0.40,
                    "features": ["basic", "curvature", "tide"]  # 加入潮汐特征
                },
                {
                    "name": "急弯+潮汐",
                    "description": "完整复杂度",
                    "max_vessels": 20,
                    "max_berths": 15,
                    "traffic_intensity": 1.0,
                    "weather_complexity": 0.4,
                    "episodes": 40,
                    "success_threshold": 0.37,  # 临时运营阈值
                    "features": ["basic", "curvature", "tide", "risk_weighted"]  # 风险加权图
                }
            ]
        elif self.port_name == 'new_orleans':
            return [
                {
                    "name": "宽航道",
                    "description": "宽直航道基础训练", 
                    "max_vessels": 6,
                    "max_berths": 4,
                    "traffic_intensity": 0.3,
                    "weather_complexity": 0.1,
                    "episodes": 25,
                    "success_threshold": 0.40,
                    "features": ["basic"]
                },
                {
                    "name": "窄航道",
                    "description": "增加航道约束",
                    "max_vessels": 10,
                    "max_berths": 6,
                    "traffic_intensity": 0.5,
                    "weather_complexity": 0.2,
                    "episodes": 30,
                    "success_threshold": 0.38,
                    "features": ["basic", "curvature"]
                },
                {
                    "name": "急弯",
                    "description": "窄弯道训练",
                    "max_vessels": 15,
                    "max_berths": 8,
                    "traffic_intensity": 0.7,
                    "weather_complexity": 0.3,
                    "episodes": 35,
                    "success_threshold": 0.47,  # 校准后的阈值
                    "features": ["basic", "curvature", "tide"]
                },
                {
                    "name": "急弯+潮汐",
                    "description": "完整复杂度",
                    "max_vessels": 18,
                    "max_berths": 12,
                    "traffic_intensity": 0.9,
                    "weather_complexity": 0.4,
                    "episodes": 40,
                    "success_threshold": 0.40,
                    "features": ["basic", "curvature", "tide", "risk_weighted"]
                }
            ]
        else:
            log.warning(f"港口 {port_name} 不支持递进式训练")
            return []
    
    def _load_gulfport_pretrained(self) -> GATPPOAgent:
        """加载Gulfport预训练模型作为warm-start"""
        log.info("加载Gulfport预训练模型...")
        
        # 尝试加载Gulfport预训练模型
        gulfport_model_path = Path(f"../../models/curriculum_v2/gulfport/best_model.pth")
        
        if gulfport_model_path.exists():
            config = create_default_config(self.port_name)
            agent = GATPPOAgent(port_name=self.port_name, config=config)
            agent.load_state_dict(torch.load(gulfport_model_path, map_location='cpu'))
            log.info("✅ 成功加载Gulfport预训练模型")
            return agent
        else:
            log.warning("⚠️ Gulfport预训练模型不存在，使用随机初始化")
            config = create_default_config(self.port_name)
            return GATPPOAgent(port_name=self.port_name, config=config)
    
    def _check_convergence(self, stage_name: str, performance: Dict) -> bool:
        """检查阶段是否收敛"""
        # 从performance中正确提取胜率
        if isinstance(performance, dict):
            # 尝试从final_performance中获取
            final_perf = performance.get('final_performance', {})
            win_rate = final_perf.get('win_rate', final_perf.get('completion_rate', 0.0))
            
            # 从stage_config中获取阈值（通过training_results传递）
            threshold = performance.get('config', {}).get('success_threshold', 0.5)
        else:
            win_rate = 0.0
            threshold = 0.5
        
        # 收敛条件：胜率超过阈值的80%（稍微宽松一点）
        converged = win_rate >= threshold * 0.8
        
        log.info(f"阶段 {stage_name} 收敛检查: 胜率={win_rate:.3f}, 阈值={threshold:.3f}, 收敛={converged}")
        
        return converged
    
    def _train_stage_with_config(self, stage_config: Dict, agent: GATPPOAgent) -> Tuple[GATPPOAgent, Dict]:
        """使用指定配置训练单个阶段"""
        stage_name = stage_config["name"]
        log.info(f"开始训练阶段: {stage_name}")
        
        # 创建课程阶段对象
        curriculum_stage = CurriculumStage(
            name=stage_name,
            description=stage_config["description"],
            max_vessels=stage_config["max_vessels"],
            max_berths=stage_config["max_berths"],
            traffic_intensity=stage_config["traffic_intensity"],
            weather_complexity=stage_config["weather_complexity"],
            episodes=stage_config["episodes"],
            success_threshold=stage_config["success_threshold"]
        )
        
        # 训练当前阶段
        trained_agent, performance = self.base_trainer.train_stage(agent, curriculum_stage)
        
        log.info(f"阶段 {stage_name} 训练完成: {performance}")
        
        return trained_agent, performance
    
    def progressive_training(self) -> Dict:
        """执行递进式训练"""
        log.info(f"开始 {self.port_name} 递进式训练")
        
        training_results = {}
        
        # 第一阶段：使用Gulfport预训练模型warm-start
        current_agent = self._load_gulfport_pretrained()
        
        for i, stage_config in enumerate(self.progressive_stages):
            stage_name = stage_config["name"]
            log.info(f"\n=== 第{i+1}阶段: {stage_name} ===")
            
            try:
                # 训练当前阶段
                current_agent, performance = self._train_stage_with_config(stage_config, current_agent)
                
                # 记录结果
                training_results[stage_name] = {
                    "config": stage_config,
                    "performance": performance,
                    "agent_saved": True
                }
                
                # 将config信息合并到performance中，供收敛检查使用
                if isinstance(performance, dict):
                    performance['config'] = stage_config
                
                # 保存当前阶段模型
                stage_save_path = self.base_trainer.save_dir / f"{stage_name}_model.pth"
                torch.save(current_agent.state_dict(), stage_save_path)
                log.info(f"✅ 阶段 {stage_name} 模型已保存到: {stage_save_path}")
                
                # 检查收敛性
                if not self._check_convergence(stage_name, performance):
                    log.warning(f"⚠️ 阶段 {stage_name} 未收敛，停止递进训练")
                    break
                
                log.info(f"✅ 阶段 {stage_name} 训练成功，继续下一阶段")
                
            except Exception as e:
                log.error(f"❌ 阶段 {stage_name} 训练失败: {e}")
                training_results[stage_name] = {
                    "config": stage_config,
                    "error": str(e),
                    "agent_saved": False
                }
                break
        
        # 保存最终模型
        final_save_path = self.base_trainer.save_dir / "curriculum_final_model.pt"
        torch.save(current_agent.state_dict(), final_save_path)
        log.info(f"✅ 递进训练完成，最终模型已保存到: {final_save_path}")
        
        return training_results

def main():
    """主函数"""
    import argparse
    
    ap = argparse.ArgumentParser(description="递进式训练脚本")
    ap.add_argument("--port", required=True,
                   choices=['baton_rouge', 'new_orleans', 'gulfport', 'south_louisiana'], 
                   help="目标港口")
    ap.add_argument("--output", default="training_results.json", help="结果输出文件")
    
    args = ap.parse_args()
    
    # 根据港口类型选择训练策略
    if args.port in ("gulfport", "south_louisiana"):
        print(f"ℹ️ 该港口未定义递进式课程，改用 CurriculumTrainer 全流程。")
        # 导入并创建课程训练器
        sys.path.append("src/federated")
        from curriculum_trainer import CurriculumTrainer
        
        trainer = CurriculumTrainer(args.port)
        results = trainer.run_all_stages()
        
        # 保存结果
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        log.info(f"课程训练结果已保存到: {args.output}")
    else:
        # 创建递进训练器
        trainer = ProgressiveTrainer(args.port)
        
        # 执行递进训练
        results = trainer.progressive_training()
        
        # 保存结果
        import json
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        log.info(f"递进训练结果已保存到: {args.output}")

if __name__ == "__main__":
    main() 