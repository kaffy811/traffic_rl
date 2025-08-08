#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
四港口全量训练脚本
按顺序训练所有港口的课程学习模型
"""
import os
import sys
import time
import logging
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent / "src" / "federated"))

def setup_logging():
    """设置日志"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("logs/training")
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"train_all_ports_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def train_port(port_name, logger):
    """训练单个港口"""
    logger.info(f"🚢 开始训练港口: {port_name}")
    start_time = time.time()
    
    try:
        # 导入并训练
        from curriculum_trainer import CurriculumTrainer
        trainer = CurriculumTrainer(port_name)
        trainer.train_curriculum()
        
        elapsed = time.time() - start_time
        logger.info(f"✅ {port_name} 训练完成，耗时: {elapsed/60:.1f}分钟")
        return True
        
    except Exception as e:
        elapsed = time.time() - start_time
        logger.error(f"❌ {port_name} 训练失败，耗时: {elapsed/60:.1f}分钟")
        logger.error(f"错误详情: {str(e)}")
        return False

def main():
    logger = setup_logging()
    
    # 四个港口按复杂度排序（简单到复杂）
    ports = [
        'gulfport',        # 2个阶段，最简单
        'south_louisiana', # 3个阶段
        'baton_rouge',     # 3个阶段  
        'new_orleans'      # 5个阶段，最复杂
    ]
    
    logger.info("🚀 开始四港口全量训练")
    logger.info(f"训练顺序: {' → '.join(ports)}")
    
    total_start = time.time()
    results = {}
    
    for i, port in enumerate(ports, 1):
        logger.info(f"\n{'='*50}")
        logger.info(f"进度: {i}/{len(ports)} - {port}")
        logger.info(f"{'='*50}")
        
        success = train_port(port, logger)
        results[port] = success
        
        if success:
            logger.info(f"🎉 {port} 训练成功")
        else:
            logger.warning(f"⚠️ {port} 训练失败，继续下一个港口")
    
    # 总结
    total_elapsed = time.time() - total_start
    successful = sum(results.values())
    
    logger.info(f"\n{'='*50}")
    logger.info("📊 训练总结")
    logger.info(f"{'='*50}")
    logger.info(f"总耗时: {total_elapsed/3600:.1f}小时")
    logger.info(f"成功率: {successful}/{len(ports)} ({successful/len(ports)*100:.1f}%)")
    
    for port, success in results.items():
        status = "✅" if success else "❌"
        logger.info(f"  {status} {port}")
    
    if successful == len(ports):
        logger.info("🎉 所有港口训练完成！")
        return 0
    else:
        logger.warning(f"⚠️ {len(ports)-successful}个港口训练失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())