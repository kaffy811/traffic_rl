#!/usr/bin/env python3
"""
海事GAT-PPO联邦学习服务器启动脚本
基于FedML框架的分布式服务器端
"""

import sys
import os
from pathlib import Path
import logging

# 设置项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "FedML" / "python"))

# FedML imports
import fedml
from fedml import FedMLRunner

# 自定义模块导入
from maritime_data_loader import load_maritime_data
from maritime_model_creator import create_maritime_model

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if __name__ == "__main__":
    logger.info("🚢 启动海事GAT-PPO联邦学习服务器...")
    
    # 设置FedML全局变量（直接设置）
    import fedml
    fedml._global_training_type = "cross_silo"
    fedml._global_comm_backend = "MQTT_S3"
    
    try:
        # 设置配置文件路径和命令行参数
        config_path = os.path.join(os.path.dirname(__file__), "config", "fedml_config.yaml")
        
        # 设置sys.argv以包含配置文件路径
        import sys
        if len(sys.argv) == 1:
            sys.argv.extend(["--cf", config_path, "--rank", "0", "--role", "server"])
        
        # 初始化FedML参数
        args = fedml.init()
        
        # 获取设备
        device = fedml.device.get_device(args)
        
        # 加载海事数据（服务器也需要数据用于测试）
        logger.info("📊 加载海事数据...")
        dataset = load_maritime_data(args)
        
        # 创建海事模型
        logger.info("🤖 创建海事GAT-PPO模型...")
        output_dim = 4  # 4个动作空间
        model = create_maritime_model(args, output_dim)
        
        # 启动联邦学习
        logger.info("🚀 启动联邦学习服务器...")
        fedml_runner = FedMLRunner(args, device, dataset, model)
        fedml_runner.run()
        
    except Exception as e:
        logger.error(f"❌ 服务器启动失败: {e}")
        import traceback
        traceback.print_exc()