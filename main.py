#!/usr/bin/env python3
"""
GAT-FedPPO 海上交通管制系统
主程序入口：整合AIS数据处理、CityFlow仿真和FedML联邦学习

实验流程：
1. AIS数据预处理
2. 生成CityFlow流量配置
3. 运行交通仿真
4. 启动联邦强化学习训练
"""

import os
import sys
import subprocess
import logging
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """检查必要的依赖库"""
    logger.info("检查依赖库...")
    
    required_packages = ['pandas', 'shapely', 'numpy', 'torch']
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            logger.info(f"✅ {package} 已安装")
        except ImportError:
            missing_packages.append(package)
            logger.warning(f"❌ {package} 未安装")
    
    if missing_packages:
        logger.error(f"缺少依赖库: {missing_packages}")
        logger.info("请运行: pip install " + " ".join(missing_packages))
        return False
    
    return True


def preprocess_ais_data():
    """运行AIS数据预处理"""
    logger.info("=== 步骤1: AIS数据预处理 ===")
    
    try:
        # 切换到data目录并运行预处理脚本
        result = subprocess.run([
            sys.executable, 'data/preprocess_ais.py'
        ], capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            logger.info("AIS数据预处理完成")
            logger.info(result.stdout)
            return True
        else:
            logger.error("AIS数据预处理失败")
            logger.error(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"预处理过程出错: {e}")
        return False


def generate_flows():
    """生成CityFlow流量配置"""
    logger.info("=== 步骤2: 生成CityFlow流量配置 ===")
    
    try:
        result = subprocess.run([
            sys.executable, 'data/build_flows.py'
        ], capture_output=True, text=True, cwd='.')
        
        if result.returncode == 0:
            logger.info("流量配置生成完成")
            logger.info(result.stdout)
            return True
        else:
            logger.error("流量配置生成失败")
            logger.error(result.stderr)
            return False
            
    except Exception as e:
        logger.error(f"流量生成过程出错: {e}")
        return False


def run_cityflow_simulation():
    """运行CityFlow交通仿真"""
    logger.info("=== 步骤3: CityFlow交通仿真 ===")
    
    # 检查CityFlow是否可用
    if not os.path.exists('FedML/CityFlow'):
        logger.warning("CityFlow目录不存在，跳过仿真步骤")
        return True
    
    try:
        # 这里需要根据具体的CityFlow配置来运行
        logger.info("CityFlow仿真准备中...")
        # result = subprocess.run([
        #     sys.executable, 'FedML/CityFlow/run_cityflow.py'
        # ], capture_output=True, text=True, cwd='.')
        
        logger.info("CityFlow仿真模块待实现")
        return True
        
    except Exception as e:
        logger.error(f"仿真过程出错: {e}")
        return False


def setup_fedml_training():
    """设置FedML联邦学习训练"""
    logger.info("=== 步骤4: FedML联邦学习设置 ===")
    
    if not os.path.exists('FedML'):
        logger.warning("FedML目录不存在，跳过联邦学习设置")
        return True
    
    try:
        logger.info("FedML联邦学习准备中...")
        # 这里需要根据具体的FedML配置来设置
        logger.info("FedML训练模块待实现")
        return True
        
    except Exception as e:
        logger.error(f"FedML设置过程出错: {e}")
        return False


def main():
    """主函数：执行完整的实验流程"""
    
    print("=" * 60)
    print("    GAT-FedPPO 海上交通管制系统")
    print("    联邦强化学习 + 图注意力网络")
    print("=" * 60)
    
    # 1. 检查依赖
    if not check_dependencies():
        logger.error("依赖检查失败，程序退出")
        sys.exit(1)
    
    # 2. AIS数据预处理
    if not preprocess_ais_data():
        logger.error("AIS数据预处理失败，程序退出")
        sys.exit(1)
    
    # 3. 生成流量配置
    if not generate_flows():
        logger.error("流量配置生成失败，程序退出")
        sys.exit(1)
    
    # 4. 运行仿真
    if not run_cityflow_simulation():
        logger.error("仿真运行失败，程序退出")
        sys.exit(1)
    
    # 5. 联邦学习设置
    if not setup_fedml_training():
        logger.error("联邦学习设置失败，程序退出")
        sys.exit(1)
    
    logger.info("🎉 实验流程完成!")
    logger.info("现在可以开始联邦强化学习训练")


if __name__ == "__main__":
    main()
