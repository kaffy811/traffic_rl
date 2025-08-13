#!/usr/bin/env python3
"""
Flower自动评测监听器
监听Flower保存目录，出现新的global_round_*.pt就触发夜测
"""

import time
import glob
import subprocess
import os
import sys
import logging
from pathlib import Path

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [autoeval] %(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)

def get_save_dir():
    """获取Flower保存目录"""
    try:
        tag = open("models/flw/LAST_SUCCESS.tag").read().strip()
        save_dir = f"models/flw/{tag}"
        if os.path.exists(save_dir):
            return save_dir
        else:
            logger.error(f"保存目录不存在: {save_dir}")
            return None
    except Exception as e:
        logger.error(f"读取LAST_SUCCESS.tag失败: {e}")
        return None

def run_nightly_ci():
    """运行夜测"""
    try:
        logger.info("🚀 触发夜测...")
        result = subprocess.run([
            "python", "scripts/nightly_ci.py",
            "--ports", "all",
            "--samples", "800",
            "--seeds", "42,123,2025",
            "--no-cache"
        ], check=False, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info("✅ 夜测完成")
        else:
            logger.error(f"❌ 夜测失败: {result.stderr}")
            
    except Exception as e:
        logger.error(f"❌ 运行夜测异常: {e}")

def main():
    """主循环"""
    logger.info("🎧 Flower自动评测监听器启动")
    
    save_dir = get_save_dir()
    if not save_dir:
        logger.error("无法获取保存目录，退出")
        return
    
    logger.info(f"📁 监听目录: {save_dir}")
    
    # 记录已见过的轮次
    seen_rounds = set()
    
    while True:
        try:
            # 查找所有轮次文件
            round_files = sorted(glob.glob(f"{save_dir}/global_round_*.pt"))
            
            # 检查新文件
            for round_file in round_files:
                if round_file not in seen_rounds:
                    seen_rounds.add(round_file)
                    round_num = Path(round_file).stem.split('_')[-1]
                    logger.info(f"🆕 发现新轮次: {round_num} → {round_file}")
                    
                    # 触发夜测
                    run_nightly_ci()
            
            # 等待20秒
            time.sleep(20)
            
        except KeyboardInterrupt:
            logger.info("👋 收到中断信号，退出")
            break
        except Exception as e:
            logger.error(f"❌ 监听循环异常: {e}")
            time.sleep(20)

if __name__ == "__main__":
    main() 