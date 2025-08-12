#!/usr/bin/env python3
"""
联邦学习评测回调脚本 - 定期评测和早停检测
"""

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

def setup_logging(log_level: str = "INFO"):
    """设置日志"""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def run_nightly_ci(ports: str = "all", samples: int = 800, 
                   seeds: str = "42,123,2025", no_cache: bool = True, k: int = 120) -> str:
    """运行夜间CI评测"""
    logger = logging.getLogger(__name__)
    
    cmd = [
        "python", "scripts/nightly_ci.py",
        "--ports", ports,
        "--samples", str(samples),
        "--seeds", seeds,
        "--k", str(k)
    ]
    
    if no_cache:
        cmd.append("--no-cache")
    
    logger.info(f"执行评测命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("评测完成")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"评测失败: {e}")
        logger.error(f"错误输出: {e.stderr}")
        return ""

def run_monitoring_dashboard() -> str:
    """运行监控面板"""
    logger = logging.getLogger(__name__)
    
    cmd = ["python", "scripts/monitoring_dashboard.py"]
    logger.info(f"执行监控命令: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logger.info("监控面板完成")
        return result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"监控面板失败: {e}")
        logger.error(f"错误输出: {e.stderr}")
        return ""

def parse_ci_results(output: str) -> Dict:
    """解析CI评测结果"""
    logger = logging.getLogger(__name__)
    
    # 尝试从输出中提取结果文件路径
    result_files = []
    for line in output.split('\n'):
        if 'consistency_' in line and '.json' in line:
            # 提取JSON文件路径
            parts = line.split()
            for part in parts:
                if 'consistency_' in part and '.json' in part:
                    result_files.append(part.strip())
                    break
    
    logger.info(f"找到结果文件: {result_files}")
    
    # 解析每个结果文件
    all_results = {}
    for result_file in result_files:
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
                port_name = data.get('port_name', 'unknown')
                all_results[port_name] = data
        except Exception as e:
            logger.warning(f"解析结果文件 {result_file} 失败: {e}")
    
    return all_results

def calculate_average_win_rate(results: Dict) -> float:
    """计算所有港口的平均胜率"""
    if not results:
        return 0.0
    
    total_wr = 0.0
    count = 0
    
    for port_data in results.values():
        stages = port_data.get('stages', [])
        for stage in stages:
            if 'win_rate' in stage:
                total_wr += stage['win_rate']
                count += 1
    
    return total_wr / count if count > 0 else 0.0

def check_early_stopping(current_wr: float, previous_wr: float, 
                         threshold: float = 0.015) -> bool:
    """检查是否需要早停"""
    if previous_wr == 0.0:  # 第一次评测
        return False
    
    drop = previous_wr - current_wr
    logger = logging.getLogger(__name__)
    
    logger.info(f"胜率变化: {previous_wr:.4f} → {current_wr:.4f} (变化: {drop:+.4f})")
    
    if drop > threshold:
        logger.warning(f"胜率下降 {drop:.4f} > {threshold:.4f}，触发早停条件")
        return True
    
    return False

def main():
    parser = argparse.ArgumentParser(description="联邦学习评测回调脚本")
    
    parser.add_argument("--round", type=int, required=True,
                       help="当前联邦学习轮次")
    parser.add_argument("--ports", type=str, default="all",
                       help="评测港口")
    parser.add_argument("--samples", type=int, default=800,
                       help="评测样本数")
    parser.add_argument("--seeds", type=str, default="42,123,2025",
                       help="评测种子")
    parser.add_argument("--k", type=int, default=120,
                       help="去噪基线K值")
    parser.add_argument("--no-cache", action="store_true",
                       help="禁用缓存")
    parser.add_argument("--early-stop-threshold", type=float, default=0.015,
                       help="早停阈值（胜率下降超过此值触发早停）")
    parser.add_argument("--history-file", type=str, 
                       default="logs/fl/eval_history.json",
                       help="评测历史文件")
    
    args = parser.parse_args()
    
    logger = setup_logging()
    logger.info(f"联邦学习评测回调 - 第 {args.round} 轮")
    
    # 运行评测
    ci_output = run_nightly_ci(
        ports=args.ports,
        samples=args.samples,
        seeds=args.seeds,
        no_cache=args.no_cache,
        k=args.k
    )
    
    if not ci_output:
        logger.error("评测失败，无法继续")
        sys.exit(1)
    
    # 运行监控面板
    dashboard_output = run_monitoring_dashboard()
    
    # 解析评测结果
    results = parse_ci_results(ci_output)
    
    if not results:
        logger.warning("无法解析评测结果")
        sys.exit(1)
    
    # 计算平均胜率
    current_wr = calculate_average_win_rate(results)
    logger.info(f"当前平均胜率: {current_wr:.4f}")
    
    # 检查早停条件
    history_file = Path(args.history_file)
    history_file.parent.mkdir(parents=True, exist_ok=True)
    
    previous_wr = 0.0
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
                if history:
                    previous_wr = history[-1].get('average_win_rate', 0.0)
        except Exception as e:
            logger.warning(f"读取历史文件失败: {e}")
    
    # 检查早停
    should_stop = check_early_stopping(
        current_wr, previous_wr, args.early_stop_threshold
    )
    
    # 保存评测历史
    eval_record = {
        'round': args.round,
        'timestamp': __import__('time').strftime('%Y-%m-%d %H:%M:%S'),
        'average_win_rate': current_wr,
        'port_results': {port: {
            'stages': data.get('stages', []),
            'pass': data.get('pass', False)
        } for port, data in results.items()},
        'early_stop_triggered': should_stop
    }
    
    history = []
    if history_file.exists():
        try:
            with open(history_file, 'r') as f:
                history = json.load(f)
        except Exception as e:
            logger.warning(f"读取历史文件失败: {e}")
    
    history.append(eval_record)
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2, ensure_ascii=False)
    
    logger.info(f"评测历史已保存到: {history_file}")
    
    # 输出结果摘要
    print("\n" + "="*50)
    print(f"联邦学习第 {args.round} 轮评测结果")
    print("="*50)
    
    for port, data in results.items():
        status = "✅" if data.get('pass', False) else "❌"
        print(f"{status} {port}: {'通过' if data.get('pass', False) else '失败'}")
    
    print(f"\n平均胜率: {current_wr:.4f}")
    
    if should_stop:
        print(f"\n⚠️  早停条件触发！胜率下降超过 {args.early_stop_threshold:.4f}")
        print("建议：停止联邦学习，回滚到上一最优全局权重")
        sys.exit(1)
    else:
        print(f"\n✅ 评测完成，继续联邦学习")
    
    # 输出监控面板摘要
    if dashboard_output:
        print("\n监控面板摘要:")
        print("-" * 30)
        lines = dashboard_output.split('\n')
        for line in lines:
            if 'baton_rouge' in line or '中级阶段' in line:
                print(line.strip())

if __name__ == "__main__":
    main() 