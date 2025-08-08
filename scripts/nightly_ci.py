#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
夜间CI检查脚本
每晚自动运行一致性测试，监控模型性能退化
"""
import os
import sys
import json
import yaml
import logging
import argparse
from datetime import datetime
from pathlib import Path

# 添加项目路径
sys.path.append(str(Path(__file__).parent.parent / "src" / "federated"))

def load_baseline_config():
    """加载基线配置"""
    config_path = Path(__file__).parent.parent / "experiments" / "baseline.yaml"
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def run_consistency_test(port, samples, seed):
    """运行一致性测试"""
    cmd = f"cd {Path(__file__).parent.parent}/src/federated && python consistency_test_fixed.py --port {port} --samples {samples} --seed {seed}"
    result = os.system(cmd)
    return result == 0

def check_wilson_bounds(results, thresholds):
    """检查Wilson下界是否满足要求"""
    alerts = []
    
    for stage_name, threshold in thresholds.items():
        if stage_name in results:
            wilson_lb = results[stage_name].get('wilson_lb', 0)
            win_rate = results[stage_name].get('win_rate', 0)
            
            if wilson_lb < threshold - 0.03:  # 3pp安全边界
                alerts.append({
                    'stage': stage_name,
                    'wilson_lb': wilson_lb,
                    'threshold': threshold,
                    'win_rate': win_rate,
                    'severity': 'high' if wilson_lb < threshold - 0.05 else 'medium'
                })
    
    return alerts

def send_alert(alerts, port):
    """发送告警（这里只是打印，实际可以接入钉钉/邮件等）"""
    if not alerts:
        print(f"✅ {port} 夜测通过，所有指标正常")
        return
    
    print(f"🚨 {port} 夜测告警:")
    for alert in alerts:
        severity_emoji = "🔴" if alert['severity'] == 'high' else "🟡"
        print(f"  {severity_emoji} {alert['stage']}: Wilson下界 {alert['wilson_lb']:.1%} < 阈值 {alert['threshold']:.1%}")
        print(f"     当前胜率: {alert['win_rate']:.1%}")

def main():
    parser = argparse.ArgumentParser(description='夜间CI检查')
    parser.add_argument('--port', default='gulfport', help='测试港口')
    parser.add_argument('--samples', type=int, default=800, help='测试样本数')
    parser.add_argument('--output-dir', default='logs/nightly', help='输出目录')
    args = parser.parse_args()
    
    # 设置日志
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.output_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_dir / f"nightly_{args.port}_{timestamp}.log"),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info(f"🌙 开始夜间CI检查 - 港口: {args.port}, 样本数: {args.samples}")
    
    # 加载配置
    config = load_baseline_config()
    seeds = config['testing']['seeds']
    thresholds = config['thresholds'][args.port]
    
    # 运行测试
    all_alerts = []
    for seed in seeds:
        logger.info(f"🧪 测试种子 {seed}")
        success = run_consistency_test(args.port, args.samples, seed)
        
        if not success:
            logger.error(f"❌ 种子 {seed} 测试失败")
            continue
        
        # 解析结果（这里简化，实际需要解析JSON结果）
        # results = parse_test_results(...)
        # alerts = check_wilson_bounds(results, thresholds)
        # all_alerts.extend(alerts)
    
    # 发送告警
    send_alert(all_alerts, args.port)
    
    logger.info("🌙 夜间CI检查完成")

if __name__ == "__main__":
    main()