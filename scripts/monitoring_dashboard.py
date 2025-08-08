#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
监控仪表板
查看夜间CI的运行状态和历史结果
"""
import json
import glob
import os
from datetime import datetime, timedelta
from pathlib import Path

def load_monitoring_status():
    """加载监控状态"""
    status_file = Path("logs/nightly/monitoring_status.json")
    if status_file.exists():
        with open(status_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def get_recent_logs(days=7):
    """获取最近几天的日志"""
    log_dir = Path("logs/nightly")
    if not log_dir.exists():
        return []
    
    cutoff_date = datetime.now() - timedelta(days=days)
    logs = []
    
    for log_file in log_dir.glob("nightly_*.log"):
        try:
            # 从文件名提取时间戳
            timestamp_str = log_file.stem.split('_')[-1]
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            
            if timestamp >= cutoff_date:
                logs.append({
                    'file': log_file,
                    'timestamp': timestamp,
                    'size': log_file.stat().st_size
                })
        except:
            continue
    
    return sorted(logs, key=lambda x: x['timestamp'], reverse=True)

def check_crontab_status():
    """检查crontab状态"""
    result = os.popen("crontab -l 2>/dev/null | grep nightly_ci").read()
    return bool(result.strip())

def display_dashboard():
    """显示监控仪表板"""
    print("🌙 GAT-FedPPO 夜间CI监控仪表板")
    print("=" * 50)
    
    # 监控状态
    status = load_monitoring_status()
    if status:
        config = status['configuration']
        print(f"📊 监控状态: {'🟢 启用' if status['monitoring']['enabled'] else '🔴 禁用'}")
        print(f"🏷️  版本: {status['monitoring']['version']}")
        print(f"⏰ 调度: {status['monitoring']['schedule']} (每晚2点)")
        print(f"🚢 测试港口: {config['port']}")
        print(f"📈 样本数: {config['samples']}")
        print(f"🎲 种子: {config['seeds']}")
        print()
        
        # 阈值信息
        print("🎯 当前阈值:")
        for stage, threshold in config['thresholds'].items():
            print(f"   {stage}: {threshold:.1%}")
        print()
    
    # Crontab状态
    cron_active = check_crontab_status()
    print(f"⏰ Crontab状态: {'🟢 已安装' if cron_active else '🔴 未安装'}")
    
    # 最近日志
    recent_logs = get_recent_logs(7)
    print(f"\n📋 最近7天日志 ({len(recent_logs)}个文件):")
    
    if recent_logs:
        for log in recent_logs[:5]:  # 只显示最近5个
            size_kb = log['size'] / 1024
            print(f"   📄 {log['timestamp'].strftime('%Y-%m-%d %H:%M')} - {size_kb:.1f}KB")
    else:
        print("   📭 暂无日志文件")
    
    # 下次运行时间
    now = datetime.now()
    if now.hour >= 2:
        next_run = now.replace(hour=2, minute=0, second=0, microsecond=0) + timedelta(days=1)
    else:
        next_run = now.replace(hour=2, minute=0, second=0, microsecond=0)
    
    print(f"\n⏭️  下次运行: {next_run.strftime('%Y-%m-%d %H:%M')}")
    
    # 操作提示
    print("\n🔧 管理命令:")
    print("   查看最新日志: tail -f logs/nightly/cron.log")
    print("   手动测试: python scripts/nightly_ci.py --port gulfport --samples 100")
    print("   停用监控: crontab -r")
    print("   重新启用: crontab scripts/crontab.active")

if __name__ == "__main__":
    display_dashboard()