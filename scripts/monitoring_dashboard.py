#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json, sys
from pathlib import Path
from datetime import datetime

ROOT = Path("/Users/kaffy/Documents/GAT-FedPPO")
LOG_DIR = ROOT / "logs" / "nightly"

def main():
    status = LOG_DIR / "monitoring_status.json"
    hist = LOG_DIR / "history.csv"
    if not status.exists():
        print("⚠️ 未发现监控状态文件，请先跑 nightly_ci.py")
        sys.exit(0)
    data = json.loads(status.read_text())

    print("\n================= Nightly Dashboard =================")
    print(f"最近运行时间: {data.get('run_time')}")
    print(f"样本/种子: {data.get('samples')} / {data.get('seeds')}")
    print(f"阈值偏移: {data.get('thr_offset',0)} | LB 安全边界: {data.get('lb_slack',0)}")
    print("-----------------------------------------------------")
    ports = data.get("ports", {})
    for p, st in ports.items():
        mark = "🟢" if st.get("ok") else "🔴"
        
        # 检查是否有缓存结果
        has_cached = False
        for seed_data in st.get("seeds", []):
            if seed_data.get("data", {}).get("from_cache", False):
                has_cached = True
                break
        
        # 显示缓存状态
        cache_flag = "🟡 cached" if has_cached else "🟢 fresh"
        print(f"{mark} {p} | alerts={len(st.get('alerts',[]))} | {cache_flag}")
        
        for a in st.get("alerts", []):
            # 获取详细数据
            stage_data = a.get('data', {})
            win_rate = a.get('win_rate', 0)
            wilson_lb = stage_data.get('wilson_lb', 0)
            threshold = a.get('thr_config', 0)
            threshold_source = stage_data.get('threshold_source', 'default')
            n_samples = stage_data.get('n_samples', 0)
            k_baseline = stage_data.get('k_baseline', 0)
            recheck_used = stage_data.get('recheck_used', False)
            
            # 显示详细信息
            thr_display = f"thr={threshold*100:.1f}% ({threshold_source})"
            n_k_display = f"n={n_samples},k={k_baseline}"
            recheck_display = "recheck=yes" if recheck_used else "recheck=no"
            
            print(f"    - {a['stage']}: wr={win_rate*100:.1f}% | LB={wilson_lb*100:.1f}% | {thr_display} | {n_k_display} | {recheck_display}")
    print("-----------------------------------------------------")
    if hist.exists():
        lines = hist.read_text().strip().splitlines()[-6:]  # 最近5条 + 头
        print("最近历史：")
        for ln in lines:
            print("  ", ln)
    print("=====================================================\n")

if __name__ == "__main__":
    main()