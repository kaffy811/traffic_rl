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
        print(f"{mark} {p} | alerts={len(st.get('alerts',[]))}")
        for a in st.get("alerts", []):
            print(f"    - {a['stage']}: wr={a['win_rate']*100:.1f}% | LB={a['wilson_lb']*100:.1f}% | thr={a['thr_config']*100:.1f}%")
    print("-----------------------------------------------------")
    if hist.exists():
        lines = hist.read_text().strip().splitlines()[-6:]  # 最近5条 + 头
        print("最近历史：")
        for ln in lines:
            print("  ", ln)
    print("=====================================================\n")

if __name__ == "__main__":
    main()