#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, glob, time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULT_ROOT = REPO_ROOT / "models" / "releases"

def _latest_result_path(port: str):
    pats = sorted(glob.glob(str(RESULT_ROOT / "*" / f"consistency_{port}_*.json")),
                  key=os.path.getmtime)
    return Path(pats[-1]) if pats else None

def _load_result(p: Path):
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 兼容：补默认值（老结果不含新字段）
    for s in data.get("stages", []):
        s["pass"] = bool(s.get("pass", s.get("ok", False)))
        s["wilson_lb"] = float(s.get("wilson_lb", s.get("wilson_lower_bound", 0.0)))
        s["n_samples"] = int(s.get("n_samples", s.get("n", 0)))
        s["k_baseline"] = int(s.get("k_baseline", s.get("k", 0)))
        s["threshold_source"] = s.get("threshold_source", "default")
        s["recheck_used"] = bool(s.get("recheck_used", False))
        s["win_rate"] = float(s.get("win_rate", 0.0))
        s["threshold"] = float(s.get("threshold", 0.0))
    data["from_cache"] = bool(data.get("from_cache", False))
    return data

def _print_port(port: str):
    rp = _latest_result_path(port)
    if not rp:
        print(f"🔴 {port} | 无结果"); return
    data = _load_result(rp)
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(rp)))
    status = "🟢 fresh" if not data["from_cache"] else "🟡 cached"
    alerts = [s for s in data.get("stages", []) if not s["pass"]]
    color = "🟢" if not alerts else "🔴"
    print(f"{color} {port} | alerts={len(alerts)} | {status} | 文件时间: {ts}")
    for s in data["stages"]:
        print(
          f"    - {s['stage']}: wr={s['win_rate']*100:.1f}% | "
          f"LB={s['wilson_lb']*100:.1f}% | "
          f"thr={s['threshold']*100:.1f}% ({s['threshold_source']}) | "
          f"n={s['n_samples']},k={s['k_baseline']} | "
          f"recheck={'yes' if s['recheck_used'] else 'no'}"
        )

def main():
    ports = ["baton_rouge", "new_orleans", "south_louisiana", "gulfport"]
    print("\n================= Nightly Dashboard =================")
    # 最近一次文件时间
    latest = sorted(glob.glob(str(RESULT_ROOT / "*" / "consistency_*.json")),
                    key=os.path.getmtime)
    last_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(latest[-1]))) if latest else "N/A"
    print(f"最近运行时间: {last_ts}")
    print("-----------------------------------------------------")
    for p in ports: _print_port(p)
    print("-----------------------------------------------------")

if __name__ == "__main__":
    main()