#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, re, json, time, argparse, subprocess, logging
from pathlib import Path
from datetime import datetime
from math import sqrt
from glob import glob

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path("/Users/kaffy/Documents/GAT-FedPPO")   # 绝对路径
RUN_DIR    = REPO_ROOT                             
LOG_DIR    = REPO_ROOT / "logs" / "nightly"
REL_DIR    = REPO_ROOT / "models" / "releases"

DEFAULT_PORTS = ["gulfport"]  # 夜测先盯 gulfport；想跑全端就传 --ports all
DEFAULT_SEEDS = [42, 123, 2025]

# 阈值（与当前 CurriculumTrainer 保持一致；可用 --thr-offset 微调）
THRESHOLDS = {
    "baton_rouge": {"基础阶段": 0.41, "中级阶段": 0.45, "高级阶段": 0.39},
    "new_orleans": {"基础阶段": 0.35, "初级阶段": 0.40, "中级阶段": 0.50, "高级阶段": 0.40, "专家阶段": 0.30},
    "south_louisiana": {"基础阶段": 0.41, "中级阶段": 0.45, "高级阶段": 0.39},
    "gulfport": {"标准阶段": 0.49, "完整阶段": 0.38},
}

def wilson_lb(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n
    denom = 1 + z*z/n
    centre = p + z*z/(2*n)
    margin = z*sqrt((p*(1-p) + z*z/(4*n))/n)
    lb = (centre - margin)/denom
    return max(0.0, min(1.0, lb))

def run_consistency_once(port: str, samples: int, seed: int, timeout: int = 1800) -> Path | None:
    """调用 consistency_test_fixed.py 并返回它保存的 JSON 路径"""
    cmd = [
        sys.executable, "consistency_test_fixed.py",
        "--port", port, "--samples", str(samples), "--seed", str(seed)
    ]
    log.info("▶️ 运行: %s", " ".join(cmd))
    try:
        # 在src/federated目录下运行
        proc = subprocess.run(cmd, cwd=RUN_DIR / "src" / "federated", capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log.error("⏱️ 超时: %s", " ".join(cmd))
        return None

    stdout = proc.stdout
    # 把本次 stdout 附带保存一下
    (LOG_DIR / f"stdout_{port}_{seed}_{int(time.time())}.log").write_text(stdout)

    # 先尝试从stdout中提取路径
    m = re.search(r"测试结果已保存到:\s*(.+\.json)", stdout)
    if m:
        json_path = Path(m.group(1)).resolve()
        if json_path.exists():
            log.info("📄 结果: %s", json_path)
            return json_path
    
    # 如果stdout中没找到，尝试查找当天最新的JSON文件
    today_dir = REL_DIR / datetime.now().strftime("%Y-%m-%d")
    if today_dir.exists():
        pattern = f"consistency_{port}_*.json"
        json_files = sorted(today_dir.glob(pattern), key=lambda x: x.stat().st_mtime, reverse=True)
        if json_files:
            log.info("📄 兜底找到结果: %s", json_files[0])
            return json_files[0]
    
    log.error("❌ 未找到测试结果JSON文件")
    return None

def collect_latest_jsons(port: str) -> list[Path]:
    """兜底：当天 releases 目录下该港口的 JSON（按时间排序）"""
    today_dir = REL_DIR / datetime.now().strftime("%Y-%m-%d")
    files = sorted(today_dir.glob(f"consistency_{port}_*.json"))
    return files

def load_json(p: Path) -> dict:
    try:
        return json.loads(p.read_text())
    except Exception:
        return {}

def decide_alert(port: str, stages: list[dict], samples: int, thr_offset: float, lb_slack: float):
    """基于 Wilson 下界做告警判定"""
    alerts = []
    thr_cfg = THRESHOLDS.get(port, {})
    for st in stages:
        name = st.get("stage", "")
        wr = float(st.get("win_rate", 0.0))
        thr = float(st.get("threshold", thr_cfg.get(name, 0.0))) + thr_offset
        n = samples
        k = int(round(wr * n))
        lb = wilson_lb(k, n)
        ok = (lb >= max(0.0, thr - lb_slack))
        if not ok:
            alerts.append({
                "stage": name,
                "win_rate": wr,
                "thr_config": thr,
                "wilson_lb": lb
            })
    return alerts

def main():
    ap = argparse.ArgumentParser(description="Nightly CI for consistency test")
    ap.add_argument("--ports", default="gulfport", help="逗号分隔的港口名或 all")
    ap.add_argument("--samples", type=int, default=800, help="每种子样本数")
    ap.add_argument("--seeds", default="42,123,2025", help="逗号分隔的种子")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--thr-offset", type=float, default=0.0, help="统一阈值偏移（如 -0.02 放宽2pp）")
    ap.add_argument("--lb-slack", type=float, default=0.03, help="Wilson 下界安全边界（默认 3pp）")
    ap.add_argument("--only-run", action="store_true", help="只执行测试，不做告警判定")
    args = ap.parse_args()

    LOG_DIR.mkdir(parents=True, exist_ok=True)

    ports = [p.strip() for p in (DEFAULT_PORTS if args.ports.lower() == "gulfport"
              else (THRESHOLDS.keys() if args.ports.lower()=="all" else args.ports.split(",")))]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    run_stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    run_id = int(time.time())

    run_summary = {
        "run_time": run_stamp,
        "ports": {},
        "samples": args.samples,
        "seeds": seeds,
        "thr_offset": args.thr_offset,
        "lb_slack": args.lb_slack,
    }

    exit_code = 0
    for port in ports:
        per_seed = []
        for sd in seeds:
            jp = run_consistency_once(port, args.samples, sd, args.timeout)
            if not jp:
                per_seed.append({"seed": sd, "status": "error"})
                exit_code = max(exit_code, 1)
                continue
            data = load_json(jp)
            per_seed.append({"seed": sd, "json": str(jp), "data": data})

        # 聚合告警
        worst_alerts = []
        if not args.only_run:
            # 对每个种子各自判定，再取"最坏"的告警集合
            for row in per_seed:
                if "data" not in row: 
                    continue
                stages = row["data"].get("stages", [])
                alerts = decide_alert(port, stages, args.samples, args.thr_offset, args.lb_slack)
                if len(alerts) > len(worst_alerts):
                    worst_alerts = alerts

        port_status = {
            "seeds": per_seed,
            "alerts": worst_alerts,
            "ok": len(worst_alerts) == 0
        }
        run_summary["ports"][port] = port_status
        if not port_status["ok"]:
            exit_code = max(exit_code, 2)

    # 写状态与历史
    (LOG_DIR / "monitoring_status.json").write_text(json.dumps(run_summary, indent=2, ensure_ascii=False))
    hist = LOG_DIR / "history.csv"
    if not hist.exists():
        hist.write_text("ts,port,ok,alerts,seeds,samples,thr_offset,lb_slack\n")
    for port, st in run_summary["ports"].items():
        hist.open("a").write(
            f'{run_stamp},{port},{int(st["ok"])},{len(st["alerts"])},{"/".join(map(str,seeds))},{args.samples},{args.thr_offset},{args.lb_slack}\n'
        )

    # 控制台提示
    log.info("\n===== 夜测总结 @ %s =====", run_stamp)
    for port, st in run_summary["ports"].items():
        mark = "✅" if st["ok"] else "❌"
        log.info("  %s %s (%d alert)", port, mark, len(st["alerts"]))
        for a in st["alerts"]:
            log.info("    - %s: wr=%.1f%%, LB=%.1f%% < thr=%.1f%%",
                     a["stage"], a["win_rate"]*100, a["wilson_lb"]*100, a["thr_config"]*100)

    sys.exit(exit_code)

if __name__ == "__main__":
    main()