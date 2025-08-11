#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os, sys, re, json, time, argparse, subprocess, logging
from pathlib import Path
from datetime import datetime
from math import sqrt
from glob import glob
import yaml

logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")
log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[1]  # 动态获取仓库根目录
CONSISTENCY_SCRIPT = REPO_ROOT / "src" / "federated" / "consistency_test_fixed.py"
LOG_DIR    = REPO_ROOT / "logs" / "nightly"
REL_DIR    = REPO_ROOT / "models" / "releases"

DEFAULT_PORTS = ["gulfport"]  # 夜测先盯 gulfport；想跑全端就传 --ports all
DEFAULT_SEEDS = [42, 123, 2025]

# 阈值（与当前 CurriculumTrainer 保持一致；可用 --thr-offset 微调）
THRESHOLDS = {
    "baton_rouge": {"基础阶段": 0.41, "中级阶段": 0.45, "高级阶段": 0.39},
    "new_orleans": {"基础阶段": 0.35, "初级阶段": 0.40, "中级阶段": 0.50, "高级阶段": 0.40, "专家阶段": 0.30},
    "south_louisiana": {"基础阶段": 0.41, "中级阶段": 0.45, "高级阶段": 0.39},
    "gulfport": {"标准阶段": 0.49, "完整阶段": 0.37},
}

def load_sample_size_config():
    """加载样本量配置文件"""
    config_path = Path(__file__).parent / "sample_size_config.yaml"
    if config_path.exists():
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except Exception as e:
            log.warning(f"加载样本量配置失败: {e}")
    return {}

def get_optimal_sample_size(port: str, stage: str, default_samples: int = 400) -> int:
    """基于港口和阶段返回最优样本量"""
    config = load_sample_size_config()
    port_config = config.get('ports', {}).get(port, {})
    
    # 阶段名称映射
    stage_mapping = {
        'gulfport': {'标准阶段': 'standard_stage', '完整阶段': 'complete_stage'},
        'new_orleans': {'高级阶段': 'advanced_stage'},
        'south_louisiana': {'高级阶段': 'advanced_stage'},
        'baton_rouge': {'中级阶段': 'intermediate_stage', '高级阶段': 'advanced_stage'}
    }
    
    mapped_stage = stage_mapping.get(port, {}).get(stage, stage)
    return port_config.get(mapped_stage, default_samples)

def wilson_lb(k, n, z=1.96):
    if n == 0: return 0.0
    p = k / n
    denom = 1 + z*z/n
    centre = p + z*z/(2*n)
    margin = z*sqrt((p*(1-p) + z*z/(4*n))/n)
    lb = (centre - margin)/denom
    return max(0.0, min(1.0, lb))

def run_consistency_once(port: str, samples: int, seed: int, timeout: int = 1800, no_cache: bool = False) -> Path | None:
    """调用 consistency_test_fixed.py 并返回它保存的 JSON 路径"""
    cmd = [
        sys.executable, str(CONSISTENCY_SCRIPT),
        "--port", port, "--samples", str(samples), "--seed", str(seed)
    ]
    if no_cache:
        cmd.append("--no-cache")
    log.info("▶️ 运行: %s", " ".join(cmd))
    try:
        # 注意 cwd=REPO_ROOT，避免找不到脚本/模型
        proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        log.error("⏱️ 超时: %s", " ".join(cmd))
        return None
    
    # 检查返回码
    if proc.returncode != 0:
        log.error("❌ 子进程返回码: %d", proc.returncode)
        log.error("stderr: %s", proc.stderr)
        return None

    stdout = proc.stdout
    stderr = proc.stderr
    # 把本次 stdout 和 stderr 附带保存一下
    (LOG_DIR / f"stdout_{port}_{seed}_{int(time.time())}.log").write_text(stdout)
    (LOG_DIR / f"stderr_{port}_{seed}_{int(time.time())}.log").write_text(stderr)
    
    # 合并stdout和stderr来查找输出
    combined_output = stdout + stderr

    # 先尝试从combined_output中提取路径
    m = re.search(r"测试结果已保存到:\s*(.+\.json)", combined_output)
    if m:
        json_path_str = m.group(1).strip()
        # 处理相对路径 ../../models/releases/...
        if json_path_str.startswith("../../"):
            json_path = REPO_ROOT / json_path_str[6:]  # 去掉 "../../"
        else:
            json_path = Path(json_path_str)
        if json_path.exists():
            log.info("📄 结果: %s", json_path)
            return json_path
    
    # 如果stdout中没找到，尝试查找当天最新的JSON文件（仅在不禁用缓存时）
    if not no_cache:
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

def _read_result(path: str):
    """读取结果文件并兼容旧格式"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # 兼容/归一
    for s in data.get("stages", []):
        s["pass"] = bool(s.get("pass", s.get("ok", False)))
    return data

def _summarize_port(result_file: str):
    """基于JSON的pass字段汇总港口状态"""
    data = _read_result(result_file)
    alerts = [s for s in data.get("stages", []) if not s["pass"]]
    return len(alerts), alerts

def main():
    ap = argparse.ArgumentParser(description="Nightly CI for consistency test")
    ap.add_argument("--ports", default="gulfport", help="逗号分隔的港口名或 all")
    ap.add_argument("--samples", type=int, default=400, help="每种子样本数（基础值，会根据港口和阶段自动调整）")
    ap.add_argument("--seeds", default="42,123,2025", help="逗号分隔的种子")
    ap.add_argument("--timeout", type=int, default=1800)
    ap.add_argument("--thr-offset", type=float, default=0.0, help="统一阈值偏移（如 -0.02 放宽2pp）")
    ap.add_argument("--lb-slack", type=float, default=0.03, help="Wilson 下界安全边界（默认 3pp）")
    ap.add_argument("--only-run", action="store_true", help="只执行测试，不做告警判定")
    ap.add_argument("--no-cache", action="store_true", help="禁用缓存，强制重算")
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
            # 根据港口和阶段动态调整样本量
            optimal_samples = get_optimal_sample_size(port, "标准阶段", args.samples)  # 默认使用标准阶段
            log.info(f"港口 {port} 种子 {sd} 使用样本量: {optimal_samples}")
            jp = run_consistency_once(port, optimal_samples, sd, args.timeout, args.no_cache)
            if not jp:
                per_seed.append({"seed": sd, "status": "error"})
                exit_code = max(exit_code, 1)
                continue
            data = load_json(jp)
            # 添加from_cache标记
            if "from_cache" not in data:
                data["from_cache"] = False  # 默认标记为新鲜结果
            per_seed.append({"seed": sd, "json": str(jp), "data": data})

        # 聚合告警 - 使用JSON的pass字段
        worst_alerts = []
        if not args.only_run:
            # 对每个种子各自判定，再取"最坏"的告警集合
            for row in per_seed:
                if "data" not in row: 
                    continue
                alerts_count, alerts = _summarize_port(row["json"])
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
            wr = a.get("win_rate", 0.0)*100
            thr = a.get("threshold", 0.0)*100
            lb = a.get("wilson_lb", 0.0)*100
            log.info("    - %s: wr=%.1f%%, LB=%.1f%% < thr=%.1f%%",
                     a["stage"], wr, lb, thr)

    sys.exit(exit_code)

if __name__ == "__main__":
    main()