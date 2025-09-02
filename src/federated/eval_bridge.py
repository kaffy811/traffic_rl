# src/federated/eval_bridge.py
from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

# 仓库根
REPO_ROOT = Path(__file__).resolve().parents[2]


def _date_dirs(base: Path) -> List[Path]:
    """返回 releases 下的日期目录，按时间新→旧排."""
    if not base.exists():
        return []
    cand = [p for p in base.iterdir() if p.is_dir()]
    def to_ts(p: Path):
        try:
            return time.strptime(p.name, "%Y-%m-%d")
        except Exception:
            return time.gmtime(0)
    return sorted(cand, key=to_ts, reverse=True)


def _latest_json_for(port: str) -> Optional[Path]:
    """查找最近的 consistency_{port}_*.json."""
    base = REPO_ROOT / "models" / "releases"
    for daydir in _date_dirs(base):
        files = sorted(
            daydir.glob(f"consistency_{port}_*.json"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if files:
            return files[0]
    return None


def _parse_consistency_json(p: Path) -> Optional[Dict[str, Any]]:
    """解析 consistency JSON，产出 {port, success_rate, avg_reward(None), num_samples, source}."""
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

    stages = data.get("stages") or []
    win_rates: List[float] = []
    weights: List[float] = []

    for st in stages:
        # 兼容两种结构：[(name, wr, thr, ok), ...] 或 [{'win_rate':..,'n_samples':..}, ...]
        if isinstance(st, (list, tuple)) and len(st) >= 2:
            wr = st[1]
            n = st[3] if len(st) >= 4 and isinstance(st[3], (int, float)) else None
        elif isinstance(st, dict):
            wr = st.get("win_rate") or st.get("completion_rate") or st.get("accuracy")
            n = st.get("n_samples")
        else:
            wr, n = None, None

        if wr is None:
            continue
        try:
            wr = float(wr)
        except Exception:
            continue

        win_rates.append(wr)
        weights.append(float(n) if isinstance(n, (int, float)) and n else 1.0)

    if not win_rates:
        return None

    totw = sum(weights)
    if totw > 0:
        sr = sum(w * wr for w, wr in zip(weights, win_rates)) / totw
    else:
        sr = sum(win_rates) / len(win_rates)

    # 若 stages 里提供了 n_samples 就求和，否则留 0
    num = 0
    for st in stages:
        if isinstance(st, dict):
            n = st.get("n_samples")
            if isinstance(n, (int, float)):
                num += int(n)

    return {
        "port": data.get("port"),
        "success_rate": float(sr),
        "avg_reward": None,
        "num_samples": int(num),
        "source": str(p),
    }


def eval_port_via_consistency(port: str, samples: int = 200) -> Dict[str, Any]:
    """
    从最近的 consistency_{port}_*.json 读取并计算 success_rate。
    """
    p = _latest_json_for(port)
    if p is None:
        raise FileNotFoundError(f"No consistency JSON found for port={port}")
    out = _parse_consistency_json(p)
    if out is None:
        raise RuntimeError(f"Cannot parse metrics from {p}")
    return out


def eval_port_with_fed_mlp(
    port: str,
    ckpt: str,
    samples: int = 200,
    seed: int = 42,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    用联邦 MLP 的 global_round_*.pt 做直评。
    依赖 src/federated/fed_mlp_adapter.py，要求其中存在：
      - evaluate_with_ckpt(...) 或
      - eval_with_ckpt(...)
    返回统一字段：{port, success_rate, avg_reward, num_samples, source}
    """
    if not ckpt:
        raise FileNotFoundError("ckpt path is empty")
    ckpt_path = Path(ckpt)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")

    # 导入适配器（两个函数名任选其一）
    adapter_eval = None
    try:
        from .fed_mlp_adapter import evaluate_with_ckpt as adapter_eval  # type: ignore
    except Exception:
        try:
            from .fed_mlp_adapter import eval_with_ckpt as adapter_eval  # type: ignore
        except Exception as e:
            raise ImportError(
                "fed_mlp_adapter.py must provide evaluate_with_ckpt(...) or eval_with_ckpt(...)"
            ) from e

    # 调用适配器
    result = adapter_eval(
        port=port,
        ckpt_path=str(ckpt_path),
        samples=int(samples),
        seed=int(seed),
        verbose=bool(verbose),
    )

    # 归一化字段
    sr = (
        result.get("success_rate")
        or result.get("win_rate")
        or result.get("accuracy")
    )
    try:
        sr_f = float(sr) if sr is not None else None
    except Exception:
        sr_f = None

    num = result.get("num_samples") or result.get("samples") or samples

    out = {
        "port": port,
        "success_rate": sr_f,
        "avg_reward": result.get("avg_reward") or result.get("reward"),
        "num_samples": int(num) if isinstance(num, (int, float)) else samples,
        "source": str(ckpt_path),
    }
    return out
