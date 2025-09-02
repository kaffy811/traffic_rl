#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按轮评测曲线（每K轮），画 per-port 和宏平均收敛图
- 读取 models/flw/<TAG>/global_round_*.pt
- 每 stride 轮评测一次（默认10）
- 调用 eval_port_with_fed_mlp(port, ckpt, samples, seed)
- 产出 CSV + PNG 到 reports/RELEASE_<STAMP>/curves/
"""

import os, re, glob, json, argparse, math
from pathlib import Path
from typing import List, Dict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.federated.eval_bridge import eval_port_with_fed_mlp

def natural_sort_key(s: str):
    m = re.search(r"(\d+)", s)
    return int(m.group(1)) if m else -1

def wilson_ci(p_hat: float, n: int, z: float = 1.96):
    if n <= 0:
        return (0.0, 0.0)
    denom = 1 + z*z/n
    center = p_hat + z*z/(2*n)
    adj = z * math.sqrt(p_hat*(1-p_hat)/n + (z*z)/(4*n*n))
    lo = max(0.0, (center - adj) / denom)
    hi = min(1.0, (center + adj) / denom)
    return lo, hi

def eval_ckpt_for_ports(tag: str, ckpt: str, ports: List[str], seeds: List[int], samples: int) -> List[Dict]:
    out = []
    for port in ports:
        acc = []
        n_sum = 0
        for sd in seeds:
            res = eval_port_with_fed_mlp(port, ckpt, samples=samples, seed=sd, verbose=False)
            sr = res.get("success_rate")
            n  = int(res.get("num_samples", samples))
            if sr is not None:
                acc.append((sr, n))
                n_sum += n
        if acc:
            # 加权平均 p̂
            num = sum(sr*n for sr, n in acc)
            den = sum(n for _, n in acc)
            p_hat = num / max(1, den)
            lo, hi = wilson_ci(p_hat, den)
            out.append(dict(tag=tag, ckpt=ckpt, port=port, p_hat=p_hat, n=den, ci_lo=lo, ci_hi=hi))
        else:
            out.append(dict(tag=tag, ckpt=ckpt, port=port, p_hat=np.nan, n=0, ci_lo=np.nan, ci_hi=np.nan))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tags", nargs="+", required=False,
                    default=[
                        os.getenv("BAL","flw_20250821_055747_Balance200"),
                        os.getenv("UNB","flw_20250821_060608_Unbalanced200_8_8_vs_2_2"),
                        os.getenv("FAIR","flw_20250821_061455_epMix_fair_unbalanced200"),
                    ], help="实验TAG列表（文件夹名）")
    ap.add_argument("--ports", nargs="+", default=["gulfport","south_louisiana","baton_rouge","new_orleans"])
    ap.add_argument("--stride", type=int, default=10, help="每几轮评一次")
    ap.add_argument("--samples", type=int, default=int(os.getenv("FLW_EVAL_SAMPLES", "400")))
    ap.add_argument("--seeds", nargs="+", type=int, default=[42], help="默认只用一个seed，快")
    ap.add_argument("--stamp", default=os.getenv("STAMP","2025-08-21"))
    args = ap.parse_args()

    repo = Path(".").resolve()
    out_dir = repo / f"reports/RELEASE_{args.stamp}/curves"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for tag in args.tags:
        ckpts = sorted(glob.glob(f"models/flw/{tag}/global_round_*.pt"), key=natural_sort_key)
        if not ckpts:
            print(f"[WARN] 无 ckpt: {tag}")
            continue
        # 按 stride 取子集
        pick = [c for i, c in enumerate(ckpts, start=1) if (i % args.stride == 0 or i == len(ckpts))]
        print(f"[{tag}] 评测 ckpt 数: {len(pick)} / {len(ckpts)} (stride={args.stride})")
        for ck in pick:
            rd = natural_sort_key(Path(ck).name)
            rs = eval_ckpt_for_ports(tag, ck, args.ports, args.seeds, args.samples)
            for r in rs:
                r["round"] = rd
                rows.append(r)

    if not rows:
        print("无数据可画，退出")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "curve_all.csv", index=False)
    print("[OK] 写出:", out_dir / "curve_all.csv")

    # per-port 曲线
    for port in args.ports:
        sub = df[df.port == port].copy()
        if sub.empty: 
            continue
        plt.figure(figsize=(8,3.2), dpi=140)
        for tag in args.tags:
            g = sub[sub.tag == tag].sort_values("round")
            if g.empty: 
                continue
            plt.plot(g["round"], g["p_hat"], marker="o", label=tag, linewidth=2)
        plt.title(f"Training Curve: {port}")
        plt.xlabel("Round"); plt.ylabel("Success Rate")
        plt.ylim(0.35, 0.6); plt.grid(alpha=0.25, linestyle="--")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / f"curve_{port}.png")
        plt.close()
        print("[OK] 写出:", out_dir / f"curve_{port}.png")

    # 宏平均曲线（四港口平均）
    macro = (df.groupby(["tag","round"])
               .apply(lambda g: pd.Series({"p_hat": np.average(g["p_hat"], weights=g["n"])}))
               .reset_index())
    plt.figure(figsize=(8,3.2), dpi=140)
    for tag in args.tags:
        g = macro[macro.tag == tag].sort_values("round")
        if g.empty: continue
        plt.plot(g["round"], g["p_hat"], marker="o", label=tag, linewidth=2)
    plt.title("Training Curve: Macro-average across ports")
    plt.xlabel("Round"); plt.ylabel("Success Rate")
    plt.ylim(0.35, 0.6); plt.grid(alpha=0.25, linestyle="--")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out_dir / f"curve_macro.png")
    plt.close()
    print("[OK] 写出:", out_dir / "curve_macro.png")

if __name__ == "__main__":
    main()
