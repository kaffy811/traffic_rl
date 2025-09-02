#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
消融与敏感性（无需重训）：
- 读取 reports/FLW_<TAG>/nightly/forced_*.json
- 统计跨seed均值/方差/95%CI，写 CSV 与图
- 若 TAG 名里含 alphaX（如 alpha0.3/alpha0.7），自动做 α-敏感性图
"""

import os, re, glob, json, math
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def wilson_ci(p_hat: float, n: int, z: float = 1.96):
    if n <= 0:
        return (0.0, 0.0)
    denom = 1 + z*z/n
    center = p_hat + z*z/(2*n)
    adj = z * math.sqrt(p_hat*(1-p_hat)/n + (z*z)/(4*n*n))
    lo = max(0.0, (center - adj) / denom)
    hi = min(1.0, (center + adj) / denom)
    return lo, hi

def load_nightly_rows(tags):
    rows = []
    for t in tags:
        for f in glob.glob(f"reports/FLW_{t}/nightly/forced_*.json"):
            try:
                obj = json.load(open(f))
                rows.append({
                    "tag": t,
                    "port": obj.get("port"),
                    "seed": int(obj.get("seed", 0)),
                    "p": float(obj.get("success_rate")),
                    "n": int(obj.get("num_samples", 0)),
                })
            except Exception:
                pass
    return pd.DataFrame(rows)

def parse_alpha(tag: str):
    m = re.search(r"alpha([0-9.]+)", tag.lower())
    if m: 
        try: return float(m.group(1))
        except: return None
    return None

def main():
    STAMP = os.getenv("STAMP", "2025-08-21")
    OUT = Path(f"reports/RELEASE_{STAMP}/ablation")
    OUT.mkdir(parents=True, exist_ok=True)

    tags = [
        os.getenv("BAL","flw_20250821_055747_Balance200"),
        os.getenv("UNB","flw_20250821_060608_Unbalanced200_8_8_vs_2_2"),
        os.getenv("FAIR","flw_20250821_061455_epMix_fair_unbalanced200"),
    ]
    df = load_nightly_rows(tags)
    if df.empty:
        print("未找到 nightly JSON，退出")
        return

    # 1) 跨seed稳定性表：mean/std/CI
    agg = (df.groupby(["tag","port"])
             .apply(lambda g: pd.Series({
                 "mean": g["p"].mean(),
                 "std":  g["p"].std(ddof=1),
                 "n_tot": int(g["n"].sum()),
                 "ci_lo": wilson_ci(np.average(g["p"], weights=g["n"]), int(g["n"].sum()))[0],
                 "ci_hi": wilson_ci(np.average(g["p"], weights=g["n"]), int(g["n"].sum()))[1],
             }))
             .reset_index())
    agg.to_csv(OUT / "seed_stability.csv", index=False)
    print("[OK] 写出:", OUT / "seed_stability.csv")

    # 可视化：每港口跨seed std 对比
    plt.figure(figsize=(8,3), dpi=140)
    for i, port in enumerate(sorted(df["port"].unique())):
        sub = agg[agg.port == port]
        x = np.arange(len(sub))
        plt.bar(x + i*0.25, sub["std"].values, width=0.25, label=port if i==0 else None)
    # 更清晰：每个tag一组
    plt.clf()
    tags_order = tags
    width = 0.18
    ports = sorted(df["port"].unique())
    x = np.arange(len(tags_order))
    for j, port in enumerate(ports):
        s = agg[agg.port == port].set_index("tag").reindex(tags_order)
        plt.bar(x + j*width, s["std"].values, width=width, label=port)
    plt.xticks(x + width*(len(ports)-1)/2, tags_order, rotation=15)
    plt.ylabel("Across-seed Std. of Success Rate")
    plt.title("Seed Sensitivity per Tag (lower is better)")
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUT / "seed_std_per_tag.png")
    plt.close()
    print("[OK] 写出:", OUT / "seed_std_per_tag.png")

    # 2) α-敏感性（若 tag 名包含 alphaX）
    df["alpha"] = df["tag"].map(parse_alpha)
    has_alpha = df["alpha"].notna().any()
    if has_alpha:
        sens = (df.groupby(["alpha","tag","port"])
                  .apply(lambda g: pd.Series({
                      "mean": g["p"].mean(),
                      "n_tot": int(g["n"].sum()),
                  }))
                  .reset_index())

        # 宏平均与 minority 平均
        minority = ["gulfport","south_louisiana"]
        macro = (sens.groupby(["alpha","tag"])
                   .apply(lambda g: pd.Series({
                       "macro": g["mean"].mean(),
                       "minority": g[g.port.isin(minority)]["mean"].mean()
                   }))
                   .reset_index())
        macro.to_csv(OUT / "alpha_sensitivity.csv", index=False)

        plt.figure(figsize=(6,3), dpi=140)
        plt.plot(macro["alpha"], macro["macro"], "-o", label="Macro across ports")
        plt.plot(macro["alpha"], macro["minority"], "-o", label="Minority avg")
        plt.xlabel("alpha (fairness weight)")
        plt.ylabel("Success Rate")
        plt.title("Alpha Sensitivity")
        plt.grid(alpha=0.25, linestyle="--")
        plt.legend()
        plt.tight_layout()
        plt.savefig(OUT / "alpha_sensitivity.png")
        plt.close()
        print("[OK] 写出:", OUT / "alpha_sensitivity.png")
    else:
        print("[INFO] 未检测到 alphaX 形式的TAG，跳过 α-敏感性图。")

    # 3) 聚合策略对比（已有：Unbalanced vs Fair vs Balanced）
    # 做一个“最差港口”指标对比（公平论文常用）
    worst = (df.groupby(["tag","seed"])
               .apply(lambda g: pd.Series({"worst_port_acc": g.groupby("port")["p"].mean().min()}))
               .reset_index())
    wp = worst.groupby("tag")["worst_port_acc"].agg(["mean","std"]).reset_index()
    wp.to_csv(OUT / "worst_port_metric.csv", index=False)
    plt.figure(figsize=(6,3), dpi=140)
    x = np.arange(len(wp))
    plt.bar(x, wp["mean"], yerr=wp["std"], capsize=4)
    plt.xticks(x, wp["tag"], rotation=10)
    plt.ylabel("Min-over-ports Success (↑)")
    plt.title("Fairness (Worst-port) vs Baselines")
    plt.grid(axis="y", alpha=0.25, linestyle="--")
    plt.tight_layout()
    plt.savefig(OUT / "worst_port_metric.png")
    plt.close()
    print("[OK] 写出:", OUT / "worst_port_metric.png")

if __name__ == "__main__":
    main()
