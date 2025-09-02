#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
plot_pubready_suite_v2.py
- 输入: by_port_mean.csv, minority_gain.csv, per_port_wilson_ci.csv
- 输出(默认到 reports/RELEASE_${STAMP}/ 或自动回退到 reports/SUMMARY_JSON/):
  1) pub_per_port_ci.png            # 各港口三组实验 + 95% CI
  2) pub_forest_gain.png            # (Fair - Unbalanced) 提升的 forest plot + 95% CI
  3) pub_minority_bars.png          # 少数域(2/2)的提升柱状图 + 误差线
  4) pub_overall_macro.png          # 各实验宏平均(跨港口) + 误差
  5) pub_table_overview.csv         # 表格汇总（含相对提升%）
  6) README_plot_suite.txt          # 输出说明
"""
import os
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.rcParams.update({
    "figure.dpi": 140,
    "savefig.dpi": 140,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white",
})
import matplotlib.pyplot as plt

# ------------------------ I/O 目录与读表 ------------------------
STAMP = os.getenv("STAMP", "").strip()
CANDIDATES = []
if STAMP:
    CANDIDATES.append(Path(f"reports/RELEASE_{STAMP}"))
CANDIDATES.append(Path("reports/SUMMARY_JSON"))  # fallback

IN_DIR = next((p for p in CANDIDATES if p.exists()), None)
if IN_DIR is None:
    raise SystemExit("❌ 找不到输入目录：既没有 reports/RELEASE_${STAMP} 也没有 reports/SUMMARY_JSON")

OUT_DIR = IN_DIR  # 输出就放同一目录，便于打包发布
print(f"[I/O] INPUT = {IN_DIR}")
print(f"[I/O] OUTPUT = {OUT_DIR}")

# 必要文件
f_byport   = IN_DIR / "by_port_mean.csv"
f_minority = IN_DIR / "minority_gain.csv"
f_wilson   = IN_DIR / "per_port_wilson_ci.csv"

for f in [f_byport, f_minority, f_wilson]:
    if not f.exists():
        raise SystemExit(f"❌ 缺少必要文件: {f}")

byport = pd.read_csv(f_byport)               # columns: port, Balanced200, FairUnbalanced200, Unbalanced200
minor  = pd.read_csv(f_minority)             # columns: port, Unbalanced200, FairUnbalanced200, fair_gain
ci     = pd.read_csv(f_wilson)               # columns: port, tag, p_hat, n, CI_low, CI_high

# 统一实验名称顺序/配色
EXPS = ["Balanced200", "FairUnbalanced200", "Unbalanced200"]
PALETTE = {
    "Balanced200":       "#4C78A8",
    "FairUnbalanced200": "#F58518",
    "Unbalanced200":     "#54A24B",
}

# ------------------------ 工具函数 ------------------------
def _ensure_order_ports(df):
    # 使用 by_port_mean 的行顺序，保证各图一致
    return list(byport["port"].tolist())

def _wilson_se_from_pn(p: float, n: int) -> float:
    """用二项分布近似的标准误差 sqrt(p*(1-p)/n)"""
    n = max(int(n), 1)
    return float(np.sqrt(max(p*(1-p), 0.0) / n))

def _diff_ci(p1, n1, p2, n2, z=1.96):
    """两比例差的 95% CI： (p1 - p2) ± z*sqrt(se1^2 + se2^2)"""
    se1 = _wilson_se_from_pn(p1, n1)
    se2 = _wilson_se_from_pn(p2, n2)
    diff = p1 - p2
    se   = np.sqrt(se1**2 + se2**2)
    return diff, diff - z*se, diff + z*se

# 方便从 per_port_wilson_ci 取到 (p, n)
def _p_n(ci_df, port, tag):
    row = ci_df[(ci_df["port"] == port) & (ci_df["tag"] == tag)]
    if row.empty:
        return None, None
    return float(row["p_hat"].iloc[0]), int(row["n"].iloc[0])

PORTS = _ensure_order_ports(ci)

# ------------------------ (1) 各港口 + 95% CI ------------------------
def plot_per_port_ci():
    # 组装矩阵
    width = 0.25
    xs = np.arange(len(PORTS))
    fig, ax = plt.subplots(figsize=(7.2, 3.6))

    for i, exp in enumerate(EXPS):
        means, lo_err, hi_err = [], [], []
        for port in PORTS:
            p, n = _p_n(ci, port, exp)
            if p is None:
                p, n = float(byport.loc[byport["port"] == port, exp]), 1000
            se = _wilson_se_from_pn(p, n)
            means.append(p)
            lo_err.append(1.96*se)
            hi_err.append(1.96*se)
        pos = xs + (i-1)*width
        ax.bar(pos, means, width=width, color=PALETTE[exp], edgecolor="#333", linewidth=0.6, label=exp)
        ax.errorbar(pos, means, yerr=[lo_err, hi_err], fmt="none", ecolor="#333", capsize=3, lw=1)

    ax.set_xticks(xs)
    ax.set_xticklabels(PORTS, rotation=15)
    ax.set_ylabel("Success Rate")
    ax.set_title("Per-port Success Rate with 95% CIs")
    ax.legend(ncol=3, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.20))
    fig.tight_layout()
    out = OUT_DIR / "pub_per_port_ci.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")

# ------------------------ (2) Forest plot: Fair - Unbalanced ------------------------
def plot_forest_gain():
    rows = []
    for port in PORTS:
        pf, nf = _p_n(ci, port, "FairUnbalanced200")
        pu, nu = _p_n(ci, port, "Unbalanced200")
        if pf is None or pu is None:
            # 兜底：没有ci就从byport取均值，n 设一个保守值
            pf = float(byport.loc[byport["port"] == port, "FairUnbalanced200"])
            pu = float(byport.loc[byport["port"] == port, "Unbalanced200"])
            nf = nu = 1000
        d, lo, hi = _diff_ci(pf, nf, pu, nu)
        rows.append({"port": port, "diff": d, "lo": lo, "hi": hi})
    fdf = pd.DataFrame(rows).set_index("port").loc[PORTS]

    fig, ax = plt.subplots(figsize=(6.8, 3.8))
    y = np.arange(len(fdf))[::-1]
    ax.hlines(y, fdf["lo"], fdf["hi"], color="#444", lw=2)
    ax.plot(fdf["diff"], y, "o", color="#D62728", markersize=6)
    ax.axvline(0, color="#999", lw=1, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(fdf.index)
    ax.set_xlabel("Effect size: Fair − Unbalanced (Δ Success Rate)")
    ax.set_title("Forest Plot of Fairness Gains (95% CI)")
    fig.tight_layout()
    out = OUT_DIR / "pub_forest_gain.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")

# ------------------------ (3) 少数域提升柱状图 ------------------------
def plot_minority_bars():
    # minority_gain.csv: port, Unbalanced200, FairUnbalanced200, fair_gain
    m = minor.copy()
    # 计算误差线（基于 ci 的两比例差）
    errs = []
    for _, r in m.iterrows():
        port = r["port"]
        pf, nf = _p_n(ci, port, "FairUnbalanced200")
        pu, nu = _p_n(ci, port, "Unbalanced200")
        if pf is None or pu is None:
            pf, pu = float(r["FairUnbalanced200"]), float(r["Unbalanced200"])
            nf = nu = 1000
        d, lo, hi = _diff_ci(pf, nf, pu, nu)
        errs.append((d, d-lo, hi-d))
    diffs, loerr, hierr = zip(*errs)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    xs = np.arange(len(m))
    ax.bar(xs, diffs, color="#F58518", edgecolor="#333", linewidth=0.6)
    ax.errorbar(xs, diffs, yerr=[loerr, hierr], fmt="none", ecolor="#333", capsize=3, lw=1)
    ax.axhline(0, color="#999", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(m["port"], rotation=15)
    ax.set_ylabel("Fair − Unbalanced (Δ Success Rate)")
    ax.set_title("Minority Ports: Fairness Uplift (95% CI)")
    fig.tight_layout()
    out = OUT_DIR / "pub_minority_bars.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")

# ------------------------ (4) 宏平均 (跨港口) ------------------------
def plot_overall_macro():
    # 用 by_port_mean 按列对各实验取“跨港口的均值”作为宏平均
    macro_mean = byport[EXPS].mean(axis=0)
    # 标准误：跨港口的 std / sqrt(K)
    K = len(byport)
    macro_se = byport[EXPS].std(axis=0, ddof=1) / np.sqrt(max(K,1))
    lo = macro_mean - 1.96*macro_se
    hi = macro_mean + 1.96*macro_se

    xs = np.arange(len(EXPS))
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ys = macro_mean.values
    ax.bar(xs, ys, color=[PALETTE[e] for e in EXPS], edgecolor="#333", linewidth=0.6)
    yerr = np.vstack([ys - lo.values, hi.values - ys])
    ax.errorbar(xs, ys, yerr=yerr, fmt="none", ecolor="#333", capsize=3, lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(EXPS, rotation=15)
    ax.set_ylabel("Macro-average Success Rate")
    ax.set_title("Overall Comparison (Macro across Ports)")
    fig.tight_layout()
    out = OUT_DIR / "pub_overall_macro.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"[OK] {out}")

# ------------------------ (5) 输出总览表 ------------------------
def write_overview_table():
    out = OUT_DIR / "pub_table_overview.csv"
    tbl = byport.copy()
    # 绝对提升 & 相对提升(%)
    tbl["FairGain_abs"] = tbl["FairUnbalanced200"] - tbl["Unbalanced200"]
    tbl["FairGain_rel_%"] = np.where(
        tbl["Unbalanced200"] > 1e-8,
        100.0 * tbl["FairGain_abs"] / tbl["Unbalanced200"],
        np.nan,
    )
    tbl.to_csv(out, index=False)
    print(f"[OK] {out}")

# ------------------------ 主流程 ------------------------
if __name__ == "__main__":
    plot_per_port_ci()
    plot_forest_gain()
    plot_minority_bars()
    plot_overall_macro()
    write_overview_table()

    # 小的说明文件
    readme = OUT_DIR / "README_plot_suite.txt"
    readme.write_text(
        "This folder contains publication-ready figures:\n"
        "- pub_per_port_ci.png           : Per-port success with 95% CIs\n"
        "- pub_forest_gain.png           : Forest plot of (Fair - Unbalanced) improvements (95% CI)\n"
        "- pub_minority_bars.png         : Fairness uplift on minority ports (95% CI)\n"
        "- pub_overall_macro.png         : Macro-average across ports with error bars\n"
        "- pub_table_overview.csv        : Table of per-port metrics and gains\n",
        encoding="utf-8",
    )
    print(f"[OK] {readme}")
