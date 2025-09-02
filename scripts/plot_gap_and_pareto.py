#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成三张图：
1) pub_fairness_gap.png        —— 全局公平性间距（越小越好）
2) pub_pareto_macro_gap.png    —— 宏平均 vs 间距的 Pareto 散点
3) pub_heat_gain.png           —— 各港口 Fair−Unbalanced 提升热力图
同时导出一个汇总表：pub_overall_macro_gap_table.csv
"""
import os, json, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams["figure.dpi"] = 160

def load_by_port_mean(release_dir: str) -> pd.DataFrame:
    """优先读 by_port_mean.csv；若缺失则从 nightly JSON 汇聚"""
    csv = os.path.join(release_dir, "by_port_mean.csv")
    if os.path.isfile(csv):
        df = pd.read_csv(csv)
        df.columns = [c.strip() for c in df.columns]
        need = {"port","Balanced200","FairUnbalanced200","Unbalanced200"}
        missing = need - set(df.columns)
        if missing:
            raise RuntimeError(f"{csv} 缺少列: {missing}")
        return df

    # 兜底：从 nightly/forced_*.json 聚合
    print(f"[INFO] {csv} 不存在，回退聚合 nightly JSON …")
    nightly = os.path.join(release_dir, "FLW_*", "nightly", "forced_*.json")
    rows = []
    for f in glob.glob(nightly):
        try:
            r = json.load(open(f))
            rows.append(r)
        except Exception:
            pass
    if not rows:
        raise RuntimeError("未发现 nightly JSON，无法聚合")
    raw = pd.DataFrame(rows)  # columns: port, success_rate, source, …
    # 粗略识别实验tag：从路径或文件夹名里取三类关键词
    def tag_from_path(p: str) -> str:
        p = str(p)
        if "Balance200" in p or "Balanced200" in p: return "Balanced200"
        if "fair_unbalanced" in p or "FairUnbalanced" in p: return "FairUnbalanced200"
        return "Unbalanced200"
    raw["exp"] = raw["source"].map(tag_from_path)
    agg = (raw.groupby(["port","exp"])["success_rate"].mean()
              .unstack("exp")[["Balanced200","FairUnbalanced200","Unbalanced200"]]
              .reset_index().rename_axis(None, axis=1))
    return agg

def compute_macro(df: pd.DataFrame) -> dict:
    return {
        "Balanced200":       df["Balanced200"].mean(),
        "FairUnbalanced200": df["FairUnbalanced200"].mean(),
        "Unbalanced200":     df["Unbalanced200"].mean(),
    }

def compute_gap(df: pd.DataFrame) -> dict:
    return {
        "Balanced200":       df["Balanced200"].max()       - df["Balanced200"].min(),
        "FairUnbalanced200": df["FairUnbalanced200"].max() - df["FairUnbalanced200"].min(),
        "Unbalanced200":     df["Unbalanced200"].max()     - df["Unbalanced200"].min(),
    }

def main():
    stamp = os.getenv("STAMP", "2025-08-21")
    rel = f"reports/RELEASE_{stamp}"
    outd = rel
    os.makedirs(outd, exist_ok=True)

    df = load_by_port_mean(rel)

    # 1) 公平性间距条形图
    gap = compute_gap(df)
    tags = ["Balanced200","FairUnbalanced200","Unbalanced200"]
    vals = [gap[t] for t in tags]

    plt.figure(figsize=(6.4,3.8))
    ax = plt.gca()
    bars = ax.bar(tags, vals, alpha=0.9)
    ax.set_ylabel("Fairness Gap = max(port) - min(port)")
    ax.set_title("Overall Fairness Gap (lower is better)")
    for b,v in zip(bars, vals):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.001, f"{v:.3f}",
                ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    p1 = os.path.join(outd, "pub_fairness_gap.png")
    plt.savefig(p1); plt.close()
    print("[OK]", p1)

    # 2) Pareto：宏平均 vs 间距
    macro = compute_macro(df)
    xs = [macro[t] for t in tags]
    ys = [gap[t]   for t in tags]
    colors = {"Balanced200":"#4e79a7","FairUnbalanced200":"#f28e2b","Unbalanced200":"#59a14f"}

    plt.figure(figsize=(7.2,4.0))
    ax = plt.gca()
    for t,x,y in zip(tags,xs,ys):
        ax.scatter(x,y, s=120, c=colors.get(t,"#666"), edgecolors="k", linewidths=0.6)
        ax.text(x+0.001, y+0.001, t, fontsize=9)
    ax.set_xlabel("Macro-average Success Rate (↑)")
    ax.set_ylabel("Fairness Gap (↓)")
    ax.set_title("Pareto of Overall vs Fairness")
    ax.grid(True, alpha=0.25, linestyle="--")
    plt.tight_layout()
    p2 = os.path.join(outd, "pub_pareto_macro_gap.png")
    plt.savefig(p2); plt.close()
    print("[OK]", p2)

    # 3) 热力图：Fair−Unbalanced 提升
    df_h = df.copy()
    df_h["gain"] = df_h["FairUnbalanced200"] - df_h["Unbalanced200"]
    mat = df_h.set_index("port")[["gain"]]
    plt.figure(figsize=(4.2,3.0))
    im = plt.imshow(mat.values, cmap="RdYlGn", aspect="auto", vmin=-0.05, vmax=0.05)
    plt.yticks(range(len(mat.index)), mat.index)
    plt.xticks([0], ["Fair − Unbalanced"])
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    cb.set_label("Δ Success Rate")
    plt.title("Per-port Fairness Uplift")
    plt.tight_layout()
    p3 = os.path.join(outd, "pub_heat_gain.png")
    plt.savefig(p3); plt.close()
    print("[OK]", p3)

    # 汇总表：macro + gap + 合成指标
    tab = (pd.DataFrame({"exp": tags, "macro": xs, "gap": ys})
           .assign(pareto_score=lambda d: d["macro"] - d["gap"]))
    tpath = os.path.join(outd, "pub_overall_macro_gap_table.csv")
    tab.to_csv(tpath, index=False)
    print("[OK]", tpath)
    print("\n--- SUMMARY ---")
    print(tab.to_string(index=False))

if __name__ == "__main__":
    main()
