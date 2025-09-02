#!/usr/bin/env python3
import os, glob, json
from pathlib import Path
from typing import Dict, Any
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def load_forced(tag: str) -> pd.DataFrame:
    rows = []
    for f in glob.glob(f"reports/FLW_{tag}/nightly/forced_*.json"):
        o = json.load(open(f, "r", encoding="utf-8"))
        rows.append({
            "tag": tag,
            "port": o.get("port"),
            "seed": o.get("seed"),
            "success_rate": o.get("success_rate"),
            "avg_reward": o.get("avg_reward"),
            "num_samples": o.get("num_samples"),
        })
    return pd.DataFrame(rows)

def main():
    tags = os.environ.get("TAGS")  # 逗号分隔
    if not tags:
        raise SystemExit("set TAGS='tag1,tag2,tag3'")
    tags = [t.strip() for t in tags.split(",") if t.strip()]
    dfs = [load_forced(t) for t in tags]
    df = pd.concat(dfs, ignore_index=True)
    out = Path(f"reports/SUMMARY_{pd.Timestamp.now():%Y%m%d_%H%M%S}")
    out.mkdir(parents=True, exist_ok=True)

    # 表 1：按 port × tag 的均值
    by_port = df.groupby(["port","tag"], as_index=False)["success_rate"].mean().pivot(index="port", columns="tag", values="success_rate")
    by_port.to_csv(out/"by_port_mean.csv")

    # 表 2：少数域收益（你可按项目定义少数域端口）
    minority_ports = ["gulfport","south_louisiana"]
    g = df[df["port"].isin(minority_ports)].groupby(["port","tag"], as_index=False)["success_rate"].mean()
    g = g.pivot(index="port", columns="tag", values="success_rate")
    g.to_csv(out/"minority_gain_raw.csv")
    if len(tags) >= 2:
        # 假定 tags[1] 是 Unbalanced，tags[2] 是 Fair（按你传入顺序决定）
        base = tags[1] if len(tags)>1 else tags[0]
        fair = tags[2] if len(tags)>2 else tags[-1]
        gain = (g[fair] - g[base]).to_frame("fair_gain")
        gain.to_csv(out/"minority_gain.csv")

    # 图 1：每个 port 三个 tag 的柱状图
    for port, sub in df.groupby("port"):
        ax = sub.pivot_table(index="seed", columns="tag", values="success_rate").mean().plot(kind="bar")
        ax.set_ylim(0,1); ax.set_title(f"Success rate by tag - {port}")
        ax.set_ylabel("success_rate")
        fig = ax.get_figure()
        fig.tight_layout()
        fig.savefig(out/f"fig_per_port_{port}.png"); plt.close(fig)

    # 图 2：少数域收益
    if len(tags) >= 2:
        ax = g.plot(kind="bar")
        ax.set_ylim(0,1); ax.set_title("Minority ports - success rate")
        ax.set_ylabel("success_rate")
        fig = ax.get_figure(); fig.tight_layout()
        fig.savefig(out/"fig_minority_ports.png"); plt.close(fig)

    print("OK ->", out)

if __name__ == "__main__":
    main()
