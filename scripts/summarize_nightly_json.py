import os, glob, json, re
from pathlib import Path
import pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAL=os.environ.get("BAL",""); UNB=os.environ.get("UNB",""); FAIR=os.environ.get("FAIR","")
EXP = [(BAL,"Balanced200"), (UNB,"Unbalanced200"), (FAIR,"FairUnbalanced200")]
ports = ["baton_rouge","new_orleans","south_louisiana","gulfport"]

def latest_release_files(n=12):
    relroot = Path("models/releases")
    if not relroot.exists(): return []
    days = sorted([p for p in relroot.iterdir() if p.is_dir()], key=lambda p:p.name, reverse=True)
    for d in days:
        fs = sorted(d.glob("consistency_*.json"), key=lambda p:p.stat().st_mtime, reverse=True)
        if fs: return [str(x) for x in fs[:n]]
    return []

def flatten(d, prefix=""):
    out={}
    if isinstance(d, dict):
        for k,v in d.items():
            out.update(flatten(v, f"{prefix}{k}/"))
    elif isinstance(d, list):
        for i,v in enumerate(d):
            out.update(flatten(v, f"{prefix}{i}/"))
    else:
        out[prefix[:-1]] = d
    return out

def pick_metrics(obj):
    flat = flatten(obj)
    # 先找成功率/准确率
    sr = None
    for k,v in flat.items():
        kl = k.lower()
        if isinstance(v,(int,float)):
            if any(x in kl for x in ["success_rate","succ_rate","win_rate","accuracy","acc"]):
                sr = float(v); break
    # success/total 兜底
    if sr is None:
        succ = next((float(v) for k,v in flat.items() if k.lower().endswith("/success") and isinstance(v,(int,float))), None)
        tot  = next((float(v) for k,v in flat.items() if k.lower().endswith("/total")   and isinstance(v,(int,float))), None)
        if succ is not None and tot and tot>0: sr = succ/tot
    # 奖励
    rew = None
    for k,v in flat.items():
        if isinstance(v,(int,float)) and ("avg_reward" in k.lower() or re.search(r"(reward|return)(_avg)?$", k.lower())):
            rew = float(v); break
    # 港口与种子
    port = obj.get("port") or obj.get("site")
    if port is None:
        # 尝试从文件名解析
        port = next((p for p in ports if p in json.dumps(obj)), "unknown")
    seed = obj.get("seed")
    if seed is None:
        # 从扁平键里猜
        for k,v in flat.items():
            if k.lower().endswith("/seed") and isinstance(v,(int,float)): seed=int(v); break
    return port, seed, sr, rew

def collect_from(tag, limit=12):
    files = sorted(glob.glob(f"reports/FLW_{tag}/nightly/*.json"), key=lambda f: Path(f).stat().st_mtime, reverse=True)[:limit]
    if not files and tag:  # 回退到 releases
        files = latest_release_files(limit)
    rows=[]
    bad=0
    for f in files:
        try:
            o = json.load(open(f))
        except Exception:
            bad+=1; continue
        port, seed, sr, rew = pick_metrics(o)
        if sr is None and rew is None:
            bad+=1; continue
        rows.append(dict(file=f, port=port, seed=seed, sr=sr, reward=rew))
    return rows, len(files), bad

all_rows=[]
diag=[]
for tag, name in EXP:
    if not tag: continue
    rows, total, bad = collect_from(tag, limit=12)
    for r in rows: r.update(exp=name, tag=tag)
    all_rows.extend(rows)
    diag.append((name, total, bad))

if not all_rows:
    print("❌ 没读到 nightly/release JSON 的有效指标。诊断：")
    for name,total,bad in diag:
        print(f"  - {name}: files={total}, unreadable_or_nometrics={bad}")
    raise SystemExit(1)

df = pd.DataFrame(all_rows)
outdir = Path(f"reports/SUMMARY_JSON"); outdir.mkdir(parents=True, exist_ok=True)

# 每港口每实验的均值
by_port = df.groupby(["exp","port"])["sr"].mean().unstack(0)
by_port = by_port.reindex(index=ports)  # 固定端口顺序
by_port.to_csv(outdir/"by_port_mean.csv")

# 少数域提升（以 gulfport / south_louisiana 为例）
minority = ["gulfport","south_louisiana"]
part = by_port.loc[minority, ["Unbalanced200","FairUnbalanced200"]].copy()
part["fair_gain"] = part["FairUnbalanced200"] - part["Unbalanced200"]
part.to_csv(outdir/"minority_gain.csv")

# 图1：四港口 × 三实验
plt.figure(figsize=(8,4))
by_port.plot(kind="bar", rot=0)
plt.ylabel("Success Rate")
plt.title("Per-Port Mean Success Rate")
plt.tight_layout()
plt.savefig(outdir/"fig_per_port.png"); plt.close()

# 图2：公平聚合对少数域的提升
plt.figure(figsize=(5,3))
part["fair_gain"].plot(kind="bar", rot=0)
plt.ylabel("Δ Success Rate (Fair - Unbalanced)")
plt.title("Minority Ports Gain")
plt.tight_layout()
plt.savefig(outdir/"fig_fair_gain_minority.png"); plt.close()

print("OK ->", outdir)
