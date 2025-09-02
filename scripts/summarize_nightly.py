import os, glob, json
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAL=os.environ["BAL"]; UNB=os.environ["UNB"]; FAIR=os.environ["FAIR"]
EXP=[(BAL,"Balanced200"),(UNB,"Unbalanced200"),(FAIR,"FairUnbalanced200")]

def latest_n(pat, n):
    fs=glob.glob(pat); fs.sort(key=lambda f: Path(f).stat().st_mtime, reverse=True)
    return fs[:n]

def pick_num(d, keys):
    for k in keys:
        if k in d and d[k] is not None:
            try: return float(d[k])
            except: pass
    return None

def pick_success_rate(obj):
    # 直接在根找
    sr = pick_num(obj, ["success_rate","successRate","sr","succ_rate","episode_success_rate","win_rate","accuracy","acc"])
    if sr is None and obj.get("total"):
        return float(obj.get("success",0))/float(obj["total"])
    # 转到 stages
    stages=obj.get("stages",[])
    for st in reversed(stages):  # 优先取最后一个有指标的 stage
        if isinstance(st, dict):
            sr = pick_num(st, ["success_rate","successRate","sr","succ_rate","episode_success_rate","win_rate","accuracy","acc"])
            if sr is None and st.get("total"):
                sr=float(st.get("success",0))/float(st["total"])
            if sr is not None:
                return sr
    return None

def pick_reward(obj):
    r = pick_num(obj, ["avg_reward","reward","reward_avg","mean_reward"])
    if r is not None: return r
    for st in reversed(obj.get("stages",[])):
        if isinstance(st, dict):
            r = pick_num(st, ["avg_reward","reward","reward_avg","mean_reward"])
            if r is not None: return r
    return None

def pick_port(obj):
    for k in ["port","site","city","name"]:
        if k in obj: return obj[k]
    return "unknown"

def pick_seed(obj):
    for k in ["seed","random_seed","rng_seed"]:
        if k in obj: return obj[k]
    for st in reversed(obj.get("stages",[])):
        if isinstance(st, dict):
            for k in ["seed","random_seed","rng_seed"]:
                if k in st: return st[k]
    return -1

rows=[]; miss={}
for tag,label in EXP:
    miss[tag]=0
    # 取最近 64 份（足够覆盖 4 ports × 3 seeds）
    for f in latest_n(f"reports/FLW_{tag}/nightly/*.json", 64):
        try: o=json.load(open(f))
        except Exception: miss[tag]+=1; continue
        sr=pick_success_rate(o)
        if sr is None: miss[tag]+=1; continue
        rows.append(dict(exp=label, tag=tag, port=pick_port(o), seed=pick_seed(o),
                         success_rate=sr, avg_reward=pick_reward(o)))

if not rows:
    print("❌ 没读到 nightly JSON 指标。诊断：")
    for tag,_ in EXP:
        cnt=len(glob.glob(f"reports/FLW_{tag}/nightly/*.json"))
        print(f"  - {tag}: files={cnt}, unreadable_or_no_metric={miss.get(tag,0)}")
    raise SystemExit(1)

df=pd.DataFrame(rows)
out=Path("reports/SUMMARY_MINI"); out.mkdir(parents=True, exist_ok=True)

# === 表1：各港口 × 实验 的成功率均值 ===
m=(df.groupby(["port","exp"]).success_rate.mean()
     .unstack("exp")
     .sort_index())
m.to_csv(out/"by_port_success_mean.csv", float_format="%.4f")

# === 图1：各港口三组实验柱状对比 ===
ports=list(m.index); exps=list(m.columns)
x=np.arange(len(ports)); w=0.26
plt.figure(figsize=(10,4))
for i,e in enumerate(exps):
    plt.bar(x+(i-1)*w, m[e].values, w, label=e)
plt.xticks(x, ports, rotation=15); plt.ylim(0,1); plt.ylabel("Success Rate"); plt.legend()
plt.tight_layout(); plt.savefig(out/"fig_per_port_success.png"); plt.close()

# === 图2：少数域公平增益（FAIR-UNB）仅 gulfport/south_louisiana ===
minor=[p for p in ports if p in ("gulfport","south_louisiana")]
if minor:
    g=m.loc[minor, ["Unbalanced200","FairUnbalanced200"]].copy()
    g["fair_gain"]=g["FairUnbalanced200"]-g["Unbalanced200"]
    g.to_csv(out/"minority_gain.csv", float_format="%.4f")
    plt.figure(figsize=(5,3))
    plt.bar(np.arange(len(minor)), g["fair_gain"].values)
    plt.xticks(np.arange(len(minor)), minor, rotation=10)
    plt.ylabel("FAIR - UNB")
    plt.tight_layout(); plt.savefig(out/"fig_fair_gain_minority.png"); plt.close()

print("✅ 生成完成 →", out)
