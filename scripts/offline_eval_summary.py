#!/usr/bin/env python3
import os, json, glob, numpy as np, pandas as pd
from pathlib import Path
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from src.federated.fedavg_client import FedAvgClient

BAL=os.environ["BAL"]; UNB=os.environ["UNB"]; FAIR=os.environ["FAIR"]
EXPS=[(BAL,"Balanced200"),(UNB,"Unbalanced200"),(FAIR,"FairUnbalanced200")]
PORTS=["baton_rouge","new_orleans","south_louisiana","gulfport"]
SEEDS=[42,123,2025]

def ckpt(tag):
    for name in ("global_best.pt","global_round_200.pt","global_round_030.pt"):
        p=Path(f"models/flw/{tag}/{name}")
        if p.exists(): return str(p)
    raise SystemExit(f"[ERR] checkpoint not found for {tag}")

def eval_once(tag, port, seed):
    np.random.seed(seed)
    try:
        import torch; torch.manual_seed(seed)
    except Exception:
        pass
    cli=FedAvgClient(port=port, init_weights=ckpt(tag))
    out=cli.evaluate()
    # success rate 优先；否则回退 avg_reward/reward
    sr=None
    for k in ("success_rate","successRate","sr","win_rate","accuracy","acc"):
        if k in out and out[k] is not None:
            try: sr=float(out[k]); break
            except: pass
    if sr is None and out.get("total"):
        sr=float(out.get("success",0))/float(out["total"])
    metric = sr if sr is not None else out.get("avg_reward", out.get("reward"))
    return {"port":port,"seed":seed,"tag":tag,"exp":dict(EXPS)[tag],
            "success_rate":sr,"avg_reward":out.get("avg_reward"),"metric":metric}

rows=[]
for tag,_ in EXPS:
    for p in PORTS:
        for s in SEEDS:
            rows.append(eval_once(tag,p,s))

df=pd.DataFrame(rows)
dst=Path("reports/SUMMARY_OFFLINE"); dst.mkdir(parents=True, exist_ok=True)
(df.sort_values(["exp","port","seed"])
   .to_csv(dst/"raw_eval.csv", index=False))

m=(df.groupby(["port","exp"])["metric"].mean().unstack("exp").sort_index())
m.to_csv(dst/"by_port_mean.csv")

plt.figure(figsize=(10,4))
x=np.arange(len(m.index)); w=0.26
for i,e in enumerate(m.columns):
    plt.bar(x+(i-1)*w, m[e].values, w, label=e)
plt.xticks(x, m.index, rotation=15); plt.ylabel("Success / Reward")
plt.ylim(0,1 if m.max().max()<=1.05 else None)
plt.legend(); plt.tight_layout()
plt.savefig(dst/"fig_per_port.png"); plt.close()

minor=[p for p in m.index if p in ("gulfport","south_louisiana")]
if minor:
    g=m.loc[minor, ["Unbalanced200","FairUnbalanced200"]].copy()
    g["fair_gain"]=g["FairUnbalanced200"]-g["Unbalanced200"]
    g.to_csv(dst/"minority_gain.csv")
    plt.figure(figsize=(5,3))
    plt.bar(np.arange(len(minor)), g["fair_gain"].values)
    plt.xticks(range(len(minor)), minor, rotation=10); plt.ylabel("FAIR-UNB")
    plt.tight_layout(); plt.savefig(dst/"fig_fair_gain_minority.png"); plt.close()

print("OK ->", dst)
