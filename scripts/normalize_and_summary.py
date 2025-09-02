import os,glob,json,re
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

BAL=os.environ["BAL"]; UNB=os.environ["UNB"]; FAIR=os.environ["FAIR"]
EXP=[(BAL,"Balanced200"),(UNB,"Unbalanced200"),(FAIR,"FairUnbalanced200")]
PORTS=["baton_rouge","new_orleans","south_louisiana","gulfport"]

def _num(x):
    try: return float(x)
    except: return None

def infer_metrics(o):
    # 先顶层
    sr = None
    for k in ["success_rate","successRate","sr","accuracy","acc","win_rate"]:
        if k in o: sr=_num(o[k]); 
        if sr is not None: break
    rew=None
    for k in ["avg_reward","reward","reward_avg","mean_reward"]:
        if k in o: rew=_num(o[k]); 
        if rew is not None: break
    total=int(o.get("total") or o.get("num_samples") or 0)
    succ=_num(o.get("success"))
    if sr is None and succ is not None and total:
        sr = succ/float(total)

    # 再 metrics
    m=o.get("metrics") if isinstance(o.get("metrics"),dict) else None
    if m:
        if sr is None:
            for k in ["success_rate","successRate","sr","accuracy","acc","win_rate"]:
                if k in m: sr=_num(m[k]); 
                if sr is not None: break
        if rew is None:
            for k in ["avg_reward","reward","reward_avg","mean_reward"]:
                if k in m: rew=_num(m[k]); 
                if rew is not None: break
        if sr is None and (m.get("success") is not None) and (m.get("total") or m.get("num_samples")):
            den=float(m.get("total") or m.get("num_samples") or 0)
            if den>0: sr=_num(m["success"])/den

    # 再看 stages（倒着找最后一个阶段）
    if (sr is None or rew is None) and isinstance(o.get("stages"),list):
        for s in reversed(o["stages"]):
            cand=[s]
            if isinstance(s,dict) and isinstance(s.get("metrics"),dict):
                cand.append(s["metrics"])
            tmp_sr,tmp_rew=None,None
            for d in cand:
                if tmp_sr is None:
                    for k in ["success_rate","successRate","sr","accuracy","acc","win_rate"]:
                        if isinstance(d,dict) and k in d:
                            tmp_sr=_num(d[k]); 
                            if tmp_sr is not None: break
                if tmp_rew is None:
                    for k in ["avg_reward","reward","reward_avg","mean_reward"]:
                        if isinstance(d,dict) and k in d:
                            tmp_rew=_num(d[k]); 
                            if tmp_rew is not None: break
                if tmp_sr is None and isinstance(d,dict) and (d.get("success") is not None) and (d.get("total") or d.get("num_samples")):
                    den=float(d.get("total") or d.get("num_samples") or 0)
                    if den>0: tmp_sr=_num(d["success"])/den
            if tmp_sr is not None or tmp_rew is not None:
                sr = tmp_sr if sr is None else sr
                rew= tmp_rew if rew is None else rew
                break
    return sr,rew

def parse_seed_from_name(name):
    m=re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else -1

rows=[]
for tag,ename in EXP:
    d=f"reports/FLW_{tag}/nightly"
    files=glob.glob(f"{d}/*.json")
    for f in files:
        try:
            o=json.load(open(f))
        except Exception:
            continue
        # 补标准字段（就地写回）
        sr,rew = infer_metrics(o)
        if sr is not None: o["success_rate"]=sr
        if rew is not None: o["avg_reward"]=rew
        if ("success_rate" in o) or ("avg_reward" in o):
            with open(f,"w") as wf: json.dump(o,wf,ensure_ascii=False,indent=2)

        port = o.get("port") or o.get("site")
        if not port:
            # 从文件名推断
            for p in PORTS:
                if p in Path(f).name: port=p; break
        seed = o.get("seed", parse_seed_from_name(Path(f).name))
        rows.append(dict(exp=ename, tag=tag, port=port or "unknown",
                         seed=seed, success_rate=o.get("success_rate"),
                         avg_reward=o.get("avg_reward"), file=f))

df=pd.DataFrame(rows)
# 过滤掉 sr/reward 全空的行
df = df[(~df["success_rate"].isna()) | (~df["avg_reward"].isna())].copy()
if df.empty:
    print("❌ 仍未在 nightly JSON 中找到可用指标。")
    raise SystemExit(1)

out=Path("reports/SUMMARY_JSON"); out.mkdir(parents=True,exist_ok=True)
df.to_csv(out/"raw_flatten_rows.csv", index=False)

# === 表1：每港口×实验 成功率均值 ===
by_port=df.groupby(["port","exp"],as_index=False)["success_rate"].mean()
pivot=by_port.pivot(index="port",columns="exp",values="success_rate").reindex(PORTS)
pivot.to_csv(out/"by_port_mean.csv")

# === 表2：少数域公平提升 ===
minor=["gulfport","south_louisiana"]
tbl=pivot.loc[minor]
if {"Unbalanced200","FairUnbalanced200"}.issubset(tbl.columns):
    mg=pd.DataFrame({
        "port":tbl.index,
        "Unbalanced200":tbl["Unbalanced200"].values,
        "FairUnbalanced200":tbl["FairUnbalanced200"].values,
        "fair_gain":(tbl["FairUnbalanced200"]-tbl["Unbalanced200"]).values
    })
    mg.to_csv(out/"minority_gain.csv",index=False)
    # 图：提升条形图
    ax=mg.set_index("port")[["Unbalanced200","FairUnbalanced200"]].plot(kind="bar")
    ax.set_ylabel("Success Rate"); ax.set_title("Minority Ports: Fair vs Unbalanced")
    plt.tight_layout(); plt.savefig(out/"fig_fair_gain_minority.png"); plt.close()

# === 图：每港口三实验并排 ===
ax=pivot.plot(kind="bar")
ax.set_ylabel("Success Rate"); ax.set_title("Per-port Success Rate across Experiments")
plt.tight_layout(); plt.savefig(out/"fig_per_port.png"); plt.close()

print("OK ->", out)
