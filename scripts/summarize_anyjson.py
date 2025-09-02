import os, glob, json, re
from pathlib import Path
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 实验标签（你已 export）
BAL  = os.environ["BAL"]
UNB  = os.environ["UNB"]
FAIR = os.environ["FAIR"]

EXP = [(BAL,"Balanced200"), (UNB,"Unbalanced200"), (FAIR,"FairUnbalanced200")]
PORTS = ["baton_rouge","new_orleans","south_louisiana","gulfport"]

# ---------- 工具：递归在任意嵌套里找第一个匹配的数字 ----------
def _as_num(x):
    try: return float(x)
    except: return None

def find_first(obj, keys):
    # 倒序优先“最后阶段”的值（list 反向）
    if isinstance(obj, dict):
        # 先查本层
        for k in keys:
            if k in obj:
                v = obj[k]
                if isinstance(v, (int, float)): return float(v)
                if isinstance(v, str):
                    try: return float(v)
                    except: pass
        # 再递归子节点
        for v in obj.values():
            val = find_first(v, keys)
            if val is not None: return val
    elif isinstance(obj, list):
        for it in reversed(obj):
            val = find_first(it, keys)
            if val is not None: return val
    return None

def find_success_over_total(obj):
    s = find_first(obj, ["success","num_success","wins"])
    t = find_first(obj, ["total","num_total","episodes","num_samples","N"])
    if s is not None and t and t>0:
        return s/float(t)
    return None

def parse_seed_from_name(name):
    m = re.search(r"seed(\d+)", name)
    return int(m.group(1)) if m else -1

# ---------- 收集并归一化 ----------
rows = []
for tag, expname in EXP:
    nightly_dir = f"reports/FLW_{tag}/nightly"
    files = sorted(glob.glob(f"{nightly_dir}/*.json"))
    for f in files:
        try:
            o = json.load(open(f))
        except Exception:
            continue
        # port/seed 优先从内容取，缺了就从文件名猜
        port = o.get("port") or o.get("site")
        if not port:
            for p in PORTS:
                if p in Path(f).name:
                    port = p; break
        seed = o.get("seed", parse_seed_from_name(Path(f).name))

        # success_rate：优先找常见键；找不到就 success/total
        sr = find_first(o, ["success_rate","successRate","sr","accuracy","acc","win_rate","episode_success_rate"])
        if sr is None:
            sr = find_success_over_total(o)

        # avg_reward：各种可能的命名
        rew = find_first(o, ["avg_reward","reward","reward_avg","mean_reward","return","avg_return"])

        rows.append(dict(exp=expname, tag=tag, port=port or "unknown",
                         seed=seed, success_rate=sr, avg_reward=rew, file=f))

df = pd.DataFrame(rows)
# 至少要有一个指标
df = df[(~df["success_rate"].isna()) | (~df["avg_reward"].isna())].copy()
if df.empty:
    print("❌ 仍未在 nightly/forced JSON 中找到可用指标。请打印一个样例 JSON 看看键名：")
    print("  python - <<'PY'\nimport os,glob,json; t=os.environ['BAL']; f=sorted(glob.glob(f\"reports/FLW_{t}/nightly/*.json\"))[0]; print('SAMPLE=',f); print(json.dumps(json.load(open(f)) ,indent=2, ensure_ascii=False));\nPY")
    raise SystemExit(1)

out = Path("reports/SUMMARY_JSON"); out.mkdir(parents=True, exist_ok=True)
df.to_csv(out/"raw_rows.csv", index=False)

# 表1：每港口×实验 成功率均值
by_port = df.groupby(["port","exp"], as_index=False)["success_rate"].mean()
pivot = by_port.pivot(index="port", columns="exp", values="success_rate").reindex(PORTS)
pivot.to_csv(out/"by_port_mean.csv")

# 表2：少数域公平提升（gulfport、south_louisiana）
minor = ["gulfport","south_louisiana"]
if set(minor).issubset(set(pivot.index)) and {"Unbalanced200","FairUnbalanced200"}.issubset(set(pivot.columns)):
    mg = pd.DataFrame({
        "port": minor,
        "Unbalanced200": pivot.loc[minor, "Unbalanced200"].values,
        "FairUnbalanced200": pivot.loc[minor, "FairUnbalanced200"].values,
    })
    mg["fair_gain"] = mg["FairUnbalanced200"] - mg["Unbalanced200"]
    mg.to_csv(out/"minority_gain.csv", index=False)
    ax = mg.set_index("port")[["Unbalanced200","FairUnbalanced200"]].plot(kind="bar")
    ax.set_ylabel("Success Rate"); ax.set_title("Minority Ports: Fair vs Unbalanced")
    plt.tight_layout(); plt.savefig(out/"fig_fair_gain_minority.png"); plt.close()

# 图：每港口三实验并排
ax = pivot.plot(kind="bar")
ax.set_ylabel("Success Rate"); ax.set_title("Per-port Success Rate across Experiments")
plt.tight_layout(); plt.savefig(out/"fig_per_port.png"); plt.close()

print("OK ->", out)
