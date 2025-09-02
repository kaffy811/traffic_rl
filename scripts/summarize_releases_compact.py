import os, glob, json, math, csv, statistics as st
from pathlib import Path

BAL=os.environ.get("BAL"); UNB=os.environ.get("UNB"); FAIR=os.environ.get("FAIR")
assert BAL and UNB and FAIR, "请先 export BAL/UNB/FAIR 三个标签"

exp_order=[(BAL,"Balanced200"),(UNB,"Unbalanced200"),(FAIR,"FairUnbalanced200")]
ports=["baton_rouge","new_orleans","south_louisiana","gulfport"]

def latest_n(pat,n):
    fs=glob.glob(pat); fs.sort(key=lambda f: Path(f).stat().st_mtime, reverse=True); return fs[:n]

# 读每个 JSON，取各 stage 的 win_rate 做平均（也可加权）
def file_wr_mean(f):
    try:
        o=json.load(open(f))
        stages=o.get("stages",[])
        if not stages: return None
        # 加权平均：按 n_samples
        num=0.0; den=0.0
        for s in stages:
            wr=s.get("win_rate"); n=s.get("n_samples",0)
            if wr is None: continue
            if n: num+=wr*n; den+=n
        if den>0: return num/den
        # 退化为简单平均
        vals=[s.get("win_rate") for s in stages if s.get("win_rate") is not None]
        return sum(vals)/len(vals) if vals else None
    except Exception:
        return None

# 收集：每个实验×港口，取最近 12 个文件里的该港口样本（通常是 3 seeds × 4 ports）
acc={}  # (exp, port) -> list of wr
for tag,ename in exp_order:
    fs=latest_n(f"reports/FLW_{tag}/nightly/*.json", 40)  # 多取点，再按港口筛
    for p in ports:
        wrs=[]
        for f in fs:
            if f"_{p}_" not in f: continue
            wr=file_wr_mean(f)
            if wr is not None: wrs.append(wr)
        acc[(ename,p)]=wrs[:3] if len(wrs)>=3 else wrs  # 最多取3个当作3个seed

outdir=Path("reports/SUMMARY_RELEASES"); outdir.mkdir(parents=True, exist_ok=True)

# 表1：各港口三实验均值
path1=outdir/"by_port_mean.csv"
with open(path1,"w",newline="") as fp:
    w=csv.writer(fp); w.writerow(["port","Balanced200","FairUnbalanced200","Unbalanced200"])
    for p in ports:
        def mean_of(name):
            xs=acc.get((name,p),[])
            return f"{(st.mean(xs) if xs else float('nan')):.3f}"
        w.writerow([p, mean_of("Balanced200"), mean_of("FairUnbalanced200"), mean_of("Unbalanced200")])

# 表2：少数域增益（gulfport/south_louisiana）
path2=outdir/"minority_gain.csv"
with open(path2,"w",newline="") as fp:
    w=csv.writer(fp); w.writerow(["port","Unbalanced200","FairUnbalanced200","fair_gain"])
    for p in ["gulfport","south_louisiana"]:
        u=acc.get(("Unbalanced200",p),[])
        f=acc.get(("FairUnbalanced200",p),[])
        um=st.mean(u) if u else float("nan")
        fm=st.mean(f) if f else float("nan")
        gain=(fm-um) if (not math.isnan(um) and not math.isnan(fm)) else float("nan")
        w.writerow([p, f"{um:.3f}" if not math.isnan(um) else "nan",
                       f"{fm:.3f}" if not math.isnan(fm) else "nan",
                       f"{gain:.3f}" if not math.isnan(gain) else "nan"])

print("OK ->", outdir)
