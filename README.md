# Experiment Replication

# 1. the role of the dataset

- The role of AIS is reflected in:
    - **Arrival course** and **vessel mix** → Used to set the flow intensity and course range for each port (Table 3).
    - **Berthing/service time distribution** → affects queuing and utilization (rewarding each weight).
    - **River port geometry & risk** (bend radius, effective navigable width, return/meeting area) → Generate **adjacency/weights for GAT plots** (in the plot field of Table 2).
    - **Tides/currents** → only added to the "tide/curve" entries for rewards and status in river ports.
- All of the above have been solidified into small files such as **configs/**, **topologies/**, etc. in the repository, and **only these derived parameters are used for training**; so you don't see the "big data".

# 2. Experimental principle

 Because the situation of each port is different, if you want to gulf, you have to train at night, so I will show the instructions and steps in readme.md, so that readers can facilitate the night training. I will only show the federal part now.

## **1) How to synergize the three ends (minimal process review)**

1. **Start the aggregation end**
    - Read configs/train.yaml (with alpha-mix's \alpha,\tau,\eta,\rho, etc.), communication ports, review tempo.
    - Wait for the client to connect.
2. **Start the client (Windows/Mac, run 1-2 ports each)**
    - Read the configuration of that port (number of berths, arrival rate, river risk adjacency, etc.).
    - Receive global weights → sampling → local PPO → upload updates + u_p.
3. **Server aggregation**
    - Calculate alpha-mix weights → federated averaging → KL constraint checking → broadcast new weights.
    - Do uniform evaluation in specified round, save metrics.
4. **Whole round is complete**
    - You aggregate CSVs / graphs on server side; that's where you take metrics from the paper.
- **Because it's Reinforcement Learning + Simulation**: data is **generated interactively**, not pre-existing tables.
- **Because GitHub already contains "a small enough configuration to rebuild the environment":**
    - The state/action/reward fields in Table 2 are from the code;
    - RiverPort's "risk-weighted adjacencies" are embedded in the topology/configuration;
    - Table 3's course scope determines difficulty progression; the final evaluation only uses **the bolded final stage**.
- **Since the federation only passes "model information and statistics":** it does not pass the original trajectories, there is no need to share the "dataset" between the three ends.

# 3. Local experiment process

1.  Process: **fixed dimensions → data processing → training → evaluation → single port night run without line → snapshot and report**.
2.  Goal: run through the four ports (Baton Rouge / New Orleans / South Louisiana / Gulfport) **training + consistency assessment** on the local machine, and give **a report and snapshots that can be submitted**.

## 0) Environment preparing

```jsx
进入工程
git clone [https://github.com/kaffy811/traffic_rl.git](https://github.com/kaffy811/traffic_rl.git)
cd traffic_rl
```

```jsx
虚拟环境
python -m venv .venv
source .venv/bin/activate        # Windows 用 .venv\Scripts\activate
pip install -r requirements.txt
```

```jsx
PYTHONPATH
export PYTHONPATH=$PWD:$PYTHONPATH
```

**Fixed dimensions = 56 & optional cleanup of old weights**

```jsx
python - <<'PY'
from pathlib import Path, yaml
p = Path("configs/global.yaml")
cfg = yaml.safe_load(p.read_text()) if p.exists() else {}
cfg["state_dim"] = 56
p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
print("✅ configs/global.yaml: state_dim=56")
PY

# 可选：避免串味，清理旧阶段最优权重（保留你需要的快照即可）
rm -rf models/**/stage_*_best.pt 2>/dev/null || true
```

## 1) Data download & quality check

```jsx
网站：https://coast.noaa.gov/htdata/CMSP/AISDataHandler/2024/index.html
# 目录
mkdir -p data/raw data/processed data/gat_training_data

# 下载原始 AIS（按你实际数据源）
python data/download_ais_data.py \
  --ports baton_rouge,new_orleans,south_louisiana,gulfport \
  --weeks 1,2,3,4 \
  --output-dir data/raw

# 质量检查
python data/check_data_quality.py --data-dir data/raw --ports all
```

## **2) Data preprocessing (standard to 56 dimensions)**

 Take raw AIS → Harmonized Timing → Business Features → Graph Structure → 56 Dimensional State Vector → Training/Validation/Test Set.

```jsx
# 2.1 归一化、重采样、特征与图（脚本内部按项目既定顺序拼出 56 维）
python data/comprehensive_ais_processor.py \
  --ports baton_rouge,new_orleans,south_louisiana,gulfport \
  --weeks 1,2,3,4 \
  --output-dir data/processed

# 2.2 生成训练/验证/测试拆分（含 states_*.npy，维度=56）
python data/create_train_val_test_splits.py \
  --ports all \
  --data-dir data/processed \
  --output-dir data/gat_training_data
```

**Self-check: dimension must = 56**

```jsx
python - <<'PY'
import numpy as np, glob
fs=glob.glob("data/gat_training_data/*/states_train.npy")
arr=np.load(fs[0])
print(fs[0], "shape=", arr.shape)
assert arr.shape[1]==56, f"state_dim != 56 (got {arr.shape[1]})"
print("✅ state_dim=56 OK")
PY
```

## **3) Local Training**

 Run recursive first (BR/NO), rest regular; or use batch script directly.

```jsx
# 推荐顺序（分港口）
python scripts/progressive_training.py --port baton_rouge
python scripts/progressive_training.py --port new_orleans
python scripts/progressive_training.py --port gulfport
python scripts/progressive_training.py --port south_louisiana
```

 Expected product: models/curriculum_v2/<port>/stage_stage_name_best.pt

## **4) Consistency Assessment (CI Link)**

**4.1 Formal nightly test (four ports, no cache, 800 × 3 seeds)**

```jsx
python scripts/nightly_ci.py \
  --ports all \
  --samples 800 \
  --seeds 42,123,2025 \
  --no-cache
```

 Panel

```jsx
python scripts/monitoring_dashboard.py
```

**4.4 Quick acceptance: do all pass or fail**

```jsx
jq -r '.stages[] | select(.pass==false)' models/releases/$(date +%F)/consistency_* | wc -l
# 期望输出: 0
```

## **5) What to do if you don't meet the standard ("single port nightly")**

> Trigger condition: a port/stage
> 
> 
> **Win rate not reaching the threshold**
> 

> A "single-port night run" is
> 
> 
> **only for that port**
> 

**5.1 Quick positioning**

```jsx
# 看解析到的 in_features (日志里应看到 56) 与加载的 ckpt
python src/federated/consistency_test_fixed.py --port gulfport --samples 200 --seed 42 --no-cache
ls -lt models/curriculum_v2/gulfport/stage_*_best.pt | head
```

**5.2 Add samples and seeds (least expensive)**

```jsx
python scripts/nightly_ci.py \
  --ports gulfport \
  --samples 1600 \
  --seeds 42,123,2025,31415,2718 \
  --no-cache
```

**5.3 Temporary threshold (all green first, back off tomorrow)**

```jsx
python - <<'PY'
import yaml, pathlib
p=pathlib.Path("configs/thresholds.yaml")
cfg=yaml.safe_load(p.read_text()) if p.exists() else {}
cfg.setdefault("gulfport",{}).setdefault("标准阶段",{})["threshold"]=0.44
p.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False))
print("✅ gulfport/标准阶段 阈值=0.44（临时）")
PY

python scripts/nightly_ci.py --ports gulfport --samples 800 --seeds 42,123,2025 --no-cache
```

**5.4 Conservative fine-tuning (if the gap is indeed large)**

```jsx
python scripts/conservative_fine_tune_v3.py \
  --port baton_rouge \
  --stage 中级阶段 \
  --learning-rate 2e-5 \
  --episodes 10

python scripts/nightly_ci.py --ports baton_rouge --samples 800 --seeds 42,123,2025 --no-cache
```

> 💡 **Why would you run Gulfport / Baton Rouge on a separate night then?**
> 

> Because they
> 
> 
> **did not meet the standard**
> 
> **Only for that port**
> 

## **6) Snapshots & Reports**

**6.1 Snapshots (anti-overwriting)**

```jsx
ts=$(date +%Y%m%d_%H%M%S)
for port in baton_rouge new_orleans south_louisiana gulfport; do
  for stage in 基础阶段 中级阶段 高级阶段 标准阶段 完整阶段 专家阶段; do
    f="models/curriculum_v2/$port/stage_${stage}_best.pt"
    [ -f "$f" ] && cp "$f" "${f%.pt}_$ts.pt" && echo "✅ $port/$stage snapshot"
  done
done
```

**6.2 Report archiving**

```jsx
mkdir -p reports/local_training_$(date +%Y%m%d)
cp -v models/releases/$(date +%F)/consistency_* reports/local_training_$(date +%Y%m%d)/

cat > reports/local_training_$(date +%Y%m%d)/SUMMARY.md << EOF
# 本地训练完成报告
- 日期：$(date)
- 样本量：800
- 种子：42,123,2025
- 缓存：--no-cache
- 阈值配置：
$(cat configs/thresholds.yaml)
EOF
```

## **7) "Attainment" determination (what you see in the panel "turning green")**

- **Default**: **Win Rate** for the phase **≥ threshold** (see configs/thresholds.yaml)
- **or**: **Wilson 95% lower bound ≥ threshold - 0.04** (more stable with enough samples)
- Requirements: --no-cache, **multiple seeds** (≥3), **enough samples** (≥800)

# 4.Flower Federated Experiment (Balanced/Unbalanced/Fair)

 Machine: Server, Mac+WSL=Client

## (1) Pre-run preparation (three machines)

1.  Enter your repository root directory

```jsx
cd ~/traffic_rl    # 按你的真实路径
```

### Suggested virtual environment

```jsx
python -m venv .venv && source .venv/bin/activate
```

### Dependencies

```jsx
pip install -q flwr torch numpy pandas matplotlib pyyaml
```

### Unify PYTHONPATH

```jsx
export PYTHONPATH=$PWD:$PYTHONPATH
```

### Route data and cache (local data if available)

```jsx
export DATA_ROOT=$PWD/data/ports
export ROLLOUT_CACHE=$PWD/data/cache/rollouts
mkdir -p "$DATA_ROOT" "$ROLLOUT_CACHE"
```

### Real reviews can be turned off during the training phase to speed up (strong reviews are done uniformly at the end)

```jsx
export FLW_EVAL_OFF=1
```

## (2) Clearance

## **Server:**

```jsx
   source .venv/bin/activate
   export PYTHONPATH=$PWD:$PYTHONPATH
```

### Clear the old strong evaluation JSON

```jsx
rm -f reports/FLW_flw_20250821_*/nightly/forced_*.json || true
```

### Training run catalog

```jsx
rm -rf models/flw/flower_run
mkdir -p models/flw/flower_run
```

# **One-time preparation (all machines run)**

```
# 进入工程并激活虚拟环境
cd ~/traffic_rl          # 按你的真实路径
source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH

# 保险装依赖
pip install -q flwr torch numpy pandas matplotlib pyyaml

# 训练阶段关闭真实评测可提速（可选）
export FLW_EVAL_OFF=1

# 日志/模型目录（不存在就建）
mkdir -p logs models/flw/flower_run
```

---

# **1) Balanced 200 rounds (same strength on all four ends, baseline)**

## **1.1 Aggregation end (Ubuntu, tmux backend run)**

```
cd ~/traffic_rl && source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
tmux kill-session -t flower 2>/dev/null || true
tmux new -d -s flower -n server \
"bash -lc 'cd ~/traffic_rl && source .venv/bin/activate && export PYTHONPATH=\$PWD:\$PYTHONPATH && \
 python scripts/flower/server.py \
   --rounds 200 --min-clients 4 --fair-agg fedavg --alpha 0.5 \
   --save-dir models/flw/flower_run \
   2>&1 | tee logs/server_\$(date +%Y%m%d_%H%M%S)_balanced200.log'"
tmux capture-pane -p -t flower:server | tail -n 20
```

## **1.2 Mac client (gulfport, new_orleans)**

```
cd ~/traffic_rl && source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
S="43.163.97.188:8080"   # 改成你的服务器:端口
for p in gulfport new_orleans; do
  python scripts/flower/client.py --server "$S" --port "$p" \
    --episodes 8 --ppo-epochs 4 --batch-size 64 --entropy-coef 0.01 \
    2>&1 | tee -a logs/client_${p}.log &
done
wait
```

## **1.3 Windows/WSL client (south_louisiana, baton_rouge)**

```
cd ~/traffic_rl && source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
S="43.163.97.188:8080"
for p in south_louisiana baton_rouge; do
  python scripts/flower/client.py --server "$S" --port "$p" \
    --episodes 8 --ppo-epochs 4 --batch-size 64 --entropy-coef 0.01 \
    2>&1 | tee -a logs/client_${p}.log &
done
wait
```

> Observe if there are repeated occurrences in the logs of the aggregation side
> 

> configure_fit: strategy sampled 4 clients... with aggregate_fit: received 4 results...
> 

## **1.4 Archive Balanced200 (aggregation side)**

```
cd ~/traffic_rl && source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH

export BAL=flw_$(date +%Y%m%d_%H%M%S)_Balance200
mkdir -p models/flw/$BAL reports/FLW_$BAL
cp models/flw/flower_run/*.pt models/flw/$BAL/
python - <<'PY'
import os,glob,hashlib,json,time
from pathlib import Path
tag=os.environ["BAL"]; d=Path(f"models/flw/{tag}")
files=sorted(glob.glob(str(d/"global_round_*.pt")))
items=[]
for f in files:
    b=open(f,'rb').read()
    items.append({"file":Path(f).name,"size":len(b),"sha256":hashlib.sha256(b).hexdigest()})
manifest={"tag":tag,"ts":time.strftime("%F %T"),"files":items,"notes":"Balanced200 (8/8 per client)"}
json.dump(manifest, open(d/"MANIFEST.json","w"), indent=2)
print("WROTE", d/"MANIFEST.json")
PY
```

---

# **2) Unbalanced 200 rounds (most 8/8, few 2/2)**

## **2.1 Aggregation side (fedavg, unchanged)**

```
tmux kill-session -t flower 2>/dev/null || true
tmux new -d -s flower -n server \
"bash -lc 'cd ~/traffic_rl && source .venv/bin/activate && export PYTHONPATH=\$PWD:\$PYTHONPATH && \
 python scripts/flower/server.py \
   --rounds 200 --min-clients 4 --fair-agg fedavg --alpha 0.5 \
   --save-dir models/flw/flower_run \
   2>&1 | tee logs/server_\$(date +%Y%m%d_%H%M%S)_unbalanced200.log'"
tmux capture-pane -p -t flower:server | tail -n 20
```

## **2.2 Client side (Mac = majority domain 8/8; WSL = minority domain 2/2)**

```
# Mac（多数域 8/8）
cd ~/traffic_rl && source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH; S="43.163.97.188:8080"
for p in baton_rouge new_orleans; do
  python scripts/flower/client.py --server "$S" --port "$p" \
    --episodes 8 --ppo-epochs 4 --batch-size 64 --entropy-coef 0.01 \
    2>&1 | tee -a logs/client_${p}.log &
done
wait
```

```
# Windows/WSL（少数域 2/2）
cd ~/traffic_rl && source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH; S="43.163.97.188:8080"
for p in south_louisiana gulfport; do
  python scripts/flower/client.py --server "$S" --port "$p" \
    --episodes 2 --ppo-epochs 4 --batch-size 64 --entropy-coef 0.01 \
    2>&1 | tee -a logs/client_${p}.log &
done
wait
```

## **2.3 Archive Unbalanced 200 (aggregation end)**

```
cd ~/traffic_rl && source .venv/bin/activate

export PYTHONPATH=$PWD:$PYTHONPATH

export UNB=flw_$(date +%Y%m%d_%H%M%S)_Unbalanced200_8_8_vs_2_2

mkdir -p models/flw/$UNB reports/FLW_$UNB

cp models/flw/flower_run/*.pt models/flw/$UNB/

python - <<'PY'
import os,glob,hashlib,json,time
from pathlib import Path
tag=os.environ["UNB"]; d=Path(f"models/flw/{tag}")
items=[]
for f in sorted(glob.glob(str(d/"global_round_*.pt"))):
    b=open(f,'rb').read()
    items.append({"file":Path(f).name,"size":len(b),"sha256":hashlib.sha256(b).hexdigest()})
manifest={"tag":tag,"ts":time.strftime("%F %T"),"files":items,"notes":"Unbalanced200 (majority 8/8, minority 2/2)"}
json.dump(manifest, open(d/"MANIFEST.json","w"), indent=2)
print("WROTE", d/"MANIFEST.json")
PY
```

---

# **3) Fair-Unbalanced 200 round (fair aggregation)**

## **3.1 Aggregation end (invsize fair aggregation)**

```
tmux kill-session -t flower 2>/dev/null || true

tmux new -d -s flower -n server \
"bash -lc 'cd ~/traffic_rl && source .venv/bin/activate && export PYTHONPATH=\$PWD:\$PYTHONPATH && \
 python scripts/flower/server.py \
   --rounds 200 --min-clients 4 --fair-agg invsize --alpha 0.5 \
   --save-dir models/flw/flower_run \
   2>&1 | tee logs/server_\$(date +%Y%m%d_%H%M%S)_fair_unbalanced200.log'"

tmux capture-pane -p -t flower:server | tail -n 20
```

## **3.2 Client side (consistent with "Unbalanced": Mac=8, WSL=2)**

```jsx
Mac（多数域 8/8）
cd ~/traffic_rl && source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH; S="43.163.97.188:8080"
for p in baton_rouge new_orleans; do
python scripts/flower/client.py --server "$S" --port "$p" \
--episodes 8 --ppo-epochs 4 --batch-size 64 --entropy-coef 0.01 \
2>&1 | tee -a logs/client_${p}.log &
done
wait
```

```jsx
Windows/WSL（少数域 2/2）
cd ~/traffic_rl && source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH; S="43.163.97.188:8080"
for p in south_louisiana gulfport; do
python scripts/flower/client.py --server "$S" --port "$p" \
--episodes 2 --ppo-epochs 4 --batch-size 64 --entropy-coef 0.01 \
2>&1 | tee -a logs/client_${p}.log &
done
wait
```

## **3.3 Archive FairUnbalanced200 (aggregation side)**

```
cd ~/traffic_rl && source .venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
export FAIR=flw_$(date +%Y%m%d_%H%M%S)_FairUnbalanced200_invsize
mkdir -p models/flw/$FAIR reports/FLW_$FAIR
cp models/flw/flower_run/*.pt models/flw/$FAIR/
python - <<'PY'
import os,glob,hashlib,json,time
from pathlib import Path
tag=os.environ["FAIR"]; d=Path(f"models/flw/{tag}")
items=[]
for f in sorted(glob.glob(str(d/"global_round_*.pt"))):
    b=open(f,'rb').read()
    items.append({"file":Path(f).name,"size":len(b),"sha256":hashlib.sha256(b).hexdigest()})
manifest={"tag":tag,"ts":time.strftime("%F %T"),"files":items,"notes":"FairUnbalanced200 (invsize on server; majority 8/8, minority 2/2)"}
json.dump(manifest, open(d/"MANIFEST.json","w"), indent=2)
print("WROTE", d/"MANIFEST.json")
PY
```

---

# **4) Fixed two steps after training (archive + nightly test/panel)**

```
# A) 记录最后一次成功 tag（可选）
echo "$FAIR" | tee models/flw/LAST_SUCCESS.tag

# B) 夜测（一致性回归，非联邦，只做离线评测与面板刷新）
source .venv/bin/activate && export PYTHONPATH=$PWD:$PYTHONPATH
python scripts/nightly_ci.py --ports all --samples 800 --seeds 42,123,2025 --no-cache
python scripts/monitoring_dashboard.py
```

---

### Configure the evaluation density

```jsx
若没在本会话里设置过，重新 export 一次三组 tag：
# 若没在本会话里设置过，重新 export 一次三组 tag：
# export BAL=flw_YYYYmmdd_HHMMSS_Balance200
# export UNB=flw_YYYYmmdd_HHMMSS_Unbalanced200_8_8_vs_2_2
# export FAIR=flw_YYYYmmdd_HHMMSS_FairUnbalanced200_invsize

export FLW_EVAL_SAMPLES=1600
export SEEDS="42 123 2025 31415 2718"
```

---

**Clear nightly, generate 60 strong reviews JSON**

```jsx
for T in "$BAL" "$UNB" "$FAIR"; do
  rm -f reports/FLW_${T}/nightly/forced_*.json
done

for TAG in "$BAL" "$UNB" "$FAIR"; do
  echo "== Force eval for $TAG =="
  CKPT=$(ls models/flw/${TAG}/global_round_*.pt | sort | tail -n1)
  [ -z "$CKPT" ] && echo "❌ 没找到 ckpt for $TAG" && continue
  for p in baton_rouge new_orleans south_louisiana gulfport; do
    for s in $SEEDS; do
      TAG="$TAG" CKPT="$CKPT" PORT="$p" SEED="$s" SAMPLES="$FLW_EVAL_SAMPLES" \
      python - <<'PY'
import os, json, time
from pathlib import Path
from src.federated.eval_bridge import eval_port_with_fed_mlp
TAG=os.environ["TAG"]; CKPT=os.environ["CKPT"]; PORT=os.environ["PORT"]
SEED=int(os.environ["SEED"]); SAMPLES=int(os.environ["SAMPLES"])
res = eval_port_with_fed_mlp(PORT, CKPT, samples=SAMPLES, seed=SEED, verbose=False)
out = {"port":PORT,"seed":SEED,"samples":SAMPLES,
       "success_rate":res.get("success_rate"),
       "avg_reward":res.get("avg_reward"),
       "num_samples":res.get("num_samples",SAMPLES),
       "source":res.get("source",CKPT),
       "ts":time.strftime("%F %T")}
path = Path(f"reports/FLW_{TAG}/nightly/forced_{PORT}_seed{SEED}.json")
path.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(path,"w"), indent=2, ensure_ascii=False)
print("WROTE", path)
PY
    done
  done
done
```

```jsx
python - <<'PY'
import os, glob, json
tags=[os.environ['BAL'],os.environ['UNB'],os.environ['FAIR']]
tot=0
for T in tags:
  fs=glob.glob(f"reports/FLW_{T}/nightly/forced_*.json")
  miss=sum(1 for f in fs if json.load(open(f)).get("success_rate") is None)
  print(f"{T}: files={len(fs)}, empty_success_rate={miss}")
  tot+=len(fs)
print("TOTAL:", tot, "(expect 60)")
PY
```

 Statistical significance and confidence intervals

```jsx
python scripts/stats_sigcheck.py
```

 Plot

```jsx
python scripts/summarize_anyjson.py
column -t -s, reports/SUMMARY_JSON/by_port_mean.csv
echo
column -t -s, reports/SUMMARY_JSON/minority_gain.csv
```

 Ready

```jsx
python scripts/make_camera_ready.py \
  --input reports/SUMMARY_JSON \
  --out reports/CAMERA_READY_$(date +%F)
```

**4) Package and Publish & Share Download Link**

```jsx
STAMP=$(date +%F)
DEST="reports/RELEASE_${STAMP}"
mkdir -p "$DEST"

# 汇总 CSV / 置信区间 / 显著性
cp -v reports/SUMMARY_JSON/* "$DEST"/ 2>/dev/null || true

# 相册（如已生成）
cp -rv reports/CAMERA_READY_${STAMP}/figs "$DEST"/figs 2>/dev/null || true
cp -rv reports/CAMERA_READY_${STAMP}/tables "$DEST"/tables 2>/dev/null || true

# nightly JSON + MANIFEST
for T in "$BAL" "$UNB" "$FAIR"; do
  mkdir -p "$DEST/FLW_${T}/nightly"
  cp -v reports/FLW_${T}/nightly/forced_*.json "$DEST/FLW_${T}/nightly/" 2>/dev/null || true
  cp -v models/flw/${T}/MANIFEST.json "$DEST/MANIFEST_${T}.json" 2>/dev/null || true
done

# 打 zip
cd "$DEST"/..
ZIP="pub_suite_${STAMP}.zip"; rm -f "$ZIP"
zip -9r "$ZIP" "RELEASE_${STAMP}" >/dev/null
sha256sum "$ZIP"; ls -lh "$ZIP"
```

 Upload

```jsx
服务器：
curl -fsS --connect-timeout 5 --retry 5 --retry-delay 2 \
  -F "file=@${ZIP}" https://0x0.st
Mac：
cd ~/Downloads
curl -L -o pub_suite_${STAMP}.zip 'https://0x0.st/XXXXX.zip'
```

## **Cheat Sheet & Quick Troubleshooting**

- Look at the last 20 lines of the tmux server log:

```
tmux capture-pane -p -t flower:server | tail -n 20
```

- Confirm that the model is down:

```
ls -lh models/flw/flower_run/global_round_200.pt
```

# 5. Results

## **Overall takeaway**

**Chinese**: Our fair federation strategy, **FairUnbalanced**, can significantly improve the completion rate of small ports (a few domains) without sacrificing the performance of large ports; at the system level, its **macro averages** are the same as those of Balanced, but it reduces **the fairness gap (the maximum-minimum difference in the success rates of ports)** from the larger level of Unbalanced back to the level of Balanced; the cross-random seeding **variance** is the same as that of Balanced. At the system level, it has the same macro average as Balanced, but reduces the fairness gap (the maximum-minimum difference in success rates across ports) from the larger level of Unbalanced back to the level of Balanced; the variance across random seeds is the same as Balanced, making training more stable. Deployment cost is the same as FedAvg (only server-side aggregation rules changed). non-identically distributed (non-IID)

**English**: Our fairness-aware federated strategy **FairUnbalanced** boosts minority ports without hurting major ports. At the system level, it matches the **macro average** of Balanced while **reducing the fairness gap** back from the larger Unbalanced level to the Balanced level. Seed-to-seed variability is on par with Balanced, indicating stable training. Seed-to-seed variability is on par with Balanced, indicating stable training. Deployment cost is unchanged from FedAvg (server-side aggregation only).

 Simple: **FairUnbalanced is the best**: it brings up the performance of small ports **without** affecting large ports; **the overall average** is as good as Balanced; **the port-to-port variability** is smaller than that of Unbalanced; **the stability** is about the same as that of Balanced; and the deployment cost is unchanged from FedAvg (server-side weights only).

## **Figure 2: Per-port success (95% CI)**

![pub_port_bar_ci.png](Experiment%20Replication%20270c7512318a807fbeeec6bb34cef00a/pub_port_bar_ci.png)

**What this graph says**: Look at the success rate of each port under the three scenarios, with error bars at the 95% confidence interval.

**Success rate**: scheduling all ships in the current batch within the specified timeframe without deadlocks, boundary violations, or hard constraint violations.

**Chinese term**:

- In **Gulfport** and **South Louisiana** (a few domains), the orange bar (FairUnbalanced) is higher than the green bar (Unbalanced) and the error bars do not overlap/barely overlap, indicating that **the lift is statistically reliable**.
- In **Baton Rouge** & **New Orleans** (most domains), the orange is at the same level as the blue (Balanced), indicating that **no strong customers are sacrificed**.
- This graph directly answers the central question: **fair aggregation gives weight to weak ports, but does not depress strong ports**.

**English**.

- At **Gulfport** and **South Louisiana** (minority ports), the orange bars (FairUnbalanced) are above the green bars (Unbalanced). The 95% CIs do not overlap (or only marginally), indicating **statistically reliable gains**.
- At **Baton Rouge** and **New Orleans** (majority ports), orange is at the **same level as blue** (Balanced), so **no degradation** for strong clients.
- This figure answers the main question: **the mixer helps the weak without hurting the strong**.

## **Figure 3: FairUnbalanced - Unbalanced (95% CI, Newcombe)**

**This figure says something**: replace Figure 2 with the "difference view", where the right side of the 0 line is boosted.

![Per-port improvement with 95% CI (Newcombe).png](Experiment%20Replication%20270c7512318a807fbeeec6bb34cef00a/Per-port_improvement_with_95_CI_(Newcombe).png)

**In Chinese**:

- The points for all four ports are to the right of 0, and the **Newcombe 95% CIdoes not cross 0** for any of them, so **the boosts are significant** rather than sampling noise.
- **The largest boost** occurs in **Gulfport**, followed by **South Louisiana**; most domains also have a **small positive direction**.
- This figure demonstrates that **the gain of the method is reproducible and directional**, and is especially effective in a few domains.

**English**.

- All four effects are **positive** and the **95% Newcombe CIs** **exclude zero**, so the lifts are **statistically significant**.
- The **largest effect** is at **Gulfport**, followed by **South Louisiana**; majority ports show **smaller positive deltas**.
- This confirms the gains are **robust and targeted**, with the strongest benefit on minority ports.

## **Table 4 + Figure 4: Macro average & Fairness gap**

**This group is about** system-level "utility-fairness" trade-offs.

![fig_macro_gap_lines.png](Experiment%20Replication%20270c7512318a807fbeeec6bb34cef00a/fig_macro_gap_lines.png)

**The Chinese terminology**:

- **Macro average (simple average of success rates across ports)**: FairUnbalanced = **0.512**, **exactly the same** as Balanced, higher than Unbalanced's **0.492** - **no utility cost**.
- **Fairness gap (maximum success rate - minimum success rate)**: **0.090** for Unbalanced (a large gap), compared to **0.069** for both Balanced and FairUnbalanced, suggesting that **FairUnbalanced reduces the cross-port gap back to Balanced's level**.
- Conclusion: FairUnbalanced **reduces the cross-port gapwithout reducing overall utility**, and **is more fairly deployable** than Unbalanced.

**English**.

- **Macro average**: FairUnbalanced = **0.512**, identical to Balanced and higher than Unbalanced **(0.492** ). **No utility tax**.
- **Fairness gap**: Unbalanced **0.090** vs. Balanced and FairUnbalanced **0.069**. FairUnbalanced **reduces disparity** to the Balanced range.
- Takeaway: **Same utility, smaller gap-a** better utility-equity trade-off than Unbalanced.

## **Figure 5: Seed-to-seed variability (std; smaller is more stable)**

**This figure is about** stability and reproducibility across random seeds.

![fig_seed_std_bars_simple.png](Experiment%20Replication%20270c7512318a807fbeeec6bb34cef00a/fig_seed_std_bars_simple.png)

**The Chinese version**:

- In **Baton Rouge** and **South Louisiana**, Unbalanced has higher variance; FairUnbalanced **pulls the variance back into the Balanced range**.
- No deterioration in variance was observed in the other ports.
- This means that **FairUnbalanced training is more stable, results are more reproducible**, and there is less risk of landing a project.

**English**.

- At **Baton Rouge** and **South Louisiana**, Unbalanced has higher variability; FairUnbalanced **returns the std to the Balanced range**.
- No ports show a variance increase under FairUnbalanced.
- Hence **training is stable and reproducible**, which matters for engineering deployment.

## **Final concluding line**

**Chinese**: Putting all four graphs and tables together, **FairUnbalanced is the best of the three**: a few domains are significantly raised, but most domains are not lowered; the macro average is equivalent to Balanced; the fairness gap is lower than Unbalanced; the stability is equivalent to Balanced; and the deployment cost is the same as that of FedAvg, with the change of only the server-side aggregation weights.

**English**: Putting all evidence together, **FairUnbalanced is the preferred setting**: it raises minority ports, does not hurt majority ports, matches Balanced in macro utility, reduces the fairness gap vs. Unbalanced, keeps seed-level stability, and preserves FedAvg-level deployment cost with a server-only change.

# 6. Core Algorithms

## 1) GAT

 We model each port as a graph. Nodes are berths, anchorages, channels, terminals, and a few vessel placeholders. Edges are binary (connected or not) and Edges are binary (connected or not) and can change with vessel status, so links act like on/off switches. This keeps structure editable and stable; risk factors (width, depth, current, congestion) are learned via node features instead of hard-coding edge weights. Every node gets the same 8-dimensional feature template: the first entry is a type id; the remaining slots hold the key normalized attributes for that node type (e.g., berth occupancy/queue/ utilization/efficiency; channel depth/current/congestion). utilization/efficiency; channel depth/width/traffic/tide/current; vessel length/beam/draught/speed/type/active). Missing fields are filled with zeros. Faster, global, time-varying signals (throughput, weather, overall congestion) are put into a Faster, global, time-varying signals (throughput, weather, overall congestion) are put into a separate 56-dimensional global vector per timestep. Attention is computed only between neighbors allowed by the binary adjacency. After each layer we apply LayerNorm; if input and output widths match we add After each layer we apply LayerNorm; if input and output widths match we add a residual skip, otherwise we only normalize. We then mean-pool all node embeddings (simple dimension-wise average) to get a We then mean-pool all node embeddings (simple dimension-wise average) to get a single graph embedding that is robust across ports. fused vector through a small MLP (Linear → ReLU → LayerNorm) to blend features, and feed it to the actor and critic heads for the decision at that timestep.

- MHA (Multi-Head Attention): multiple attention heads in parallel assign weights to neighboring nodes from different perspectives (using adjacency masks in the graph to look only at connectable neighbors), and then aggregate.
- + LN (LayerNorm): do layer normalization for each layer output to stabilize the distribution and accelerate convergence; when the input/output channels are the same, also add the residuals and do normalization again.

 2) PPO

 PPO is safe trial-and-error learning. In each round we run the current policy for a short rollout and record states, actions, rewards, and the policy's In each round we run the current policy for a short rollout and record states, actions, rewards, and the policy's log-probs (its confidence). We then turn the trajectory into advantages with GAE-an estimate of how much better than usual each action was, smoothed to reduce noise while still looking ahead. We then turn the trajectory into advantages with GAE-an estimate of how much better than usual each action was, smoothed to reduce noise while still looking ahead. Next we update the policy with a clipped objective: if the new/old probability ratio moves too far, we clip it before computing Next we update the policy with a clipped objective: if the new/old probability ratio moves too far, we clip it before computing the loss so each step stays small and safe. We also add an entropy bonus to keep exploring and train a value network to predict how good each state is, which stabilizes the advantages. stabilizes the advantages. Repeating this collect → score → small, clipped update loop makes learning steady, efficient, and robust.

 3) Fairness

 Use inverse-size aggregation instead of FedAvg. Give each client a weight that is inversely proportional to its data size, raised to a power alpha, then normalize the weights and take the weights of each client. Give each client a weight that is inversely proportional to its data size, raised to a power alpha, then normalize the weights and take the weighted average. Smaller clients get larger weights. Alpha = 0 means almost equal weights; a moderate alpha gently Alpha = 0 means almost equal weights; a moderate alpha gently boosts small clients; a large alpha strongly boosts them but can amplify noise. This prevents big clients from dominating when data are imbalanced, and This prevents big clients from dominating when data are imbalanced, and behaves close to FedAvg when data are already balanced.

# 7. File roles

### **@configs/ - Configuration file directory**

- **thresholds.yaml**: Configuration of success thresholds for different training phases for each port
- Defines performance thresholds for 4 ports (gulfport, baton_rouge, new_orleans, south_louisiana) for different training phases.
- Used for stage switching judgment in course learning

### **@data/ - data processing module**

- **comprehensive_ais_processor.py**: comprehensive AIS data processor (25KB, 643 lines)
- Processes raw AIS ship track data
- Data cleaning, feature engineering, format conversion
- **create_train_val_test_splits.py**: dataset splitting script
- Split the processed data into training, validation and test sets.
- **DATA_PROCESSING_REPORT.md**: Data processing report.
- Detailed records of data cleaning statistics, quality assessment, and transformation results.
- **gat_training_data/**: GAT training data storage.
- **processed/**: processed data file
- **raw/**: raw AIS data file

### **@docs/ - documentation directory**

- **FL_OPERATION_MANUAL.md**: Federal Learning Operations Manual
- Detailed FL startup, operation, and monitoring guide
- **threshold_rollback_plan.md**: threshold rollback plan
- **weekly_fix_plan.md**: weekly fix plan

### **@logs/ - log file directory**

- **client_*.log**: Individual client training logs
- client_baton_rouge.log, client_gulfport.log, client_new_orleans.log, etc.
- **flower_autoeval.out**: Flower autoevaluation output
- **quick_test_results.json**: quick test results
- **nightly/**: nightly batch log
- **fl/**: federated learning log
- **training/**: training process log

### **@models/ - model storage directory**

- **curriculum_v2/**: curriculum v2 models
- **fine_tuned/**: fine-tuned models
- **fl/**: federated learning model
- **flw/**: Flower federated learning model
- **releases/**: Release version model (824 files)
- **single_port/**: Single port training model

### **@reports/ - report generation directory**

- **consistency_***: consistency test reports (CSV, HTML, MD formats)
- **summary_*.json**: summary of test results
- **EOD_20250812/**: end-of-day report
- **FL_***: Federal Learning Report
- **FLW_***: Flower Federal Learning Report

### **@results/ - Experiment Results Catalog**

- **federated/**: federated learning experiment results
- **alpha_sensitivity/**: alpha parameter sensitivity analysis results

### **@runs/ - training run logs**

- **Aug12_16-05-32_*/**: Timestamped training runs

### **@scripts/ - scripts tools directory**

- **nightly_ci.py**: nightly CI check scripts
- **fl_train.py**: federal learning training scripts
- **fl_eval_callback.py**: FL evaluation callbacks
- **conservative_fine_tune*.py**: conservative fine-tuning script family
- **aggressive_fine_tune.py**: aggressive fine-tuning scripts
- **progressive_training.py**: progressive training
- **weekly_fix_checklist.py**: weekly fix checklist
- **flower/**: Flower-related scripts
- **crontab.***: Timed task configuration

### **@src/ - source code directory**

- **federated/**: federated learning core code (39 files)
- GAT-PPO intelligences, data processing, trainers, evaluators, etc.
- **models/**: model definition code (4 files)
- GAT networks, PPO algorithms, fairness rewards, etc.

### **@topologies/ - topology configuration directory**

- **maritime_3x3_*.json**: 3×3 maritime topology configuration
- **maritime_4x4_*.json**: 4×4 maritime topology configuration
- **maritime_5x5_*.json**: 5×5 maritime topology configuration
- **maritime_6x6_*.json**: 6×6 maritime topology configuration
- Contains configuration, traffic, network, and statistical information for each port.

### **Root directory files**

- **ci_check.py**: CI check script for quick verification of system status.
- **hyperparameters_table.py**: script to generate hyperparameters table
- **main.py**: main program entry, integrates the whole experiment process.
- **training_results.json**: training results log (528 rows), containing detailed performance data for the 4 training phases.
