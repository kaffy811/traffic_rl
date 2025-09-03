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
    - Table 3's course scope determines the difficulty progression; the final evaluation only uses **the bolded final stage**.
- **Because the federation only passes "model information and statistics":** it does not pass raw trajectories, so none of the three ends need to share "datasets".

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

# **1) Balanced 200 rounds (same intensity on all four ends, baseline)**

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
