#!/usr/bin/env bash
set -Eeuo pipefail

# === 基本路径 ===
ROOT="/Users/kaffy/Documents/GAT-FedPPO"
cd "$ROOT"

# === Python 环境 & 路径 ===
ENV_ACT='source .venv/bin/activate'
PYSET='export PYTHONPATH=$PWD:$PYTHONPATH'

# === 可调参数（支持环境变量覆盖）===
ROUNDS="${ROUNDS:-200}"
MINC="${MIN_CLIENTS:-4}"
SERVER_ADDR="${SERVER_ADDR:-127.0.0.1:8080}"

EPISODES="${EPISODES:-8}"
PPO_EPOCHS="${PPO_EPOCHS:-4}"
BATCH_SIZE="${BATCH_SIZE:-64}"
ENTROPY_COEF="${ENTROPY_COEF:-0.01}"

# === 运行目录 & 日志 ===
RUN_TAG="flw_$(date +%Y%m%d_%H%M%S)"
SAVE_DIR="models/flw/${RUN_TAG}"
SAVE_LINK="models/flw/flower_run"    # 监听器监控这个目录
mkdir -p "$SAVE_DIR" logs/tmux models/flw
ln -sfn "$SAVE_DIR" "$SAVE_LINK"

# === 选择初始权重（优先用 FL 基线，没有就用上次 FLW）===
INIT=""
if [[ -f models/fl/LAST_SUCCESS.tag ]]; then
  FLTAG=$(cat models/fl/LAST_SUCCESS.tag)
  CANDIDATE="models/fl/${FLTAG}/global_best.pt"
  [[ -f "$CANDIDATE" ]] && INIT="$CANDIDATE"
fi
if [[ -z "${INIT}" && -f models/flw/LAST_SUCCESS.tag ]]; then
  FLWTAG=$(cat models/flw/LAST_SUCCESS.tag)
  CANDIDATE="models/flw/${FLWTAG}/global_best.pt"
  [[ -f "$CANDIDATE" ]] && INIT="$CANDIDATE"
fi

# === 启动 tmux 会话 ===
SESSION="flower"
if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "⚠️ tmux 会话 '$SESSION' 已存在。可执行：tmux attach -t $SESSION 或 tmux kill-session -t $SESSION"
  exit 1
fi

tmux new-session -d -s "$SESSION" -n server

# --- Server 窗格 ---
tmux send-keys -t "$SESSION":server "cd $ROOT" C-m
tmux send-keys -t "$SESSION":server "$ENV_ACT ; $PYSET" C-m
tmux send-keys -t "$SESSION":server \
  "python scripts/flower/server.py --rounds $ROUNDS --min-clients $MINC --save-dir '$SAVE_DIR' |& tee 'logs/tmux/server.log'" C-m

# --- Clients 窗口 ---
tmux new-window -t "$SESSION" -n clients
ports=(gulfport new_orleans south_louisiana baton_rouge)

for i in {0..3}; do
  (( i==0 )) || tmux split-window -t "$SESSION":clients -v
  tmux select-layout -t "$SESSION":clients tiled >/dev/null
  p="${ports[$i]}"

  tmux send-keys -t "$SESSION":clients.$i "cd $ROOT" C-m
  tmux send-keys -t "$SESSION":clients.$i "$ENV_ACT ; $PYSET" C-m

  base_cmd="python scripts/flower/client.py --server $SERVER_ADDR --port $p \
    --episodes $EPISODES --ppo-epochs $PPO_EPOCHS --batch-size $BATCH_SIZE --entropy-coef $ENTROPY_COEF"

  if [[ -n "$INIT" ]]; then
    base_cmd="$base_cmd --init '$INIT'"
  fi

  tmux send-keys -t "$SESSION":clients.$i "$base_cmd |& tee 'logs/tmux/client_${p}.log'" C-m
done

# --- 监听器窗口 ---
tmux new-window -t "$SESSION" -n listener
tmux send-keys -t "$SESSION":listener "cd $ROOT" C-m
tmux send-keys -t "$SESSION":listener "$ENV_ACT ; $PYSET" C-m
tmux send-keys -t "$SESSION":listener \
  "python scripts/flower/autoeval_listener.py |& tee 'logs/flower_autoeval.out'" C-m

# --- Info 窗口（展示关键信息）---
tmux new-window -t "$SESSION" -n info
tmux send-keys -t "$SESSION":info "echo RUN_TAG=$RUN_TAG" C-m
tmux send-keys -t "$SESSION":info "echo SAVE_DIR=$SAVE_DIR (→ $SAVE_LINK)" C-m
tmux send-keys -t "$SESSION":info "echo SERVER=$SERVER_ADDR  ROUNDS=$ROUNDS  MIN_CLIENTS=$MINC" C-m
tmux send-keys -t "$SESSION":info "echo INIT=$INIT" C-m
tmux send-keys -t "$SESSION":info "ls -la '$SAVE_DIR'" C-m

echo "✅ tmux 会话 '$SESSION' 已启动"
echo "   进入会话： tmux attach -t $SESSION"
echo "   退出不杀： 按 Ctrl-b 然后 d"
echo "   查看日志： tail -f logs/tmux/server.log" 