# Flower联邦学习使用指南

## 🚀 一键启动

### 基本启动
```bash
./scripts/flower/tmux_up.sh
```

### 自定义参数启动
```bash
# 调整训练轮数和客户端参数
ROUNDS=300 EPISODES=12 PPO_EPOCHS=6 ./scripts/flower/tmux_up.sh

# 调整学习参数
BATCH_SIZE=128 ENTROPY_COEF=0.02 ./scripts/flower/tmux_up.sh
```

## 📊 tmux会话管理

### 进入会话
```bash
tmux attach -t flower
```

### 会话内操作
- **切换窗口**: `Ctrl-b` + `数字键` 或 `Ctrl-b` + `n/p`
- **分割窗格**: `Ctrl-b` + `%` (垂直) 或 `Ctrl-b` + `"` (水平)
- **离开会话**: `Ctrl-b` + `d` (不终止进程)

### 会话外操作
```bash
# 列出所有会话
tmux ls

# 关闭会话
tmux kill-session -t flower

# 查看会话状态
tmux list-sessions
```

## 🔍 日志查看

### 实时日志
```bash
# 服务器日志
tail -f logs/tmux/server.log

# 客户端日志
tail -f logs/tmux/client_baton_rouge.log
tail -f logs/tmux/client_gulfport.log
tail -f logs/tmux/client_new_orleans.log
tail -f logs/tmux/client_south_louisiana.log

# 自动评测监听器日志
tail -f logs/flower_autoeval.out
```

### 历史日志
```bash
# 查看最近的日志
tail -n 100 logs/tmux/server.log
```

## ⚙️ 可调参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `ROUNDS` | 200 | 训练轮数 |
| `MIN_CLIENTS` | 4 | 最小客户端数量 |
| `SERVER_ADDR` | 127.0.0.1:8080 | 服务器地址 |
| `EPISODES` | 8 | 每轮训练episodes数 |
| `PPO_EPOCHS` | 4 | PPO训练轮数 |
| `BATCH_SIZE` | 64 | 批次大小 |
| `ENTROPY_COEF` | 0.01 | 熵系数 |

## 📁 目录结构

```
models/flw/
├── flower_run -> flw_YYYYMMDD_HHMMSS/  # 软链接，监听器监控
├── flw_YYYYMMDD_HHMMSS/               # 实际保存目录
│   ├── global_best.pt                 # 最佳模型
│   ├── global_round_001.pt            # 第1轮模型
│   ├── global_round_002.pt            # 第2轮模型
│   └── ...
└── LAST_SUCCESS.tag                   # 最新成功标签

logs/
├── tmux/                              # tmux会话日志
│   ├── server.log                     # 服务器日志
│   ├── client_baton_rouge.log         # BR客户端日志
│   ├── client_gulfport.log            # GP客户端日志
│   ├── client_new_orleans.log         # NO客户端日志
│   └── client_south_louisiana.log     # SL客户端日志
└── flower_autoeval.out                # 自动评测日志
```

## 🔄 工作流程

1. **启动**: `./scripts/flower/tmux_up.sh`
2. **自动创建**: 新的运行目录和软链接
3. **启动服务**: Server等待4个客户端连接
4. **客户端连接**: 4个港口客户端自动连接并开始训练
5. **自动评测**: 监听器监控新模型并触发夜测
6. **模型保存**: 每轮保存到指定目录

## 🛠️ 故障排除

### 常见问题

1. **tmux会话已存在**
   ```bash
   tmux kill-session -t flower
   ./scripts/flower/tmux_up.sh
   ```

2. **客户端连接失败**
   - 检查服务器是否启动: `tail -f logs/tmux/server.log`
   - 确认端口8080未被占用: `lsof -i :8080`

3. **模型保存失败**
   - 检查目录权限: `ls -la models/flw/`
   - 确认磁盘空间: `df -h`

### 调试模式
```bash
# 详细日志
DEBUG=1 ./scripts/flower/tmux_up.sh

# 查看环境变量
env | grep -E "(ROUNDS|EPISODES|BATCH_SIZE)"
```

## 📈 性能监控

### 训练进度
```bash
# 查看当前轮数
grep "ROUND" logs/tmux/server.log | tail -1

# 查看客户端状态
grep "configure_fit" logs/tmux/server.log | tail -5
```

### 资源使用
```bash
# 查看进程
ps aux | grep -E "(flower|python.*flower)" | grep -v grep

# 查看内存使用
ps aux | grep autoeval_listener | grep -v grep
```

## 🎯 最佳实践

1. **定期归档**: 训练完成后及时归档模型
2. **参数调优**: 根据性能调整训练参数
3. **日志管理**: 定期清理旧日志文件
4. **监控告警**: 设置关键指标监控
5. **备份策略**: 重要模型定期备份 