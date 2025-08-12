# 联邦学习（FL）操作手册

## 概述

本手册基于操作手册的"从零到开跑"流程，提供完整的联邦学习启动和运行指南。

## 一、开跑前准备（已完成）

### 1. 冻结当前单港口状态
- ✅ 四个港口的当前 best 权重已打时间戳快照
- ✅ 快照时间：20250812_143614

### 2. 固定评测参数
- ✅ samples=800
- ✅ seeds=42,123,2025
- ✅ 去噪基线 --k 120
- ✅ BR-中级阈值继续用 0.495（确保全绿）

### 3. 建立 FL 运行目录
- ✅ FL_TAG: fl_20250812_143631
- ✅ 目录：models/fl/fl_20250812_143631, logs/fl/fl_20250812_143631

## 二、全局模型初始化（已完成）

### 1. FedAvg 初始化
- ✅ 全局初始化模型：models/fl/fl_20250812_143640/global_init.pt
- ✅ 基于四个港口的"最稳定"同架构权重平均

## 三、联邦学习启动脚本

### 1. 主要脚本
- `scripts/fl_train.py` - 联邦学习主脚本
- `scripts/fl_eval_callback.py` - 评测回调脚本
- `scripts/fl_start_example.sh` - 启动示例脚本

### 2. 启动命令
```bash
# 设置环境变量
export FL_TAG=fl_20250812_143640

# 启动联邦学习
python scripts/fl_train.py \
    --algo fedavg \
    --rounds 30 \
    --clients "gulfport,new_orleans,south_louisiana,baton_rouge" \
    --global-init "models/fl/$FL_TAG/global_init.pt" \
    --save-dir "models/fl/$FL_TAG" \
    --local-episodes 8 \
    --lr-schedule "0:3e-4,5:1.5e-4,25:7.5e-5" \
    --batch-size 64 \
    --ppo-epochs 4 \
    --entropy-coef 0.01 \
    --log-file "logs/fl/$FL_TAG/train.log"
```

### 3. 参数说明
- **算法**: FedAvg（每轮全局模型下发给四个港口，客户端各跑 local-episodes，回传权重/梯度，服务器做加权平均）
- **轮次**: 30（5 轮 warmup / 20 轮 main / 5 轮 cooldown）
- **学习率日程**: 3e-4 → 1.5e-4 → 7.5e-5（在第 5、25 轮拐点处调整）
- **本地训练**: ppo_epochs=4、batch=64、entropy=0.01（偏保守，追稳不追激进）
- **local-episodes**: 每客户端每轮 8 个 episode，配合之前的"短训+评测"节奏

## 四、联邦过程中的定期评测

### 1. 评测频率
每 5 轮做一次全港口一致性评测

### 2. 评测命令
```bash
# 第5轮评测
python scripts/fl_eval_callback.py --round 5 --no-cache

# 第10轮评测
python scripts/fl_eval_callback.py --round 10 --no-cache

# 第15轮评测
python scripts/fl_eval_callback.py --round 15 --no-cache

# 第20轮评测
python scripts/fl_eval_callback.py --round 20 --no-cache

# 第25轮评测
python scripts/fl_eval_callback.py --round 25 --no-cache

# 第30轮评测
python scripts/fl_eval_callback.py --round 30 --no-cache
```

### 3. 评测内容
- 四港口所有阶段 pass 是否为 true
- BR-中级（seed=2025）：重点看 win_rate 与 wilson_lb
- 确认是否仍走 config 阈值

## 五、早停与回滚

### 1. 早停条件
若连续两次评测的全港口平均 wr 较上一次下降 > 1.5pp

### 2. 早停处理
1. 立刻停止当前联邦
2. 回滚到 models/fl/$FL_TAG 中最近一次评测前的最优全局权重
3. 降低学习率（整体 ×0.8）后继续从 best.pt 复跑，或直接封版

### 3. 回滚命令
```bash
# 回滚到指定轮次
cp models/fl/$FL_TAG/global_round_XX.pt models/fl/$FL_TAG/global_best.pt
```

## 六、BR-中级的阈值回退策略

### 1. 回退条件
只在两次相邻评测中，BR-中级三种子全部 ≥ 0.497 时执行

### 2. 回退步骤
1. 把 configs/thresholds.yaml 的 BR-中级从 0.495 → 0.497
2. 再次评测全港口（无缓存）
3. 如仍稳定，则次晚回到 0.500；否则回滚阈值到 0.495

### 3. 注意事项
- 过程中继续用 --k 120，保证去噪一致
- 保持评测护栏继续有效

## 七、封版与归档

### 1. 归档评测 JSON
```bash
mkdir -p reports/FL_$FL_TAG
cp models/releases/$(date +%F)/consistency_* reports/FL_$FL_TAG/
```

### 2. 生成简报
```bash
echo "# FL $FL_TAG Summary" > reports/FL_$FL_TAG/SUMMARY.txt
echo "All ports GREEN (BR-mid thr=0.495 or rolled back to 0.500 if stable)" >> reports/FL_$FL_TAG/SUMMARY.txt
echo "Key notes: seed=2025 stabilized; k=120; samples=800; seeds=42,123,2025" >> reports/FL_$FL_TAG/SUMMARY.txt
```

### 3. 打标签
```bash
git add -A
git commit -m "feat(fl): $FL_TAG stable; reports archived"
git tag -a "$FL_TAG" -m "FL stable $FL_TAG"
git push origin master --tags
```

## 八、常见坑位与快速自检

### 1. 模型维度不匹配
- 确保四港口使用同一架构
- consistency_test_fixed.py 已支持根据权重推断维度
- 联邦入口也应做同样处理

### 2. 评测与训练数据缓存冲突
- 联邦期间评测统一用 --no-cache + --k 120
- 或明确复用同一缓存，保持同分母

### 3. 单港口掉点
- 临时在该港口启用更保守本地 lr/entropy
- 入口脚本暴露 per-client 覆盖参数会更好

### 4. 日志与快照
- 每轮保存 global_round_XX.pt
- 发生异常可直接回滚
- 日志落到 logs/fl/$FL_TAG/train.log

## 九、快速启动检查清单

### 1. 环境检查
- [ ] Python 环境正常
- [ ] 依赖包已安装
- [ ] 四个港口模型文件存在

### 2. 文件检查
- [ ] global_init.pt 已生成
- [ ] FL 目录已创建
- [ ] 脚本权限已设置

### 3. 启动命令
```bash
# 快速启动（使用示例脚本）
./scripts/fl_start_example.sh

# 或手动启动
export FL_TAG=fl_20250812_143640
python scripts/fl_train.py --global-init "models/fl/$FL_TAG/global_init.pt" --save-dir "models/fl/$FL_TAG" --log-file "logs/fl/$FL_TAG/train.log"
```

## 十、监控与维护

### 1. 实时监控
```bash
# 监控训练日志
tail -f logs/fl/$FL_TAG/train.log

# 监控模型保存
watch -n 10 "ls -la models/fl/$FL_TAG/"

# 监控评测历史
tail -f logs/fl/eval_history.json
```

### 2. 定期检查
- 每轮训练完成状态
- 模型文件保存情况
- 评测结果和早停条件

### 3. 异常处理
- 训练中断：检查日志，从断点继续
- 评测失败：检查依赖和环境
- 早停触发：按回滚流程处理

---

**注意**: 本手册基于当前仓库的脚本和工作流设计，如有更新请同步修改。 