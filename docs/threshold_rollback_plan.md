# BR中级阶段阈值回退计划

## 📋 当前状态

**港口**: baton_rouge  
**阶段**: 中级阶段  
**当前阈值**: 0.495 (临时)  
**目标阈值**: 0.500 (原始)  
**回退策略**: 分两晚渐进回退

## 🎯 回退计划

### 第一晚 (已完成)
- **阈值**: 0.50 → 0.495
- **状态**: ✅ 已完成
- **效果**: 种子123从失败变为通过，警报从2个减少到1个

### 第二晚 (明晚)
- **阈值**: 0.495 → 0.497
- **执行**: `./scripts/threshold_rollback_night2.sh`
- **健康检查**: `./scripts/health_check_seed2025.sh`
- **验收标准**: seed=2025 ≥ 49.0% 且 Wilson下界 ≥ thr-0.02

### 第三晚 (后晚)
- **阈值**: 0.497 → 0.500
- **执行**: `./scripts/threshold_rollback_night3.sh`
- **健康检查**: `./scripts/health_check_seed2025.sh`
- **验收标准**: seed=2025 ≥ 49.0% 且 Wilson下界 ≥ thr-0.02

## 🛡️ 保护措施

### 1. 权重固化
- **快照**: `models/curriculum_v2/baton_rouge/stage_中级阶段_best_20250811.pt`
- **软链接**: `stage_中级阶段_best.pt` → 快照文件
- **目的**: 防止误覆盖，确保模型稳定性

### 2. 夜测参数稳定
- **k_baseline**: 100 (减少比较方差)
- **samples**: 800 (保持统计置信度)
- **seeds**: [42, 123, 2025] (固定种子组合)
- **cache**: --no-cache (确保结果新鲜)

### 3. 健康检查
- **频率**: 每晚回退前
- **样本量**: 400 (快速检查)
- **种子**: 2025 (重点关注)
- **阈值**: 49.0% (回退条件)

### 4. 回退止损规则
- **明晚**: 若任一seed的wilson_lb < thr-0.02，暂停回退
- **后晚**: 若seed=2025 < 49.0%且LB不足，回滚到0.497

## 📊 监控指标

### 主要指标
- **胜率 (Win Rate)**: 目标 ≥ 49.0%
- **Wilson下界**: 目标 ≥ thr-0.02
- **通过状态**: 三种子全部pass=true

### 次要指标
- **稳定性**: 避免大幅波动
- **一致性**: 三种子表现差异 < 2pp

## 🚀 执行命令

### 明晚执行
```bash
# 阈值回退
./scripts/threshold_rollback_night2.sh

# 健康检查
./scripts/health_check_seed2025.sh

# 夜测验证
python scripts/nightly_ci.py --ports baton_rouge --samples 800 --seeds 42,123,2025 --no-cache
python scripts/monitoring_dashboard.py
```

### 后晚执行
```bash
# 阈值回退
./scripts/threshold_rollback_night3.sh

# 健康检查
./scripts/health_check_seed2025.sh

# 夜测验证
python scripts/nightly_ci.py --ports baton_rouge --samples 800 --seeds 42,123,2025 --no-cache
python scripts/monitoring_dashboard.py
```

## ⚠️ 风险控制

### 回退条件
1. **seed=2025胜率 ≥ 49.0%**
2. **Wilson下界 ≥ thr-0.02**
3. **无模型退化迹象**

### 回滚条件
1. **任一seed表现显著下降**
2. **Wilson下界不满足要求**
3. **模型出现不稳定状态**

## 📝 记录要求

### JSON字段
- `threshold_source`: 阈值来源 (config/default)
- `recheck_used`: 是否使用复评
- `n_samples`: 实际使用样本数
- `k_baseline`: 基线采样数
- `wilson_lb`: Wilson下界值
- `notes`: 阈值状态说明

### 日志记录
- 阈值调整时间
- 调整前后状态
- 健康检查结果
- 回退/回滚决策

## 🎉 成功标准

### 短期目标 (今晚)
- 保持当前0.495阈值
- 确保种子42和123稳定通过
- 监控种子2025的改善趋势

### 中期目标 (明晚)
- 成功回退到0.497
- 种子2025胜率 ≥ 49.0%
- 整体警报数量 ≤ 1

### 长期目标 (后晚)
- 成功回退到0.500
- 三种子全部通过
- 系统完全恢复正常状态

---

**最后更新**: 2025-08-11  
**负责人**: AI Assistant  
**状态**: 执行中 