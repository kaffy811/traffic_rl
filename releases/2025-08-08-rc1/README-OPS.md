# GAT-FedPPO 运维手册 - Release 2025-08-08-rc1

## 快速部署

```bash
# 解压部署包
tar xzf gat-fedppo-2025-08-08-rc1.tgz
cd deploy/2025-08-08-rc1

# Docker部署
docker-compose up -d

# 验证部署
./smoke_test.sh
```

## 完整性验证

```bash
# 校验模型文件
shasum -c ../../releases/2025-08-08-rc1/manifest.sha256

# 检查模型数量
find releases/2025-08-08-rc1/models -name "*_best.pt" | wc -l
# 应该输出: 13
```

## 监控命令

```bash
# 查看系统状态
python3 scripts/monitoring_dashboard.py

# 夜测结果汇总
bash scripts/morning_summary.sh

# 单港口测试
python3 scripts/nightly_ci.py --ports <port_name> --samples 400 --seeds 42,123,2025
```

## 紧急操作

### 快速回滚单阶段
```bash
# 回滚指定港口的指定阶段模型
cp releases/2025-08-08-rc1/models/curriculum_v2/<port>/stage_<阶段>_best.pt \
   models/curriculum_v2/<port>/stage_<阶段>_best.pt

# 示例: 回滚new_orleans的中级阶段
cp releases/2025-08-08-rc1/models/curriculum_v2/new_orleans/stage_中级阶段_best.pt \
   models/curriculum_v2/new_orleans/stage_中级阶段_best.pt
```

### New Orleans 补针
```bash
# 仅在需要时执行
bash scripts/new_orleans_emergency_fix.sh
```

### 完整系统回滚
```bash
# 回滚所有模型到RC版本
cp -r releases/2025-08-08-rc1/models/curriculum_v2/* models/curriculum_v2/

# 回滚配置
cp releases/2025-08-08-rc1/config/thresholds.txt config/
```

## 故障排除

### 模型加载失败
```bash
# 检查PyTorch版本兼容性
python3 -c "import torch; print(torch.__version__)"

# 验证模型文件
python3 -c "import torch; torch.load('path/to/model.pt', weights_only=False)"
```

### 测试脚本路径问题
```bash
# 确保在项目根目录执行
cd /path/to/GAT-FedPPO
python3 scripts/nightly_ci.py --help
```

## 版本信息

- **发布版本**: 2025-08-08-rc1
- **Git标签**: release-2025-08-08-rc1
- **PyTorch兼容**: 2.6+ (weights_only=False)
- **Python版本**: 3.11+

## 联系信息

- **紧急联系**: 检查logs/nightly/目录下的最新日志
- **状态监控**: 运行monitoring_dashboard.py
- **问题报告**: 检查Git提交历史和标签