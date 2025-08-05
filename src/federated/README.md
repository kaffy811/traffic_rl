# 海事GAT-PPO联邦学习系统

## 📋 项目概述

本系统实现了基于FedML框架的海事GAT-PPO联邦学习解决方案，用于多港口/航道节点的协同智能交通控制。

## 🏗️ 系统架构

```
海事联邦学习系统
├── 联邦服务器 (FedML Server)
│   ├── 模型聚合 (MaritimeFedAggregator)
│   ├── 全局模型管理
│   └── 性能监控
│
└── 联邦客户端 (FedML Clients)
    ├── NodeA - 新奥尔良港主入口
    ├── NodeB - 密西西比河口  
    ├── NodeC - 河道中段
    └── NodeD - 近海锚地
    
每个客户端包含:
├── GAT-PPO智能体 (MaritimeGATPPOAgent)
├── 联邦训练器 (MaritimeFedTrainer)
├── 公平性奖励计算器
└── 本地环境仿真
```

## 🚀 核心特性

### 1. 智能化聚合策略
- **地理位置权重**: 相邻港口节点获得更高协作权重
- **性能权重**: 表现优秀的客户端获得更高影响力
- **公平性权重**: 促进系统整体公平性
- **自适应权重**: 根据训练进度动态调整

### 2. 海事专用设计
- **真实AIS数据**: 基于新奥尔良港2024年7月6日实际数据
- **CityFlow集成**: 与交通仿真平台无缝对接
- **多维度奖励**: 效率、公平性、安全性、环境友好
- **实时决策**: 支持港口信号控制实时决策

### 3. 隐私保护
- **本地训练**: 敏感数据不离开本地节点
- **模型聚合**: 仅交换模型参数，不传输原始数据
- **差分隐私**: 可选的隐私保护机制

## 📁 文件结构

```
src/federated/
├── maritime_fed_trainer.py      # 联邦训练器
├── maritime_fed_aggregator.py   # 联邦聚合器
├── maritime_server.py           # 服务器启动脚本
├── maritime_client.py           # 客户端启动脚本
├── run_server.sh               # 服务器启动Shell脚本
├── run_client.sh               # 客户端启动Shell脚本
├── test_maritime_federated.py  # 完整集成测试
├── config/
│   ├── maritime_fedml_config.yaml  # 主配置文件
│   └── grpc_ipconfig.csv           # 网络配置
└── README.md                   # 本文档
```

## 🛠️ 快速开始

### 1. 环境准备

```bash
# 确保已安装依赖
pip install torch fedml numpy pyyaml

# 检查项目结构
cd /path/to/GAT-FedPPO
ls -la src/federated/
```

### 2. 运行集成测试

```bash
# 进入联邦学习目录
cd src/federated

# 运行完整测试
python test_maritime_federated.py
```

预期输出：
```
🚢 海事GAT-PPO联邦学习完整集成测试
==================================================
🧪 测试1: 联邦训练器创建
✅ 节点0训练器创建成功
✅ 节点1训练器创建成功
...
🎯 测试总结: 6/6 通过
🎉 所有测试通过！海事联邦学习系统准备就绪！
```

### 3. 启动联邦训练

**步骤1: 启动服务器**
```bash
cd src/federated
./run_server.sh
```

**步骤2: 启动客户端（每个在单独终端）**
```bash
# 终端1 - NodeA
./run_client.sh 1

# 终端2 - NodeB  
./run_client.sh 2

# 终端3 - NodeC
./run_client.sh 3

# 终端4 - NodeD
./run_client.sh 4
```

### 4. 监控训练进度

联邦训练将自动运行，观察日志输出：
- 本地训练进度
- 模型聚合过程
- 全局性能指标
- 公平性分数变化

## ⚙️ 配置说明

### 主要配置项 (maritime_fedml_config.yaml)

```yaml
# 训练配置
train_args:
  comm_round: 50                    # 联邦通信轮数
  client_num_per_round: 4          # 每轮参与客户端数
  episodes_per_epoch: 5            # 每轮本地训练episode数
  
# PPO配置
model_args:
  ppo_config:
    learning_rate: 0.0003          # 学习率
    gamma: 0.99                    # 折扣因子
    ppo_batch_size: 20             # 批量大小
    
# 聚合策略配置
train_args:
  use_geographic_weights: true     # 启用地理权重
  use_performance_weights: true    # 启用性能权重
  use_fairness_weights: true       # 启用公平性权重
```

## 📊 性能指标

系统监控以下关键指标：

### 训练指标
- **平均奖励**: 各客户端的平均episode奖励
- **策略损失**: PPO策略网络损失
- **价值损失**: PPO价值网络损失
- **KL散度**: 策略更新的KL散度

### 聚合指标
- **聚合权重**: 各客户端在聚合中的权重
- **参与度**: 客户端参与联邦训练的频率
- **收敛速度**: 全局模型收敛所需轮数

### 海事指标
- **通行效率**: 船舶平均等待时间
- **吞吐量**: 单位时间通过船舶数量
- **公平性分数**: 主航道与辅航道的服务平衡
- **安全指标**: 冲突事件数量

## 🔧 故障排除

### 常见问题

1. **Import错误**
   ```
   解决: 确保PYTHONPATH包含项目根目录和FedML路径
   export PYTHONPATH="$PYTHONPATH:../..:../../FedML/python"
   ```

2. **配置文件未找到**
   ```
   解决: 检查config/maritime_fedml_config.yaml是否存在
   ```

3. **客户端连接失败**
   ```
   解决: 确保服务器先启动，检查IP配置文件
   ```

4. **GAT模块导入失败**
   ```
   解决: 确保FedML/externals/pytorch-GAT有__init__.py文件
   ```

### 调试模式

启用详细日志：
```bash
export FEDML_LOG_LEVEL=DEBUG
python test_maritime_federated.py
```

## 🚦 下一步开发

### 短期目标
- [ ] 集成真实CityFlow仿真环境
- [ ] 添加模型剪枝与知识蒸馏
- [ ] 实现3×3、5×5拓扑自动生成
- [ ] 性能评估和基准测试

### 长期目标
- [ ] 支持更多港口节点
- [ ] 实时AIS数据流集成
- [ ] 云端部署和边缘计算支持
- [ ] 安全聚合和差分隐私增强

## 📞 联系支持

如遇到技术问题或需要功能扩展，请参考：
1. FedML官方文档: https://doc.fedml.ai
2. PyTorch官方文档: https://pytorch.org/docs/
3. 项目源代码注释和测试用例

---

🚢 **祝您在海事智能交通联邦学习的探索中一帆风顺！** ⚓
