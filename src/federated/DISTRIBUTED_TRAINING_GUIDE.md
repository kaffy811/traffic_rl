# 分布式多端口联邦学习指南

## 🎯 系统概述

这是一个**真正的分布式训练系统**，每个港口在**不同的终端或服务器**上运行，通过网络进行联邦学习通信。

### 系统架构
```
终端1: 联邦学习服务器     (协调中心)
  ↓ HTTP通信
终端2: New Orleans       (端口客户端)
终端3: South Louisiana   (端口客户端)  
终端4: Baton Rouge       (端口客户端)
终端5: Gulfport          (端口客户端)
```

## 📋 环境准备

### 1. 安装依赖
```bash
cd /Users/kaffy/Documents/GAT-FedPPO
pip install -r src/federated/requirements_distributed.txt
```

### 2. 检查端口可用性
确保端口8888没有被占用：
```bash
lsof -i :8888
```

## 🚀 使用方法

### 方法1: 自动启动 (推荐)

在一个终端运行自动化脚本：
```bash
cd /Users/kaffy/Documents/GAT-FedPPO
python src/federated/start_distributed_training.py --rounds 10 --episodes 3
```

### 方法2: 手动分布式启动

#### 步骤1: 启动服务器 (终端1)
```bash
cd /Users/kaffy/Documents/GAT-FedPPO
./src/federated/start_server.sh
```
或者：
```bash
python src/federated/distributed_federated_server.py --host localhost --port 8888 --max_rounds 10
```

#### 步骤2: 启动港口客户端

**终端2 - New Orleans港口:**
```bash
cd /Users/kaffy/Documents/GAT-FedPPO
./src/federated/start_port_new_orleans.sh
```

**终端3 - South Louisiana港口:**
```bash
cd /Users/kaffy/Documents/GAT-FedPPO
./src/federated/start_port_south_louisiana.sh
```

**终端4 - Baton Rouge港口:**
```bash
cd /Users/kaffy/Documents/GAT-FedPPO
./src/federated/start_port_baton_rouge.sh
```

**终端5 - Gulfport港口:**
```bash
cd /Users/kaffy/Documents/GAT-FedPPO
./src/federated/start_port_gulfport.sh
```

### 方法3: 自定义参数启动

#### 服务器：
```bash
python src/federated/distributed_federated_server.py \
    --host localhost \
    --port 8888 \
    --min_clients 2 \
    --max_rounds 15
```

#### 客户端：
```bash
python src/federated/distributed_port_client.py \
    --port_id 0 \
    --port_name new_orleans \
    --server_host localhost \
    --server_port 8888 \
    --rounds 15 \
    --episodes 5
```

## 🌐 跨服务器分布式训练

### 如果要在不同服务器上运行：

#### 服务器A (运行联邦服务器):
```bash
python src/federated/distributed_federated_server.py \
    --host 0.0.0.0 \
    --port 8888 \
    --min_clients 2 \
    --max_rounds 10
```

#### 服务器B (运行港口客户端):
```bash
python src/federated/distributed_port_client.py \
    --port_id 0 \
    --port_name new_orleans \
    --server_host <服务器A的IP> \
    --server_port 8888 \
    --rounds 10 \
    --episodes 3
```

#### 服务器C (运行另一个港口客户端):
```bash
python src/federated/distributed_port_client.py \
    --port_id 1 \
    --port_name south_louisiana \
    --server_host <服务器A的IP> \
    --server_port 8888 \
    --rounds 10 \
    --episodes 3
```

## 📊 系统特性

### 1. 真正的分布式架构
- ✅ 每个港口在独立的进程/终端中运行
- ✅ 通过HTTP RESTful API进行通信
- ✅ 支持跨网络的分布式部署
- ✅ 容错机制，客户端可以随时加入/退出

### 2. 联邦学习流程
```
1. 客户端注册到服务器
2. 服务器等待最少数量的客户端
3. 客户端获取全局模型
4. 客户端进行本地训练
5. 客户端上传本地模型
6. 服务器执行联邦聚合
7. 重复步骤3-6直到完成
```

### 3. 网络通信API

#### 客户端注册: POST /register
```json
{
  "client_id": "port_0_new_orleans",
  "port_id": 0,
  "port_name": "new_orleans",
  "capabilities": {...}
}
```

#### 获取全局模型: GET /get_global_model
```json
{
  "has_model": true,
  "model_params": {...},
  "version": 5
}
```

#### 上传本地模型: POST /upload_model
```json
{
  "client_id": "port_0_new_orleans",
  "model_params": {...},
  "training_result": {...}
}
```

### 4. 数据隐私保护
- ✅ 原始数据永不离开本地港口
- ✅ 只交换模型参数
- ✅ 支持差分隐私（可扩展）
- ✅ 安全的HTTP通信

## 📈 监控和日志

### 服务器日志
```
src/federated/logs/federated_server_YYYYMMDD_HHMMSS.log
```

### 客户端日志
```
src/federated/logs/port_new_orleans_YYYYMMDD_HHMMSS.log
src/federated/logs/port_south_louisiana_YYYYMMDD_HHMMSS.log
src/federated/logs/port_baton_rouge_YYYYMMDD_HHMMSS.log
src/federated/logs/port_gulfport_YYYYMMDD_HHMMSS.log
```

### 实时状态检查
```bash
curl http://localhost:8888/status
```

## 🔧 故障排除

### 常见问题

1. **端口被占用**
```bash
# 检查端口
lsof -i :8888
# 杀死进程
kill -9 <PID>
```

2. **客户端无法连接服务器**
- 检查服务器是否启动
- 检查网络连接
- 检查防火墙设置

3. **联邦聚合不开始**
- 确保有足够的客户端连接
- 检查服务器日志中的错误信息

4. **内存不足**
- 减少episodes_per_round
- 使用更小的拓扑配置

### 调试模式
启动时添加详细日志：
```bash
python src/federated/distributed_federated_server.py --host localhost --port 8888 --min_clients 1
```

## 🎯 高级配置

### 1. 性能优化
```bash
# 增加并发episodes
--episodes 5

# 使用更大的拓扑
--topology 4x4

# 调整聚合频率
--min_clients 4
```

### 2. 安全配置
```bash
# 使用HTTPS (需要SSL证书)
# 添加认证机制
# 启用模型加密
```

### 3. 容错配置
```bash
# 客户端重连机制
# 动态客户端加入
# 异步聚合
```

## 📊 实验结果

训练完成后，可以生成可视化：
```bash
python src/federated/visualization_generator.py
```

结果文件位置：
- **实验数据**: `src/federated/experiment_data/`
- **可视化结果**: `src/federated/visualization_results/`
- **日志文件**: `src/federated/logs/`

## 🎉 分布式训练优势

### 相比单机多进程训练：
1. **真正的分布式**: 可以跨多台服务器部署
2. **更好的隔离**: 每个港口完全独立运行
3. **更强的扩展性**: 可以动态添加/移除港口
4. **更真实的环境**: 模拟真实的分布式部署场景
5. **容错能力**: 单个港口失败不影响其他港口

### 适用场景：
- ✅ 多台服务器的真实分布式部署
- ✅ 模拟真实的港口分布式环境
- ✅ 研究联邦学习的网络通信影响
- ✅ 验证系统的容错和扩展能力

---

**这就是真正的分布式多端口联邦学习系统！每个港口在独立的终端/服务器上运行，通过网络进行协作学习。** 🌐