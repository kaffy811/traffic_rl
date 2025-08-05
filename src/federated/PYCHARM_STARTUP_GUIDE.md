# PyCharm中运行FedML的解决方案

## 🔍 问题分析

您遇到的错误 `KeyError: 2` 已经解决！这个错误是因为数据加载器没有为所有客户端创建数据索引。

## ✅ 已修复的问题

1. **数据索引不匹配**: 之前数据加载器只为当前客户端创建数据，现在为所有4个客户端创建数据(索引0-3)
2. **FedML客户端分配**: 服务器现在正确分配`data_silo_index = 0, 1, 2, 3`给4个客户端

## 🚀 在PyCharm中正确启动多客户端的方法

### 方法1: 使用终端脚本（推荐）

```bash
# 1. 启动服务器
cd /Users/kaffy/Documents/GAT-FedPPO/src/federated
./run_server.sh

# 2. 分别启动4个客户端（新终端窗口）
./run_client.sh 1  # 客户端1 (new_orleans)
./run_client.sh 2  # 客户端2 (south_louisiana) 
./run_client.sh 3  # 客户端3 (baton_rouge)
./run_client.sh 4  # 客户端4 (gulfport)
```

### 方法2: PyCharm运行配置

在PyCharm中创建5个运行配置：

#### 服务器配置
- **Name**: FedML Server
- **Script path**: `/Users/kaffy/Documents/GAT-FedPPO/src/federated/maritime_server.py`
- **Environment variables**:
  ```
  FEDML_TRAINING_TYPE=cross_silo
  FEDML_BACKEND=MQTT_S3
  ```

#### 客户端配置 (创建4个)
- **Name**: FedML Client 1
- **Script path**: `/Users/kaffy/Documents/GAT-FedPPO/src/federated/maritime_client.py`
- **Environment variables**:
  ```
  FEDML_TRAINING_TYPE=cross_silo
  FEDML_BACKEND=MQTT_S3
  CLIENT_RANK=1
  ```

**重复为客户端2、3、4**，只需修改`CLIENT_RANK=2,3,4`

## 📋 启动顺序

1. **先启动服务器** - 等待看到 "FedMLDebug server.wait START = True"
2. **再启动客户端** - 按顺序启动1、2、3、4
3. **等待连接** - 每个客户端会显示 "communication backend is alive"

## 🔧 为什么之前只能启动2个客户端？

1. **数据索引问题**: 客户端3和4因为`KeyError: 2`而启动失败
2. **现在已修复**: 数据加载器为所有4个客户端(索引0-3)创建数据

## 📊 验证修复

运行测试脚本确认数据加载器工作正常：
```bash
cd /Users/kaffy/Documents/GAT-FedPPO/src/federated
python test_data_loader.py
```

应该看到:
```
✅ 客户端 0: 训练 100 episodes, 测试 20 episodes
✅ 客户端 1: 训练 100 episodes, 测试 20 episodes  
✅ 客户端 2: 训练 100 episodes, 测试 20 episodes
✅ 客户端 3: 训练 100 episodes, 测试 20 episodes
```

## 🎯 关键修复点

- **修复文件**: `maritime_data_loader.py`
- **关键改动**: 为所有`client_num_in_total`个客户端创建数据，而不只是当前客户端
- **数据索引**: 现在正确创建索引0、1、2、3的数据字典

现在您应该能够在PyCharm中成功启动1个服务器和4个客户端了！🎉