# GAT-FedPPO Production Deployment
# Release: 2025-08-08-rc1
FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 复制requirements文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码和模型
COPY models/ ./models/
COPY config/ ./config/
COPY smoke_test.sh .

# 创建日志目录
RUN mkdir -p logs/nightly

# 设置环境变量
ENV PYTHONPATH=/app
ENV TORCH_WEIGHTS_ONLY=False

# 健康检查
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python3 -c "import torch; print('PyTorch OK')" || exit 1

# 默认入口点
ENTRYPOINT ["./smoke_test.sh"]

# 元数据
LABEL version="2025-08-08-rc1"
LABEL description="GAT-FedPPO Federated Learning System"
LABEL maintainer="GAT-FedPPO Team"