#!/usr/bin/env python3
"""
分布式联邦学习服务器
协调多个分布式港口客户端进行联邦学习
"""

import sys
import os
import json
import time
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import logging

# Web服务
from flask import Flask, request, jsonify
import torch
import numpy as np

# 设置项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 自定义模块导入
from src.federated.real_data_collector import RealDataCollector, initialize_data_collector

class DistributedFederatedServer:
    """分布式联邦学习服务器"""
    
    def __init__(self, host: str = "localhost", port: int = 8888, min_clients: int = 2):
        self.host = host
        self.port = port
        self.min_clients = min_clients
        
        # 客户端管理
        self.registered_clients = {}  # client_id -> client_info
        self.client_models = {}       # client_id -> model_params
        self.client_training_results = {}  # client_id -> training_results
        
        # 全局模型管理
        self.global_model_params = None
        self.global_model_version = 0
        self.aggregation_weights = {}  # client_id -> weight
        
        # 联邦学习状态
        self.current_round = 0
        self.max_rounds = 10
        self.round_start_time = None
        self.federated_training_active = False
        
        # 数据收集器
        self.data_collector = initialize_data_collector("distributed_federated_experiment")
        
        # 设置日志
        self.logger = self._setup_logging()
        
        # Flask应用
        self.app = Flask(__name__)
        self._setup_routes()
        
        self.logger.info(f"🏢 分布式联邦学习服务器初始化")
        self.logger.info(f"📡 服务地址: {host}:{port}")
        self.logger.info(f"👥 最少客户端数: {min_clients}")
    
    def _setup_logging(self):
        """设置日志"""
        logger = logging.getLogger("FederatedServer")
        logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        log_dir = project_root / "src" / "federated" / "logs"
        log_dir.mkdir(exist_ok=True)
        
        handler = logging.FileHandler(log_dir / f"federated_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _setup_routes(self):
        """设置Flask路由"""
        
        @self.app.route('/register', methods=['POST'])
        def register_client():
            """客户端注册"""
            try:
                client_data = request.json
                client_id = client_data['client_id']
                
                # 注册客户端
                self.registered_clients[client_id] = {
                    'client_id': client_id,
                    'port_id': client_data['port_id'],
                    'port_name': client_data['port_name'],
                    'topology_size': client_data['topology_size'],
                    'capabilities': client_data['capabilities'],
                    'register_time': datetime.now().isoformat(),
                    'last_seen': datetime.now().isoformat()
                }
                
                self.logger.info(f"✅ 客户端注册: {client_id} ({client_data['port_name']})")
                
                return jsonify({
                    'status': 'success',
                    'message': f'客户端 {client_id} 注册成功',
                    'server_info': {
                        'current_round': self.current_round,
                        'max_rounds': self.max_rounds,
                        'registered_clients': len(self.registered_clients)
                    }
                })
                
            except Exception as e:
                self.logger.error(f"❌ 客户端注册失败: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 400
        
        @self.app.route('/get_global_model', methods=['GET'])
        def get_global_model():
            """获取全局模型"""
            try:
                client_id = request.args.get('client_id')
                
                if client_id in self.registered_clients:
                    self.registered_clients[client_id]['last_seen'] = datetime.now().isoformat()
                
                if self.global_model_params is not None:
                    # 将tensor转换为可序列化格式
                    serializable_params = {}
                    for key, value in self.global_model_params.items():
                        if torch.is_tensor(value):
                            serializable_params[key] = value.tolist()
                        else:
                            serializable_params[key] = value
                    
                    return jsonify({
                        'status': 'success',
                        'has_model': True,
                        'model_params': serializable_params,
                        'version': self.global_model_version,
                        'round': self.current_round
                    })
                else:
                    return jsonify({
                        'status': 'success',
                        'has_model': False,
                        'message': '全局模型尚未初始化',
                        'version': 0,
                        'round': self.current_round
                    })
                    
            except Exception as e:
                self.logger.error(f"❌ 获取全局模型失败: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/upload_model', methods=['POST'])
        def upload_model():
            """上传客户端模型"""
            try:
                data = request.json
                client_id = data['client_id']
                
                if client_id not in self.registered_clients:
                    return jsonify({'status': 'error', 'message': '客户端未注册'}), 400
                
                # 更新客户端最后活跃时间
                self.registered_clients[client_id]['last_seen'] = datetime.now().isoformat()
                
                # 存储客户端模型和训练结果
                self.client_models[client_id] = data['model_params']
                self.client_training_results[client_id] = data['training_result']
                
                self.logger.info(f"📤 收到客户端模型: {client_id} ({data['port_name']})")
                
                # 检查是否可以进行聚合
                if len(self.client_models) >= self.min_clients:
                    threading.Thread(target=self._try_federated_aggregation).start()
                
                return jsonify({
                    'status': 'success',
                    'message': f'模型上传成功',
                    'uploaded_clients': len(self.client_models),
                    'required_clients': self.min_clients
                })
                
            except Exception as e:
                self.logger.error(f"❌ 模型上传失败: {e}")
                return jsonify({'status': 'error', 'message': str(e)}), 500
        
        @self.app.route('/status', methods=['GET'])
        def get_status():
            """获取服务器状态"""
            return jsonify({
                'status': 'success',
                'server_info': {
                    'current_round': self.current_round,
                    'max_rounds': self.max_rounds,
                    'global_model_version': self.global_model_version,
                    'registered_clients': len(self.registered_clients),
                    'uploaded_models': len(self.client_models),
                    'training_active': self.federated_training_active,
                    'clients': list(self.registered_clients.keys())
                }
            })
    
    def _try_federated_aggregation(self):
        """尝试执行联邦聚合"""
        try:
            if not self.federated_training_active:
                return
            
            # 检查是否有足够的客户端模型
            if len(self.client_models) < self.min_clients:
                return
            
            self.logger.info(f"🔄 开始联邦聚合 - 轮次 {self.current_round + 1}")
            self.logger.info(f"   参与客户端: {list(self.client_models.keys())}")
            
            # 执行联邦聚合
            aggregated_params = self._federated_averaging()
            
            if aggregated_params:
                self.global_model_params = aggregated_params
                self.global_model_version += 1
                self.current_round += 1
                
                # 收集聚合数据
                if self.data_collector:
                    self._collect_aggregation_data()
                
                self.logger.info(f"✅ 联邦聚合完成 - 全局模型 v{self.global_model_version}")
                
                # 清空客户端模型，准备下一轮
                self.client_models.clear()
                
                # 检查是否完成训练
                if self.current_round >= self.max_rounds:
                    self._finish_federated_training()
            
        except Exception as e:
            self.logger.error(f"❌ 联邦聚合失败: {e}")
    
    def _federated_averaging(self) -> Dict:
        """联邦平均聚合算法"""
        if not self.client_models:
            return {}
        
        aggregated_params = {}
        num_clients = len(self.client_models)
        
        # 计算客户端权重 (简单平均，可以基于数据量调整)
        weights = {client_id: 1.0 / num_clients for client_id in self.client_models.keys()}
        self.aggregation_weights = weights
        
        self.logger.info(f"   ⚖️ 聚合权重: {weights}")
        
        # 获取参数结构
        first_client_params = next(iter(self.client_models.values()))
        
        for param_name in first_client_params.keys():
            param_sum = None
            total_weight = 0
            
            for client_id, client_params in self.client_models.items():
                if param_name in client_params:
                    weight = weights[client_id]
                    
                    # 转换为tensor
                    if isinstance(client_params[param_name], list):
                        param_tensor = torch.tensor(client_params[param_name])
                    else:
                        param_tensor = torch.tensor(client_params[param_name])
                    
                    if param_sum is None:
                        param_sum = param_tensor * weight
                    else:
                        param_sum += param_tensor * weight
                    
                    total_weight += weight
            
            if param_sum is not None and total_weight > 0:
                aggregated_params[param_name] = param_sum / total_weight
        
        self.logger.info(f"   ✅ 聚合了 {num_clients} 个客户端的 {len(aggregated_params)} 个参数")
        return aggregated_params
    
    def _collect_aggregation_data(self):
        """收集聚合数据"""
        try:
            # 计算平均客户端奖励
            avg_client_reward = 0
            if self.client_training_results:
                rewards = [result.get('avg_reward', 0) for result in self.client_training_results.values()]
                avg_client_reward = np.mean(rewards)
            
            aggregation_data = {
                'participating_clients': len(self.client_models),
                'total_samples': sum(result.get('episodes', 0) for result in self.client_training_results.values()),
                'aggregation_weights': self.aggregation_weights,
                'avg_client_reward': avg_client_reward,
                'global_model_version': self.global_model_version
            }
            
            self.data_collector.collect_aggregation_data(aggregation_data)
            
            # 收集客户端训练数据
            for client_id, result in self.client_training_results.items():
                self.data_collector.collect_training_data(client_id, {
                    'avg_reward': result.get('avg_reward', 0),
                    'avg_policy_loss': result.get('avg_policy_loss', 0),
                    'avg_value_loss': result.get('avg_value_loss', 0),
                    'total_episodes': result.get('episodes', 0)
                })
            
        except Exception as e:
            self.logger.error(f"❌ 数据收集失败: {e}")
    
    def start_federated_training(self, max_rounds: int = 10):
        """启动联邦训练"""
        self.max_rounds = max_rounds
        self.current_round = 0
        self.federated_training_active = True
        self.round_start_time = datetime.now()
        
        # 启动数据收集
        if self.data_collector:
            self.data_collector.start_experiment(max_rounds, "Distributed-Multi-Port-FedPPO")
        
        self.logger.info(f"🚀 联邦训练启动 - {max_rounds} 轮")
        self.logger.info(f"   等待客户端连接... (最少 {self.min_clients} 个)")
    
    def _finish_federated_training(self):
        """完成联邦训练"""
        self.federated_training_active = False
        
        # 完成数据收集
        if self.data_collector:
            timestamp = self.data_collector.finish_experiment()
            self.logger.info(f"📊 实验数据已保存: {timestamp}")
        
        # 生成训练总结
        self._generate_training_summary()
        
        self.logger.info(f"🎉 分布式联邦训练完成!")
    
    def _generate_training_summary(self):
        """生成训练总结"""
        self.logger.info("📋 分布式联邦训练总结:")
        self.logger.info(f"   总轮次: {self.current_round}")
        self.logger.info(f"   参与客户端: {len(self.registered_clients)}")
        self.logger.info(f"   全局模型版本: {self.global_model_version}")
        
        # 客户端统计
        for client_id, client_info in self.registered_clients.items():
            port_name = client_info['port_name']
            self.logger.info(f"   🏭 {port_name}: 最后活跃 {client_info['last_seen']}")
    
    def run(self):
        """运行服务器"""
        self.logger.info(f"🌐 启动Flask服务器: {self.host}:{self.port}")
        
        # 启动联邦训练
        self.start_federated_training()
        
        # 启动Flask应用
        self.app.run(
            host=self.host,
            port=self.port,
            debug=False,
            threaded=True
        )

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="分布式联邦学习服务器")
    
    parser.add_argument("--host", type=str, default="localhost",
                       help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8888,
                       help="服务器端口")
    parser.add_argument("--min_clients", type=int, default=2,
                       help="最少客户端数")
    parser.add_argument("--max_rounds", type=int, default=10,
                       help="最大联邦轮次")
    
    args = parser.parse_args()
    
    print(f"🏢 启动分布式联邦学习服务器")
    print(f"   地址: {args.host}:{args.port}")
    print(f"   最少客户端: {args.min_clients}")
    print(f"   最大轮次: {args.max_rounds}")
    
    # 创建服务器
    server = DistributedFederatedServer(
        host=args.host,
        port=args.port,
        min_clients=args.min_clients
    )
    
    try:
        # 设置最大轮次
        server.max_rounds = args.max_rounds
        
        # 运行服务器
        server.run()
        
    except KeyboardInterrupt:
        print("⚠️ 服务器被用户中断")
    except Exception as e:
        print(f"❌ 服务器运行异常: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()