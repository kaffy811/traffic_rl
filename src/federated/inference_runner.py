#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
推理运行器 - 用于生产环境的模型推理
支持多种格式：PyTorch state_dict, TorchScript, ONNX
"""
import os
import sys
import json
import time
import logging
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import torch
import torch.nn as nn

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from curriculum_trainer import CurriculumTrainer, build_agent
from gat_ppo_agent import GATPPOAgent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InferenceRunner:
    """推理运行器类"""
    
    def __init__(self, port_name: str, device: str = "cpu"):
        self.port_name = port_name
        self.device = torch.device(device)
        self.trainer = None
        self.agent = None
        self.models = {}  # 存储不同阶段的模型
        self.current_stage = None
        
        logger.info(f"初始化推理运行器 - 港口: {port_name}, 设备: {device}")
    
    def load_pytorch_model(self, stage_name: str, model_path: str):
        """加载PyTorch模型"""
        try:
            if self.trainer is None:
                self.trainer = CurriculumTrainer(self.port_name)
            
            if self.agent is None:
                self.agent = build_agent(self.port_name, device=self.device)
            
            # 加载模型权重
            checkpoint = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in checkpoint:
                self.agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
            else:
                self.agent.actor_critic.load_state_dict(checkpoint)
            
            self.agent.actor_critic.eval()
            self.models[stage_name] = {
                'type': 'pytorch',
                'model': self.agent.actor_critic,
                'path': model_path
            }
            
            logger.info(f"✅ 加载PyTorch模型: {stage_name} <- {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 加载PyTorch模型失败: {e}")
            return False
    
    def export_torchscript(self, stage_name: str, output_path: str, 
                          sample_input: Optional[Tuple] = None):
        """导出TorchScript模型"""
        try:
            if stage_name not in self.models:
                logger.error(f"阶段 {stage_name} 的模型未加载")
                return False
            
            model = self.models[stage_name]['model']
            
            # 创建示例输入
            if sample_input is None:
                # 使用默认的示例输入
                batch_size = 1
                state_dim = 20  # 根据实际状态维度调整
                node_features_dim = (10, 8)  # (num_nodes, feature_dim)
                adj_matrix_dim = (10, 10)  # (num_nodes, num_nodes)
                
                sample_input = (
                    torch.randn(batch_size, state_dim, device=self.device),
                    torch.randn(batch_size, *node_features_dim, device=self.device),
                    torch.randn(batch_size, *adj_matrix_dim, device=self.device)
                )
            
            # 导出TorchScript
            traced_model = torch.jit.trace(model, sample_input)
            traced_model.save(output_path)
            
            self.models[stage_name + '_torchscript'] = {
                'type': 'torchscript',
                'model': traced_model,
                'path': output_path
            }
            
            logger.info(f"✅ 导出TorchScript模型: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 导出TorchScript失败: {e}")
            return False
    
    def export_onnx(self, stage_name: str, output_path: str,
                   sample_input: Optional[Tuple] = None):
        """导出ONNX模型"""
        try:
            if stage_name not in self.models:
                logger.error(f"阶段 {stage_name} 的模型未加载")
                return False
            
            model = self.models[stage_name]['model']
            
            # 创建示例输入
            if sample_input is None:
                batch_size = 1
                state_dim = 20
                node_features_dim = (10, 8)
                adj_matrix_dim = (10, 10)
                
                sample_input = (
                    torch.randn(batch_size, state_dim, device=self.device),
                    torch.randn(batch_size, *node_features_dim, device=self.device),
                    torch.randn(batch_size, *adj_matrix_dim, device=self.device)
                )
            
            # 导出ONNX
            torch.onnx.export(
                model,
                sample_input,
                output_path,
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=['state', 'node_features', 'adj_matrix'],
                output_names=['action_probs', 'value'],
                dynamic_axes={
                    'state': {0: 'batch_size'},
                    'node_features': {0: 'batch_size'},
                    'adj_matrix': {0: 'batch_size'},
                    'action_probs': {0: 'batch_size'},
                    'value': {0: 'batch_size'}
                }
            )
            
            self.models[stage_name + '_onnx'] = {
                'type': 'onnx',
                'path': output_path
            }
            
            logger.info(f"✅ 导出ONNX模型: {output_path}")
            return True
            
        except Exception as e:
            logger.error(f"❌ 导出ONNX失败: {e}")
            return False
    
    def predict(self, observation: Dict, stage_name: str = None, 
                model_type: str = 'pytorch') -> Tuple[int, float]:
        """执行推理预测"""
        try:
            # 选择模型
            if stage_name is None:
                stage_name = self.current_stage
            if stage_name is None:
                raise ValueError("未指定阶段名称")
            
            model_key = stage_name if model_type == 'pytorch' else f"{stage_name}_{model_type}"
            if model_key not in self.models:
                raise ValueError(f"模型 {model_key} 未加载")
            
            # 预处理输入
            state, node_features, adj_matrix = self._preprocess_observation(observation)
            
            # 执行推理
            if model_type == 'pytorch':
                action, value = self._predict_pytorch(
                    self.models[model_key]['model'], state, node_features, adj_matrix
                )
            elif model_type == 'torchscript':
                action, value = self._predict_torchscript(
                    self.models[model_key]['model'], state, node_features, adj_matrix
                )
            else:
                raise ValueError(f"不支持的模型类型: {model_type}")
            
            return action, value
            
        except Exception as e:
            logger.error(f"推理失败: {e}")
            return 0, 0.0
    
    def _preprocess_observation(self, observation: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """预处理观测数据"""
        if self.trainer is None:
            self.trainer = CurriculumTrainer(self.port_name)
        
        # 提取状态特征
        state = self.trainer._extract_state_from_data(observation)
        node_features, adj_matrix = self.trainer._extract_graph_features_from_data(observation)
        
        # 转换为张量
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        node_features_tensor = torch.as_tensor(node_features, dtype=torch.float32, device=self.device).unsqueeze(0)
        adj_matrix_tensor = torch.as_tensor(adj_matrix, dtype=torch.float32, device=self.device).unsqueeze(0)
        
        return state_tensor, node_features_tensor, adj_matrix_tensor
    
    def _predict_pytorch(self, model: nn.Module, state: torch.Tensor, 
                        node_features: torch.Tensor, adj_matrix: torch.Tensor) -> Tuple[int, float]:
        """PyTorch模型推理"""
        with torch.no_grad():
            action_probs, value = model(state, node_features, adj_matrix)
            
            # 处理NaN和无穷值
            action_probs = torch.nan_to_num(action_probs, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 归一化概率
            sum_ = action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            action_probs = action_probs / sum_
            
            # 选择动作
            action = torch.argmax(action_probs, dim=-1).item()
            value_scalar = value.item() if value.numel() == 1 else value.mean().item()
            
            return action, value_scalar
    
    def _predict_torchscript(self, model, state: torch.Tensor,
                           node_features: torch.Tensor, adj_matrix: torch.Tensor) -> Tuple[int, float]:
        """TorchScript模型推理"""
        with torch.no_grad():
            action_probs, value = model(state, node_features, adj_matrix)
            
            # 处理NaN和无穷值
            action_probs = torch.nan_to_num(action_probs, nan=0.0, posinf=0.0, neginf=0.0)
            
            # 归一化概率
            sum_ = action_probs.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            action_probs = action_probs / sum_
            
            # 选择动作
            action = torch.argmax(action_probs, dim=-1).item()
            value_scalar = value.item() if value.numel() == 1 else value.mean().item()
            
            return action, value_scalar
    
    def set_current_stage(self, stage_name: str):
        """设置当前阶段"""
        self.current_stage = stage_name
        logger.info(f"设置当前阶段: {stage_name}")
    
    def get_model_info(self) -> Dict:
        """获取模型信息"""
        info = {
            'port_name': self.port_name,
            'device': str(self.device),
            'current_stage': self.current_stage,
            'loaded_models': {}
        }
        
        for name, model_info in self.models.items():
            info['loaded_models'][name] = {
                'type': model_info['type'],
                'path': model_info.get('path', 'N/A')
            }
        
        return info
    
    def benchmark(self, observation: Dict, stage_name: str, 
                 num_runs: int = 1000) -> Dict:
        """性能基准测试"""
        logger.info(f"开始性能基准测试: {num_runs} 次推理")
        
        results = {}
        
        for model_type in ['pytorch', 'torchscript']:
            model_key = stage_name if model_type == 'pytorch' else f"{stage_name}_{model_type}"
            if model_key not in self.models:
                continue
            
            times = []
            for _ in range(num_runs):
                start_time = time.time()
                self.predict(observation, stage_name, model_type)
                end_time = time.time()
                times.append((end_time - start_time) * 1000)  # 转换为毫秒
            
            results[model_type] = {
                'avg_time_ms': np.mean(times),
                'std_time_ms': np.std(times),
                'min_time_ms': np.min(times),
                'max_time_ms': np.max(times),
                'total_runs': num_runs
            }
        
        logger.info("基准测试完成")
        return results

def main():
    parser = argparse.ArgumentParser(description="推理运行器")
    parser.add_argument("--port", required=True, help="港口名称")
    parser.add_argument("--stage", required=True, help="阶段名称")
    parser.add_argument("--model-path", required=True, help="模型文件路径")
    parser.add_argument("--device", default="cpu", help="设备类型")
    parser.add_argument("--export-torchscript", help="导出TorchScript路径")
    parser.add_argument("--export-onnx", help="导出ONNX路径")
    parser.add_argument("--benchmark", action="store_true", help="运行性能基准测试")
    parser.add_argument("--test-data", help="测试数据文件路径")
    
    args = parser.parse_args()
    
    # 初始化推理运行器
    runner = InferenceRunner(args.port, args.device)
    
    # 加载模型
    success = runner.load_pytorch_model(args.stage, args.model_path)
    if not success:
        logger.error("模型加载失败")
        sys.exit(1)
    
    runner.set_current_stage(args.stage)
    
    # 导出其他格式
    if args.export_torchscript:
        runner.export_torchscript(args.stage, args.export_torchscript)
    
    if args.export_onnx:
        runner.export_onnx(args.stage, args.export_onnx)
    
    # 性能基准测试
    if args.benchmark and args.test_data:
        with open(args.test_data, 'r') as f:
            test_observation = json.load(f)
        
        benchmark_results = runner.benchmark(test_observation, args.stage)
        print("性能基准测试结果:")
        print(json.dumps(benchmark_results, indent=2, ensure_ascii=False))
    
    # 显示模型信息
    model_info = runner.get_model_info()
    print("模型信息:")
    print(json.dumps(model_info, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()