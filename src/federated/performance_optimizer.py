#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
性能优化器 - 优化评测和推理速度
包括多进程评测、缓存优化、内存管理等
"""
import os
import sys
import time
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp
import psutil
import torch
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from consistency_test_fixed import eval_one_stage
from curriculum_trainer import CurriculumTrainer, build_agent

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PerformanceOptimizer:
    """性能优化器"""
    
    def __init__(self):
        self.system_info = self._get_system_info()
        logger.info(f"系统信息: {self.system_info}")
    
    def _get_system_info(self) -> Dict:
        """获取系统信息"""
        return {
            'cpu_count': mp.cpu_count(),
            'memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3),
            'torch_threads': torch.get_num_threads(),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    
    def optimize_torch_settings(self):
        """优化PyTorch设置"""
        # 设置线程数
        optimal_threads = min(self.system_info['cpu_count'], 8)
        torch.set_num_threads(optimal_threads)
        
        # 设置内存分配策略
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cudnn.deterministic = False
        
        # 禁用梯度计算（推理模式）
        torch.set_grad_enabled(False)
        
        logger.info(f"优化PyTorch设置: threads={optimal_threads}")
    
    def profile_single_evaluation(self, port_name: str, stage_name: str, 
                                 n_samples: int = 100) -> Dict:
        """性能分析单次评估"""
        logger.info(f"🔍 性能分析: {port_name}/{stage_name}")
        
        # 初始化
        start_time = time.time()
        trainer = CurriculumTrainer(port_name)
        agent = build_agent(port_name, device="cpu")
        init_time = time.time() - start_time
        
        # 找到对应阶段
        stage = None
        for s in trainer.curriculum_stages:
            if s.name == stage_name:
                stage = s
                break
        
        if stage is None:
            return {'error': f'未找到阶段: {stage_name}'}
        
        # 加载模型
        model_load_start = time.time()
        model_path = f"../../models/curriculum_v2/{port_name}/stage_{stage_name}_best.pt"
        try:
            checkpoint = torch.load(model_path, map_location="cpu")
            if 'model_state_dict' in checkpoint:
                agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
            else:
                agent.actor_critic.load_state_dict(checkpoint)
            agent.actor_critic.eval()
        except Exception as e:
            return {'error': f'模型加载失败: {e}'}
        
        model_load_time = time.time() - model_load_start
        
        # 数据生成
        data_gen_start = time.time()
        test_data = trainer._generate_stage_data(stage, num_samples=n_samples)
        data_gen_time = time.time() - data_gen_start
        
        # 基线计算
        baseline_start = time.time()
        baseline_data = trainer._generate_stage_data(stage, num_samples=50)
        baseline_rewards = []
        for data in baseline_data:
            reward = trainer._calculate_baseline_reward(data)
            baseline_rewards.append(reward)
        baseline_time = time.time() - baseline_start
        
        # 模型推理
        inference_start = time.time()
        inference_times = []
        
        for i, data in enumerate(test_data[:min(50, len(test_data))]):  # 只测试前50个
            single_start = time.time()
            
            # 提取特征
            state = trainer._extract_state_from_data(data)
            node_features, adj_matrix = trainer._extract_graph_features_from_data(data)
            
            # 转换为张量
            state_tensor = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0)
            node_features_tensor = torch.as_tensor(node_features, dtype=torch.float32).unsqueeze(0)
            adj_matrix_tensor = torch.as_tensor(adj_matrix, dtype=torch.float32).unsqueeze(0)
            
            # 推理
            with torch.no_grad():
                action_probs, value = agent.actor_critic(
                    state_tensor, node_features_tensor, adj_matrix_tensor
                )
            
            single_time = time.time() - single_start
            inference_times.append(single_time * 1000)  # 转换为毫秒
        
        total_inference_time = time.time() - inference_start
        
        # 完整评估
        eval_start = time.time()
        eval_result = eval_one_stage(trainer, agent, stage, n_samples=n_samples, device="cpu")
        eval_time = time.time() - eval_start
        
        total_time = time.time() - start_time
        
        profile_result = {
            'port': port_name,
            'stage': stage_name,
            'samples': n_samples,
            'timing': {
                'total_time': total_time,
                'init_time': init_time,
                'model_load_time': model_load_time,
                'data_generation_time': data_gen_time,
                'baseline_calculation_time': baseline_time,
                'inference_time': total_inference_time,
                'evaluation_time': eval_time
            },
            'inference_stats': {
                'avg_inference_ms': np.mean(inference_times),
                'std_inference_ms': np.std(inference_times),
                'min_inference_ms': np.min(inference_times),
                'max_inference_ms': np.max(inference_times),
                'samples_tested': len(inference_times)
            },
            'memory_usage': {
                'peak_memory_mb': psutil.Process().memory_info().rss / (1024*1024),
                'available_memory_gb': psutil.virtual_memory().available / (1024**3)
            },
            'evaluation_result': eval_result
        }
        
        return profile_result
    
    def benchmark_parallel_evaluation(self, ports_stages: List[Tuple[str, str]], 
                                    n_samples: int = 100, max_workers: Optional[int] = None) -> Dict:
        """基准测试并行评估"""
        if max_workers is None:
            max_workers = min(len(ports_stages), self.system_info['cpu_count'] - 1)
        
        logger.info(f"🚀 并行评估基准测试: {len(ports_stages)} 个任务, {max_workers} 个进程")
        
        # 串行基准
        serial_start = time.time()
        serial_results = []
        for port, stage in ports_stages[:2]:  # 只测试前2个
            result = self.profile_single_evaluation(port, stage, n_samples)
            if 'error' not in result:
                serial_results.append(result)
        serial_time = time.time() - serial_start
        
        # 并行基准
        parallel_start = time.time()
        parallel_results = []
        
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_task = {
                executor.submit(self.profile_single_evaluation, port, stage, n_samples): (port, stage)
                for port, stage in ports_stages[:max_workers]  # 限制任务数
            }
            
            for future in as_completed(future_to_task):
                port, stage = future_to_task[future]
                try:
                    result = future.result()
                    if 'error' not in result:
                        parallel_results.append(result)
                except Exception as e:
                    logger.error(f"并行任务失败 {port}/{stage}: {e}")
        
        parallel_time = time.time() - parallel_start
        
        # 计算加速比
        if serial_results and parallel_results:
            avg_serial_time = np.mean([r['timing']['total_time'] for r in serial_results])
            avg_parallel_time = np.mean([r['timing']['total_time'] for r in parallel_results])
            speedup = serial_time / parallel_time if parallel_time > 0 else 0
            efficiency = speedup / max_workers if max_workers > 0 else 0
        else:
            avg_serial_time = avg_parallel_time = speedup = efficiency = 0
        
        benchmark_result = {
            'test_config': {
                'tasks': len(ports_stages),
                'samples_per_task': n_samples,
                'max_workers': max_workers
            },
            'serial_benchmark': {
                'total_time': serial_time,
                'avg_task_time': avg_serial_time,
                'tasks_completed': len(serial_results)
            },
            'parallel_benchmark': {
                'total_time': parallel_time,
                'avg_task_time': avg_parallel_time,
                'tasks_completed': len(parallel_results)
            },
            'performance_metrics': {
                'speedup': speedup,
                'efficiency': efficiency,
                'optimal_workers': max_workers
            },
            'system_info': self.system_info
        }
        
        return benchmark_result
    
    def optimize_evaluation_pipeline(self, results_dir: str) -> Dict:
        """优化评估流水线"""
        logger.info("🔧 优化评估流水线...")
        
        # 分析现有结果找出瓶颈
        bottlenecks = self._analyze_bottlenecks(results_dir)
        
        # 生成优化建议
        recommendations = self._generate_optimization_recommendations(bottlenecks)
        
        # 实施优化
        optimizations_applied = self._apply_optimizations(recommendations)
        
        return {
            'bottlenecks': bottlenecks,
            'recommendations': recommendations,
            'optimizations_applied': optimizations_applied
        }
    
    def _analyze_bottlenecks(self, results_dir: str) -> Dict:
        """分析性能瓶颈"""
        # 这里可以分析日志文件、性能数据等
        # 简化实现
        return {
            'data_generation': 'medium',
            'model_loading': 'low',
            'inference': 'high',
            'baseline_calculation': 'medium'
        }
    
    def _generate_optimization_recommendations(self, bottlenecks: Dict) -> List[str]:
        """生成优化建议"""
        recommendations = []
        
        if bottlenecks.get('inference') == 'high':
            recommendations.extend([
                "使用TorchScript编译模型以提升推理速度",
                "批量处理推理请求",
                "使用GPU加速（如果可用）",
                "优化模型架构减少计算复杂度"
            ])
        
        if bottlenecks.get('data_generation') in ['medium', 'high']:
            recommendations.extend([
                "预生成和缓存测试数据",
                "使用更高效的数据结构",
                "并行化数据生成过程"
            ])
        
        if bottlenecks.get('baseline_calculation') in ['medium', 'high']:
            recommendations.extend([
                "缓存基线计算结果",
                "使用近似算法加速基线计算",
                "减少基线样本数量"
            ])
        
        return recommendations
    
    def _apply_optimizations(self, recommendations: List[str]) -> List[str]:
        """应用优化措施"""
        applied = []
        
        # 优化PyTorch设置
        self.optimize_torch_settings()
        applied.append("优化PyTorch线程和内存设置")
        
        # 设置环境变量
        os.environ['OMP_NUM_THREADS'] = str(min(self.system_info['cpu_count'], 8))
        os.environ['MKL_NUM_THREADS'] = str(min(self.system_info['cpu_count'], 8))
        applied.append("设置OpenMP和MKL线程数")
        
        return applied
    
    def create_performance_report(self, profile_results: List[Dict], 
                                benchmark_result: Dict, output_path: str):
        """创建性能报告"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'system_info': self.system_info,
            'profile_results': profile_results,
            'benchmark_result': benchmark_result,
            'summary': self._generate_performance_summary(profile_results, benchmark_result)
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"性能报告已保存: {output_path}")
        return report
    
    def _generate_performance_summary(self, profile_results: List[Dict], 
                                    benchmark_result: Dict) -> Dict:
        """生成性能摘要"""
        if not profile_results:
            return {}
        
        # 计算平均指标
        avg_total_time = np.mean([r['timing']['total_time'] for r in profile_results])
        avg_inference_time = np.mean([r['inference_stats']['avg_inference_ms'] for r in profile_results])
        avg_memory_usage = np.mean([r['memory_usage']['peak_memory_mb'] for r in profile_results])
        
        summary = {
            'avg_evaluation_time_seconds': avg_total_time,
            'avg_inference_time_ms': avg_inference_time,
            'avg_memory_usage_mb': avg_memory_usage,
            'parallel_speedup': benchmark_result.get('performance_metrics', {}).get('speedup', 0),
            'parallel_efficiency': benchmark_result.get('performance_metrics', {}).get('efficiency', 0),
            'recommendations': [
                f"平均评估时间: {avg_total_time:.2f}秒",
                f"平均推理时间: {avg_inference_time:.2f}毫秒",
                f"并行加速比: {benchmark_result.get('performance_metrics', {}).get('speedup', 0):.2f}x"
            ]
        }
        
        return summary

def main():
    parser = argparse.ArgumentParser(description="性能优化器")
    parser.add_argument("--profile", action="store_true", help="性能分析模式")
    parser.add_argument("--benchmark", action="store_true", help="基准测试模式")
    parser.add_argument("--port", help="港口名称（性能分析用）")
    parser.add_argument("--stage", help="阶段名称（性能分析用）")
    parser.add_argument("--samples", type=int, default=100, help="测试样本数")
    parser.add_argument("--max-workers", type=int, help="最大并行进程数")
    parser.add_argument("--output", default="../../reports/performance_report.json", 
                       help="输出报告路径")
    
    args = parser.parse_args()
    
    optimizer = PerformanceOptimizer()
    
    if args.profile and args.port and args.stage:
        # 单个阶段性能分析
        logger.info(f"🔍 性能分析: {args.port}/{args.stage}")
        result = optimizer.profile_single_evaluation(args.port, args.stage, args.samples)
        
        if 'error' in result:
            logger.error(f"性能分析失败: {result['error']}")
        else:
            print("\n📊 性能分析结果:")
            print(f"总时间: {result['timing']['total_time']:.2f}秒")
            print(f"模型加载: {result['timing']['model_load_time']:.2f}秒")
            print(f"数据生成: {result['timing']['data_generation_time']:.2f}秒")
            print(f"推理时间: {result['timing']['inference_time']:.2f}秒")
            print(f"平均单次推理: {result['inference_stats']['avg_inference_ms']:.2f}毫秒")
            print(f"内存使用: {result['memory_usage']['peak_memory_mb']:.1f}MB")
    
    elif args.benchmark:
        # 并行基准测试
        # 使用一些示例任务
        test_tasks = [
            ("baton_rouge", "基础阶段"),
            ("baton_rouge", "中级阶段"),
            ("new_orleans", "基础阶段"),
            ("new_orleans", "初级阶段")
        ]
        
        logger.info("🚀 并行基准测试")
        benchmark_result = optimizer.benchmark_parallel_evaluation(
            test_tasks, args.samples, args.max_workers
        )
        
        print("\n📊 并行基准测试结果:")
        print(f"串行时间: {benchmark_result['serial_benchmark']['total_time']:.2f}秒")
        print(f"并行时间: {benchmark_result['parallel_benchmark']['total_time']:.2f}秒")
        print(f"加速比: {benchmark_result['performance_metrics']['speedup']:.2f}x")
        print(f"效率: {benchmark_result['performance_metrics']['efficiency']:.2f}")
        
        # 保存基准测试报告
        optimizer.create_performance_report([], benchmark_result, args.output)
    
    else:
        # 完整性能优化分析
        logger.info("🔧 完整性能优化分析")
        
        # 分析几个代表性阶段
        test_cases = [
            ("baton_rouge", "基础阶段"),
            ("new_orleans", "初级阶段")
        ]
        
        profile_results = []
        for port, stage in test_cases:
            result = optimizer.profile_single_evaluation(port, stage, args.samples)
            if 'error' not in result:
                profile_results.append(result)
        
        # 并行基准测试
        benchmark_result = optimizer.benchmark_parallel_evaluation(test_cases, args.samples)
        
        # 生成完整报告
        report = optimizer.create_performance_report(profile_results, benchmark_result, args.output)
        
        print("\n📋 性能优化报告:")
        summary = report['summary']
        for rec in summary.get('recommendations', []):
            print(f"  • {rec}")

if __name__ == "__main__":
    main()