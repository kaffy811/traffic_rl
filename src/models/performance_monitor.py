#!/usr/bin/env python3
"""
完整的性能监控模块
支持推理延迟测试、资源消耗评估、模型分析、实时监控等
"""

import time
import psutil
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Callable, Any, Union
import threading
import multiprocessing
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import json
import pickle
import logging
from collections import defaultdict, deque
import gc
import sys
import tracemalloc
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import GPUtil
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# 设置图表样式
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


@dataclass
class PerformanceConfig:
    """性能监控配置"""
    monitor_interval: float = 0.1
    warmup_runs: int = 10
    test_runs: int = 100
    max_history_size: int = 10000
    enable_gpu_monitoring: bool = True
    enable_memory_profiling: bool = True
    enable_realtime_plot: bool = False
    save_detailed_logs: bool = True
    log_level: str = "INFO"

    # 测试配置
    batch_sizes: List[int] = field(default_factory=lambda: [1, 2, 4, 8, 16, 32, 64])
    sequence_lengths: List[int] = field(default_factory=lambda: [10, 50, 100, 200])
    precision_modes: List[str] = field(default_factory=lambda: ['float32', 'float16'])

    # 阈值设置
    latency_warning_threshold: float = 100.0  # ms
    memory_warning_threshold: float = 80.0  # %
    cpu_warning_threshold: float = 90.0  # %

    # 输出配置
    output_dir: str = "performance_reports"
    report_format: List[str] = field(default_factory=lambda: ['json', 'csv', 'html'])


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: float
    latency_ms: float
    cpu_percent: float
    memory_mb: float
    memory_percent: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    throughput_qps: Optional[float] = None
    batch_size: Optional[int] = None
    model_flops: Optional[int] = None

    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'latency_ms': self.latency_ms,
            'cpu_percent': self.cpu_percent,
            'memory_mb': self.memory_mb,
            'memory_percent': self.memory_percent,
            'gpu_memory_mb': self.gpu_memory_mb,
            'gpu_utilization': self.gpu_utilization,
            'throughput_qps': self.throughput_qps,
            'batch_size': self.batch_size,
            'model_flops': self.model_flops
        }


class SystemMonitor:
    """系统资源监控器"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.monitoring = False
        self.metrics_history = deque(maxlen=config.max_history_size)
        self.monitor_thread = None

        # GPU检测
        self.gpu_available = torch.cuda.is_available() and config.enable_gpu_monitoring
        if self.gpu_available:
            try:
                import GPUtil
                self.gpus = GPUtil.getGPUs()
            except ImportError:
                self.gpu_available = False
                logging.warning("GPUtil not available, GPU monitoring disabled")

        # 进程监控
        self.process = psutil.Process()

        # 性能计数器
        self.start_time = None
        self.total_inferences = 0

    @contextmanager
    def monitor_context(self, description: str = "performance_test"):
        """性能监控上下文管理器"""
        print(f"🔍 开始监控: {description}")

        self.start_monitoring()
        start_time = time.time()

        try:
            yield self
        finally:
            end_time = time.time()
            self.stop_monitoring()

            # 计算统计信息
            duration = end_time - start_time
            avg_metrics = self.get_average_metrics()

            print(f"📊 {description} 性能报告:")
            print(f"  总耗时: {duration:.4f}秒")
            print(f"  平均CPU: {avg_metrics.get('cpu_percent', 0):.2f}%")
            print(f"  平均内存: {avg_metrics.get('memory_mb', 0):.1f}MB ({avg_metrics.get('memory_percent', 0):.1f}%)")

            if self.gpu_available and 'gpu_memory_mb' in avg_metrics:
                print(f"  平均GPU内存: {avg_metrics.get('gpu_memory_mb', 0):.1f}MB")
                print(f"  平均GPU利用率: {avg_metrics.get('gpu_utilization', 0):.1f}%")

            if 'throughput_qps' in avg_metrics:
                print(f"  平均吞吐量: {avg_metrics.get('throughput_qps', 0):.2f} QPS")

    def start_monitoring(self):
        """开始监控"""
        self.monitoring = True
        self.metrics_history.clear()
        self.start_time = time.time()
        self.total_inferences = 0

        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()

    def stop_monitoring(self):
        """停止监控"""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)

    def _monitor_loop(self):
        """监控循环"""
        while self.monitoring:
            try:
                metrics = self._collect_system_metrics()
                self.metrics_history.append(metrics)

                # 检查警告阈值
                self._check_warning_thresholds(metrics)

            except Exception as e:
                logging.error(f"监控过程出错: {e}")

            time.sleep(self.config.monitor_interval)

    def _collect_system_metrics(self) -> PerformanceMetrics:
        """收集系统指标"""
        # CPU和内存
        cpu_percent = self.process.cpu_percent()
        memory_info = self.process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        memory_percent = self.process.memory_percent()

        # GPU指标
        gpu_memory_mb = None
        gpu_utilization = None

        if self.gpu_available and torch.cuda.is_available():
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 / 1024
                if self.gpus:
                    gpu_utilization = self.gpus[0].load * 100
            except Exception as e:
                logging.warning(f"GPU监控失败: {e}")

        # 吞吐量计算
        throughput_qps = None
        if self.start_time and self.total_inferences > 0:
            elapsed = time.time() - self.start_time
            throughput_qps = self.total_inferences / elapsed if elapsed > 0 else 0

        return PerformanceMetrics(
            timestamp=time.time(),
            latency_ms=0,  # 在推理测试中单独设置
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            memory_percent=memory_percent,
            gpu_memory_mb=gpu_memory_mb,
            gpu_utilization=gpu_utilization,
            throughput_qps=throughput_qps
        )

    def _check_warning_thresholds(self, metrics: PerformanceMetrics):
        """检查警告阈值"""
        if metrics.cpu_percent > self.config.cpu_warning_threshold:
            logging.warning(f"高CPU使用率: {metrics.cpu_percent:.1f}%")

        if metrics.memory_percent > self.config.memory_warning_threshold:
            logging.warning(f"高内存使用率: {metrics.memory_percent:.1f}%")

        if metrics.latency_ms > self.config.latency_warning_threshold:
            logging.warning(f"高延迟: {metrics.latency_ms:.2f}ms")

    def record_inference(self, latency_ms: float):
        """记录推理"""
        self.total_inferences += 1

        # 更新最新指标的延迟
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
            latest_metrics.latency_ms = latency_ms

    def get_average_metrics(self) -> Dict[str, float]:
        """获取平均指标"""
        if not self.metrics_history:
            return {}

        # 计算平均值
        metrics_dict = defaultdict(list)
        for metrics in self.metrics_history:
            for key, value in metrics.to_dict().items():
                if value is not None and key != 'timestamp':
                    metrics_dict[key].append(value)

        avg_metrics = {}
        for key, values in metrics_dict.items():
            if values:
                avg_metrics[key] = np.mean(values)

        return avg_metrics

    def get_metrics_dataframe(self) -> pd.DataFrame:
        """获取指标DataFrame"""
        if not self.metrics_history:
            return pd.DataFrame()

        data = [metrics.to_dict() for metrics in self.metrics_history]
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        return df


class InferenceLatencyTester:
    """推理延迟测试器"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.monitor = SystemMonitor(config)

    def test_model_latency(self,
                           model: nn.Module,
                           input_generator: Callable,
                           input_args: Dict = None,
                           device: str = 'cpu') -> Dict[str, Any]:
        """
        测试模型推理延迟

        Args:
            model: 要测试的模型
            input_generator: 输入生成函数
            input_args: 输入生成参数
            device: 设备

        Returns:
            详细的延迟统计结果
        """

        input_args = input_args or {}
        model = model.to(device)
        model.eval()

        print(f"🚀 测试模型推理延迟")
        print(f"  设备: {device}")
        print(f"  预热轮次: {self.config.warmup_runs}")
        print(f"  测试轮次: {self.config.test_runs}")

        # 预热
        print("🔥 模型预热中...")
        for _ in range(self.config.warmup_runs):
            with torch.no_grad():
                inputs = input_generator(**input_args)
                if isinstance(inputs, (list, tuple)):
                    inputs = [inp.to(device) if torch.is_tensor(inp) else inp for inp in inputs]
                elif torch.is_tensor(inputs):
                    inputs = inputs.to(device)

                if isinstance(inputs, (list, tuple)):
                    _ = model(*inputs)
                else:
                    _ = model(inputs)

        if device.startswith('cuda'):
            torch.cuda.synchronize()

        # 测试延迟
        latencies = []

        with self.monitor.monitor_context("推理延迟测试"):
            for i in range(self.config.test_runs):
                # 生成输入
                inputs = input_generator(**input_args)
                if isinstance(inputs, (list, tuple)):
                    inputs = [inp.to(device) if torch.is_tensor(inp) else inp for inp in inputs]
                elif torch.is_tensor(inputs):
                    inputs = inputs.to(device)

                if device.startswith('cuda'):
                    torch.cuda.synchronize()

                start_time = time.time()

                with torch.no_grad():
                    if isinstance(inputs, (list, tuple)):
                        output = model(*inputs)
                    else:
                        output = model(inputs)

                if device.startswith('cuda'):
                    torch.cuda.synchronize()

                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                latencies.append(latency_ms)

                # 记录到监控器
                self.monitor.record_inference(latency_ms)

                if (i + 1) % (self.config.test_runs // 10) == 0:
                    print(f"  完成 {i + 1}/{self.config.test_runs} 次推理")

        # 计算详细统计
        latencies = np.array(latencies)

        results = {
            'device': device,
            'test_runs': self.config.test_runs,
            'warmup_runs': self.config.warmup_runs,

            # 延迟统计
            'mean_latency_ms': float(np.mean(latencies)),
            'std_latency_ms': float(np.std(latencies)),
            'min_latency_ms': float(np.min(latencies)),
            'max_latency_ms': float(np.max(latencies)),
            'median_latency_ms': float(np.median(latencies)),
            'p95_latency_ms': float(np.percentile(latencies, 95)),
            'p99_latency_ms': float(np.percentile(latencies, 99)),

            # 吞吐量
            'throughput_qps': 1000 / np.mean(latencies),
            'peak_throughput_qps': 1000 / np.min(latencies),

            # 资源使用
            'avg_system_metrics': self.monitor.get_average_metrics(),

            # 原始数据
            'raw_latencies': latencies.tolist(),
            'metrics_history': [m.to_dict() for m in self.monitor.metrics_history]
        }

        return results

    def benchmark_batch_sizes(self,
                              model: nn.Module,
                              input_generator: Callable,
                              batch_sizes: List[int] = None,
                              device: str = 'cpu') -> Dict[int, Dict]:
        """
        测试不同批量大小的性能

        Args:
            model: 模型
            input_generator: 输入生成函数，需要接受batch_size参数
            batch_sizes: 批量大小列表
            device: 设备

        Returns:
            各批量大小的性能结果
        """

        batch_sizes = batch_sizes or self.config.batch_sizes
        results = {}

        print("📊 批量大小性能测试")
        print(f"  测试批量: {batch_sizes}")

        for batch_size in batch_sizes:
            print(f"\n🔍 测试批量大小: {batch_size}")

            # 测试当前批量大小
            batch_results = self.test_model_latency(
                model=model,
                input_generator=input_generator,
                input_args={'batch_size': batch_size},
                device=device
            )

            # 添加批量相关指标
            batch_results['batch_size'] = batch_size
            batch_results['per_sample_latency_ms'] = batch_results['mean_latency_ms'] / batch_size
            batch_results['samples_per_second'] = batch_size * batch_results['throughput_qps']

            results[batch_size] = batch_results

            print(f"  平均延迟: {batch_results['mean_latency_ms']:.2f}ms")
            print(f"  单样本延迟: {batch_results['per_sample_latency_ms']:.2f}ms")
            print(f"  样本吞吐量: {batch_results['samples_per_second']:.1f} samples/s")

            # 清理GPU缓存
            if device.startswith('cuda'):
                torch.cuda.empty_cache()
                gc.collect()

        return results

    def benchmark_precision_modes(self,
                                  model: nn.Module,
                                  input_generator: Callable,
                                  precision_modes: List[str] = None,
                                  device: str = 'cpu') -> Dict[str, Dict]:
        """
        测试不同精度模式的性能

        Args:
            model: 模型
            input_generator: 输入生成函数
            precision_modes: 精度模式列表 ['float32', 'float16', 'int8']
            device: 设备

        Returns:
            各精度模式的性能结果
        """

        precision_modes = precision_modes or self.config.precision_modes
        results = {}

        print("🎯 精度模式性能测试")
        print(f"  测试精度: {precision_modes}")

        original_model = model

        for precision in precision_modes:
            print(f"\n🔍 测试精度模式: {precision}")

            try:
                # 转换模型精度
                if precision == 'float16' and device.startswith('cuda'):
                    test_model = original_model.half()
                elif precision == 'float32':
                    test_model = original_model.float()
                else:
                    print(f"  跳过不支持的精度: {precision}")
                    continue

                # 测试性能
                precision_results = self.test_model_latency(
                    model=test_model,
                    input_generator=lambda **kwargs: self._convert_input_precision(
                        input_generator(**kwargs), precision, device
                    ),
                    device=device
                )

                precision_results['precision'] = precision
                results[precision] = precision_results

                print(f"  平均延迟: {precision_results['mean_latency_ms']:.2f}ms")
                print(f"  吞吐量: {precision_results['throughput_qps']:.1f} QPS")

            except Exception as e:
                print(f"  精度 {precision} 测试失败: {e}")
                continue

        return results

    def _convert_input_precision(self, inputs, precision: str, device: str):
        """转换输入精度"""
        if isinstance(inputs, torch.Tensor):
            if precision == 'float16':
                return inputs.half().to(device)
            elif precision == 'float32':
                return inputs.float().to(device)
            else:
                return inputs.to(device)
        elif isinstance(inputs, (list, tuple)):
            return [self._convert_input_precision(inp, precision, device) for inp in inputs]
        else:
            return inputs


class ModelProfiler:
    """模型分析器"""

    def __init__(self, config: PerformanceConfig):
        self.config = config

    def profile_model_complexity(self,
                                 model: nn.Module,
                                 input_generator: Callable,
                                 input_args: Dict = None) -> Dict[str, Any]:
        """
        分析模型复杂度

        Returns:
            模型复杂度分析结果
        """

        input_args = input_args or {}

        print("🔬 分析模型复杂度")

        # 获取模型信息
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        # 获取模型大小
        param_size = sum(p.numel() * p.element_size() for p in model.parameters())
        buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
        model_size_mb = (param_size + buffer_size) / 1024 / 1024

        # 分析模型结构
        layer_info = self._analyze_model_layers(model)

        # 估算FLOPs
        try:
            sample_input = input_generator(**input_args)
            flops = self._estimate_flops(model, sample_input)
        except Exception as e:
            print(f"FLOPs估算失败: {e}")
            flops = None

        results = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': model_size_mb,
            'estimated_flops': flops,
            'layer_analysis': layer_info,
            'model_summary': str(model)
        }

        print(f"  总参数量: {total_params:,}")
        print(f"  可训练参数: {trainable_params:,}")
        print(f"  模型大小: {model_size_mb:.2f}MB")
        if flops:
            print(f"  估算FLOPs: {flops:,}")

        return results

    def _analyze_model_layers(self, model: nn.Module) -> Dict[str, Any]:
        """分析模型层"""
        layer_info = {
            'total_layers': 0,
            'layer_types': defaultdict(int),
            'layer_details': []
        }

        for name, module in model.named_modules():
            if len(list(module.children())) == 0:  # 叶子节点
                layer_info['total_layers'] += 1
                layer_type = type(module).__name__
                layer_info['layer_types'][layer_type] += 1

                # 详细信息
                layer_detail = {
                    'name': name,
                    'type': layer_type,
                    'parameters': sum(p.numel() for p in module.parameters())
                }

                # 特定层的信息
                if isinstance(module, nn.Linear):
                    layer_detail['in_features'] = module.in_features
                    layer_detail['out_features'] = module.out_features
                elif isinstance(module, nn.Conv2d):
                    layer_detail['in_channels'] = module.in_channels
                    layer_detail['out_channels'] = module.out_channels
                    layer_detail['kernel_size'] = module.kernel_size

                layer_info['layer_details'].append(layer_detail)

        return layer_info

    def _estimate_flops(self, model: nn.Module, sample_input) -> Optional[int]:
        """估算FLOPs"""
        try:
            # 这里可以集成 thop, fvcore 等库来精确计算FLOPs
            # 简化实现：基于参数量的粗略估算
            total_params = sum(p.numel() for p in model.parameters())

            # 假设每个参数平均执行2次运算（1次乘法，1次加法）
            estimated_flops = total_params * 2

            return estimated_flops
        except Exception:
            return None

    def memory_profiling(self,
                         model: nn.Module,
                         input_generator: Callable,
                         input_args: Dict = None,
                         device: str = 'cpu') -> Dict[str, Any]:
        """
        内存使用分析

        Returns:
            内存使用分析结果
        """

        input_args = input_args or {}
        model = model.to(device)

        print("🧠 分析内存使用")

        # 启用内存追踪
        if self.config.enable_memory_profiling:
            tracemalloc.start()

        # 基准内存
        if device.startswith('cuda'):
            torch.cuda.reset_peak_memory_stats()
            baseline_gpu_memory = torch.cuda.memory_allocated()

        baseline_cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024

        # 前向传播
        inputs = input_generator(**input_args)
        if isinstance(inputs, (list, tuple)):
            inputs = [inp.to(device) if torch.is_tensor(inp) else inp for inp in inputs]
        elif torch.is_tensor(inputs):
            inputs = inputs.to(device)

        with torch.no_grad():
            if isinstance(inputs, (list, tuple)):
                output = model(*inputs)
            else:
                output = model(inputs)

        # 内存使用
        peak_cpu_memory = psutil.Process().memory_info().rss / 1024 / 1024
        cpu_memory_usage = peak_cpu_memory - baseline_cpu_memory

        gpu_memory_usage = None
        peak_gpu_memory = None

        if device.startswith('cuda'):
            peak_gpu_memory = torch.cuda.max_memory_allocated()
            gpu_memory_usage = (peak_gpu_memory - baseline_gpu_memory) / 1024 / 1024

        # CPU内存追踪
        cpu_memory_trace = None
        if self.config.enable_memory_profiling:
            current, peak = tracemalloc.get_traced_memory()
            cpu_memory_trace = {
                'current_mb': current / 1024 / 1024,
                'peak_mb': peak / 1024 / 1024
            }
            tracemalloc.stop()

        results = {
            'device': device,
            'cpu_memory_usage_mb': cpu_memory_usage,
            'gpu_memory_usage_mb': gpu_memory_usage,
            'peak_gpu_memory_mb': peak_gpu_memory / 1024 / 1024 if peak_gpu_memory else None,
            'cpu_memory_trace': cpu_memory_trace,
            'baseline_cpu_memory_mb': baseline_cpu_memory,
            'peak_cpu_memory_mb': peak_cpu_memory
        }

        print(f"  CPU内存使用: {cpu_memory_usage:.2f}MB")
        if gpu_memory_usage is not None:
            print(f"  GPU内存使用: {gpu_memory_usage:.2f}MB")

        return results


class PerformanceReporter:
    """性能报告生成器"""

    def __init__(self, config: PerformanceConfig):
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def generate_comprehensive_report(self,
                                      latency_results: Dict,
                                      batch_results: Dict = None,
                                      precision_results: Dict = None,
                                      complexity_results: Dict = None,
                                      memory_results: Dict = None) -> str:
        """
        生成综合性能报告

        Returns:
            报告文件路径
        """

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_name = f"performance_report_{timestamp}"

        print(f"📝 生成综合性能报告: {report_name}")

        # 生成各种格式的报告
        report_files = []

        if 'json' in self.config.report_format:
            json_file = self._generate_json_report(
                report_name, latency_results, batch_results,
                precision_results, complexity_results, memory_results
            )
            report_files.append(json_file)

        if 'csv' in self.config.report_format:
            csv_file = self._generate_csv_report(
                report_name, latency_results, batch_results, precision_results
            )
            report_files.append(csv_file)

        if 'html' in self.config.report_format:
            html_file = self._generate_html_report(
                report_name, latency_results, batch_results,
                precision_results, complexity_results, memory_results
            )
            report_files.append(html_file)

        # 生成可视化图表
        plots_dir = self.output_dir / f"{report_name}_plots"
        plots_dir.mkdir(exist_ok=True)

        self._generate_performance_plots(
            plots_dir, latency_results, batch_results, precision_results
        )

        print(f"  报告已保存到: {self.output_dir}")
        for file in report_files:
            print(f"    - {file}")

        return str(self.output_dir / f"{report_name}.html")

    def _generate_json_report(self, report_name: str, *results) -> str:
        """生成JSON报告"""
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'config': self.config.__dict__,
            'latency_results': results[0] if len(results) > 0 else None,
            'batch_results': results[1] if len(results) > 1 else None,
            'precision_results': results[2] if len(results) > 2 else None,
            'complexity_results': results[3] if len(results) > 3 else None,
            'memory_results': results[4] if len(results) > 4 else None
        }

        json_file = self.output_dir / f"{report_name}.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)

        return str(json_file)

    def _generate_csv_report(self, report_name: str, *results) -> str:
        """生成CSV报告"""
        csv_data = []

        # 延迟结果
        if results[0]:
            csv_data.append({
                'test_type': 'latency',
                'metric': 'mean_latency_ms',
                'value': results[0]['mean_latency_ms']
            })
            csv_data.append({
                'test_type': 'latency',
                'metric': 'throughput_qps',
                'value': results[0]['throughput_qps']
            })

        # 批量大小结果
        if results[1]:
            for batch_size, data in results[1].items():
                csv_data.append({
                    'test_type': 'batch_size',
                    'batch_size': batch_size,
                    'metric': 'mean_latency_ms',
                    'value': data['mean_latency_ms']
                })
                csv_data.append({
                    'test_type': 'batch_size',
                    'batch_size': batch_size,
                    'metric': 'samples_per_second',
                    'value': data['samples_per_second']
                })

        # 精度结果
        if results[2]:
            for precision, data in results[2].items():
                csv_data.append({
                    'test_type': 'precision',
                    'precision': precision,
                    'metric': 'mean_latency_ms',
                    'value': data['mean_latency_ms']
                })

        df = pd.DataFrame(csv_data)
        csv_file = self.output_dir / f"{report_name}.csv"
        df.to_csv(csv_file, index=False)

        return str(csv_file)

    def _generate_html_report(self, report_name: str, *results) -> str:
        """生成HTML报告"""
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>性能测试报告 - {report_name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .header {{ background-color: #f0f8ff; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }}
                .metric {{ margin: 10px 0; }}
                .value {{ font-weight: bold; color: #2e8b57; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🚀 性能测试报告</h1>
                <p>生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
        """

        # 延迟测试结果
        if results[0]:
            latency_data = results[0]
            html_content += f"""
            <div class="section">
                <h2>📊 延迟测试结果</h2>
                <div class="metric">平均延迟: <span class="value">{latency_data['mean_latency_ms']:.2f}ms</span></div>
                <div class="metric">P95延迟: <span class="value">{latency_data['p95_latency_ms']:.2f}ms</span></div>
                <div class="metric">P99延迟: <span class="value">{latency_data['p99_latency_ms']:.2f}ms</span></div>
                <div class="metric">吞吐量: <span class="value">{latency_data['throughput_qps']:.2f} QPS</span></div>
                <div class="metric">设备: <span class="value">{latency_data['device']}</span></div>
            </div>
            """

        # 批量大小测试结果
        if results[1]:
            html_content += """
            <div class="section">
                <h2>📈 批量大小测试结果</h2>
                <table>
                    <tr>
                        <th>批量大小</th>
                        <th>平均延迟 (ms)</th>
                        <th>单样本延迟 (ms)</th>
                        <th>样本吞吐量 (samples/s)</th>
                    </tr>
            """

            for batch_size, data in results[1].items():
                html_content += f"""
                    <tr>
                        <td>{batch_size}</td>
                        <td>{data['mean_latency_ms']:.2f}</td>
                        <td>{data['per_sample_latency_ms']:.2f}</td>
                        <td>{data['samples_per_second']:.1f}</td>
                    </tr>
                """

            html_content += "</table></div>"

        # 模型复杂度结果
        if results[3]:
            complexity_data = results[3]
            html_content += f"""
            <div class="section">
                <h2>🔬 模型复杂度分析</h2>
                <div class="metric">总参数量: <span class="value">{complexity_data['total_parameters']:,}</span></div>
                <div class="metric">可训练参数: <span class="value">{complexity_data['trainable_parameters']:,}</span></div>
                <div class="metric">模型大小: <span class="value">{complexity_data['model_size_mb']:.2f}MB</span></div>
                <div class="metric">总层数: <span class="value">{complexity_data['layer_analysis']['total_layers']}</span></div>
            """

            if complexity_data['estimated_flops']:
                html_content += f"""
                <div class="metric">估算FLOPs: <span class="value">{complexity_data['estimated_flops']:,}</span></div>
                """

            html_content += "</div>"

        html_content += "</body></html>"

        html_file = self.output_dir / f"{report_name}.html"
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return str(html_file)

    def _generate_performance_plots(self,
                                    plots_dir: Path,
                                    latency_results: Dict,
                                    batch_results: Dict = None,
                                    precision_results: Dict = None):
        """生成性能可视化图表"""

        # 延迟分布图
        if latency_results and 'raw_latencies' in latency_results:
            plt.figure(figsize=(10, 6))
            plt.hist(latency_results['raw_latencies'], bins=50, alpha=0.7, edgecolor='black')
            plt.axvline(latency_results['mean_latency_ms'], color='red', linestyle='--',
                        label=f"平均值: {latency_results['mean_latency_ms']:.2f}ms")
            plt.axvline(latency_results['p95_latency_ms'], color='orange', linestyle='--',
                        label=f"P95: {latency_results['p95_latency_ms']:.2f}ms")
            plt.xlabel('延迟 (ms)')
            plt.ylabel('频次')
            plt.title('推理延迟分布')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(plots_dir / 'latency_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 批量大小性能图
        if batch_results:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            batch_sizes = list(batch_results.keys())
            latencies = [batch_results[bs]['mean_latency_ms'] for bs in batch_sizes]
            throughputs = [batch_results[bs]['samples_per_second'] for bs in batch_sizes]

            # 延迟图
            ax1.plot(batch_sizes, latencies, 'o-', linewidth=2, markersize=8)
            ax1.set_xlabel('批量大小')
            ax1.set_ylabel('平均延迟 (ms)')
            ax1.set_title('批量大小 vs 延迟')
            ax1.grid(True, alpha=0.3)
            ax1.set_xscale('log', base=2)

            # 吞吐量图
            ax2.plot(batch_sizes, throughputs, 'o-', linewidth=2, markersize=8, color='green')
            ax2.set_xlabel('批量大小')
            ax2.set_ylabel('样本吞吐量 (samples/s)')
            ax2.set_title('批量大小 vs 吞吐量')
            ax2.grid(True, alpha=0.3)
            ax2.set_xscale('log', base=2)

            plt.tight_layout()
            plt.savefig(plots_dir / 'batch_size_performance.png', dpi=300, bbox_inches='tight')
            plt.close()

        # 精度模式对比图
        if precision_results:
            precisions = list(precision_results.keys())
            latencies = [precision_results[p]['mean_latency_ms'] for p in precisions]
            throughputs = [precision_results[p]['throughput_qps'] for p in precisions]

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            # 延迟对比
            bars1 = ax1.bar(precisions, latencies, color=['skyblue', 'lightcoral'])
            ax1.set_ylabel('平均延迟 (ms)')
            ax1.set_title('精度模式延迟对比')
            ax1.grid(True, alpha=0.3)

            # 添加数值标签
            for bar, latency in zip(bars1, latencies):
                ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f'{latency:.2f}', ha='center', va='bottom')

            # 吞吐量对比
            bars2 = ax2.bar(precisions, throughputs, color=['lightgreen', 'orange'])
            ax2.set_ylabel('吞吐量 (QPS)')
            ax2.set_title('精度模式吞吐量对比')
            ax2.grid(True, alpha=0.3)

            # 添加数值标签
            for bar, throughput in zip(bars2, throughputs):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                         f'{throughput:.1f}', ha='center', va='bottom')

            plt.tight_layout()
            plt.savefig(plots_dir / 'precision_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()


class ComprehensivePerformanceTester:
    """综合性能测试器"""

    def __init__(self, config: PerformanceConfig = None):
        self.config = config or PerformanceConfig()
        self.latency_tester = InferenceLatencyTester(self.config)
        self.profiler = ModelProfiler(self.config)
        self.reporter = PerformanceReporter(self.config)

        # 设置日志
        logging.basicConfig(
            level=getattr(logging, self.config.log_level),
            format='%(asctime)s - %(levelname)s - %(message)s'
        )

    def run_full_benchmark(self,
                           model: nn.Module,
                           input_generator: Callable,
                           device: str = 'cpu',
                           run_batch_test: bool = True,
                           run_precision_test: bool = True,
                           run_complexity_analysis: bool = True,
                           run_memory_analysis: bool = True) -> str:
        """
        运行完整的性能基准测试

        Args:
            model: 要测试的模型
            input_generator: 输入生成函数
            device: 测试设备
            run_batch_test: 是否运行批量大小测试
            run_precision_test: 是否运行精度测试
            run_complexity_analysis: 是否运行复杂度分析
            run_memory_analysis: 是否运行内存分析

        Returns:
            报告文件路径
        """

        print("🚀 开始综合性能基准测试")
        print("=" * 60)

        results = {}

        # 1. 基础延迟测试
        print("\n1️⃣ 基础延迟测试")
        results['latency'] = self.latency_tester.test_model_latency(
            model=model,
            input_generator=input_generator,
            device=device
        )

        # 2. 批量大小测试
        if run_batch_test:
            print("\n2️⃣ 批量大小性能测试")
            results['batch_sizes'] = self.latency_tester.benchmark_batch_sizes(
                model=model,
                input_generator=input_generator,
                device=device
            )

        # 3. 精度模式测试
        if run_precision_test and device.startswith('cuda'):
            print("\n精度模式性能测试")
            results['precision_modes'] = self.latency_tester.benchmark_precision_modes(
                model=model,
                input_generator=input_generator,
                device=device
            )

        # 4. 模型复杂度分析
        if run_complexity_analysis:
            print("\n模型复杂度分析")
            results['complexity'] = self.profiler.profile_model_complexity(
                model=model,
                input_generator=input_generator
            )

        # 5. 内存使用分析
        if run_memory_analysis:
            print("\n内存使用分析")
            results['memory'] = self.profiler.memory_profiling(
                model=model,
                input_generator=input_generator,
                device=device
            )

        # 6. 生成综合报告
        print("\n生成综合报告")
        report_path = self.reporter.generate_comprehensive_report(
            latency_results=results.get('latency'),
            batch_results=results.get('batch_sizes'),
            precision_results=results.get('precision_modes'),
            complexity_results=results.get('complexity'),
            memory_results=results.get('memory')
        )

        print("\n综合性能基准测试完成！")
        print(f"报告已保存到: {report_path}")

        return report_path


def create_test_input_generator():
    """创建测试输入生成器（用于示例）"""

    def generator(batch_size=1, seq_length=50, feature_dim=128):
        return torch.randn(batch_size, seq_length, feature_dim)

    return generator


def comprehensive_test():
    """完整的性能监控测试"""

    print("完整性能监控模块测试")
    print("=" * 60)

    # 创建测试模型
    test_model = nn.Sequential(
        nn.Linear(128, 256),
        nn.ReLU(),
        nn.LayerNorm(256),
        nn.Dropout(0.1),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.Linear(128, 4)
    )

    # 创建配置
    config = PerformanceConfig(
        test_runs=50,
        warmup_runs=5,
        batch_sizes=[1, 2, 4, 8],
        enable_memory_profiling=True,
        report_format=['json', 'html']
    )

    # 创建性能测试器
    tester = ComprehensivePerformanceTester(config)

    # 创建输入生成器
    input_gen = create_test_input_generator()

    # 运行完整基准测试
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    report_path = tester.run_full_benchmark(
        model=test_model,
        input_generator=input_gen,
        device=device,
        run_batch_test=True,
        run_precision_test=torch.cuda.is_available(),
        run_complexity_analysis=True,
        run_memory_analysis=True
    )

    print(f"\n测试完成！详细报告请查看: {report_path}")


if __name__ == "__main__":
    comprehensive_test()