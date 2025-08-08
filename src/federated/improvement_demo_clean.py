#!/usr/bin/env python3
"""
改进方向验证试验台 - 可重复、可量化的改进效果模拟
"""

import numpy as np
import torch

# 配置matplotlib（必须在pyplot导入前）
import matplotlib
matplotlib.use("Agg")  # 非交互后端
from matplotlib import rcParams
rcParams['font.sans-serif'] = [
    'Noto Sans CJK SC', 'PingFang SC', 'Heiti SC',
    'Hiragino Sans GB', 'Source Han Sans SC', 'Arial Unicode MS'
]
rcParams['axes.unicode_minus'] = False

import matplotlib.pyplot as plt
import argparse
import random
import json
import os
from typing import Dict, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging

# 配置日志
def setup_logging(log_file: str = None):
    handlers = [logging.StreamHandler()]
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)

def set_seed(seed: int = 42):
    """设置随机种子确保可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description="GAT-FedPPO改进效果验证试验台")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--alpha", type=float, default=1.0, help="α-fair聚合的α参数")
    parser.add_argument("--mc", type=int, default=200, help="Monte Carlo采样次数")
    parser.add_argument("--ports", type=str, default="all", 
                       help="测试港口，逗号分隔或'all'")
    parser.add_argument("--output-dir", type=str, default="../../results/improvement_demo",
                       help="结果输出目录")
    parser.add_argument("--enable-transfer", action="store_true", default=True,
                       help="启用转移学习")
    parser.add_argument("--enable-reward", action="store_true", default=True,
                       help="启用奖励优化")
    parser.add_argument("--enable-curriculum", action="store_true", default=True,
                       help="启用分阶段训练")
    parser.add_argument("--enable-hpo", action="store_true", default=True,
                       help="启用超参数优化")
    parser.add_argument("--enable-fed", action="store_true", default=True,
                       help="启用联邦学习")
    parser.add_argument("--visualize", action="store_true", default=True,
                       help="生成可视化图表")
    return parser.parse_args()

@dataclass
class GainDist:
    """增益分布定义"""
    mean: float    # 期望相对增益，例如 0.25 表示 +25%
    std: float     # 不确定性（标准差）
    
@dataclass 
class PortStatus:
    """港口状态"""
    name: str
    completion_rate: float
    avg_reward: float
    n_samples: int  # 有效样本量，影响不确定性

class ImprovementValidationBench:
    """改进效果验证试验台"""
    
    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        
        # 港口状态数据（包含样本量信息）
        self.port_status = {
            'gulfport': PortStatus('gulfport', 0.9488, 38.19, 162),
            'baton_rouge': PortStatus('baton_rouge', 0.3271, -1365.46, 874),
            'new_orleans': PortStatus('new_orleans', 0.1438, -3190.48, 1723),
            'south_louisiana': PortStatus('south_louisiana', 0.4373, -925.49, 603)
        }
        
        # 改进技术的增益分布定义
        self.gain_distributions = self._define_gain_distributions()
        
        # 结果存储
        self.results = {}
        
        # 输出目录
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("初始化改进效果验证试验台")
    
    def _define_gain_distributions(self) -> Dict:
        """定义各种改进技术的增益分布"""
        return {
            "transfer": {
                "baton_rouge": GainDist(0.22, 0.07),
                "new_orleans": GainDist(0.35, 0.10),
                "south_louisiana": GainDist(0.25, 0.08),
            },
            "reward": {
                "new_orleans": GainDist(0.18, 0.08),
                "baton_rouge": GainDist(0.12, 0.05),
                "south_louisiana": GainDist(0.10, 0.05),
                "gulfport": GainDist(0.03, 0.02),
            },
            "curriculum": {
                "new_orleans": GainDist(0.15, 0.06),
                "baton_rouge": GainDist(0.08, 0.04),
            },
            "hpo": {  # 超参数优化 - 所有港口都受益
                "all": GainDist(0.08, 0.03),
            },
            "federated": {  # 联邦学习额外收益
                "all": GainDist(0.05, 0.02),
            }
        }
    
    def sample_gain(self, gain_dist: GainDist, n_samples: int = None) -> float:
        """
        从增益分布中采样
        n_samples: 有效样本量，影响不确定性
        """
        if n_samples is not None and n_samples > 0:
            # 有效样本越多，std 越小；加一个下限避免过于自信
            eff_std = max(0.4 * gain_dist.std, gain_dist.std / np.sqrt(max(n_samples, 1)))
        else:
            eff_std = gain_dist.std
        
        return max(0.0, np.random.normal(gain_dist.mean, eff_std))
    
    def combine_completion_rates(self, baseline: float, gains: List[float]) -> float:
        """
        组合多个改进的效果，使用边际收益递减模型
        作用在"未完成率"上：new = 1 - (1 - baseline) * Π(1 - gain_k)
        """
        remaining_failure_rate = 1.0 - baseline
        for gain in gains:
            remaining_failure_rate *= (1.0 - gain)
        
        new_completion_rate = 1.0 - remaining_failure_rate
        return min(0.999, max(0.0, new_completion_rate))
    
    def get_applicable_gains(self, port_name: str) -> List[Tuple[str, GainDist]]:
        """获取适用于指定港口的改进技术"""
        applicable_gains = []
        
        # 转移学习（除了gulfport作为源港口）
        if self.args.enable_transfer and port_name != "gulfport":
            if port_name in self.gain_distributions["transfer"]:
                applicable_gains.append(("transfer", self.gain_distributions["transfer"][port_name]))
        
        # 奖励函数优化
        if self.args.enable_reward and port_name in self.gain_distributions["reward"]:
            applicable_gains.append(("reward", self.gain_distributions["reward"][port_name]))
        
        # 分阶段训练
        if self.args.enable_curriculum and port_name in self.gain_distributions["curriculum"]:
            applicable_gains.append(("curriculum", self.gain_distributions["curriculum"][port_name]))
        
        # 超参数优化（所有港口）
        if self.args.enable_hpo:
            applicable_gains.append(("hpo", self.gain_distributions["hpo"]["all"]))
        
        # 联邦学习（所有港口）
        if self.args.enable_fed:
            applicable_gains.append(("federated", self.gain_distributions["federated"]["all"]))
        
        return applicable_gains
    
    def mc_estimate_completion(self, port_name: str, mc_samples: int = None) -> Tuple[float, float, float]:
        """
        Monte Carlo估计改进后的完成率
        返回: (均值, 5%分位数, 95%分位数)
        """
        if mc_samples is None:
            mc_samples = self.args.mc
        
        port_status = self.port_status[port_name]
        baseline = port_status.completion_rate
        applicable_gains = self.get_applicable_gains(port_name)
        
        simulations = []
        
        for _ in range(mc_samples):
            # 为每次模拟采样增益
            gains = []
            for technique_name, gain_dist in applicable_gains:
                gain = self.sample_gain(gain_dist, port_status.n_samples)
                gains.append(gain)
            
            # 组合增益效果
            improved_rate = self.combine_completion_rates(baseline, gains)
            simulations.append(improved_rate)
        
        simulations = np.array(simulations)
        mean_rate = float(np.mean(simulations))
        p5_rate = float(np.percentile(simulations, 5))
        p95_rate = float(np.percentile(simulations, 95))
        
        return mean_rate, p5_rate, p95_rate
    
    def alpha_fair_weights(self, performance: Dict[str, float], alpha: float, 
                          temp: float = 1.5, floor: float = 0.02, 
                          uniform_mix: float = 0.1) -> Dict[str, float]:
        """
        计算α-fair权重分配（修正版 - 确保弱者权重随α上升）
        performance: 港口 -> 完成率
        alpha: 公平性参数（越大越偏向弱者）
        temp: 温度参数（>1拉平，<1拉尖）
        floor: 最小权重地板
        uniform_mix: 与均匀分布的混合比例
        """
        ports = list(performance.keys())
        eps = 1e-6
        
        # 转换为"缺口"（弱者更大）
        perf_values = np.array([performance[p] for p in ports], dtype=float)
        perf_values = np.clip(perf_values, eps, 0.999999)
        
        # 用"缺口"刻画弱者（越弱越大）
        utilities = 1.0 - perf_values
        
        self.logger.debug(f"α={alpha}: 性能值 = {perf_values}")
        self.logger.debug(f"α={alpha}: 缺口值 = {utilities}")
        
        # α 越大越偏弱者：raw ∝ utilities^alpha
        alpha = max(0.0, float(alpha))
        raw_weights = np.power(utilities + eps, alpha)
        raw_weights = np.nan_to_num(raw_weights, nan=eps, posinf=1e6, neginf=eps)
        
        self.logger.debug(f"α={alpha}: 原始权重 = {raw_weights}")
        
        # 归一化
        norm_weights = raw_weights / max(raw_weights.sum(), eps)
        
        # 温度调节（降低权重抖动）
        if temp != 1.0:
            norm_weights = np.power(norm_weights, 1.0 / temp)
            norm_weights = norm_weights / max(norm_weights.sum(), eps)
        
        # 权重地板（避免某些港口被完全忽略）
        norm_weights = np.maximum(norm_weights, floor)
        norm_weights = norm_weights / max(norm_weights.sum(), eps)
        
        # 与均匀分布混合（避免极端分配）
        if uniform_mix > 0:
            uniform_weights = np.ones_like(norm_weights) / len(norm_weights)
            norm_weights = (1 - uniform_mix) * norm_weights + uniform_mix * uniform_weights
            norm_weights = norm_weights / max(norm_weights.sum(), eps)
        
        self.logger.debug(f"α={alpha}: 最终权重 = {norm_weights}, 总和 = {np.sum(norm_weights):.6f}")
        
        # 计算公平性指标
        weight_ratio = np.max(norm_weights) / max(np.min(norm_weights), eps)
        weight_range = np.max(norm_weights) - np.min(norm_weights)
        
        self.logger.debug(f"α={alpha}: 权重比 = {weight_ratio:.2f}, 权重范围 = {weight_range:.3f}")
        
        return dict(zip(ports, norm_weights))
    
    def sanity_check(self, before: float, after: float, port_name: str):
        """安全性检查"""
        assert 0 <= before <= 1 and 0 <= after <= 1, f"{port_name} 完成率超出范围 [0,1]"
        
        # 允许最多2%的轻微回退（模拟domain shift等因素）
        min_allowed = before - 0.02
        assert after >= min_allowed, f"{port_name} 过度退化: {before:.3f} -> {after:.3f}"
        
        # Gulfport作为最佳港口，改进上限检查
        if port_name == "gulfport":
            assert after <= 0.99, f"{port_name} 改进超出合理上限"
            # 相对增长上限（防止"火箭升天"）
            relative_gain = (after - before) / before if before > 0 else 0
            assert relative_gain <= 0.05, f"{port_name} 相对增长过大: {relative_gain:.1%}"
        
        # 增益合成上限保护（避免过度叠加）
        if port_name == "gulfport":
            max_allowed = 0.99  # Gulfport特殊上限
        else:
            max_allowed = min(0.95, before + 0.6)  # 其他港口天然上限
        assert after <= max_allowed, f"{port_name} 改进超出天然上限: {after:.3f} > {max_allowed:.3f}"
    
    def run_monte_carlo_analysis(self) -> Dict:
        """运行Monte Carlo分析"""
        self.logger.info("开始Monte Carlo分析...")
        
        # 确定要分析的港口
        if self.args.ports == "all":
            target_ports = list(self.port_status.keys())
        else:
            target_ports = [p.strip() for p in self.args.ports.split(",")]
        
        mc_results = {}
        
        for port_name in target_ports:
            if port_name not in self.port_status:
                self.logger.warning(f"未知港口: {port_name}")
                continue
            
            self.logger.info(f"分析港口: {port_name}")
            
            # 当前状态
            current_rate = self.port_status[port_name].completion_rate
            
            # Monte Carlo估计
            mean_rate, p5_rate, p95_rate = self.mc_estimate_completion(port_name)
            
            # 安全性检查
            self.sanity_check(current_rate, mean_rate, port_name)
            
            # 计算改进指标
            absolute_improvement = mean_rate - current_rate
            relative_improvement = (absolute_improvement / current_rate) * 100 if current_rate > 0 else 0
            
            # 获取适用的改进技术
            applicable_gains = self.get_applicable_gains(port_name)
            techniques = [name for name, _ in applicable_gains]
            
            mc_results[port_name] = {
                'current_rate': current_rate,
                'estimated_mean': mean_rate,
                'confidence_interval': [p5_rate, p95_rate],
                'absolute_improvement': absolute_improvement,
                'relative_improvement': relative_improvement,
                'applicable_techniques': techniques,
                'n_samples': self.port_status[port_name].n_samples
            }
            
            self.logger.info(f"  当前: {current_rate:.2%}")
            self.logger.info(f"  预估: {mean_rate:.2%} [{p5_rate:.2%}, {p95_rate:.2%}]")
            self.logger.info(f"  改进: +{absolute_improvement:.2%} ({relative_improvement:.1f}%)")
            self.logger.info(f"  技术: {', '.join(techniques)}")
        
        return mc_results
    
    def analyze_alpha_fairness(self, mc_results: Dict) -> Dict:
        """分析α-fair权重分配"""
        self.logger.info("分析α-fair权重分配...")
        
        # 提取当前和改进后的完成率
        current_performance = {port: results['current_rate'] 
                             for port, results in mc_results.items()}
        improved_performance = {port: results['estimated_mean'] 
                              for port, results in mc_results.items()}
        
        self.logger.info("基础性能数据:")
        for port in current_performance.keys():
            self.logger.info(f"  {port}: 当前 {current_performance[port]:.3f} -> 改进后 {improved_performance[port]:.3f}")
        
        alpha_values = [0.0, 0.5, 1.0, 1.5, 2.0]
        fairness_analysis = {}
        
        for alpha in alpha_values:
            self.logger.info(f"\n--- α={alpha} 权重分析 ---")
            
            current_weights = self.alpha_fair_weights(current_performance, alpha)
            improved_weights = self.alpha_fair_weights(improved_performance, alpha)
            
            # 计算权重变化
            weight_changes = {}
            for port in current_performance.keys():
                change = improved_weights[port] - current_weights[port]
                weight_changes[port] = change
            
            fairness_analysis[alpha] = {
                'current_weights': current_weights,
                'improved_weights': improved_weights,
                'weight_changes': weight_changes
            }
            
            # 详细输出
            self.logger.info("权重分配 (当前 -> 改进后 | 变化):")
            for port in sorted(current_performance.keys()):
                curr_w = current_weights[port]
                impr_w = improved_weights[port]
                change = weight_changes[port]
                self.logger.info(f"  {port:15s}: {curr_w:.3f} -> {impr_w:.3f} | {change:+.3f}")
            
            # 公平性指标
            curr_min, curr_max = min(current_weights.values()), max(current_weights.values())
            impr_min, impr_max = min(improved_weights.values()), max(improved_weights.values())
            
            self.logger.info(f"权重范围: 当前 [{curr_min:.3f}, {curr_max:.3f}] -> 改进后 [{impr_min:.3f}, {impr_max:.3f}]")
            self.logger.info(f"权重比: 当前 {curr_max/max(curr_min, 1e-6):.2f} -> 改进后 {impr_max/max(impr_min, 1e-6):.2f}")
            
            # 验证权重和
            curr_sum = sum(current_weights.values())
            impr_sum = sum(improved_weights.values())
            self.logger.info(f"权重总和: 当前 {curr_sum:.6f}, 改进后 {impr_sum:.6f}")
        
        # 单调性检查
        self._check_alpha_monotonicity(fairness_analysis)
        
        return fairness_analysis
    
    def _is_increasing(self, xs, tolerance=1e-9):
        """检查序列是否单调递增（允许轻微数值抖动）"""
        return all(xs[i+1] >= xs[i] - tolerance for i in range(len(xs)-1))
    
    def _is_decreasing(self, xs, tolerance=1e-9):
        """检查序列是否单调递减（允许轻微数值抖动）"""
        return all(xs[i+1] <= xs[i] + tolerance for i in range(len(xs)-1))
    
    def _check_alpha_monotonicity(self, fairness_analysis: Dict):
        """检查α-fair权重的单调性"""
        self.logger.info("\n--- α单调性检查 ---")
        
        alpha_vals = sorted(fairness_analysis.keys())
        
        # 弱港口权重应随α上升
        weak_ports = ["new_orleans", "baton_rouge"]
        for port in weak_ports:
            if port in fairness_analysis[alpha_vals[0]]['improved_weights']:
                weights = [fairness_analysis[a]['improved_weights'][port] for a in alpha_vals]
                if self._is_increasing(weights):
                    self.logger.info(f"✓ 弱港口 {port} 权重随α单调上升")
                else:
                    self.logger.warning(f"✗ 弱港口 {port} 权重未随α单调上升: {[f'{w:.3f}' for w in weights]}")
        
        # 强港口权重应随α下降
        strong_ports = ["gulfport"]
        for port in strong_ports:
            if port in fairness_analysis[alpha_vals[0]]['improved_weights']:
                weights = [fairness_analysis[a]['improved_weights'][port] for a in alpha_vals]
                if self._is_decreasing(weights):
                    self.logger.info(f"✓ 强港口 {port} 权重随α单调下降")
                else:
                    self.logger.warning(f"✗ 强港口 {port} 权重未随α单调下降: {[f'{w:.3f}' for w in weights]}")
        
        # 公平性指标应随α改善
        weight_ranges = [fairness_analysis[a]['improved_weights'] for a in alpha_vals]
        range_values = [max(w.values()) - min(w.values()) for w in weight_ranges]
        
        if self._is_decreasing(range_values):
            self.logger.info("✓ 权重范围随α单调下降（更公平）")
        else:
            self.logger.warning(f"✗ 权重范围未随α单调下降: {[f'{r:.3f}' for r in range_values]}")
    
    def export_results(self, mc_results: Dict, fairness_analysis: Dict):
        """导出结构化结果"""
        self.logger.info("导出结构化结果...")
        
        # 准备导出数据
        export_data = {
            'metadata': {
                'timestamp': str(np.datetime64('now')),
                'seed': self.args.seed,
                'mc_samples': self.args.mc,
                'alpha': self.args.alpha,
                'enabled_techniques': {
                    'transfer': self.args.enable_transfer,
                    'reward': self.args.enable_reward,
                    'curriculum': self.args.enable_curriculum,
                    'hpo': self.args.enable_hpo,
                    'federated': self.args.enable_fed
                }
            },
            'monte_carlo_results': mc_results,
            'fairness_analysis': fairness_analysis,
            'gain_distributions': {
                technique: {
                    port: {'mean': dist.mean, 'std': dist.std}
                    for port, dist in ports.items()
                }
                for technique, ports in self.gain_distributions.items()
            }
        }
        
        # 保存JSON结果
        results_path = self.output_dir / "improvement_validation_results.json"
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"结果已保存: {results_path}")
        
        # 保存CSV摘要
        csv_path = self.output_dir / "improvement_summary.csv"
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write("港口,当前完成率,预估完成率,置信区间下限,置信区间上限,绝对改进,相对改进(%),适用技术\n")
            for port, results in mc_results.items():
                techniques = ';'.join(results['applicable_techniques'])
                f.write(f"{port},{results['current_rate']:.4f},{results['estimated_mean']:.4f},"
                       f"{results['confidence_interval'][0]:.4f},{results['confidence_interval'][1]:.4f},"
                       f"{results['absolute_improvement']:.4f},{results['relative_improvement']:.1f},"
                       f"{techniques}\n")
        
        self.logger.info(f"CSV摘要已保存: {csv_path}")
    
    def generate_visualizations(self, mc_results: Dict, fairness_analysis: Dict):
        """生成可视化图表"""
        if not self.args.visualize:
            return
        
        self.logger.info("生成可视化图表...")
        
        try:
            # 图1: 改进前后完成率对比
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            ports = list(mc_results.keys())
            current_rates = [mc_results[p]['current_rate'] for p in ports]
            improved_rates = [mc_results[p]['estimated_mean'] for p in ports]
            ci_lower = [mc_results[p]['confidence_interval'][0] for p in ports]
            ci_upper = [mc_results[p]['confidence_interval'][1] for p in ports]
            
            x = np.arange(len(ports))
            width = 0.35
            
            ax1.bar(x - width/2, current_rates, width, label='当前', alpha=0.8, color='lightcoral')
            ax1.bar(x + width/2, improved_rates, width, label='改进后', alpha=0.8, color='lightblue')
            ax1.errorbar(x + width/2, improved_rates, 
                        yerr=[np.array(improved_rates) - np.array(ci_lower),
                              np.array(ci_upper) - np.array(improved_rates)],
                        fmt='none', color='black', capsize=5)
            
            ax1.set_xlabel('港口')
            ax1.set_ylabel('完成率')
            ax1.set_title('改进前后完成率对比')
            ax1.set_xticks(x)
            ax1.set_xticklabels([p.replace('_', ' ').title() for p in ports], rotation=45)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, 1)
            
            # 图2: α-fair权重变化
            alpha_values = list(fairness_analysis.keys())
            colors = ['red', 'blue', 'green', 'orange']
            
            for i, port in enumerate(ports):
                current_weights = [fairness_analysis[alpha]['current_weights'][port] 
                                 for alpha in alpha_values]
                improved_weights = [fairness_analysis[alpha]['improved_weights'][port] 
                                  for alpha in alpha_values]
                
                color = colors[i % len(colors)]
                ax2.plot(alpha_values, current_weights, '--', color=color, 
                        label=f'{port.replace("_", " ").title()} (当前)', alpha=0.7)
                ax2.plot(alpha_values, improved_weights, '-', color=color, 
                        label=f'{port.replace("_", " ").title()} (改进)', linewidth=2)
            
            ax2.set_xlabel('α 参数')
            ax2.set_ylabel('权重')
            ax2.set_title('α-Fair权重分配变化')
            ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax2.grid(True, alpha=0.3)
            ax2.set_ylim(0, 1)
            
            plt.tight_layout()
            
            # 保存图表
            plot_path = self.output_dir / "improvement_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"图表已保存: {plot_path}")
            
            plt.close()
            
            # 图3: α敏感性分析
            fig, (ax3, ax4) = plt.subplots(1, 2, figsize=(15, 6))
            
            # 权重范围变化
            alpha_vals = list(fairness_analysis.keys())
            weight_ranges_curr = []
            weight_ranges_impr = []
            weight_ratios_curr = []
            weight_ratios_impr = []
            
            for alpha in alpha_vals:
                curr_weights = list(fairness_analysis[alpha]['current_weights'].values())
                impr_weights = list(fairness_analysis[alpha]['improved_weights'].values())
                
                weight_ranges_curr.append(max(curr_weights) - min(curr_weights))
                weight_ranges_impr.append(max(impr_weights) - min(impr_weights))
                
                weight_ratios_curr.append(max(curr_weights) / max(min(curr_weights), 1e-6))
                weight_ratios_impr.append(max(impr_weights) / max(min(impr_weights), 1e-6))
            
            ax3.plot(alpha_vals, weight_ranges_curr, 'o-', label='当前权重范围', color='red')
            ax3.plot(alpha_vals, weight_ranges_impr, 's-', label='改进后权重范围', color='blue')
            ax3.set_xlabel('α 参数')
            ax3.set_ylabel('权重范围 (max - min)')
            ax3.set_title('权重分布范围 vs α')
            ax3.legend()
            ax3.grid(True, alpha=0.3)
            
            ax4.plot(alpha_vals, weight_ratios_curr, 'o-', label='当前权重比', color='red')
            ax4.plot(alpha_vals, weight_ratios_impr, 's-', label='改进后权重比', color='blue')
            ax4.set_xlabel('α 参数')
            ax4.set_ylabel('权重比 (max / min)')
            ax4.set_title('权重不平衡度 vs α')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
            ax4.set_yscale('log')
            
            plt.tight_layout()
            
            # 保存敏感性分析图
            sensitivity_path = self.output_dir / "alpha_sensitivity_analysis.png"
            plt.savefig(sensitivity_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"敏感性分析图已保存: {sensitivity_path}")
            
            plt.close()
            
        except Exception as e:
            self.logger.warning(f"可视化生成失败: {e}")
    
    def run_full_validation(self):
        """运行完整验证"""
        self.logger.info("开始GAT-FedPPO改进效果验证")
        self.logger.info("=" * 60)
        
        # 1. Monte Carlo分析
        mc_results = self.run_monte_carlo_analysis()
        
        # 2. α-fair分析
        fairness_analysis = self.analyze_alpha_fairness(mc_results)
        
        # 3. 可视化
        self.generate_visualizations(mc_results, fairness_analysis)
        
        # 4. 导出结果
        self.export_results(mc_results, fairness_analysis)
        
        # 5. 总结报告
        self.generate_summary_report(mc_results)
        
        self.logger.info("=" * 60)
        self.logger.info("验证完成！")
        
        return {
            'monte_carlo': mc_results,
            'fairness': fairness_analysis
        }
    
    def generate_summary_report(self, mc_results: Dict):
        """生成总结报告"""
        self.logger.info("\n=== 改进效果总结报告 ===")
        
        total_improvement = 0
        port_count = 0
        
        for port, results in mc_results.items():
            improvement = results['relative_improvement']
            total_improvement += improvement
            port_count += 1
            
            self.logger.info(f"{port.upper()}:")
            self.logger.info(f"  当前: {results['current_rate']:.2%}")
            self.logger.info(f"  预估: {results['estimated_mean']:.2%}")
            self.logger.info(f"  改进: +{results['absolute_improvement']:.2%} ({improvement:.1f}%)")
            self.logger.info(f"  置信区间: [{results['confidence_interval'][0]:.2%}, {results['confidence_interval'][1]:.2%}]")
            self.logger.info(f"  技术: {', '.join(results['applicable_techniques'])}")
            self.logger.info("")
        
        avg_improvement = total_improvement / port_count if port_count > 0 else 0
        self.logger.info(f"平均相对改进: {avg_improvement:.1f}%")

def main():
    """主函数"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置日志
    log_file = Path(args.output_dir) / "improvement_validation.log"
    logger = setup_logging(str(log_file))
    
    # 创建验证台
    bench = ImprovementValidationBench(args, logger)
    
    # 运行验证
    results = bench.run_full_validation()
    
    return results

if __name__ == "__main__":
    main()