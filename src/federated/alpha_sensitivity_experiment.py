#!/usr/bin/env python3
"""
α敏感性网格实验 - 详细分析不同α值对权重分配的影响
"""

import numpy as np

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
from improvement_demo_clean import ImprovementValidationBench, parse_args, set_seed, setup_logging
from pathlib import Path
import json

def run_alpha_sensitivity_grid():
    """运行α敏感性网格实验"""
    
    # 设置参数
    args = parse_args()
    args.seed = 42
    args.mc = 200
    args.output_dir = "../../results/alpha_sensitivity"
    
    set_seed(args.seed)
    
    # 设置日志
    log_file = Path(args.output_dir) / "alpha_sensitivity.log"
    logger = setup_logging(str(log_file))
    
    # 创建验证台
    bench = ImprovementValidationBench(args, logger)
    
    # 运行基础MC分析
    logger.info("运行基础Monte Carlo分析...")
    mc_results = bench.run_monte_carlo_analysis()
    
    # α值网格
    alpha_grid = np.linspace(0.0, 3.0, 31)  # 0.0 到 3.0，步长0.1
    
    # 提取性能数据
    current_performance = {port: results['current_rate'] 
                         for port, results in mc_results.items()}
    improved_performance = {port: results['estimated_mean'] 
                          for port, results in mc_results.items()}
    
    logger.info(f"测试α值范围: {alpha_grid[0]:.1f} 到 {alpha_grid[-1]:.1f}")
    
    # 存储结果
    sensitivity_results = {
        'alpha_values': alpha_grid.tolist(),
        'current_weights': {},
        'improved_weights': {},
        'fairness_metrics': {
            'current': {'weight_range': [], 'weight_ratio': [], 'gini': []},
            'improved': {'weight_range': [], 'weight_ratio': [], 'gini': []}
        }
    }
    
    # 初始化港口权重存储
    for port in current_performance.keys():
        sensitivity_results['current_weights'][port] = []
        sensitivity_results['improved_weights'][port] = []
    
    # 遍历α值
    for alpha in alpha_grid:
        logger.info(f"测试α={alpha:.2f}")
        
        # 计算权重
        current_weights = bench.alpha_fair_weights(current_performance, alpha)
        improved_weights = bench.alpha_fair_weights(improved_performance, alpha)
        
        # 存储权重
        for port in current_performance.keys():
            sensitivity_results['current_weights'][port].append(current_weights[port])
            sensitivity_results['improved_weights'][port].append(improved_weights[port])
        
        # 计算公平性指标
        def calc_fairness_metrics(weights_dict):
            weights = list(weights_dict.values())
            weight_range = max(weights) - min(weights)
            weight_ratio = max(weights) / max(min(weights), 1e-6)
            
            # 基尼系数
            weights_sorted = sorted(weights)
            n = len(weights_sorted)
            cumsum = np.cumsum(weights_sorted)
            gini = (n + 1 - 2 * sum((n + 1 - i) * w for i, w in enumerate(weights_sorted, 1))) / (n * sum(weights_sorted))
            
            return weight_range, weight_ratio, gini
        
        curr_range, curr_ratio, curr_gini = calc_fairness_metrics(current_weights)
        impr_range, impr_ratio, impr_gini = calc_fairness_metrics(improved_weights)
        
        sensitivity_results['fairness_metrics']['current']['weight_range'].append(curr_range)
        sensitivity_results['fairness_metrics']['current']['weight_ratio'].append(curr_ratio)
        sensitivity_results['fairness_metrics']['current']['gini'].append(curr_gini)
        
        sensitivity_results['fairness_metrics']['improved']['weight_range'].append(impr_range)
        sensitivity_results['fairness_metrics']['improved']['weight_ratio'].append(impr_ratio)
        sensitivity_results['fairness_metrics']['improved']['gini'].append(impr_gini)
    
    # 保存结果
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results_path = output_dir / "alpha_sensitivity_results.json"
    with open(results_path, 'w') as f:
        json.dump(sensitivity_results, f, indent=2)
    
    logger.info(f"敏感性分析结果已保存: {results_path}")
    
    # 生成详细可视化
    generate_detailed_visualizations(sensitivity_results, output_dir, logger)
    
    # 分析关键发现
    analyze_key_findings(sensitivity_results, logger)
    
    return sensitivity_results

def generate_detailed_visualizations(results, output_dir, logger):
    """生成详细的可视化图表"""
    
    logger.info("生成详细可视化图表...")
    
    alpha_values = results['alpha_values']
    ports = list(results['current_weights'].keys())
    
    # 图1: 权重变化曲线（4个子图）
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    colors = ['red', 'blue', 'green', 'orange']
    
    for i, port in enumerate(ports):
        ax = axes[i]
        
        current_weights = results['current_weights'][port]
        improved_weights = results['improved_weights'][port]
        
        ax.plot(alpha_values, current_weights, '--', color=colors[i], 
                label='当前', linewidth=2, alpha=0.7)
        ax.plot(alpha_values, improved_weights, '-', color=colors[i], 
                label='改进后', linewidth=3)
        
        ax.set_xlabel('α 参数')
        ax.set_ylabel('权重')
        ax.set_title(f'{port.replace("_", " ").title()} 权重变化')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plot_path = output_dir / "alpha_sensitivity_by_port.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    logger.info(f"港口权重变化图已保存: {plot_path}")
    plt.close()
    
    # 图2: 公平性指标变化
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    metrics = ['weight_range', 'weight_ratio', 'gini']
    titles = ['权重范围 (max - min)', '权重比 (max / min)', '基尼系数']
    
    for i, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[i]
        
        current_values = results['fairness_metrics']['current'][metric]
        improved_values = results['fairness_metrics']['improved'][metric]
        
        ax.plot(alpha_values, current_values, 'o-', label='当前', color='red', alpha=0.7)
        ax.plot(alpha_values, improved_values, 's-', label='改进后', color='blue')
        
        ax.set_xlabel('α 参数')
        ax.set_ylabel(title)
        ax.set_title(f'{title} vs α')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if metric == 'weight_ratio':
            ax.set_yscale('log')
    
    plt.tight_layout()
    fairness_path = output_dir / "fairness_metrics_vs_alpha.png"
    plt.savefig(fairness_path, dpi=300, bbox_inches='tight')
    logger.info(f"公平性指标图已保存: {fairness_path}")
    plt.close()
    
    # 图3: 热力图 - 权重分布
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 当前权重热力图
    current_matrix = np.array([results['current_weights'][port] for port in ports])
    im1 = ax1.imshow(current_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    ax1.set_xlabel('α 参数索引')
    ax1.set_ylabel('港口')
    ax1.set_title('当前权重分布热力图')
    ax1.set_yticks(range(len(ports)))
    ax1.set_yticklabels([p.replace('_', ' ').title() for p in ports])
    
    # 添加α值标签
    alpha_ticks = range(0, len(alpha_values), 5)
    ax1.set_xticks(alpha_ticks)
    ax1.set_xticklabels([f'{alpha_values[i]:.1f}' for i in alpha_ticks])
    
    plt.colorbar(im1, ax=ax1, label='权重')
    
    # 改进后权重热力图
    improved_matrix = np.array([results['improved_weights'][port] for port in ports])
    im2 = ax2.imshow(improved_matrix, aspect='auto', cmap='viridis', interpolation='nearest')
    ax2.set_xlabel('α 参数索引')
    ax2.set_ylabel('港口')
    ax2.set_title('改进后权重分布热力图')
    ax2.set_yticks(range(len(ports)))
    ax2.set_yticklabels([p.replace('_', ' ').title() for p in ports])
    ax2.set_xticks(alpha_ticks)
    ax2.set_xticklabels([f'{alpha_values[i]:.1f}' for i in alpha_ticks])
    
    plt.colorbar(im2, ax=ax2, label='权重')
    
    plt.tight_layout()
    heatmap_path = output_dir / "weight_distribution_heatmap.png"
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    logger.info(f"权重分布热力图已保存: {heatmap_path}")
    plt.close()

def analyze_key_findings(results, logger):
    """分析关键发现"""
    
    logger.info("\n=== α敏感性分析关键发现 ===")
    
    alpha_values = results['alpha_values']
    ports = list(results['current_weights'].keys())
    
    # 找到最公平的α值（基于基尼系数）
    improved_gini = results['fairness_metrics']['improved']['gini']
    min_gini_idx = np.argmin(improved_gini)
    optimal_alpha = alpha_values[min_gini_idx]
    
    logger.info(f"最公平的α值: {optimal_alpha:.2f} (基尼系数: {improved_gini[min_gini_idx]:.3f})")
    
    # 分析权重单调性
    for port in ports:
        weights = results['improved_weights'][port]
        
        # 计算权重变化趋势
        diff = np.diff(weights)
        increasing = np.sum(diff > 0.001)
        decreasing = np.sum(diff < -0.001)
        
        if increasing > decreasing:
            trend = "递增"
        elif decreasing > increasing:
            trend = "递减"
        else:
            trend = "稳定"
        
        logger.info(f"{port}: 权重随α变化趋势为{trend}")
    
    # 找到权重交叉点（降噪版）
    logger.info("\n权重交叉分析:")
    
    def log_first_crossing(alpha_vals, weights1, weights2, port1, port2, logger):
        """只记录第一次真正的符号反转交叉"""
        diff = weights1 - weights2
        
        # 跳过在α=0处就相等的情况（均匀分布噪音）
        if abs(diff[0]) < 0.01:
            return
        
        prev_sign = np.sign(diff[0]) if abs(diff[0]) > 1e-9 else 1
        
        for i in range(1, len(diff)):
            cur_sign = np.sign(diff[i]) if abs(diff[i]) > 1e-9 else prev_sign
            
            # 检查符号反转
            if prev_sign * cur_sign < 0:
                # 线性插值估算精确交叉点
                a0, a1 = alpha_vals[i-1], alpha_vals[i]
                y0, y1 = diff[i-1], diff[i]
                
                if abs(y1 - y0) > 1e-10:
                    a_cross = a0 - y0 * (a1 - a0) / (y1 - y0)
                    logger.info(f"{port1} 和 {port2} 在 α≈{a_cross:.2f} 处权重交叉")
                break
            prev_sign = cur_sign
    
    for i, port1 in enumerate(ports):
        for port2 in ports[i+1:]:
            weights1 = np.array(results['improved_weights'][port1])
            weights2 = np.array(results['improved_weights'][port2])
            
            log_first_crossing(alpha_values, weights1, weights2, port1, port2, logger)
    
    # 推荐α值范围
    logger.info(f"\n推荐α值范围:")
    logger.info(f"  - 最大最小公平: α = 0.0")
    logger.info(f"  - 平衡公平性: α = {optimal_alpha:.1f}")
    logger.info(f"  - 比例公平: α = 1.0")
    logger.info(f"  - 效率优先: α = 2.0+")

if __name__ == "__main__":
    results = run_alpha_sensitivity_grid()