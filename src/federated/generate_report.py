#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
报告生成器 - 汇总一致性测试结果生成可视化报告
支持生成HTML看板、Markdown报告和CSV数据表
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import pandas as pd

def load_test_results(results_dir: str) -> List[Dict]:
    """加载所有测试结果"""
    results_path = Path(results_dir)
    test_results = []
    
    # 查找所有一致性测试结果文件
    for json_file in results_path.glob("consistency_*.json"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                data['file_name'] = json_file.name
                data['file_path'] = str(json_file)
                test_results.append(data)
        except Exception as e:
            print(f"警告: 无法加载 {json_file}: {e}")
    
    return test_results

def parse_test_results(test_results: List[Dict]) -> pd.DataFrame:
    """解析测试结果为DataFrame"""
    rows = []
    
    for result in test_results:
        port_name = result.get('port', result.get('port_name', 'unknown'))
        timestamp = result.get('timestamp', 'unknown')
        
        # 解析阶段结果 - 支持两种格式
        stages_data = result.get('stages', result.get('stage_results', []))
        
        for stage_result in stages_data:
            row = {
                'port': port_name,
                'stage': stage_result.get('stage', 'unknown'),
                'win_rate': stage_result.get('win_rate', 0.0),
                'threshold': stage_result.get('threshold', 0.0),
                'pass': stage_result.get('pass', False),
                'timestamp': timestamp,
                'file_name': result.get('file_name', ''),
                'wilson_lb': stage_result.get('wilson_lower_bound', stage_result.get('wilson_lb', 0.0)),
                'std_dev': stage_result.get('std_dev', stage_result.get('std', 0.0)),
                'stable': stage_result.get('stable', True),  # 默认为稳定
                'samples': stage_result.get('samples', stage_result.get('n_samples', 100))
            }
            
            # 计算余量
            row['margin'] = row['win_rate'] - row['threshold']
            row['wilson_margin'] = row['wilson_lb'] - row['threshold']
            
            rows.append(row)
    
    return pd.DataFrame(rows)

def generate_summary_stats(df: pd.DataFrame) -> Dict:
    """生成汇总统计"""
    if df.empty:
        return {}
    
    summary = {
        'total_tests': len(df),
        'total_ports': df['port'].nunique(),
        'total_stages': len(df),
        'pass_rate': df['pass'].mean() * 100,
        'avg_win_rate': df['win_rate'].mean() * 100,
        'avg_threshold': df['threshold'].mean() * 100,
        'avg_margin': df['margin'].mean() * 100,
        'stable_rate': df['stable'].mean() * 100 if 'stable' in df.columns else 0,
    }
    
    # 按港口统计
    port_stats = df.groupby('port').agg({
        'pass': ['count', 'sum', 'mean'],
        'win_rate': 'mean',
        'threshold': 'mean',
        'margin': 'mean'
    }).round(3)
    
    # 转换为可序列化的格式
    port_stats_dict = {}
    for port in port_stats.index:
        port_stats_dict[port] = {
            'total_stages': int(port_stats.loc[port, ('pass', 'count')]),
            'passed_stages': int(port_stats.loc[port, ('pass', 'sum')]),
            'pass_rate': float(port_stats.loc[port, ('pass', 'mean')]),
            'avg_win_rate': float(port_stats.loc[port, 'win_rate']),
            'avg_threshold': float(port_stats.loc[port, 'threshold']),
            'avg_margin': float(port_stats.loc[port, 'margin'])
        }
    
    summary['port_stats'] = port_stats_dict
    
    # 风险阶段（余量最小的）
    risk_stages = df.nsmallest(5, 'margin')[['port', 'stage', 'win_rate', 'threshold', 'margin']]
    summary['risk_stages'] = risk_stages.to_dict('records')
    
    # 最佳阶段（余量最大的）
    best_stages = df.nlargest(5, 'margin')[['port', 'stage', 'win_rate', 'threshold', 'margin']]
    summary['best_stages'] = best_stages.to_dict('records')
    
    return summary

def generate_html_report(df: pd.DataFrame, summary: Dict, output_path: str):
    """生成HTML报告"""
    html_template = """
<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GAT-FedPPO 一致性测试报告</title>
    <style>
        body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 0; padding: 20px; background-color: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
        h1 {{ color: #2c3e50; text-align: center; margin-bottom: 30px; }}
        h2 {{ color: #34495e; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
        .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 20px; margin-bottom: 30px; }}
        .summary-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 8px; text-align: center; }}
        .summary-card h3 {{ margin: 0 0 10px 0; font-size: 14px; opacity: 0.9; }}
        .summary-card .value {{ font-size: 24px; font-weight: bold; }}
        table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
        th {{ background-color: #f8f9fa; font-weight: 600; }}
        .pass {{ color: #27ae60; font-weight: bold; }}
        .fail {{ color: #e74c3c; font-weight: bold; }}
        .risk {{ background-color: #fff3cd; }}
        .good {{ background-color: #d4edda; }}
        .status-badge {{ padding: 4px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }}
        .status-pass {{ background-color: #d4edda; color: #155724; }}
        .status-fail {{ background-color: #f8d7da; color: #721c24; }}
        .progress-bar {{ width: 100%; height: 20px; background-color: #e9ecef; border-radius: 10px; overflow: hidden; }}
        .progress-fill {{ height: 100%; background: linear-gradient(90deg, #28a745, #20c997); transition: width 0.3s ease; }}
        .timestamp {{ color: #6c757d; font-size: 12px; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🎯 GAT-FedPPO 一致性测试报告</h1>
        
        <div class="summary-grid">
            <div class="summary-card">
                <h3>总通过率</h3>
                <div class="value">{pass_rate:.1f}%</div>
            </div>
            <div class="summary-card">
                <h3>测试港口数</h3>
                <div class="value">{total_ports}</div>
            </div>
            <div class="summary-card">
                <h3>测试阶段数</h3>
                <div class="value">{total_stages}</div>
            </div>
            <div class="summary-card">
                <h3>平均胜率</h3>
                <div class="value">{avg_win_rate:.1f}%</div>
            </div>
            <div class="summary-card">
                <h3>平均阈值</h3>
                <div class="value">{avg_threshold:.1f}%</div>
            </div>
            <div class="summary-card">
                <h3>稳定率</h3>
                <div class="value">{stable_rate:.1f}%</div>
            </div>
        </div>

        <h2>📊 详细测试结果</h2>
        <table>
            <thead>
                <tr>
                    <th>港口</th>
                    <th>阶段</th>
                    <th>胜率</th>
                    <th>阈值</th>
                    <th>余量</th>
                    <th>Wilson下界</th>
                    <th>状态</th>
                    <th>稳定性</th>
                </tr>
            </thead>
            <tbody>
                {table_rows}
            </tbody>
        </table>

        <h2>⚠️ 风险阶段 (余量最小)</h2>
        <table>
            <thead>
                <tr><th>港口</th><th>阶段</th><th>胜率</th><th>阈值</th><th>余量</th></tr>
            </thead>
            <tbody>
                {risk_table}
            </tbody>
        </table>

        <h2>🏆 最佳阶段 (余量最大)</h2>
        <table>
            <thead>
                <tr><th>港口</th><th>阶段</th><th>胜率</th><th>阈值</th><th>余量</th></tr>
            </thead>
            <tbody>
                {best_table}
            </tbody>
        </table>

        <div class="timestamp">
            报告生成时间: {timestamp}
        </div>
    </div>
</body>
</html>
"""
    
    # 生成表格行
    table_rows = ""
    for _, row in df.iterrows():
        status_class = "status-pass" if row['pass'] else "status-fail"
        status_text = "✅ 通过" if row['pass'] else "❌ 失败"
        
        row_class = ""
        if row['margin'] < 0.02:  # 余量小于2%
            row_class = "risk"
        elif row['margin'] > 0.10:  # 余量大于10%
            row_class = "good"
        
        stable_text = "🟢 稳定" if row.get('stable', False) else "🟡 不稳定"
        
        table_rows += f"""
        <tr class="{row_class}">
            <td>{row['port']}</td>
            <td>{row['stage']}</td>
            <td>{row['win_rate']*100:.1f}%</td>
            <td>{row['threshold']*100:.1f}%</td>
            <td>{row['margin']*100:+.1f}%</td>
            <td>{row.get('wilson_lb', 0)*100:.1f}%</td>
            <td><span class="status-badge {status_class}">{status_text}</span></td>
            <td>{stable_text}</td>
        </tr>
        """
    
    # 生成风险表格
    risk_table = ""
    for stage in summary.get('risk_stages', []):
        risk_table += f"""
        <tr class="risk">
            <td>{stage['port']}</td>
            <td>{stage['stage']}</td>
            <td>{stage['win_rate']*100:.1f}%</td>
            <td>{stage['threshold']*100:.1f}%</td>
            <td>{stage['margin']*100:+.1f}%</td>
        </tr>
        """
    
    # 生成最佳表格
    best_table = ""
    for stage in summary.get('best_stages', []):
        best_table += f"""
        <tr class="good">
            <td>{stage['port']}</td>
            <td>{stage['stage']}</td>
            <td>{stage['win_rate']*100:.1f}%</td>
            <td>{stage['threshold']*100:.1f}%</td>
            <td>{stage['margin']*100:+.1f}%</td>
        </tr>
        """
    
    # 填充模板
    html_content = html_template.format(
        pass_rate=summary.get('pass_rate', 0),
        total_ports=summary.get('total_ports', 0),
        total_stages=summary.get('total_stages', 0),
        avg_win_rate=summary.get('avg_win_rate', 0),
        avg_threshold=summary.get('avg_threshold', 0),
        stable_rate=summary.get('stable_rate', 0),
        table_rows=table_rows,
        risk_table=risk_table,
        best_table=best_table,
        timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    )
    
    # 保存HTML文件
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

def generate_markdown_report(df: pd.DataFrame, summary: Dict, output_path: str):
    """生成Markdown报告"""
    md_content = f"""# GAT-FedPPO 一致性测试报告

## 📊 测试摘要

- **总通过率**: {summary.get('pass_rate', 0):.1f}%
- **测试港口数**: {summary.get('total_ports', 0)}
- **测试阶段数**: {summary.get('total_stages', 0)}
- **平均胜率**: {summary.get('avg_win_rate', 0):.1f}%
- **平均阈值**: {summary.get('avg_threshold', 0):.1f}%
- **稳定率**: {summary.get('stable_rate', 0):.1f}%

## 📋 详细测试结果

| 港口 | 阶段 | 胜率 | 阈值 | 余量 | Wilson下界 | 状态 | 稳定性 |
|------|------|------|------|------|------------|------|--------|
"""
    
    for _, row in df.iterrows():
        status = "✅ 通过" if row['pass'] else "❌ 失败"
        stable = "🟢 稳定" if row.get('stable', False) else "🟡 不稳定"
        
        md_content += f"| {row['port']} | {row['stage']} | {row['win_rate']*100:.1f}% | {row['threshold']*100:.1f}% | {row['margin']*100:+.1f}% | {row.get('wilson_lb', 0)*100:.1f}% | {status} | {stable} |\n"
    
    md_content += f"""
## ⚠️ 风险阶段 (余量最小)

| 港口 | 阶段 | 胜率 | 阈值 | 余量 |
|------|------|------|------|------|
"""
    
    for stage in summary.get('risk_stages', []):
        md_content += f"| {stage['port']} | {stage['stage']} | {stage['win_rate']*100:.1f}% | {stage['threshold']*100:.1f}% | {stage['margin']*100:+.1f}% |\n"
    
    md_content += f"""
## 🏆 最佳阶段 (余量最大)

| 港口 | 阶段 | 胜率 | 阈值 | 余量 |
|------|------|------|------|------|
"""
    
    for stage in summary.get('best_stages', []):
        md_content += f"| {stage['port']} | {stage['stage']} | {stage['win_rate']*100:.1f}% | {stage['threshold']*100:.1f}% | {stage['margin']*100:+.1f}% |\n"
    
    md_content += f"""
---
*报告生成时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}*
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(md_content)

def main():
    parser = argparse.ArgumentParser(description="生成一致性测试报告")
    parser.add_argument("--results-dir", default="../../models/releases/2025-08-07",
                       help="测试结果目录")
    parser.add_argument("--output-dir", default="../../reports",
                       help="报告输出目录")
    parser.add_argument("--formats", nargs="+", 
                       choices=["html", "markdown", "csv"],
                       default=["html", "markdown", "csv"],
                       help="报告格式")
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"🔍 加载测试结果: {args.results_dir}")
    test_results = load_test_results(args.results_dir)
    
    if not test_results:
        print("❌ 未找到测试结果文件")
        sys.exit(1)
    
    print(f"📊 解析 {len(test_results)} 个测试结果文件")
    df = parse_test_results(test_results)
    
    if df.empty:
        print("❌ 无有效测试数据")
        sys.exit(1)
    
    print(f"📈 生成汇总统计 ({len(df)} 条记录)")
    summary = generate_summary_stats(df)
    
    # 生成不同格式的报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if "html" in args.formats:
        html_path = output_dir / f"consistency_report_{timestamp}.html"
        generate_html_report(df, summary, str(html_path))
        print(f"✅ HTML报告: {html_path}")
    
    if "markdown" in args.formats:
        md_path = output_dir / f"consistency_report_{timestamp}.md"
        generate_markdown_report(df, summary, str(md_path))
        print(f"✅ Markdown报告: {md_path}")
    
    if "csv" in args.formats:
        csv_path = output_dir / f"consistency_data_{timestamp}.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8-sig')
        print(f"✅ CSV数据: {csv_path}")
    
    # 保存汇总统计
    summary_path = output_dir / f"summary_{timestamp}.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"✅ 汇总统计: {summary_path}")
    
    print("\n" + "="*50)
    print("📋 报告摘要")
    print("="*50)
    print(f"总通过率: {summary.get('pass_rate', 0):.1f}%")
    print(f"测试港口: {summary.get('total_ports', 0)}")
    print(f"测试阶段: {summary.get('total_stages', 0)}")
    print(f"平均胜率: {summary.get('avg_win_rate', 0):.1f}%")
    print(f"稳定率: {summary.get('stable_rate', 0):.1f}%")
    print("="*50)

if __name__ == "__main__":
    main()