#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CI检查脚本 - 用于快速验证系统状态
简化版一致性测试，适用于CI/CD流水线
"""
import os
import sys
import json
import subprocess
import argparse
from pathlib import Path

def run_consistency_test(samples=100, timeout=1800):
    """运行一致性测试"""
    print(f"🔍 运行一致性测试 (samples={samples}, timeout={timeout}s)")
    
    try:
        # 确保在正确的目录运行
        script_dir = Path(__file__).parent
        federated_dir = script_dir / "src" / "federated"
        
        result = subprocess.run([
            sys.executable, "consistency_test_fixed.py", 
            "--all", "--samples", str(samples)
        ], 
        capture_output=True, 
        text=True, 
        timeout=timeout,
        cwd=federated_dir
        )
        
        return result.returncode == 0, result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return False, "", "测试超时"
    except Exception as e:
        return False, "", str(e)

def parse_test_results(stdout):
    """解析测试结果"""
    results = {
        "total_ports": 0,
        "passed_ports": 0,
        "success_rate": "0.0%",
        "port_details": {}
    }
    
    lines = stdout.split('\n')
    for line in lines:
        if "总港口数:" in line:
            results["total_ports"] = int(line.split(":")[-1].strip())
        elif "成功港口数:" in line:
            results["passed_ports"] = int(line.split(":")[-1].strip())
        elif "成功率:" in line:
            results["success_rate"] = line.split(":")[-1].strip()
        elif "港口" in line and ("✅ 通过" in line or "❌ 失败" in line):
            port_name = line.split()[1].replace(":", "")
            status = "✅ 通过" in line
            results["port_details"][port_name] = status
    
    return results

def generate_ci_report(results, success, stdout, stderr):
    """生成CI报告"""
    report = {
        "timestamp": subprocess.check_output(["date", "+%Y-%m-%d %H:%M:%S"]).decode().strip(),
        "success": success,
        "results": results,
        "logs": {
            "stdout": stdout,
            "stderr": stderr
        }
    }
    
    # 保存报告
    report_file = Path("ci_report.json")
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    
    return report

def print_summary(results, success):
    """打印测试摘要"""
    print("\n" + "="*50)
    print("🔍 CI检查结果摘要")
    print("="*50)
    
    status_emoji = "✅" if success else "❌"
    status_text = "通过" if success else "失败"
    
    print(f"状态: {status_emoji} {status_text}")
    print(f"成功率: {results['success_rate']}")
    print(f"港口状态: {results['passed_ports']}/{results['total_ports']} 通过")
    
    if results["port_details"]:
        print("\n港口详情:")
        for port, status in results["port_details"].items():
            emoji = "✅" if status else "❌"
            print(f"  {emoji} {port}")
    
    print("="*50)

def main():
    parser = argparse.ArgumentParser(description="CI一致性检查")
    parser.add_argument("--samples", type=int, default=100, help="测试样本数")
    parser.add_argument("--timeout", type=int, default=1800, help="超时时间(秒)")
    parser.add_argument("--fail-fast", action="store_true", help="遇到失败立即退出")
    
    args = parser.parse_args()
    
    print("🚀 开始CI一致性检查...")
    print(f"参数: samples={args.samples}, timeout={args.timeout}")
    
    # 检查必要文件
    script_dir = Path(__file__).parent
    federated_dir = script_dir / "src" / "federated"
    
    required_files = [
        "consistency_test_fixed.py",
        "curriculum_trainer.py"
    ]
    
    for file in required_files:
        file_path = federated_dir / file
        if not file_path.exists():
            print(f"❌ 缺少必要文件: {file_path}")
            sys.exit(1)
    
    # 运行测试
    success, stdout, stderr = run_consistency_test(args.samples, args.timeout)
    
    # 解析结果
    results = parse_test_results(stdout)
    
    # 生成报告
    report = generate_ci_report(results, success, stdout, stderr)
    
    # 打印摘要
    print_summary(results, success)
    
    # 输出详细日志（如果失败）
    if not success:
        print("\n❌ 测试失败详情:")
        if stderr:
            print("STDERR:")
            print(stderr)
        if args.fail_fast:
            print("\n💥 fail-fast模式，立即退出")
            sys.exit(1)
    
    # 设置退出码
    exit_code = 0 if success else 1
    print(f"\n🏁 CI检查完成，退出码: {exit_code}")
    sys.exit(exit_code)

if __name__ == "__main__":
    main()