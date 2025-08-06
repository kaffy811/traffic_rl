#!/usr/bin/env python3
"""
多端口CityFlow联邦学习实验运行器
支持真实的CityFlow仿真环境的多端口联邦学习
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.federated.multi_port_cityflow_system import MultiPortFederatedSystem
from src.federated.visualization_generator import VisualizationGenerator

def run_multi_port_experiment(num_ports: int = 4, 
                              topology_size: str = "3x3",
                              num_rounds: int = 10,
                              episodes_per_round: int = 5):
    """运行多端口联邦学习实验"""
    
    print("🚀 启动多端口CityFlow联邦学习实验")
    print("=" * 80)
    print(f"配置信息:")
    print(f"  端口数量: {num_ports}")
    print(f"  拓扑大小: {topology_size}")
    print(f"  联邦轮次: {num_rounds}")
    print(f"  每轮episodes: {episodes_per_round}")
    print("=" * 80)
    
    # 创建多端口联邦学习系统
    system = MultiPortFederatedSystem(
        num_ports=num_ports,
        topology_size=topology_size
    )
    
    try:
        # 运行联邦学习实验
        results = system.run_federated_experiment(
            num_rounds=num_rounds,
            episodes_per_round=episodes_per_round
        )
        
        print(f"\n✅ 实验完成!")
        return True
        
    except KeyboardInterrupt:
        print(f"\n⚠️ 实验被用户中断")
        return False
    except Exception as e:
        print(f"\n❌ 实验失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        system.close()

def generate_visualizations():
    """生成实验结果的可视化"""
    print("\n🎨 生成实验结果可视化...")
    
    generator = VisualizationGenerator()
    
    # 自动加载最新数据并生成可视化
    results = generator.run_complete_analysis()
    
    if results:
        print("✅ 可视化生成完成!")
        return True
    else:
        print("❌ 可视化生成失败")
        return False

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="多端口CityFlow联邦学习实验系统")
    
    # 实验配置参数
    parser.add_argument("--ports", type=int, default=4, choices=[2, 3, 4], 
                       help="端口数量 (默认: 4)")
    parser.add_argument("--topology", type=str, default="3x3", 
                       choices=["3x3", "4x4", "5x5", "6x6"],
                       help="拓扑大小 (默认: 3x3)")
    parser.add_argument("--rounds", type=int, default=10, 
                       help="联邦学习轮次 (默认: 10)")
    parser.add_argument("--episodes", type=int, default=5,
                       help="每轮训练episodes (默认: 5)")
    
    # 运行模式
    parser.add_argument("--experiment", action="store_true",
                       help="运行联邦学习实验")
    parser.add_argument("--visualize", action="store_true", 
                       help="生成可视化图表")
    parser.add_argument("--complete", action="store_true",
                       help="运行完整流程 (实验+可视化)")
    
    # 其他选项
    parser.add_argument("--check-cityflow", action="store_true",
                       help="检查CityFlow环境")
    
    args = parser.parse_args()
    
    if args.check_cityflow:
        # 检查CityFlow环境
        try:
            import cityflow
            print("✅ CityFlow 可用")
            
            # 检查拓扑文件
            topology_dir = project_root / "topologies"
            config_file = topology_dir / f"maritime_{args.topology}_config.json"
            if config_file.exists():
                print(f"✅ 拓扑配置文件存在: {config_file}")
            else:
                print(f"❌ 拓扑配置文件不存在: {config_file}")
                
        except ImportError:
            print("❌ CityFlow 不可用，将使用模拟环境")
            
        return
    
    if args.complete:
        # 运行完整流程
        print("🚀 运行完整多端口联邦学习流程")
        
        # 1. 运行实验
        success = run_multi_port_experiment(
            num_ports=args.ports,
            topology_size=args.topology, 
            num_rounds=args.rounds,
            episodes_per_round=args.episodes
        )
        
        if success:
            # 2. 生成可视化
            generate_visualizations()
        else:
            print("❌ 实验失败，跳过可视化生成")
            
    elif args.experiment:
        # 仅运行实验
        run_multi_port_experiment(
            num_ports=args.ports,
            topology_size=args.topology,
            num_rounds=args.rounds,
            episodes_per_round=args.episodes
        )
        
    elif args.visualize:
        # 仅生成可视化
        generate_visualizations()
        
    else:
        # 显示帮助信息
        print("""
🚀 多端口CityFlow联邦学习实验系统

这是一个基于真实CityFlow仿真的多端口海事交通联邦学习系统。
每个端口运行独立的CityFlow环境，通过联邦学习协同优化。

架构特点:
📍 多个独立端口 (New Orleans, South Louisiana, Baton Rouge, Gulfport)
🌊 每个端口运行独立的CityFlow海事交通仿真
🧠 GAT-PPO智能体进行本地决策
🤝 联邦学习实现端口间知识共享
📊 实时数据收集和性能监控

使用示例:

1. 检查环境:
   python run_multi_port_experiment.py --check-cityflow

2. 运行完整四港口实验 (推荐):
   python run_multi_port_experiment.py --complete --ports 4 --rounds 10

3. 仅运行四港口实验:
   python run_multi_port_experiment.py --experiment --ports 4 --topology 3x3

4. 仅生成可视化:
   python run_multi_port_experiment.py --visualize

5. 大规模实验:
   python run_multi_port_experiment.py --complete --ports 4 --topology 4x4 --rounds 20

配置说明:
--ports: 端口数量 (2-4)
--topology: 拓扑大小 (3x3, 4x4, 5x5, 6x6)
--rounds: 联邦学习轮次
--episodes: 每轮训练episodes数

输出位置:
📁 实验数据: src/federated/experiment_data/
📊 可视化结果: src/federated/visualization_results/
        """)
        parser.print_help()

if __name__ == "__main__":
    main()