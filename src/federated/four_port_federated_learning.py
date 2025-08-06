#!/usr/bin/env python3
"""
四港口联邦学习系统
New Orleans、South Louisiana、Baton Rouge、Gulfport 四个港口互相学习的系统
"""

import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.federated.multi_port_cityflow_system import MultiPortFederatedSystem
from src.federated.visualization_generator import VisualizationGenerator

def run_four_port_federated_learning():
    """运行四港口联邦学习实验"""
    
    print("🚀 四港口联邦学习系统启动")
    print("=" * 80)
    print("🏭 参与港口:")
    print("   1️⃣ New Orleans Port (新奥尔良港)")
    print("   2️⃣ South Louisiana Port (南路易斯安那港)")
    print("   3️⃣ Baton Rouge Port (巴吞鲁日港)")
    print("   4️⃣ Gulfport (格尔夫波特港)")
    print()
    print("🤝 联邦学习模式:")
    print("   ✅ 每个港口运行独立的CityFlow仿真环境")
    print("   ✅ 每个港口使用GAT-PPO智能体进行本地决策")
    print("   ✅ 港口间通过联邦学习共享知识，不共享原始数据")
    print("   ✅ α-Fair机制确保所有港口公平受益")
    print("=" * 80)
    
    # 创建四港口联邦学习系统
    system = MultiPortFederatedSystem(num_ports=4, topology_size="3x3")
    
    try:
        # 运行联邦学习实验
        print("\n🔄 开始四港口联邦学习训练...")
        results = system.run_federated_experiment(
            num_rounds=10,           # 10轮联邦学习
            episodes_per_round=5     # 每轮每个港口训练5个episodes
        )
        
        print("\n✅ 四港口联邦学习实验完成!")
        print("📊 实验结果:")
        
        # 显示每个港口的学习效果
        for round_idx, round_result in enumerate(results):
            if isinstance(round_result, dict) and 'round' in round_result:
                print(f"\n   轮次 {round_result['round']}:")
                for port_id, port_result in round_result.items():
                    if isinstance(port_result, dict) and 'port_name' in port_result:
                        avg_reward = port_result.get('avg_episode_reward', 0)
                        episodes = port_result.get('episodes_trained', 0)
                        print(f"     🏭 {port_result['port_name']}: "
                              f"平均奖励 {avg_reward:.2f} ({episodes} episodes)")
        
        return True
        
    except Exception as e:
        print(f"❌ 四港口联邦学习实验失败: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        system.close()

def generate_four_port_analysis():
    """生成四港口联邦学习分析报告"""
    print("\n🎨 生成四港口联邦学习分析报告...")
    
    generator = VisualizationGenerator()
    results = generator.run_complete_analysis()
    
    if results:
        print("✅ 四港口分析报告生成完成!")
        print("\n📊 报告内容:")
        print("   📈 性能演进分析 - 展示四港口学习过程")
        print("   🤝 联邦学习效果 - 港口间知识共享效果")
        print("   ⚖️ 公平性分析 - α-Fair机制效果")
        print("   🎯 收敛分析 - 联邦模型收敛情况")
        print("   📋 性能对比表 - 四港口详细对比")
        print("   🔍 消融实验 - 联邦学习 vs 独立学习")
        return True
    else:
        print("❌ 分析报告生成失败")
        return False

def main():
    """主函数"""
    print("🌊 四港口海事交通联邦学习系统")
    print("基于CityFlow仿真的GAT-FedPPO框架")
    print()
    
    # 1. 运行四港口联邦学习实验
    success = run_four_port_federated_learning()
    
    if success:
        # 2. 生成分析报告
        generate_four_port_analysis()
        
        print("\n🎉 四港口联邦学习完整流程执行完成!")
        print("📁 查看结果:")
        print("   📊 实验数据: src/federated/experiment_data/")
        print("   📈 可视化结果: src/federated/visualization_results/")
        
    else:
        print("\n❌ 四港口联邦学习实验失败")

if __name__ == "__main__":
    main()