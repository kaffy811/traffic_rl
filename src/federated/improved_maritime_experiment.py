"""
改进版海事GAT-FedPPO实验系统
结合领域知识、改进的状态表示、奖励函数和详细调试
"""

import os
import json
import torch
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path

from maritime_domain_knowledge import PORT_SPECIFICATIONS
from debug_enhanced_trainer import DebugEnhancedMaritimeAgent
from improved_ais_preprocessor import ImprovedAISPreprocessor

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ImprovedMaritimeExperiment:
    """改进版海事实验系统"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.ports = config.get('ports', ['new_orleans', 'south_louisiana', 'baton_rouge', 'gulfport'])
        self.experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 实验目录
        self.experiment_dir = f"./improved_experiments/{self.experiment_id}"
        os.makedirs(self.experiment_dir, exist_ok=True)
        
        # 初始化组件
        self.agents = {}
        self.preprocessors = {}
        self.training_data = {}
        self.test_data = {}
        
        # 实验跟踪
        self.experiment_log = {
            'experiment_id': self.experiment_id,
            'config': config,
            'start_time': datetime.now().isoformat(),
            'ports': self.ports,
            'phases': {}
        }
        
        logger.info(f"初始化改进版实验系统 - ID: {self.experiment_id}")
    
    def run_complete_experiment(self) -> Dict:
        """运行完整的改进版实验"""
        try:
            # 阶段1: 数据预处理
            logger.info("=" * 60)
            logger.info("阶段1: 高级AIS数据预处理")
            logger.info("=" * 60)
            preprocessing_results = self._run_data_preprocessing()
            self.experiment_log['phases']['preprocessing'] = preprocessing_results
            
            # 阶段2: 初始化智能体
            logger.info("=" * 60)
            logger.info("阶段2: 初始化增强版智能体")
            logger.info("=" * 60)
            agent_init_results = self._initialize_enhanced_agents()
            self.experiment_log['phases']['agent_initialization'] = agent_init_results
            
            # 阶段3: 训练阶段
            logger.info("=" * 60)
            logger.info("阶段3: 联邦学习训练")
            logger.info("=" * 60)
            training_results = self._run_federated_training()
            self.experiment_log['phases']['training'] = training_results
            
            # 阶段4: 优化测试
            logger.info("=" * 60)
            logger.info("阶段4: 模型优化测试")
            logger.info("=" * 60)
            optimization_results = self._run_optimization_test()
            self.experiment_log['phases']['optimization'] = optimization_results
            
            # 阶段5: 结果分析
            logger.info("=" * 60)
            logger.info("阶段5: 结果分析和报告")
            logger.info("=" * 60)
            analysis_results = self._analyze_results()
            self.experiment_log['phases']['analysis'] = analysis_results
            
            # 完成实验
            self.experiment_log['end_time'] = datetime.now().isoformat()
            self.experiment_log['success'] = True
            
            # 保存实验日志
            self._save_experiment_log()
            
            return self.experiment_log
            
        except Exception as e:
            logger.error(f"实验失败: {e}")
            self.experiment_log['error'] = str(e)
            self.experiment_log['success'] = False
            self._save_experiment_log()
            raise
    
    def _run_data_preprocessing(self) -> Dict:
        """运行高级数据预处理"""
        preprocessing_results = {}
        
        for port in self.ports:
            logger.info(f"预处理 {port} 数据...")
            
            try:
                # 初始化预处理器
                preprocessor = ImprovedAISPreprocessor(port)
                self.preprocessors[port] = preprocessor
                
                # 查找原始数据
                raw_data_path = self._find_raw_data_path(port)
                if not raw_data_path:
                    logger.warning(f"未找到 {port} 的原始数据，使用模拟数据")
                    raw_data_path = self._generate_mock_data(port)
                
                # 执行预处理
                output_dir = os.path.join(self.experiment_dir, f"preprocessed_{port}")
                result = preprocessor.preprocess_ais_data(raw_data_path, output_dir)
                
                # 加载训练和测试数据
                self._load_training_test_data(port, output_dir)
                
                preprocessing_results[port] = result
                logger.info(f"{port} 预处理完成: {result['processed_records']} 条记录")
                
            except Exception as e:
                logger.error(f"{port} 预处理失败: {e}")
                preprocessing_results[port] = {'error': str(e)}
        
        return preprocessing_results
    
    def _find_raw_data_path(self, port: str) -> Optional[str]:
        """查找原始数据路径"""
        possible_paths = [
            f"./data/processed/{port}_week1",
            f"./data/{port}",
            f"./data/raw/{port}",
            f"./data/processed/{port}_week1/processed_data.csv"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        return None
    
    def _generate_mock_data(self, port: str) -> str:
        """生成模拟数据用于测试"""
        logger.info(f"为 {port} 生成模拟数据...")
        
        port_spec = PORT_SPECIFICATIONS[port]
        mock_data = []
        
        # 生成7天的模拟数据
        start_time = datetime.now() - timedelta(days=7)
        
        for day in range(7):
            for hour in range(24):
                for vessel_id in range(5):  # 每小时5艘船
                    timestamp = start_time + timedelta(days=day, hours=hour, minutes=vessel_id*10)
                    
                    # 模拟船舶轨迹
                    base_lat, base_lon = port_spec.lat, port_spec.lon
                    lat = base_lat + np.random.normal(0, 0.03)
                    lon = base_lon + np.random.normal(0, 0.03)
                    
                    record = {
                        'mmsi': int(1000 + vessel_id),
                        'timestamp': timestamp.isoformat(),
                        'lat': float(lat),
                        'lon': float(lon),
                        'sog': float(np.random.uniform(0, 20)),  # 增加速度范围以触发安全检查
                        'cog': float(np.random.uniform(0, 360)),
                        'heading': float(np.random.uniform(0, 360)),
                        'length': float(np.random.uniform(100, 300)),
                        'width': float(np.random.uniform(20, 40)),
                        'draught': float(np.random.uniform(5, 15)),
                        'vessel_type': int(np.random.randint(70, 90))
                    }
                    mock_data.append(record)
        
        # 保存模拟数据
        mock_file = os.path.join(self.experiment_dir, f"mock_{port}_data.csv")
        pd.DataFrame(mock_data).to_csv(mock_file, index=False)
        
        return mock_file
    
    def _generate_mock_training_data(self, port: str, test_mode: bool = False) -> List[Dict]:
        """生成模拟训练数据，包含正确的queue_status"""
        logger.info(f"为 {port} 生成模拟{'测试' if test_mode else '训练'}数据...")
        
        port_spec = PORT_SPECIFICATIONS[port]
        training_samples = []
        
        # 生成100个训练样本
        num_samples = 50 if test_mode else 100
        
        for i in range(num_samples):
            # 模拟船舶状态
            vessel_state = {
                'mmsi': int(1000 + i),
                'lat': float(port_spec.lat + np.random.normal(0, 0.02)),
                'lon': float(port_spec.lon + np.random.normal(0, 0.02)),
                'sog': float(np.random.uniform(0, 20)),  # 包含高速情况
                'cog': float(np.random.uniform(0, 360)),
                'heading': float(np.random.uniform(0, 360)),
                'length': float(np.random.uniform(100, 300)),
                'width': float(np.random.uniform(20, 40)),
                'draught': float(np.random.uniform(5, 15)),
                'vessel_type': int(np.random.randint(70, 90)),
                'timestamp': datetime.now().isoformat()
            }
            
            # 模拟港口状态
            port_status = {
                'total_berths': port_spec.berths,
                'available_berths': np.random.randint(1, port_spec.berths),
                'current_traffic': np.random.randint(5, 20),
                'weather_condition': np.random.choice(['good', 'moderate', 'poor']),
                'tide_level': np.random.uniform(-2, 2)
            }
            
            # 模拟队列状态 - 关键修复！
            num_vessels_in_queue = np.random.randint(2, 8)  # 确保至少2艘船
            vessel_waiting_times = [
                float(np.random.uniform(0.5, 12.0))  # 0.5到12小时的等待时间
                for _ in range(num_vessels_in_queue)
            ]
            
            queue_status = {
                'queue_length': num_vessels_in_queue,
                'avg_waiting_time': float(np.mean(vessel_waiting_times)),
                'vessel_waiting_times': vessel_waiting_times,  # 关键数据！
                'est_processing_hours': float(np.random.uniform(2, 8)),
                'priority_vessels': np.random.randint(0, 3)
            }
            
            training_sample = {
                'vessel_state': vessel_state,
                'port_status': port_status,
                'queue_status': queue_status
            }
            
            training_samples.append(training_sample)
        
        logger.info(f"生成了 {len(training_samples)} 个{'测试' if test_mode else '训练'}样本")
        return training_samples
    
    def _load_training_test_data(self, port: str, preprocessed_dir: str):
        """加载训练和测试数据"""
        training_file = os.path.join(preprocessed_dir, f"{port}_training_data.json")
        
        if os.path.exists(training_file):
            with open(training_file, 'r') as f:
                all_data = json.load(f)
            
            # 按照5:2比例分割训练和测试数据
            split_index = int(len(all_data) * 0.7)
            self.training_data[port] = all_data[:split_index]
            self.test_data[port] = all_data[split_index:]
            
            logger.info(f"{port} - 训练样本: {len(self.training_data[port])}, 测试样本: {len(self.test_data[port])}")
        else:
            logger.warning(f"未找到 {port} 的训练数据文件，生成模拟训练数据")
            self.training_data[port] = self._generate_mock_training_data(port)
            self.test_data[port] = self._generate_mock_training_data(port, test_mode=True)
    
    def _initialize_enhanced_agents(self) -> Dict:
        """初始化增强版智能体"""
        agent_configs = {}
        
        # 增强的模型配置
        enhanced_config = {
            'action_dim': 4,
            'hidden_dim': 256,  # 增加隐藏层维度
            'num_heads': 6,     # 增加注意力头数
            'lr': 1e-4,         # 降低学习率
            'gamma': 0.99,
            'eps_clip': 0.2,
            'update_epochs': 10
        }
        
        for port in self.ports:
            logger.info(f"初始化 {port} 增强版智能体...")
            
            try:
                agent = DebugEnhancedMaritimeAgent(port, enhanced_config)
                self.agents[port] = agent
                
                # 打印状态样本用于调试
                if self.training_data.get(port):
                    sample_data = self.training_data[port][0]
                    self._print_enhanced_state_sample(port, sample_data, agent)
                
                agent_configs[port] = {
                    'initialized': True,
                    'state_dim': agent.state_dim,
                    'action_dim': agent.action_dim,
                    'model_parameters': sum(p.numel() for p in agent.actor.parameters())
                }
                
            except Exception as e:
                logger.error(f"{port} 智能体初始化失败: {e}")
                agent_configs[port] = {'error': str(e)}
        
        return agent_configs
    
    def _print_enhanced_state_sample(self, port: str, sample_data: Dict, agent: DebugEnhancedMaritimeAgent):
        """打印增强版状态样本"""
        try:
            vessel_state = sample_data['vessel_state']
            port_status = sample_data['port_status']
            queue_status = sample_data['queue_status']
            
            # 构建完整的状态信息
            berth_status = {
                'available_count': np.random.randint(5, 15),
                'suitable_count': np.random.randint(3, 10),
                'utilization_rate': np.random.uniform(0.5, 0.9),
                'next_available_hours': np.random.uniform(1, 12)
            }
            
            vessel_list = [vessel_state]
            
            # 获取动作和调试信息
            action, value, debug_info = agent.get_action_and_value(
                vessel_state, port_status, queue_status, berth_status, vessel_list
            )
            
            logger.info(f"\n{port} 增强版状态样本:")
            logger.info(f"  动作: {action}")
            logger.info(f"  价值估计: {value:.4f}")
            logger.info(f"  图嵌入范数: {debug_info.get('graph_embedding_norm', 0):.4f}")
            
        except Exception as e:
            logger.warning(f"打印状态样本失败: {e}")
    
    def _run_federated_training(self) -> Dict:
        """运行联邦学习训练"""
        training_results = {}
        
        # 训练配置
        num_rounds = self.config.get('training_rounds', 20)
        episodes_per_round = self.config.get('episodes_per_round', 50)
        
        global_performance = []
        
        for round_num in range(num_rounds):
            logger.info(f"联邦学习轮次 {round_num + 1}/{num_rounds}")
            
            round_results = {}
            
            # 每个港口的本地训练
            for port in self.ports:
                if port not in self.agents or not self.training_data.get(port):
                    continue
                
                agent = self.agents[port]
                training_data = self.training_data[port]
                
                port_performance = self._train_port_locally(
                    port, agent, training_data, episodes_per_round
                )
                
                round_results[port] = port_performance
            
            # 联邦平均 (简化版)
            self._federated_averaging()
            
            # 评估全局性能
            global_perf = self._evaluate_global_performance(round_num)
            global_performance.append(global_perf)
            
            logger.info(f"轮次 {round_num + 1} 完成 - 全局平均奖励: {global_perf.get('avg_reward', 0):.4f}")
        
        training_results = {
            'num_rounds': num_rounds,
            'global_performance': global_performance,
            'final_avg_reward': global_performance[-1].get('avg_reward', 0) if global_performance else 0,
            'convergence_achieved': self._check_convergence(global_performance)
        }
        
        return training_results
    
    def _train_port_locally(self, port: str, agent: DebugEnhancedMaritimeAgent, 
                           training_data: List[Dict], num_episodes: int) -> Dict:
        """港口本地训练"""
        episode_rewards = []
        successful_episodes = 0
        
        for episode in range(num_episodes):
            try:
                episode_reward = 0
                episode_steps = 0
                
                # 随机选择训练样本
                if len(training_data) > 0:
                    sample_indices = np.random.choice(len(training_data), min(20, len(training_data)), replace=False)
                    
                    for idx in sample_indices:
                        sample = training_data[idx]
                        
                        # 构建状态信息
                        vessel_state = sample['vessel_state']
                        port_status = sample['port_status']
                        queue_status = sample['queue_status']
                        
                        # 模拟泊位状态
                        berth_status = self._simulate_berth_status()
                        vessel_list = [vessel_state]
                        
                        # 获取动作
                        action, value, debug_info = agent.get_action_and_value(
                            vessel_state, port_status, queue_status, berth_status, vessel_list
                        )
                        
                        # 计算奖励
                        reward, breakdown = agent.calculate_reward(
                            vessel_state, action, port_status, queue_status
                        )
                        
                        episode_reward += reward
                        episode_steps += 1
                        
                        # 存储经验 (简化)
                        agent.store_transition(
                            debug_info['state_vector'], action, reward, 
                            debug_info['state_vector'], False
                        )
                
                # 更新策略
                if episode_steps > 0:
                    update_info = agent.update_policy()
                    episode_rewards.append(episode_reward / episode_steps)
                    successful_episodes += 1
                    
                    if episode % 10 == 0:
                        logger.debug(f"{port} Episode {episode}: 平均奖励 = {episode_reward/episode_steps:.4f}")
                
            except Exception as e:
                logger.warning(f"{port} Episode {episode} 失败: {e}")
        
        return {
            'successful_episodes': successful_episodes,
            'avg_reward': np.mean(episode_rewards) if episode_rewards else 0.0,
            'std_reward': np.std(episode_rewards) if episode_rewards else 0.0,
            'episode_rewards': episode_rewards
        }
    
    def _simulate_berth_status(self) -> Dict:
        """模拟泊位状态"""
        return {
            'available_count': np.random.randint(5, 15),
            'suitable_count': np.random.randint(3, 10),
            'utilization_rate': np.random.uniform(0.5, 0.9),
            'next_available_hours': np.random.uniform(1, 12)
        }
    
    def _federated_averaging(self):
        """联邦平均算法 (简化版)"""
        if len(self.agents) < 2:
            return
        
        # 获取所有智能体的参数
        all_params = {}
        for port, agent in self.agents.items():
            try:
                params = {}
                for name, param in agent.actor.named_parameters():
                    params[name] = param.data.clone()
                all_params[port] = params
            except Exception as e:
                logger.warning(f"获取 {port} 参数失败: {e}")
        
        if len(all_params) < 2:
            return
        
        # 计算平均参数
        avg_params = {}
        param_names = list(next(iter(all_params.values())).keys())
        
        for param_name in param_names:
            param_values = [all_params[port][param_name] for port in all_params.keys()]
            avg_params[param_name] = torch.stack(param_values).mean(dim=0)
        
        # 更新所有智能体的参数
        for port, agent in self.agents.items():
            try:
                for name, param in agent.actor.named_parameters():
                    if name in avg_params:
                        param.data.copy_(avg_params[name])
            except Exception as e:
                logger.warning(f"更新 {port} 参数失败: {e}")
    
    def _evaluate_global_performance(self, round_num: int) -> Dict:
        """评估全局性能"""
        total_rewards = []
        
        for port, agent in self.agents.items():
            if hasattr(agent, 'debug_info') and agent.debug_info.get('episode_rewards'):
                recent_rewards = agent.debug_info['episode_rewards'][-10:]  # 最近10个episode
                total_rewards.extend(recent_rewards)
        
        return {
            'round': round_num,
            'avg_reward': np.mean(total_rewards) if total_rewards else 0.0,
            'std_reward': np.std(total_rewards) if total_rewards else 0.0,
            'num_samples': len(total_rewards)
        }
    
    def _check_convergence(self, performance_history: List[Dict]) -> bool:
        """检查收敛性"""
        if len(performance_history) < 5:
            return False
        
        recent_rewards = [p['avg_reward'] for p in performance_history[-5:]]
        return np.std(recent_rewards) < 0.01  # 如果最近5轮的标准差小于0.01认为收敛
    
    def _run_optimization_test(self) -> Dict:
        """运行优化测试"""
        optimization_results = {}
        
        for port in self.ports:
            if port not in self.agents or not self.test_data.get(port):
                continue
            
            logger.info(f"测试 {port} 的优化效果...")
            
            agent = self.agents[port]
            test_data = self.test_data[port]
            
            original_performance = []
            optimized_performance = []
            optimization_success_count = 0
            
            for sample in test_data[:100]:  # 测试前100个样本
                try:
                    vessel_state = sample['vessel_state']
                    port_status = sample['port_status']
                    queue_status = sample['queue_status']
                    berth_status = self._simulate_berth_status()
                    vessel_list = [vessel_state]
                    
                    # 原始性能 (随机动作)
                    random_action = np.random.uniform(-1, 1, 4)
                    original_reward, _ = agent.calculate_reward(
                        vessel_state, random_action, port_status, queue_status
                    )
                    original_performance.append(original_reward)
                    
                    # 优化性能 (智能体动作)
                    action, _, _ = agent.get_action_and_value(
                        vessel_state, port_status, queue_status, berth_status, vessel_list
                    )
                    optimized_reward, _ = agent.calculate_reward(
                        vessel_state, action, port_status, queue_status
                    )
                    optimized_performance.append(optimized_reward)
                    
                    # 统计优化成功
                    if optimized_reward > original_reward:
                        optimization_success_count += 1
                        
                except Exception as e:
                    logger.warning(f"优化测试样本失败: {e}")
            
            if original_performance and optimized_performance:
                optimization_results[port] = {
                    'original_avg_reward': np.mean(original_performance),
                    'optimized_avg_reward': np.mean(optimized_performance),
                    'improvement_ratio': (np.mean(optimized_performance) - np.mean(original_performance)) / max(abs(np.mean(original_performance)), 1e-8),
                    'optimization_success_rate': optimization_success_count / len(original_performance),
                    'test_samples': len(original_performance)
                }
                
                logger.info(f"{port} 优化结果:")
                logger.info(f"  原始平均奖励: {optimization_results[port]['original_avg_reward']:.4f}")
                logger.info(f"  优化后平均奖励: {optimization_results[port]['optimized_avg_reward']:.4f}")
                logger.info(f"  改进率: {optimization_results[port]['improvement_ratio']:.2%}")
                logger.info(f"  优化成功率: {optimization_results[port]['optimization_success_rate']:.2%}")
            else:
                optimization_results[port] = {'error': 'No valid test samples'}
        
        return optimization_results
    
    def _analyze_results(self) -> Dict:
        """分析实验结果"""
        analysis = {
            'experiment_summary': {
                'total_ports': len(self.ports),
                'successful_ports': len([p for p in self.ports if p in self.agents]),
                'experiment_duration': str(datetime.now() - datetime.fromisoformat(self.experiment_log['start_time']))
            },
            'data_quality': {},
            'model_performance': {},
            'optimization_effectiveness': {}
        }
        
        # 数据质量分析
        for port in self.ports:
            if port in self.training_data:
                analysis['data_quality'][port] = {
                    'training_samples': len(self.training_data[port]),
                    'test_samples': len(self.test_data.get(port, [])),
                    'data_coverage': '7_days_simulated'  # 或实际数据天数
                }
        
        # 模型性能分析
        if 'training' in self.experiment_log['phases']:
            training_results = self.experiment_log['phases']['training']
            analysis['model_performance'] = {
                'convergence_achieved': training_results.get('convergence_achieved', False),
                'final_performance': training_results.get('final_avg_reward', 0),
                'training_stability': 'stable' if training_results.get('convergence_achieved') else 'unstable'
            }
        
        # 优化效果分析
        if 'optimization' in self.experiment_log['phases']:
            opt_results = self.experiment_log['phases']['optimization']
            overall_success_rate = np.mean([
                result.get('optimization_success_rate', 0) 
                for result in opt_results.values() 
                if isinstance(result, dict) and 'optimization_success_rate' in result
            ])
            
            analysis['optimization_effectiveness'] = {
                'overall_success_rate': overall_success_rate,
                'ports_with_improvement': len([
                    port for port, result in opt_results.items()
                    if isinstance(result, dict) and result.get('improvement_ratio', 0) > 0
                ]),
                'best_performing_port': max(
                    opt_results.items(),
                    key=lambda x: x[1].get('improvement_ratio', 0) if isinstance(x[1], dict) else 0,
                    default=('none', {})
                )[0]
            }
        
        # 保存调试报告
        for port, agent in self.agents.items():
            debug_dir = os.path.join(self.experiment_dir, 'debug_reports')
            agent.save_debug_report(debug_dir)
        
        return analysis
    
    def _save_experiment_log(self):
        """保存实验日志"""
        log_file = os.path.join(self.experiment_dir, 'experiment_log.json')
        with open(log_file, 'w', encoding='utf-8') as f:
            json.dump(self.experiment_log, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"实验日志已保存到: {log_file}")

def main():
    """主函数"""
    print("启动改进版海事GAT-FedPPO实验...")
    
    # 实验配置
    config = {
        'ports': ['new_orleans', 'south_louisiana', 'baton_rouge', 'gulfport'],
        'training_rounds': 15,
        'episodes_per_round': 30,
        'use_enhanced_features': True,
        'enable_detailed_logging': True
    }
    
    # 创建并运行实验
    experiment = ImprovedMaritimeExperiment(config)
    results = experiment.run_complete_experiment()
    
    # 输出结果摘要
    print("\n" + "="*80)
    print("实验完成摘要")
    print("="*80)
    print(f"实验ID: {results['experiment_id']}")
    print(f"成功状态: {results['success']}")
    
    if results['success']:
        analysis = results['phases'].get('analysis', {})
        print(f"训练轮次: {results['phases']['training']['num_rounds']}")
        print(f"收敛状态: {analysis.get('model_performance', {}).get('convergence_achieved', 'Unknown')}")
        print(f"整体优化成功率: {analysis.get('optimization_effectiveness', {}).get('overall_success_rate', 0):.2%}")
        print(f"最佳表现港口: {analysis.get('optimization_effectiveness', {}).get('best_performing_port', 'Unknown')}")
    
    print(f"实验目录: {experiment.experiment_dir}")
    print("="*80)

if __name__ == "__main__":
    main()