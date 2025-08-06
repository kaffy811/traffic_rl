#!/usr/bin/env python3
"""
分布式港口客户端
每个港口在独立的终端/服务器上运行，通过网络进行联邦学习
"""

import sys
import os
import argparse
import time
import json
import socket
import threading
from pathlib import Path
from datetime import datetime
import logging

# 设置项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 网络通信
import requests
import torch
import numpy as np

# CityFlow导入
try:
    import cityflow
    CITYFLOW_AVAILABLE = True
except ImportError:
    CITYFLOW_AVAILABLE = False

# 自定义模块导入
from src.models.maritime_gat_ppo import MaritimeGATPPOAgent
from src.models.fairness_reward import AlphaFairRewardCalculator

class DistributedPortClient:
    """分布式港口客户端 - 在独立终端/服务器上运行"""
    
    def __init__(self, port_id: int, port_name: str, server_host: str = "localhost", 
                 server_port: int = 8888, topology_size: str = "3x3"):
        self.port_id = port_id
        self.port_name = port_name
        self.server_host = server_host
        self.server_port = server_port
        self.topology_size = topology_size
        
        # 客户端标识
        self.client_id = f"port_{port_id}_{port_name}"
        
        # 设置日志
        self.logger = self._setup_logging()
        
        # CityFlow环境
        self.cityflow_env = None
        self.current_step = 0
        self.max_steps = 1000
        
        # GAT-PPO智能体
        self.state_dim = 4
        self.action_dim = 4
        self.node_num = 9 if topology_size == "3x3" else 16
        
        self.gat_ppo_agent = MaritimeGATPPOAgent(
            node_num=self.node_num,
            node_dim=self.state_dim,
            action_dim=self.action_dim
        )
        
        # 公平性奖励计算器
        self.fairness_calculator = AlphaFairRewardCalculator(alpha=0.5)
        
        # 训练历史
        self.training_history = []
        self.local_model_version = 0
        
        self.logger.info(f"🏭 分布式港口客户端初始化: {port_name} (ID: {port_id})")
        self.logger.info(f"📡 服务器地址: {server_host}:{server_port}")
        
        # 初始化CityFlow环境
        self._initialize_cityflow()
        
    def _setup_logging(self):
        """设置日志"""
        logger = logging.getLogger(f"Port_{self.port_name}")
        logger.setLevel(logging.INFO)
        
        # 创建文件处理器
        log_dir = project_root / "src" / "federated" / "logs"
        log_dir.mkdir(exist_ok=True)
        
        handler = logging.FileHandler(log_dir / f"port_{self.port_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        
        # 控制台处理器
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        
        logger.addHandler(handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _initialize_cityflow(self):
        """初始化CityFlow环境"""
        try:
            if CITYFLOW_AVAILABLE:
                # 创建港口特定的配置
                config_file = self._create_port_config()
                self.cityflow_env = cityflow.Engine(config_file, thread_num=1)
                self.logger.info(f"✅ CityFlow环境初始化成功: {config_file}")
            else:
                self.logger.warning("⚠️ CityFlow不可用，使用模拟环境")
                self._init_mock_environment()
        except Exception as e:
            self.logger.error(f"❌ CityFlow初始化失败: {e}")
            self._init_mock_environment()
    
    def _create_port_config(self) -> str:
        """创建港口特定的配置文件"""
        topology_dir = project_root / "topologies"
        base_config_file = topology_dir / f"maritime_{self.topology_size}_config.json"
        
        # 读取基础配置
        with open(base_config_file, 'r') as f:
            config = json.load(f)
        
        # 为每个港口创建唯一配置
        port_config_file = topology_dir / f"maritime_{self.topology_size}_{self.port_name}_config.json"
        
        # 修改配置
        config["seed"] = 42 + self.port_id * 100  # 不同港口使用不同种子
        config["roadnetLogFile"] = f"maritime_{self.topology_size}_{self.port_name}_replay_roadnet.json"
        config["replayLogFile"] = f"maritime_{self.topology_size}_{self.port_name}_replay.txt"
        
        # 保存港口配置
        with open(port_config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return str(port_config_file)
    
    def _init_mock_environment(self):
        """初始化模拟环境"""
        self.cityflow_env = None
        self.mock_state = {
            'vehicles': np.random.randint(10, 50),
            'waiting_vehicles': np.random.randint(5, 20),
            'average_speed': np.random.uniform(5, 15),
            'queue_lengths': np.random.randint(0, 10, size=4).tolist()
        }
        self.logger.info("🔄 使用模拟环境")
    
    def register_with_server(self) -> bool:
        """向服务器注册"""
        try:
            registration_data = {
                "client_id": self.client_id,
                "port_id": self.port_id,
                "port_name": self.port_name,
                "topology_size": self.topology_size,
                "capabilities": {
                    "cityflow_available": CITYFLOW_AVAILABLE,
                    "node_num": self.node_num,
                    "state_dim": self.state_dim,
                    "action_dim": self.action_dim
                }
            }
            
            response = requests.post(
                f"http://{self.server_host}:{self.server_port}/register",
                json=registration_data,
                timeout=10
            )
            
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"✅ 注册成功: {result['message']}")
                return True
            else:
                self.logger.error(f"❌ 注册失败: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 注册异常: {e}")
            return False
    
    def get_global_model(self) -> bool:
        """从服务器获取全局模型"""
        try:
            response = requests.get(
                f"http://{self.server_host}:{self.server_port}/get_global_model",
                params={"client_id": self.client_id},
                timeout=30
            )
            
            if response.status_code == 200:
                model_data = response.json()
                
                if model_data["has_model"]:
                    # 更新本地模型
                    global_params = model_data["model_params"]
                    self._update_local_model(global_params)
                    self.local_model_version = model_data["version"]
                    self.logger.info(f"📥 获取全局模型 v{self.local_model_version}")
                    return True
                else:
                    self.logger.info("📭 服务器暂无全局模型")
                    return True
            else:
                self.logger.error(f"❌ 获取全局模型失败: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 获取全局模型异常: {e}")
            return False
    
    def _update_local_model(self, global_params: dict):
        """更新本地模型参数"""
        try:
            # 将字符串参数转换回tensor
            torch_params = {}
            for key, value in global_params.items():
                if isinstance(value, list):
                    torch_params[key] = torch.tensor(value)
                else:
                    torch_params[key] = torch.tensor(value)
            
            self.gat_ppo_agent.set_model_parameters(torch_params)
            self.logger.info("🔄 本地模型参数已更新")
            
        except Exception as e:
            self.logger.error(f"❌ 更新本地模型失败: {e}")
    
    def train_local_episode(self) -> dict:
        """训练一个本地episode"""
        self.logger.info(f"🏃 开始本地训练 Episode")
        
        # 重置环境
        state = self._reset_environment()
        episode_reward = 0
        episode_steps = 0
        
        # 构建GAT输入
        node_features = self._build_node_features(state)
        
        for step in range(self.max_steps):
            # GAT-PPO决策
            action, action_prob = self.gat_ppo_agent.select_action(node_features)
            
            # 环境交互
            next_state, reward, done, info = self._step_environment(action)
            
            # 公平性奖励
            fairness_reward = self.fairness_calculator.calculate_reward(
                base_reward=reward,
                current_state=state,
                action=action,
                agent_id=self.port_id
            )
            
            total_reward = reward + fairness_reward
            
            # 构建下一状态特征
            next_node_features = self._build_node_features(next_state)
            
            # 存储经验
            self.gat_ppo_agent.store_experience(
                state=node_features,
                action=action,
                reward=total_reward,
                next_state=next_node_features,
                done=done,
                action_prob=action_prob
            )
            
            # 更新状态
            state = next_state
            node_features = next_node_features
            episode_reward += total_reward
            episode_steps += 1
            
            if done:
                break
        
        # 执行PPO更新
        training_stats = self.gat_ppo_agent.update()
        
        episode_result = {
            "episode_reward": episode_reward,
            "episode_steps": episode_steps,
            "avg_reward_per_step": episode_reward / max(episode_steps, 1),
            "port_name": self.port_name,
            "port_id": self.port_id,
            **training_stats
        }
        
        self.training_history.append(episode_result)
        self.logger.info(f"✅ Episode完成 - 奖励: {episode_reward:.2f}, 步数: {episode_steps}")
        
        return episode_result
    
    def upload_local_model(self, training_result: dict) -> bool:
        """上传本地模型到服务器"""
        try:
            # 获取模型参数
            local_params = self.gat_ppo_agent.get_model_parameters()
            
            # 转换tensor为list (JSON序列化)
            serializable_params = {}
            for key, value in local_params.items():
                if torch.is_tensor(value):
                    serializable_params[key] = value.tolist()
                else:
                    serializable_params[key] = value
            
            upload_data = {
                "client_id": self.client_id,
                "port_id": self.port_id,
                "port_name": self.port_name,
                "model_params": serializable_params,
                "training_result": training_result,
                "local_version": self.local_model_version
            }
            
            response = requests.post(
                f"http://{self.server_host}:{self.server_port}/upload_model",
                json=upload_data,
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                self.logger.info(f"📤 模型上传成功: {result['message']}")
                return True
            else:
                self.logger.error(f"❌ 模型上传失败: {response.status_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ 模型上传异常: {e}")
            return False
    
    def _reset_environment(self) -> np.ndarray:
        """重置环境"""
        self.current_step = 0
        
        if self.cityflow_env:
            self.cityflow_env.reset()
            return self._get_cityflow_state()
        else:
            self._init_mock_environment()
            return self._get_mock_state()
    
    def _step_environment(self, action: int) -> tuple:
        """环境步进"""
        self.current_step += 1
        
        if self.cityflow_env:
            return self._cityflow_step(action)
        else:
            return self._mock_step(action)
    
    def _cityflow_step(self, action: int) -> tuple:
        """CityFlow环境步进"""
        # 应用动作
        self._apply_cityflow_action(action)
        
        # 执行仿真步
        self.cityflow_env.next_step()
        
        # 获取状态和奖励
        state = self._get_cityflow_state()
        reward = self._calculate_reward(state, action)
        done = self.current_step >= self.max_steps
        info = {"step": self.current_step, "port": self.port_name}
        
        return state, reward, done, info
    
    def _mock_step(self, action: int) -> tuple:
        """模拟环境步进"""
        # 模拟动作效果
        effect = {0: 0.95, 1: 1.1, 2: 0.9, 3: 1.05}.get(action, 1.0)
        
        # 更新状态
        self.mock_state['vehicles'] = max(5, int(self.mock_state['vehicles'] * np.random.uniform(0.9, 1.1)))
        self.mock_state['waiting_vehicles'] = max(0, int(self.mock_state['waiting_vehicles'] * effect))
        self.mock_state['average_speed'] = np.clip(
            self.mock_state['average_speed'] * np.random.uniform(0.95, 1.05) * (2-effect), 
            2, 20
        )
        
        state = self._get_mock_state()
        reward = self._calculate_reward(state, action)
        done = self.current_step >= self.max_steps
        info = {"step": self.current_step, "port": self.port_name}
        
        return state, reward, done, info
    
    def _get_cityflow_state(self) -> np.ndarray:
        """获取CityFlow状态"""
        try:
            lane_count = self.cityflow_env.get_lane_vehicle_count()
            lane_waiting = self.cityflow_env.get_lane_waiting_vehicle_count()
            vehicle_speed = self.cityflow_env.get_vehicle_speed()
            
            total_vehicles = sum(lane_count.values()) if lane_count else 0
            total_waiting = sum(lane_waiting.values()) if lane_waiting else 0
            avg_speed = np.mean(list(vehicle_speed.values())) if vehicle_speed else 0
            
            state = np.array([
                total_vehicles / 100.0,
                total_waiting / 50.0,
                avg_speed / 20.0,
                len(lane_count) / 20.0
            ], dtype=np.float32)
            
            return state
        except:
            return np.array([0.1, 0.1, 0.5, 0.2], dtype=np.float32)
    
    def _get_mock_state(self) -> np.ndarray:
        """获取模拟状态"""
        return np.array([
            self.mock_state['vehicles'] / 100.0,
            self.mock_state['waiting_vehicles'] / 50.0,
            self.mock_state['average_speed'] / 20.0,
            np.mean(self.mock_state['queue_lengths']) / 10.0
        ], dtype=np.float32)
    
    def _apply_cityflow_action(self, action: int):
        """应用CityFlow动作"""
        if not self.cityflow_env:
            return
        try:
            intersections = self.cityflow_env.get_intersection_ids()
            if intersections:
                phase = action % 4  # 4个相位
                self.cityflow_env.set_tl_phase(intersections[0], phase)
        except:
            pass
    
    def _calculate_reward(self, state: np.ndarray, action: int) -> float:
        """计算奖励"""
        efficiency_reward = (state[2] * 10) - (state[1] * 5)
        throughput_reward = (state[0] * 2) if state[1] < 0.3 else 0
        action_reward = {0: 0, 1: 2, 2: 1, 3: 1.5}.get(action, 0)
        stability_reward = 2 if 0.2 < state[0] < 0.8 and state[1] < 0.4 else 0
        
        return efficiency_reward + throughput_reward + action_reward + stability_reward
    
    def _build_node_features(self, state: np.ndarray) -> torch.Tensor:
        """构建GAT节点特征"""
        node_features = []
        
        for i in range(self.node_num):
            base_features = state.copy()
            
            # 位置编码
            row = i // int(np.sqrt(self.node_num))
            col = i % int(np.sqrt(self.node_num))
            position_encoding = np.array([row / int(np.sqrt(self.node_num)), 
                                        col / int(np.sqrt(self.node_num))])
            
            combined_features = np.concatenate([base_features, position_encoding])
            node_features.append(combined_features)
        
        return torch.FloatTensor(node_features).unsqueeze(0)
    
    def run_federated_training(self, num_rounds: int = 10, episodes_per_round: int = 3):
        """运行分布式联邦训练"""
        self.logger.info(f"🚀 开始分布式联邦训练: {num_rounds}轮, 每轮{episodes_per_round}episodes")
        
        # 注册到服务器
        if not self.register_with_server():
            self.logger.error("❌ 服务器注册失败，退出训练")
            return
        
        for round_num in range(1, num_rounds + 1):
            self.logger.info(f"📍 开始第 {round_num}/{num_rounds} 轮联邦训练")
            
            # 1. 获取全局模型
            if not self.get_global_model():
                self.logger.warning(f"⚠️ 获取全局模型失败，使用本地模型继续训练")
            
            # 2. 本地训练
            round_results = []
            for episode in range(episodes_per_round):
                self.logger.info(f"   Episode {episode + 1}/{episodes_per_round}")
                episode_result = self.train_local_episode()
                round_results.append(episode_result)
            
            # 3. 计算轮次统计
            avg_reward = np.mean([r["episode_reward"] for r in round_results])
            round_summary = {
                "round": round_num,
                "episodes": len(round_results),
                "avg_reward": avg_reward,
                "total_steps": sum(r["episode_steps"] for r in round_results),
                "results": round_results
            }
            
            # 4. 上传本地模型
            if not self.upload_local_model(round_summary):
                self.logger.warning(f"⚠️ 模型上传失败")
            
            self.logger.info(f"✅ 第 {round_num} 轮完成 - 平均奖励: {avg_reward:.2f}")
            
            # 轮次间等待
            time.sleep(2)
        
        self.logger.info(f"🎉 分布式联邦训练完成!")
    
    def close(self):
        """关闭客户端"""
        if self.cityflow_env:
            self.cityflow_env = None
        self.logger.info(f"🔒 港口客户端 {self.port_name} 已关闭")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分布式港口客户端")
    
    parser.add_argument("--port_id", type=int, required=True, 
                       help="港口ID (0-3)")
    parser.add_argument("--port_name", type=str, required=True,
                       choices=["new_orleans", "south_louisiana", "baton_rouge", "gulfport"],
                       help="港口名称")
    parser.add_argument("--server_host", type=str, default="localhost",
                       help="服务器主机地址")
    parser.add_argument("--server_port", type=int, default=8888,
                       help="服务器端口")
    parser.add_argument("--topology", type=str, default="3x3",
                       choices=["3x3", "4x4", "5x5", "6x6"],
                       help="拓扑大小")
    parser.add_argument("--rounds", type=int, default=10,
                       help="联邦学习轮次")
    parser.add_argument("--episodes", type=int, default=3,
                       help="每轮episodes数")
    
    args = parser.parse_args()
    
    print(f"🏭 启动分布式港口客户端")
    print(f"   港口: {args.port_name} (ID: {args.port_id})")
    print(f"   服务器: {args.server_host}:{args.server_port}")
    print(f"   配置: {args.topology}, {args.rounds}轮, 每轮{args.episodes}episodes")
    
    # 创建分布式港口客户端
    client = DistributedPortClient(
        port_id=args.port_id,
        port_name=args.port_name,
        server_host=args.server_host,
        server_port=args.server_port,
        topology_size=args.topology
    )
    
    try:
        # 运行分布式联邦训练
        client.run_federated_training(
            num_rounds=args.rounds,
            episodes_per_round=args.episodes
        )
    except KeyboardInterrupt:
        print("⚠️ 用户中断训练")
    except Exception as e:
        print(f"❌ 训练异常: {e}")
        import traceback
        traceback.print_exc()
    finally:
        client.close()

if __name__ == "__main__":
    main()