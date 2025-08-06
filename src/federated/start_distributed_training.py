#!/usr/bin/env python3
"""
分布式多端口联邦学习启动脚本
自动启动服务器和多个客户端进行分布式训练
"""

import sys
import os
import time
import subprocess
import signal
import threading
from pathlib import Path
import argparse

# 设置项目路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

class DistributedTrainingManager:
    """分布式训练管理器"""
    
    def __init__(self, server_host="localhost", server_port=8888, num_rounds=10, episodes_per_round=3):
        self.server_host = server_host
        self.server_port = server_port
        self.num_rounds = num_rounds
        self.episodes_per_round = episodes_per_round
        
        # 港口配置
        self.ports = [
            {"id": 0, "name": "new_orleans", "display": "New Orleans"},
            {"id": 1, "name": "south_louisiana", "display": "South Louisiana"},
            {"id": 2, "name": "baton_rouge", "display": "Baton Rouge"},
            {"id": 3, "name": "gulfport", "display": "Gulfport"}
        ]
        
        # 进程管理
        self.server_process = None
        self.client_processes = []
        self.running = False
        
    def start_server(self):
        """启动联邦学习服务器"""
        print("🏢 启动分布式联邦学习服务器...")
        
        server_script = project_root / "src" / "federated" / "distributed_federated_server.py"
        server_cmd = [
            sys.executable, str(server_script),
            "--host", self.server_host,
            "--port", str(self.server_port),
            "--min_clients", "2",  # 最少2个客户端
            "--max_rounds", str(self.num_rounds)
        ]
        
        try:
            self.server_process = subprocess.Popen(
                server_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            print(f"✅ 服务器启动成功 (PID: {self.server_process.pid})")
            print(f"   地址: {self.server_host}:{self.server_port}")
            
            # 等待服务器启动
            time.sleep(3)
            return True
            
        except Exception as e:
            print(f"❌ 服务器启动失败: {e}")
            return False
    
    def start_client(self, port_info, delay=0):
        """启动单个港口客户端"""
        if delay > 0:
            time.sleep(delay)
        
        print(f"🏭 启动港口客户端: {port_info['display']} (ID: {port_info['id']})")
        
        client_script = project_root / "src" / "federated" / "distributed_port_client.py"
        client_cmd = [
            sys.executable, str(client_script),
            "--port_id", str(port_info['id']),
            "--port_name", port_info['name'],
            "--server_host", self.server_host,
            "--server_port", str(self.server_port),
            "--topology", "3x3",
            "--rounds", str(self.num_rounds),
            "--episodes", str(self.episodes_per_round)
        ]
        
        try:
            client_process = subprocess.Popen(
                client_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            self.client_processes.append({
                'process': client_process,
                'port_info': port_info,
                'pid': client_process.pid
            })
            
            print(f"✅ 客户端启动成功: {port_info['display']} (PID: {client_process.pid})")
            return True
            
        except Exception as e:
            print(f"❌ 客户端启动失败 {port_info['display']}: {e}")
            return False
    
    def start_all_clients(self, selected_ports=None):
        """启动所有客户端"""
        if selected_ports is None:
            selected_ports = list(range(len(self.ports)))
        
        print(f"🚀 启动 {len(selected_ports)} 个港口客户端...")
        
        # 使用线程并行启动客户端
        threads = []
        for i, port_idx in enumerate(selected_ports):
            port_info = self.ports[port_idx]
            delay = i * 2  # 每个客户端间隔2秒启动
            
            thread = threading.Thread(
                target=self.start_client,
                args=(port_info, delay)
            )
            thread.start()
            threads.append(thread)
        
        # 等待所有客户端启动
        for thread in threads:
            thread.join()
        
        print(f"✅ 所有客户端启动完成 ({len(self.client_processes)} 个)")
    
    def monitor_processes(self):
        """监控进程状态"""
        print("📊 开始监控分布式训练进程...")
        self.running = True
        
        try:
            while self.running:
                # 检查服务器进程
                if self.server_process and self.server_process.poll() is not None:
                    print("⚠️ 服务器进程已退出")
                    break
                
                # 检查客户端进程
                active_clients = 0
                for client in self.client_processes:
                    if client['process'].poll() is None:
                        active_clients += 1
                
                if active_clients == 0 and len(self.client_processes) > 0:
                    print("✅ 所有客户端训练完成")
                    break
                
                # 显示状态
                print(f"🔄 运行状态 - 服务器: {'运行中' if self.server_process and self.server_process.poll() is None else '已停止'}, "
                      f"活跃客户端: {active_clients}/{len(self.client_processes)}")
                
                time.sleep(10)  # 每10秒检查一次
                
        except KeyboardInterrupt:
            print("⚠️ 用户中断监控")
        
        print("🔚 监控结束")
    
    def stop_all_processes(self):
        """停止所有进程"""
        print("🛑 停止所有分布式训练进程...")
        self.running = False
        
        # 停止客户端
        for client in self.client_processes:
            try:
                if client['process'].poll() is None:
                    client['process'].terminate()
                    client['process'].wait(timeout=5)
                    print(f"🔒 客户端已停止: {client['port_info']['display']}")
            except Exception as e:
                print(f"⚠️ 停止客户端失败 {client['port_info']['display']}: {e}")
                try:
                    client['process'].kill()
                except:
                    pass
        
        # 停止服务器
        if self.server_process:
            try:
                if self.server_process.poll() is None:
                    self.server_process.terminate()
                    self.server_process.wait(timeout=5)
                    print("🔒 服务器已停止")
            except Exception as e:
                print(f"⚠️ 停止服务器失败: {e}")
                try:
                    self.server_process.kill()
                except:
                    pass
        
        print("✅ 所有进程已停止")
    
    def run_distributed_training(self, selected_ports=None):
        """运行完整的分布式训练"""
        print("🚀 开始分布式多端口联邦学习")
        print("=" * 80)
        print(f"配置信息:")
        print(f"  服务器地址: {self.server_host}:{self.server_port}")
        print(f"  联邦轮次: {self.num_rounds}")
        print(f"  每轮episodes: {self.episodes_per_round}")
        
        if selected_ports:
            participating_ports = [self.ports[i]['display'] for i in selected_ports]
            print(f"  参与港口: {', '.join(participating_ports)}")
        else:
            print(f"  参与港口: 所有港口")
        
        print("=" * 80)
        
        try:
            # 1. 启动服务器
            if not self.start_server():
                return False
            
            # 2. 启动客户端
            self.start_all_clients(selected_ports)
            
            if len(self.client_processes) == 0:
                print("❌ 没有客户端启动成功")
                return False
            
            # 3. 监控训练过程
            self.monitor_processes()
            
            print("🎉 分布式联邦学习完成!")
            return True
            
        except KeyboardInterrupt:
            print("\n⚠️ 用户中断训练")
            return False
        except Exception as e:
            print(f"\n❌ 分布式训练异常: {e}")
            return False
        finally:
            self.stop_all_processes()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="分布式多端口联邦学习启动器")
    
    parser.add_argument("--host", type=str, default="localhost",
                       help="服务器主机地址")
    parser.add_argument("--port", type=int, default=8888,
                       help="服务器端口")
    parser.add_argument("--rounds", type=int, default=10,
                       help="联邦学习轮次")
    parser.add_argument("--episodes", type=int, default=3,
                       help="每轮episodes数")
    parser.add_argument("--ports", type=str, default="all",
                       help="参与港口 (all, 0,1,2,3 或 0-3)")
    
    args = parser.parse_args()
    
    # 解析端口选择
    selected_ports = None
    if args.ports != "all":
        try:
            if "-" in args.ports:
                start, end = map(int, args.ports.split("-"))
                selected_ports = list(range(start, end + 1))
            else:
                selected_ports = [int(p.strip()) for p in args.ports.split(",")]
        except ValueError:
            print("❌ 端口参数格式错误，使用所有端口")
            selected_ports = None
    
    # 创建分布式训练管理器
    manager = DistributedTrainingManager(
        server_host=args.host,
        server_port=args.port,
        num_rounds=args.rounds,
        episodes_per_round=args.episodes
    )
    
    # 设置信号处理
    def signal_handler(sig, frame):
        print("\n🛑 收到中断信号，正在安全停止...")
        manager.stop_all_processes()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # 运行分布式训练
    success = manager.run_distributed_training(selected_ports)
    
    if success:
        print("\n🎯 可以使用以下命令生成可视化:")
        print("python src/federated/visualization_generator.py")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()