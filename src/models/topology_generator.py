#!/usr/bin/env python3
"""
海事交通拓扑生成器
自动生成N×N网格拓扑的CityFlow配置，支持论文实验中的可扩展性分析
"""

import json
import os
import random
from typing import Tuple, Dict, List, Any
from pathlib import Path
import logging

class MaritimeTopologyGenerator:
    """海事交通拓扑生成器"""
    
    def __init__(self, grid_size: int = 3, base_coord: Tuple[float, float] = (-90.0, 29.9)):
        """
        初始化拓扑生成器
        
        Args:
            grid_size: 网格大小 (N×N)
            base_coord: 基础坐标 (经度, 纬度)
        """
        self.grid_size = grid_size
        self.base_lon, self.base_lat = base_coord
        self.coord_step = 0.1  # 节点间距离 (度)
        self.nodes = []
        self.roads = []
        
        logging.info(f"🌐 初始化 {grid_size}×{grid_size} 海事拓扑生成器")
    
    def generate_topology(self):
        """生成完整拓扑"""
        self._generate_nodes()
        self._generate_roads()
        logging.info(f"✅ 生成完成: {len(self.nodes)} 节点, {len(self.roads)} 道路")
    
    def _generate_nodes(self):
        """生成节点"""
        self.nodes = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                node_id = f"Node_{i}_{j}"
                longitude = self.base_lon + j * self.coord_step
                latitude = self.base_lat + i * self.coord_step
                
                # 根据位置确定节点类型
                if i == 0 and j == 0:
                    node_type = "main_port"  # 主港口
                    width = 60
                    light_time = 50
                elif i == 0 or j == 0:
                    node_type = "primary_channel"  # 主航道
                    width = 45
                    light_time = 40
                elif i == self.grid_size-1 or j == self.grid_size-1:
                    node_type = "secondary_channel"  # 次航道
                    width = 35
                    light_time = 30
                else:
                    node_type = "inner_channel"  # 内部航道
                    width = 30
                    light_time = 25
                
                # 计算连接的道路
                connected_roads = self._get_connected_roads(i, j)
                
                node = {
                    "id": node_id,
                    "point": {"x": longitude, "y": latitude},
                    "width": width,
                    "roads": connected_roads,
                    "trafficLight": {"time": light_time, "availableRoadLinks": []},
                    "metadata": {
                        "type": node_type,
                        "grid_position": [i, j],
                        "maritime_features": {
                            "depth": random.uniform(5.0, 20.0),  # 水深(米)
                            "current_speed": random.uniform(0.5, 3.0),  # 水流速度(m/s)
                            "wind_exposure": random.uniform(0.3, 1.0),  # 风浪暴露度
                        }
                    }
                }
                
                self.nodes.append(node)
    
    def _get_connected_roads(self, i: int, j: int) -> List[str]:
        """获取节点连接的道路列表"""
        roads = []
        
        # 向右连接
        if j < self.grid_size - 1:
            roads.extend([f"road_{i}_{j}_to_{i}_{j+1}", f"road_{i}_{j+1}_to_{i}_{j}"])
        
        # 向下连接  
        if i < self.grid_size - 1:
            roads.extend([f"road_{i}_{j}_to_{i+1}_{j}", f"road_{i+1}_{j}_to_{i}_{j}"])
        
        # 向左连接
        if j > 0:
            roads.extend([f"road_{i}_{j-1}_to_{i}_{j}", f"road_{i}_{j}_to_{i}_{j-1}"])
        
        # 向上连接
        if i > 0:
            roads.extend([f"road_{i-1}_{j}_to_{i}_{j}", f"road_{i}_{j}_to_{i-1}_{j}"])
        
        return roads
    
    def _generate_roads(self):
        """生成道路"""
        self.roads = []
        
        for i in range(self.grid_size):
            for j in range(self.grid_size):
                # 水平道路 (向右)
                if j < self.grid_size - 1:
                    road_id = f"road_{i}_{j}_to_{i}_{j+1}"
                    reverse_road_id = f"road_{i}_{j+1}_to_{i}_{j}"
                    
                    start_node = f"Node_{i}_{j}"
                    end_node = f"Node_{i}_{j+1}"
                    
                    start_lon = self.base_lon + j * self.coord_step
                    start_lat = self.base_lat + i * self.coord_step
                    end_lon = self.base_lon + (j+1) * self.coord_step
                    end_lat = self.base_lat + i * self.coord_step
                    
                    # 根据位置确定道路等级
                    if i == 0:
                        lane_width, max_speed, num_lanes = 8, 15, 2  # 主航道
                    elif i == 1:
                        lane_width, max_speed, num_lanes = 6, 12, 2  # 次航道
                    else:
                        lane_width, max_speed, num_lanes = 4, 10, 1  # 一般航道
                    
                    # 正向道路
                    self.roads.append({
                        "id": road_id,
                        "points": [
                            {"x": start_lon, "y": start_lat},
                            {"x": end_lon, "y": end_lat}
                        ],
                        "lanes": [
                            {"id": f"{road_id}_{k}", "width": lane_width, "maxSpeed": max_speed}
                            for k in range(num_lanes)
                        ],
                        "startIntersection": start_node,
                        "endIntersection": end_node
                    })
                    
                    # 反向道路
                    self.roads.append({
                        "id": reverse_road_id,
                        "points": [
                            {"x": end_lon, "y": end_lat},
                            {"x": start_lon, "y": start_lat}
                        ],
                        "lanes": [
                            {"id": f"{reverse_road_id}_{k}", "width": lane_width, "maxSpeed": max_speed}
                            for k in range(num_lanes)
                        ],
                        "startIntersection": end_node,
                        "endIntersection": start_node
                    })
                
                # 垂直道路 (向下)
                if i < self.grid_size - 1:
                    road_id = f"road_{i}_{j}_to_{i+1}_{j}"
                    reverse_road_id = f"road_{i+1}_{j}_to_{i}_{j}"
                    
                    start_node = f"Node_{i}_{j}"
                    end_node = f"Node_{i+1}_{j}"
                    
                    start_lon = self.base_lon + j * self.coord_step
                    start_lat = self.base_lat + i * self.coord_step
                    end_lon = self.base_lon + j * self.coord_step
                    end_lat = self.base_lat + (i+1) * self.coord_step
                    
                    # 根据位置确定道路等级
                    if j == 0:
                        lane_width, max_speed, num_lanes = 8, 15, 2  # 主航道
                    elif j == 1:
                        lane_width, max_speed, num_lanes = 6, 12, 2  # 次航道
                    else:
                        lane_width, max_speed, num_lanes = 4, 10, 1  # 一般航道
                    
                    # 正向道路
                    self.roads.append({
                        "id": road_id,
                        "points": [
                            {"x": start_lon, "y": start_lat},
                            {"x": end_lon, "y": end_lat}
                        ],
                        "lanes": [
                            {"id": f"{road_id}_{k}", "width": lane_width, "maxSpeed": max_speed}
                            for k in range(num_lanes)
                        ],
                        "startIntersection": start_node,
                        "endIntersection": end_node
                    })
                    
                    # 反向道路
                    self.roads.append({
                        "id": reverse_road_id,
                        "points": [
                            {"x": end_lon, "y": end_lat},
                            {"x": start_lon, "y": start_lat}
                        ],
                        "lanes": [
                            {"id": f"{reverse_road_id}_{k}", "width": lane_width, "maxSpeed": max_speed}
                            for k in range(num_lanes)
                        ],
                        "startIntersection": end_node,
                        "endIntersection": start_node
                    })
    
    def generate_flows(self, num_flows: int = 50) -> List[Dict]:
        """生成交通流量"""
        flows = []
        
        for flow_id in range(num_flows):
            # 随机选择起始和终点节点
            start_i, start_j = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
            end_i, end_j = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
            
            # 确保起点和终点不同
            while start_i == end_i and start_j == end_j:
                end_i, end_j = random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1)
            
            start_node = f"Node_{start_i}_{start_j}"
            end_node = f"Node_{end_i}_{end_j}"
            
            # 生成路径 (简单的最短路径)
            route = self._generate_route(start_i, start_j, end_i, end_j)
            
            # 船舶类型分布
            vessel_types = {
                "container": 0.4,    # 集装箱船
                "bulk": 0.3,         # 散货船 
                "tanker": 0.2,       # 油轮
                "passenger": 0.1     # 客船
            }
            
            vessel_type = random.choices(
                list(vessel_types.keys()),
                weights=list(vessel_types.values())
            )[0]
            
            # 根据船舶类型设置参数
            if vessel_type == "container":
                length, width, max_speed = random.uniform(200, 400), random.uniform(25, 35), random.uniform(8, 12)
            elif vessel_type == "bulk":
                length, width, max_speed = random.uniform(150, 300), random.uniform(20, 30), random.uniform(6, 10)
            elif vessel_type == "tanker":
                length, width, max_speed = random.uniform(180, 350), random.uniform(22, 32), random.uniform(7, 11)
            else:  # passenger
                length, width, max_speed = random.uniform(100, 200), random.uniform(15, 25), random.uniform(10, 16)
            
            flow = {
                "id": flow_id,
                "vehicle": {
                    "length": length,
                    "width": width,
                    "maxPosAcc": random.uniform(0.5, 1.5),
                    "maxNegAcc": random.uniform(1.0, 2.5),
                    "usualPosAcc": random.uniform(0.3, 1.0),
                    "usualNegAcc": random.uniform(0.8, 2.0),
                    "minGap": random.uniform(2.0, 5.0),
                    "maxSpeed": max_speed,
                    "headwayTime": random.uniform(1.5, 3.0)
                },
                "route": route,
                "interval": random.uniform(1.0, 10.0),
                "startTime": random.uniform(0, 300),
                "endTime": random.uniform(600, 1200),
                "metadata": {
                    "vessel_type": vessel_type,
                    "cargo_capacity": random.uniform(1000, 50000),
                    "priority": random.choice(["normal", "high", "emergency"]),
                    "environmental_impact": random.uniform(0.1, 1.0)
                }
            }
            
            flows.append(flow)
        
        return flows
    
    def _generate_route(self, start_i: int, start_j: int, end_i: int, end_j: int) -> List[str]:
        """生成从起点到终点的路径"""
        route = [f"Node_{start_i}_{start_j}"]
        current_i, current_j = start_i, start_j
        
        # 简单的贪心路径：先水平移动，再垂直移动
        while current_j != end_j:
            if current_j < end_j:
                current_j += 1
            else:
                current_j -= 1
            route.append(f"Node_{current_i}_{current_j}")
        
        while current_i != end_i:
            if current_i < end_i:
                current_i += 1
            else:
                current_i -= 1
            route.append(f"Node_{current_i}_{current_j}")
        
        return route
    
    def save_topology(self, output_dir: str = "topologies"):
        """保存拓扑配置"""
        os.makedirs(output_dir, exist_ok=True)
        
        topology_name = f"maritime_{self.grid_size}x{self.grid_size}"
        
        # 保存路网配置
        roadnet = {
            "intersections": self.nodes,
            "roads": self.roads,
            "metadata": {
                "grid_size": self.grid_size,
                "total_nodes": len(self.nodes),
                "total_roads": len(self.roads),
                "coverage_area": {
                    "min_lon": self.base_lon,
                    "max_lon": self.base_lon + (self.grid_size-1) * self.coord_step,
                    "min_lat": self.base_lat,
                    "max_lat": self.base_lat + (self.grid_size-1) * self.coord_step
                }
            }
        }
        
        roadnet_file = f"{output_dir}/{topology_name}_roadnet.json"
        with open(roadnet_file, 'w', encoding='utf-8') as f:
            json.dump(roadnet, f, indent=2, ensure_ascii=False)
        
        # 生成和保存流量配置
        flows = self.generate_flows(num_flows=self.grid_size * self.grid_size * 5)
        flow_file = f"{output_dir}/{topology_name}_flows.json"
        with open(flow_file, 'w', encoding='utf-8') as f:
            json.dump(flows, f, indent=2, ensure_ascii=False)
        
        # 保存CityFlow配置
        config = {
            "interval": 1.0,
            "seed": 42,
            "dir": f"{output_dir}/",
            "roadnetFile": f"{topology_name}_roadnet.json",
            "flowFile": f"{topology_name}_flows.json",
            "rlTrafficLight": True,
            "laneChange": False,
            "saveReplay": True,
            "roadnetLogFile": f"{topology_name}_replay_roadnet.json",
            "replayLogFile": f"{topology_name}_replay.txt"
        }
        
        config_file = f"{output_dir}/{topology_name}_config.json"
        with open(config_file, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2)
        
        # 保存统计信息
        stats = self._generate_statistics()
        stats_file = f"{output_dir}/{topology_name}_stats.json"
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logging.info(f"✅ 拓扑保存完成:")
        logging.info(f"   📁 目录: {output_dir}")
        logging.info(f"   🌐 路网: {roadnet_file}")
        logging.info(f"   🚢 流量: {flow_file}")
        logging.info(f"   ⚙️ 配置: {config_file}")
        logging.info(f"   📊 统计: {stats_file}")
        
        return {
            "topology_name": topology_name,
            "files": {
                "roadnet": roadnet_file,
                "flows": flow_file,
                "config": config_file,
                "stats": stats_file
            }
        }
    
    def _generate_statistics(self) -> Dict[str, Any]:
        """生成拓扑统计信息"""
        node_types = {}
        road_lengths = []
        
        for node in self.nodes:
            node_type = node["metadata"]["type"]
            node_types[node_type] = node_types.get(node_type, 0) + 1
        
        for road in self.roads:
            start_point = road["points"][0]
            end_point = road["points"][1]
            # 简单的欧几里得距离
            length = ((end_point["x"] - start_point["x"])**2 + (end_point["y"] - start_point["y"])**2)**0.5
            road_lengths.append(length)
        
        return {
            "topology_info": {
                "grid_size": f"{self.grid_size}×{self.grid_size}",
                "total_nodes": len(self.nodes),
                "total_roads": len(self.roads),
                "node_types": node_types
            },
            "network_metrics": {
                "average_road_length": sum(road_lengths) / len(road_lengths) if road_lengths else 0,
                "total_coverage_area": (self.coord_step * (self.grid_size-1))**2,
                "network_density": len(self.roads) / len(self.nodes) if self.nodes else 0,
                "connectivity": len(self.roads) / (len(self.nodes) * (len(self.nodes)-1) / 2) if len(self.nodes) > 1 else 0
            },
            "scalability_analysis": {
                "computational_complexity": f"O(n²) where n={self.grid_size}",
                "communication_overhead": len(self.roads) * 2,  # 双向通信
                "memory_footprint": len(self.nodes) + len(self.roads)
            }
        }


def generate_multi_scale_topologies():
    """生成多种规模的拓扑用于论文实验"""
    scales = [3, 4, 5, 6]  # 3×3 到 6×6
    results = {}
    
    logging.info("🌐 开始生成多尺度海事拓扑")
    
    for scale in scales:
        logging.info(f"📐 生成 {scale}×{scale} 拓扑...")
        
        generator = MaritimeTopologyGenerator(grid_size=scale)
        generator.generate_topology()
        
        result = generator.save_topology()
        results[f"{scale}x{scale}"] = result
        
        # 生成性能预测
        estimated_agents = scale * scale
        estimated_parameters = estimated_agents * 50000  # 假设每个智能体5万参数
        estimated_training_time = scale**2 * 0.5  # 估算训练时间(分钟)
        
        logging.info(f"   📊 预估: {estimated_agents}个智能体, {estimated_parameters:,}参数, ~{estimated_training_time:.1f}分钟训练")
    
    # 保存总体比较
    comparison = {
        "generated_topologies": list(results.keys()),
        "scalability_comparison": {
            scale: {
                "nodes": scale**2,
                "roads": scale * (scale-1) * 4,  # 估算道路数
                "complexity": scale**4,
                "estimated_training_time_minutes": scale**2 * 0.5
            }
            for scale in scales
        },
        "recommended_usage": {
            "3x3": "概念验证和快速原型",
            "4x4": "小规模港区仿真",
            "5x5": "中等规模综合测试",
            "6x6": "大规模性能评估"
        }
    }
    
    with open("topologies/scalability_analysis.json", 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    logging.info("✅ 多尺度拓扑生成完成!")
    logging.info(f"📈 可扩展性分析: topologies/scalability_analysis.json")
    
    return results


def main():
    """主函数：演示拓扑生成"""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    print("🚢 海事交通拓扑生成器")
    print("=" * 50)
    
    # 生成单个拓扑示例
    print("\n📐 生成示例拓扑 (4×4)")
    generator = MaritimeTopologyGenerator(grid_size=4)
    generator.generate_topology()
    result = generator.save_topology()
    
    print(f"✅ 示例拓扑已保存: {result['topology_name']}")
    
    # 生成多尺度拓扑
    print("\n🌐 生成多尺度拓扑...")
    multi_results = generate_multi_scale_topologies()
    
    print(f"\n📊 生成完成! 共生成 {len(multi_results)} 种拓扑:")
    for scale, info in multi_results.items():
        print(f"   • {scale}: {info['topology_name']}")
    
    print("\n🎯 用途建议:")
    print("   • 3×3: 算法验证和调试")
    print("   • 4×4: 性能基准测试")
    print("   • 5×5: 可扩展性分析")
    print("   • 6×6: 大规模压力测试")
    
    print("\n📈 论文实验建议:")
    print("   1. 使用3×3验证算法正确性")
    print("   2. 使用4×4、5×5展示可扩展性")
    print("   3. 使用6×6进行极限性能测试")
    print("   4. 对比不同规模的训练收敛性")


if __name__ == "__main__":
    main()