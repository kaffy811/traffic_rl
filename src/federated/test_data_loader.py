#!/usr/bin/env python3
"""
测试海事数据加载器
验证数据索引是否正确匹配
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from maritime_data_loader import load_maritime_data
import argparse

def test_data_loader():
    """测试数据加载器"""
    
    # 模拟FedML的args对象
    args = argparse.Namespace()
    args.client_num_in_total = 4
    args.batch_size = 10
    
    print("🧪 测试海事数据加载器...")
    print(f"📋 客户端总数: {args.client_num_in_total}")
    print(f"📦 批次大小: {args.batch_size}")
    print("-" * 50)
    
    try:
        # 加载数据
        dataset = load_maritime_data(args)
        
        # 解包8元组
        train_data_num, test_data_num, train_data_global, test_data_global, \
        train_data_local_num_dict, train_data_local_dict, test_data_local_dict, class_num = dataset
        
        print("✅ 数据加载成功！")
        print(f"📊 总训练数据: {train_data_num}")
        print(f"📊 总测试数据: {test_data_num}")
        print(f"🏷️ 类别数: {class_num}")
        print()
        
        print("📋 训练数据分布:")
        for client_idx, num in train_data_local_num_dict.items():
            print(f"  客户端 {client_idx}: {num} episodes")
        
        print()
        print("🔑 数据字典键值:")
        print(f"  train_data_local_dict keys: {list(train_data_local_dict.keys())}")
        print(f"  test_data_local_dict keys: {list(test_data_local_dict.keys())}")
        
        # 测试访问每个客户端的数据
        print()
        print("🧪 测试数据访问:")
        for client_idx in range(args.client_num_in_total):
            try:
                train_loader = train_data_local_dict[client_idx]
                test_loader = test_data_local_dict[client_idx]
                print(f"  ✅ 客户端 {client_idx}: 训练 {len(train_loader.dataset)} episodes, 测试 {len(test_loader.dataset)} episodes")
            except KeyError as e:
                print(f"  ❌ 客户端 {client_idx}: KeyError - {e}")
        
        # 测试数据索引2（这是之前出错的索引）
        print()
        print("🎯 重点测试客户端索引2:")
        try:
            train_loader_2 = train_data_local_dict[2]
            test_loader_2 = test_data_local_dict[2]
            print(f"  ✅ 客户端2数据访问成功: 训练 {len(train_loader_2.dataset)} episodes")
            
            # 测试数据迭代
            for i, (x, y) in enumerate(train_loader_2):
                print(f"  ✅ 批次 {i}: 输入形状 {x.shape}, 标签形状 {y.shape}")
                if i >= 1:  # 只测试前2个批次
                    break
                    
        except Exception as e:
            print(f"  ❌ 客户端2数据访问失败: {e}")
        
        print()
        print("🎉 数据加载器测试完成！")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        raise

if __name__ == "__main__":
    test_data_loader()