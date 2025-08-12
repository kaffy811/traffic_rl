#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
简化微调脚本 - 使用现有的train_stage方法
- lr: 6e-5 起步，每轮 ×0.85 衰减
- ppo_epochs: 4 (稍加更新强度)
- entropy_coef: 0.01 (促进 exploitation)
- 轮次: 12 + 12 (带早停，2轮不升即停)
"""

import os
import sys
import json
import time
import subprocess
import re
from pathlib import Path
from datetime import datetime

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(project_root))

from src.federated.curriculum_trainer import CurriculumTrainer, build_agent
from src.federated.curriculum_trainer import CurriculumStage

def simple_fine_tune():
    """简化微调 - 使用现有的train_stage方法"""
    
    # 配置参数
    port = "baton_rouge"
    stage_name = "中级阶段"
    
    # 更保守的学习率设置
    base_lr = 3e-4 * 0.2  # 6e-5
    lr_decay = 0.85
    
    # 训练参数
    ppo_epochs = 4  # 稍加更新强度
    entropy_coef = 0.01  # 促进 exploitation
    max_rounds = 12
    early_stop_rounds = 2
    
    print(f"🎯 开始简化微调 - {port} {stage_name}")
    print(f"📊 参数: lr={base_lr:.2e}, ppo_epochs={ppo_epochs}, entropy_coef={entropy_coef}")
    print(f"⏱️  轮次: {max_rounds} + {max_rounds} (早停: {early_stop_rounds}轮)")
    
    # 初始化训练器
    trainer = CurriculumTrainer(port_name=port)
    
    # 找到对应的阶段
    target_stage = None
    for stage in trainer.curriculum_stages:
        if stage.name == stage_name:
            target_stage = stage
            break
    
    if not target_stage:
        print(f"❌ 未找到阶段: {stage_name}")
        return
    
    print(f"✅ 找到阶段: {target_stage.name} - {target_stage.description}")
    
    # 记录最佳性能
    best_win_rate = 0.0
    no_improvement_count = 0
    best_ckpt_path = None
    
    # 第一轮微调
    print(f"\n🔄 第一轮微调 ({max_rounds}轮)")
    for round_idx in range(max_rounds):
        current_lr = base_lr * (lr_decay ** round_idx)
        
        print(f"\n📈 轮次 {round_idx + 1}/{max_rounds}")
        print(f"   💡 学习率: {current_lr:.2e}")
        
        # 构建智能体
        agent = build_agent(port, learning_rate=current_lr, ppo_epochs=ppo_epochs)
        
        # 设置entropy_coef
        if hasattr(agent, 'entropy_coef'):
            agent.entropy_coef = entropy_coef
        
        # 加载现有模型
        ckpt_path = trainer.save_dir / f"stage_{stage_name}_best.pt"
        if ckpt_path.exists():
            print(f"   📂 加载现有模型: {ckpt_path}")
            try:
                checkpoint = torch.load(ckpt_path, map_location='cpu')
                if 'model_state_dict' in checkpoint:
                    agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
                elif 'actor_critic' in checkpoint:
                    agent.actor_critic.load_state_dict(checkpoint['actor_critic'])
                else:
                    agent.actor_critic.load_state_dict(checkpoint)
                print(f"   ✅ 模型加载成功")
            except Exception as e:
                print(f"   ⚠️  模型加载失败: {e}")
        
        # 微调
        print(f"   🔄 开始微调...")
        try:
            trained_agent, training_info = trainer.train_stage(agent, target_stage)
            
            # 保存模型
            save_path = trainer.save_dir / f"stage_{stage_name}_best.pt"
            torch.save({
                'model_state_dict': trained_agent.actor_critic.state_dict(),
                'training_info': training_info,
                'timestamp': datetime.now().isoformat()
            }, save_path)
            print(f"   💾 模型已保存: {save_path}")
            
        except Exception as e:
            print(f"   ❌ 微调失败: {e}")
            continue
        
        # 快速测试
        print(f"   🧪 快速测试...")
        cmd = [
            sys.executable, str(project_root / "src" / "federated" / "consistency_test_fixed.py"),
            "--port", port, "--samples", "200", "--seed", "42", "--no-cache"
        ]
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        
        # 从输出中提取JSON文件路径
        json_match = re.search(r"测试结果已保存到:\s*(.+\.json)", result.stdout + result.stderr)
        if json_match:
            json_path = json_match.group(1).strip()
            with open(json_path, 'r', encoding='utf-8') as f:
                test_result = json.load(f)
            
            # 提取中级阶段结果
            stage_result = None
            for stage_data in test_result.get('stages', []):
                if stage_data['stage'] == stage_name:
                    stage_result = stage_data
                    break
            
            if stage_result:
                win_rate = stage_result['win_rate']
                wilson_lb = stage_result['wilson_lb']
                print(f"   📊 胜率: {win_rate:.3f} (Wilson下界: {wilson_lb:.3f})")
                
                # 检查是否有改进
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    best_ckpt_path = save_path
                    no_improvement_count = 0
                    print(f"   ✅ 新最佳! 胜率提升到 {win_rate:.3f}")
                else:
                    no_improvement_count += 1
                    print(f"   ⏸️  无改进 ({no_improvement_count}/{early_stop_rounds})")
                
                # 早停检查
                if no_improvement_count >= early_stop_rounds:
                    print(f"   🛑 早停触发 ({early_stop_rounds}轮无改进)")
                    break
            else:
                print(f"   ❌ 未找到{stage_name}测试结果")
        else:
            print(f"   ❌ 无法找到测试结果文件")
    
    print(f"\n📈 第一轮最佳胜率: {best_win_rate:.3f}")
    print(f"💾 最佳检查点: {best_ckpt_path}")
    
    # 第二轮微调（如果第一轮有改进）
    if best_win_rate > 0.0:
        print(f"\n🔄 第二轮微调 ({max_rounds}轮)")
        no_improvement_count = 0
        
        for round_idx in range(max_rounds):
            current_lr = base_lr * (lr_decay ** round_idx) * 0.5  # 更保守
            
            print(f"\n📈 轮次 {round_idx + 1}/{max_rounds}")
            print(f"   💡 学习率: {current_lr:.2e}")
            
            # 构建智能体
            agent = build_agent(port, learning_rate=current_lr, ppo_epochs=ppo_epochs)
            
            # 设置entropy_coef
            if hasattr(agent, 'entropy_coef'):
                agent.entropy_coef = entropy_coef
            
            # 加载最佳模型
            if best_ckpt_path and best_ckpt_path.exists():
                print(f"   📂 加载最佳模型: {best_ckpt_path}")
                try:
                    checkpoint = torch.load(best_ckpt_path, map_location='cpu')
                    if 'model_state_dict' in checkpoint:
                        agent.actor_critic.load_state_dict(checkpoint['model_state_dict'])
                    elif 'actor_critic' in checkpoint:
                        agent.actor_critic.load_state_dict(checkpoint['actor_critic'])
                    else:
                        agent.actor_critic.load_state_dict(checkpoint)
                    print(f"   ✅ 模型加载成功")
                except Exception as e:
                    print(f"   ⚠️  模型加载失败: {e}")
            
            # 微调
            print(f"   🔄 开始微调...")
            try:
                trained_agent, training_info = trainer.train_stage(agent, target_stage)
                
                # 保存模型
                save_path = trainer.save_dir / f"stage_{stage_name}_best.pt"
                torch.save({
                    'model_state_dict': trained_agent.actor_critic.state_dict(),
                    'training_info': training_info,
                    'timestamp': datetime.now().isoformat()
                }, save_path)
                print(f"   💾 模型已保存: {save_path}")
                
            except Exception as e:
                print(f"   ❌ 微调失败: {e}")
                continue
            
            # 快速测试
            print(f"   🧪 快速测试...")
            cmd = [
                sys.executable, str(project_root / "src" / "federated" / "consistency_test_fixed.py"),
                "--port", port, "--samples", "200", "--seed", "42", "--no-cache"
            ]
            result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
            
            # 从输出中提取JSON文件路径
            json_match = re.search(r"测试结果已保存到:\s*(.+\.json)", result.stdout + result.stderr)
            if json_match:
                json_path = json_match.group(1).strip()
                with open(json_path, 'r', encoding='utf-8') as f:
                    test_result = json.load(f)
                
                # 提取中级阶段结果
                stage_result = None
                for stage_data in test_result.get('stages', []):
                    if stage_data['stage'] == stage_name:
                        stage_result = stage_data
                        break
                
                if stage_result:
                    win_rate = stage_result['win_rate']
                    wilson_lb = stage_result['wilson_lb']
                    print(f"   📊 胜率: {win_rate:.3f} (Wilson下界: {wilson_lb:.3f})")
                    
                    # 检查是否有改进
                    if win_rate > best_win_rate:
                        best_win_rate = win_rate
                        best_ckpt_path = save_path
                        no_improvement_count = 0
                        print(f"   ✅ 新最佳! 胜率提升到 {win_rate:.3f}")
                    else:
                        no_improvement_count += 1
                        print(f"   ⏸️  无改进 ({no_improvement_count}/{early_stop_rounds})")
                    
                    # 早停检查
                    if no_improvement_count >= early_stop_rounds:
                        print(f"   🛑 早停触发 ({early_stop_rounds}轮无改进)")
                        break
                else:
                    print(f"   ❌ 未找到{stage_name}测试结果")
            else:
                print(f"   ❌ 无法找到测试结果文件")
    
    # 最终测试
    print(f"\n🎯 最终测试")
    print(f"📊 最佳胜率: {best_win_rate:.3f}")
    print(f"💾 最佳检查点: {best_ckpt_path}")
    
    # 运行完整一致性测试
    print(f"🧪 运行完整一致性测试...")
    cmd = [
        sys.executable, str(project_root / "src" / "federated" / "consistency_test_fixed.py"),
        "--port", port, "--samples", "800", "--seed", "42", "--no-cache"
    ]
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    
    # 从输出中提取JSON文件路径
    json_match = re.search(r"测试结果已保存到:\s*(.+\.json)", result.stdout + result.stderr)
    if json_match:
        json_path = json_match.group(1).strip()
        with open(json_path, 'r', encoding='utf-8') as f:
            final_result = json.load(f)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = project_root / "models" / "releases" / f"simple_fine_tune_{port}_{stage_name}_{timestamp}.json"
        
        # 添加微调信息到结果
        final_result['fine_tune_info'] = {
            'version': 'simple',
            'best_win_rate': best_win_rate,
            'best_ckpt_path': str(best_ckpt_path) if best_ckpt_path else None,
            'parameters': {
                'base_lr': base_lr,
                'lr_decay': lr_decay,
                'ppo_epochs': ppo_epochs,
                'entropy_coef': entropy_coef,
                'max_rounds': max_rounds,
                'early_stop_rounds': early_stop_rounds
            }
        }
        
        result_file.parent.mkdir(parents=True, exist_ok=True)
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(final_result, f, ensure_ascii=False, indent=2)
        
        print(f"💾 结果已保存: {result_file}")
        
        # 显示最终结果
        print(f"\n📊 最终结果:")
        for stage_data in final_result.get('stages', []):
            if stage_data['stage'] == stage_name:
                print(f"   {stage_name}: 胜率 {stage_data['win_rate']:.3f} (阈值 {stage_data['threshold']:.2f})")
                print(f"   Wilson下界: {stage_data['wilson_lb']:.3f}")
                print(f"   通过: {'✅' if stage_data['pass'] else '❌'}")
                break
    else:
        print(f"❌ 无法找到最终测试结果文件")

if __name__ == "__main__":
    import torch
    simple_fine_tune() 