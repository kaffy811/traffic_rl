#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
稳态微调 v2 - 更保守 + EMA去抖
- learning_rate: 3e-4 * 0.15 (更小)
- entropy_coef: 0.008 (更低探索)
- schedule: [16,16,12] (更多轮次)
- 接受条件: min_wr 或 min_lb 任一提升 ≥ 0.2pp
- EMA去抖: α=0.7
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

def steady_fine_tune_v2():
    """稳态微调 v2 - 更保守 + EMA去抖"""
    
    # 配置参数
    port = "baton_rouge"
    stage_name = "中级阶段"
    
    # 收尾微调配置（短平快）
    base_lr = 3e-4 * 0.12  # ≈3.6e-5
    lr_decay = 0.85
    
    # 训练参数
    ppo_epochs = 4  # 稍加更新强度
    entropy_coef = 0.006  # 更低探索
    max_rounds = 8
    early_stop_rounds = 2
    
    print(f"🎯 开始稳态微调 v2 - {port} {stage_name}")
    print(f"📊 参数: lr={base_lr:.2e}, ppo_epochs={ppo_epochs}, entropy_coef={entropy_coef}")
    print(f"⏱️  轮次: {max_rounds} + {max_rounds} + 6 (早停: {early_stop_rounds}轮)")
    
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
    best_min_wr = 0.0
    best_min_lb = 0.0
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
        
        # 快速测试（三个种子）
        print(f"   🧪 快速测试（三个种子）...")
        min_wr, min_lb = quick_eval_min_wr_lb(port, stage_name)
        
        # 检查是否有改进
        improve_wr = (min_wr - best_min_wr) >= 0.002   # ≥0.2pp
        improve_lb = (min_lb - best_min_lb) >= 0.002
        if improve_wr or improve_lb:
            best_min_wr = min_wr
            best_min_lb = min_lb
            best_ckpt_path = save_path
            no_improvement_count = 0
            print(f"   ✅ 新最佳! min_wr={min_wr:.3f}, min_lb={min_lb:.3f}")
            
            # EMA去抖
            if best_ckpt_path and best_ckpt_path.exists():
                ema_result = apply_ema_smoothing(best_ckpt_path, port, stage_name)
                if ema_result:
                    print(f"   ✓ EMA去抖完成")
        else:
            no_improvement_count += 1
            print(f"   ⏸️  无改进 ({no_improvement_count}/{early_stop_rounds})")
        
        # 早停检查
        if no_improvement_count >= early_stop_rounds:
            print(f"   🛑 早停触发 ({early_stop_rounds}轮无改进)")
            break
    
    print(f"\n📈 第一轮最佳: min_wr={best_min_wr:.3f}, min_lb={best_min_lb:.3f}")
    print(f"💾 最佳检查点: {best_ckpt_path}")
    
    # 第二轮微调（如果第一轮有改进）
    if best_min_wr > 0.0:
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
            
            # 快速测试（三个种子）
            print(f"   🧪 快速测试（三个种子）...")
            min_wr, min_lb = quick_eval_min_wr_lb(port, stage_name)
            
            # 检查是否有改进
            improve_wr = (min_wr - best_min_wr) >= 0.002   # ≥0.2pp
            improve_lb = (min_lb - best_min_lb) >= 0.002
            if improve_wr or improve_lb:
                best_min_wr = min_wr
                best_min_lb = min_lb
                best_ckpt_path = save_path
                no_improvement_count = 0
                print(f"   ✅ 新最佳! min_wr={min_wr:.3f}, min_lb={min_lb:.3f}")
                
                # EMA去抖
                if best_ckpt_path and best_ckpt_path.exists():
                    ema_result = apply_ema_smoothing(best_ckpt_path, port, stage_name)
                    if ema_result:
                        print(f"   ✓ EMA去抖完成")
            else:
                no_improvement_count += 1
                print(f"   ⏸️  无改进 ({no_improvement_count}/{early_stop_rounds})")
            
            # 早停检查
            if no_improvement_count >= early_stop_rounds:
                print(f"   🛑 早停触发 ({early_stop_rounds}轮无改进)")
                break
    
    # 最终测试
    print(f"\n🎯 最终测试")
    print(f"📊 最佳: min_wr={best_min_wr:.3f}, min_lb={best_min_lb:.3f}")
    print(f"💾 最佳检查点: {best_ckpt_path}")
    
    # 运行完整一致性测试
    print(f"🧪 运行完整一致性测试...")
    cmd = [
        sys.executable, str(project_root / "src" / "federated" / "consistency_test_fixed.py"),
        "--port", port, "--samples", "800", "--seed", "42", "--no-cache"
    ]
    result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
    
    # 从输出中提取JSON文件路径
    import re
    json_match = re.search(r"测试结果已保存到:\s*(.+\.json)", result.stdout + result.stderr)
    if json_match:
        json_path = json_match.group(1).strip()
        with open(json_path, 'r', encoding='utf-8') as f:
            final_result = json.load(f)
        
        # 保存结果
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_file = project_root / "models" / "releases" / f"steady_fine_tune_v2_{port}_{stage_name}_{timestamp}.json"
        
        # 添加微调信息到结果
        final_result['fine_tune_info'] = {
            'version': 'v2',
            'best_min_wr': best_min_wr,
            'best_min_lb': best_min_lb,
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

def quick_eval_min_wr_lb(port, stage_name):
    """快速评估三个种子的最小胜率和Wilson下界"""
    seeds = [42, 123, 2025]
    wrs, lbs = [], []
    
    for seed in seeds:
        cmd = [
            sys.executable, str(project_root / "src" / "federated" / "consistency_test_fixed.py"),
            "--port", port, "--samples", "400", "--seed", str(seed), "--k", "100"
        ]
        result = subprocess.run(cmd, cwd=project_root, capture_output=True, text=True)
        
        # 从输出中提取JSON文件路径
        json_match = re.search(r"测试结果已保存到:\s*(.+\.json)", result.stdout + result.stderr)
        if json_match:
            json_path = json_match.group(1).strip()
            with open(json_path, 'r', encoding='utf-8') as f:
                test_result = json.load(f)
            
            # 提取指定阶段结果
            for stage_data in test_result.get('stages', []):
                if stage_data['stage'] == stage_name:
                    wrs.append(stage_data['win_rate'])
                    lbs.append(stage_data['wilson_lb'])
                    break
    
    min_wr = min(wrs) if wrs else 0.0
    min_lb = min(lbs) if lbs else 0.0
    print(f"   📊 胜率: {[round(w*100,2) for w in wrs]} | Wilson下界: {[round(l*100,2) for l in lbs]}")
    print(f"   📊 最小胜率: {min_wr*100:.2f}% | 最小Wilson下界: {min_lb*100:.2f}%")
    return min_wr, min_lb

def apply_ema_smoothing(ckpt_path, port, stage_name):
    """应用EMA平滑"""
    try:
        import torch
        # 读取当前检查点
        current = torch.load(ckpt_path, map_location='cpu')
        
        # 读取之前的检查点（如果存在）
        prev_path = project_root / "models" / "curriculum_v2" / port / f"stage_{stage_name}_best.pt.backup"
        if prev_path.exists():
            prev = torch.load(prev_path, map_location='cpu')
            
            # EMA合并
            alpha = 0.75
            ema_state_dict = {}
            for k, v in current['model_state_dict'].items():
                if k in prev['model_state_dict'] and prev['model_state_dict'][k].shape == v.shape:
                    ema_state_dict[k] = alpha * v + (1 - alpha) * prev['model_state_dict'][k]
                else:
                    ema_state_dict[k] = v
            
            # 保存EMA版本
            ema_checkpoint = {'model_state_dict': ema_state_dict}
            torch.save(ema_checkpoint, ckpt_path)
            
            # 备份当前版本
            torch.save(current, prev_path)
            return True
        else:
            # 第一次运行，备份当前版本
            torch.save(current, prev_path)
            return True
    except Exception as e:
        print(f"   ⚠️  EMA平滑失败: {e}")
        return False

if __name__ == "__main__":
    import torch
    steady_fine_tune_v2() 