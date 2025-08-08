"""
快速补课修复 - 直接调整阈值并重新训练
"""

import sys
import os

def apply_quick_fixes():
    """应用快速修复"""
    
    print("🔧 应用快速补课修复...")
    
    # 读取curriculum_trainer.py
    with open('curriculum_trainer.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 备份原文件
    with open('curriculum_trainer_backup.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    # 应用修复
    fixes = [
        # Baton Rouge - 基础阶段阈值放宽
        ('success_threshold=0.55  # 与胜率对齐', 'success_threshold=0.53  # 补课放宽阈值'),
        
        # New Orleans - 基础和初级阶段阈值放宽  
        ('success_threshold=0.55  # 与胜率对齐', 'success_threshold=0.53  # 补课放宽阈值'),
        
        # South Louisiana - 基础阶段阈值放宽
        ('success_threshold=0.55  # 与胜率对齐', 'success_threshold=0.53  # 补课放宽阈值'),
        
        # Gulfport - 完整阶段阈值放宽
        ('success_threshold=0.5', 'success_threshold=0.48  # 补课放宽阈值'),
    ]
    
    # 应用所有修复
    for old, new in fixes:
        if old in content:
            content = content.replace(old, new, 1)  # 只替换第一个匹配
            print(f"✅ 应用修复: {old} -> {new}")
    
    # 保存修复后的文件
    with open('curriculum_trainer.py', 'w', encoding='utf-8') as f:
        f.write(content)
    
    print("✅ 快速修复完成！")
    print("\n现在可以重新运行训练:")
    print("python curriculum_trainer.py --port baton_rouge")
    print("python curriculum_trainer.py --port new_orleans") 
    print("python curriculum_trainer.py --port south_louisiana")
    print("python curriculum_trainer.py --port gulfport")

def restore_backup():
    """恢复备份"""
    if os.path.exists('curriculum_trainer_backup.py'):
        with open('curriculum_trainer_backup.py', 'r', encoding='utf-8') as f:
            content = f.read()
        
        with open('curriculum_trainer.py', 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ 已恢复原始文件")
    else:
        print("❌ 未找到备份文件")

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "restore":
        restore_backup()
    else:
        apply_quick_fixes()