#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型导出脚本 - 批量导出所有训练好的模型为不同格式
支持导出为 TorchScript 和 ONNX 格式
"""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List
import torch
import logging

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from inference_runner import InferenceRunner
from curriculum_trainer import CurriculumTrainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_all_model_paths() -> Dict[str, Dict[str, str]]:
    """获取所有模型路径"""
    models_dir = Path("../../models/curriculum_v2")
    model_paths = {}
    
    for port_dir in models_dir.iterdir():
        if port_dir.is_dir():
            port_name = port_dir.name
            model_paths[port_name] = {}
            
            for model_file in port_dir.glob("stage_*_best.pt"):
                stage_name = model_file.stem.replace("stage_", "").replace("_best", "")
                model_paths[port_name][stage_name] = str(model_file)
    
    return model_paths

def create_sample_input(port_name: str, stage_name: str) -> tuple:
    """为指定港口和阶段创建示例输入"""
    try:
        trainer = CurriculumTrainer(port_name)
        stages = trainer.curriculum_stages
        
        # 找到对应的阶段
        target_stage = None
        for stage in stages:
            if stage.name == stage_name:
                target_stage = stage
                break
        
        if target_stage is None:
            logger.warning(f"未找到阶段 {stage_name}，使用默认输入")
            # 使用默认维度
            batch_size = 1
            state_dim = 56
            num_nodes = 10
            node_feature_dim = 8
        else:
            # 根据阶段配置创建输入
            batch_size = 1
            state_dim = 56  # 统一为56维状态
            num_nodes = min(target_stage.max_vessels + target_stage.max_berths, 20)
            node_feature_dim = 8
        
        # 创建示例张量
        device = torch.device("cpu")
        sample_input = (
            torch.randn(batch_size, state_dim, device=device),
            torch.randn(batch_size, num_nodes, node_feature_dim, device=device),
            torch.randn(batch_size, num_nodes, num_nodes, device=device)
        )
        
        return sample_input
        
    except Exception as e:
        logger.error(f"创建示例输入失败: {e}")
        # 返回默认输入
        device = torch.device("cpu")
        return (
            torch.randn(1, 56, device=device),
            torch.randn(1, 10, 8, device=device),
            torch.randn(1, 10, 10, device=device)
        )

def export_single_model(port_name: str, stage_name: str, model_path: str,
                       output_dir: Path, formats: List[str]) -> Dict:
    """导出单个模型"""
    results = {
        'port': port_name,
        'stage': stage_name,
        'source_model': model_path,
        'exports': {}
    }
    
    try:
        # 初始化推理运行器
        runner = InferenceRunner(port_name, device="cpu")
        
        # 加载PyTorch模型
        success = runner.load_pytorch_model(stage_name, model_path)
        if not success:
            results['error'] = "PyTorch模型加载失败"
            return results
        
        # 创建示例输入
        sample_input = create_sample_input(port_name, stage_name)
        
        # 创建输出目录
        port_output_dir = output_dir / port_name
        port_output_dir.mkdir(parents=True, exist_ok=True)
        
        # 导出不同格式
        if 'torchscript' in formats:
            torchscript_path = port_output_dir / f"stage_{stage_name}_best.pt"
            success = runner.export_torchscript(stage_name, str(torchscript_path), sample_input)
            results['exports']['torchscript'] = {
                'success': success,
                'path': str(torchscript_path) if success else None
            }
        
        if 'onnx' in formats:
            onnx_path = port_output_dir / f"stage_{stage_name}_best.onnx"
            success = runner.export_onnx(stage_name, str(onnx_path), sample_input)
            results['exports']['onnx'] = {
                'success': success,
                'path': str(onnx_path) if success else None
            }
        
        logger.info(f"✅ 完成导出: {port_name}/{stage_name}")
        
    except Exception as e:
        logger.error(f"❌ 导出失败 {port_name}/{stage_name}: {e}")
        results['error'] = str(e)
    
    return results

def export_all_models(output_dir: str, formats: List[str]) -> Dict:
    """批量导出所有模型"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 获取所有模型路径
    model_paths = get_all_model_paths()
    
    export_results = {
        'total_models': 0,
        'successful_exports': 0,
        'failed_exports': 0,
        'formats': formats,
        'output_directory': str(output_path),
        'results': []
    }
    
    # 逐个导出模型
    for port_name, stages in model_paths.items():
        for stage_name, model_path in stages.items():
            export_results['total_models'] += 1
            
            logger.info(f"导出模型: {port_name}/{stage_name}")
            result = export_single_model(port_name, stage_name, model_path, 
                                       output_path, formats)
            
            export_results['results'].append(result)
            
            # 统计成功/失败
            if 'error' in result:
                export_results['failed_exports'] += 1
            else:
                # 检查是否有成功的导出
                has_success = any(
                    export_info.get('success', False) 
                    for export_info in result['exports'].values()
                )
                if has_success:
                    export_results['successful_exports'] += 1
                else:
                    export_results['failed_exports'] += 1
    
    return export_results

def create_deployment_manifest(export_results: Dict, output_dir: str):
    """创建部署清单"""
    manifest = {
        'version': '1.0',
        'export_date': torch.utils.data.get_worker_info(),  # 获取时间戳的替代方法
        'summary': {
            'total_models': export_results['total_models'],
            'successful_exports': export_results['successful_exports'],
            'failed_exports': export_results['failed_exports'],
            'formats': export_results['formats']
        },
        'models': {}
    }
    
    # 组织模型信息
    for result in export_results['results']:
        if 'error' in result:
            continue
            
        port = result['port']
        stage = result['stage']
        
        if port not in manifest['models']:
            manifest['models'][port] = {}
        
        manifest['models'][port][stage] = {
            'source_model': result['source_model'],
            'exports': {}
        }
        
        for format_name, export_info in result['exports'].items():
            if export_info.get('success', False):
                manifest['models'][port][stage]['exports'][format_name] = export_info['path']
    
    # 保存清单
    manifest_path = Path(output_dir) / "deployment_manifest.json"
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    
    logger.info(f"部署清单已保存: {manifest_path}")
    return manifest_path

def main():
    parser = argparse.ArgumentParser(description="批量导出模型")
    parser.add_argument("--output-dir", default="../../exports", 
                       help="导出目录")
    parser.add_argument("--formats", nargs="+", 
                       choices=["torchscript", "onnx"],
                       default=["torchscript", "onnx"],
                       help="导出格式")
    parser.add_argument("--port", help="指定港口（可选）")
    parser.add_argument("--stage", help="指定阶段（可选，需要同时指定港口）")
    
    args = parser.parse_args()
    
    logger.info("开始批量模型导出...")
    logger.info(f"输出目录: {args.output_dir}")
    logger.info(f"导出格式: {args.formats}")
    
    if args.port and args.stage:
        # 导出单个模型
        model_paths = get_all_model_paths()
        if args.port not in model_paths or args.stage not in model_paths[args.port]:
            logger.error(f"未找到模型: {args.port}/{args.stage}")
            sys.exit(1)
        
        model_path = model_paths[args.port][args.stage]
        output_path = Path(args.output_dir)
        
        result = export_single_model(args.port, args.stage, model_path,
                                   output_path, args.formats)
        
        print("导出结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
        
    else:
        # 批量导出所有模型
        export_results = export_all_models(args.output_dir, args.formats)
        
        # 创建部署清单
        manifest_path = create_deployment_manifest(export_results, args.output_dir)
        
        # 显示结果摘要
        print("\n" + "="*50)
        print("📦 模型导出完成")
        print("="*50)
        print(f"总模型数: {export_results['total_models']}")
        print(f"成功导出: {export_results['successful_exports']}")
        print(f"失败导出: {export_results['failed_exports']}")
        print(f"导出格式: {', '.join(args.formats)}")
        print(f"输出目录: {args.output_dir}")
        print(f"部署清单: {manifest_path}")
        
        # 显示失败的导出
        failed_exports = [r for r in export_results['results'] if 'error' in r]
        if failed_exports:
            print(f"\n❌ 失败的导出 ({len(failed_exports)}):")
            for result in failed_exports:
                print(f"  - {result['port']}/{result['stage']}: {result['error']}")
        
        print("="*50)

if __name__ == "__main__":
    main()