#!/usr/bin/env python3
"""
统一分子生成器 - 主入口文件
Unified Molecule Generator - Main Entry Point

整合三种分子生成方法：
1. 基础方法 (Base) - 传统遗传算法 + PPO强化学习
2. 增强方法 (Enhanced) - 添加亲和力预测和混合策略
3. 智能方法 (Intelligent) - 集成药物相似性检查和智能决策

使用方法:
python generate_molecules.py --method base --num_molecules 100
python generate_molecules.py --method enhanced --num_molecules 100 --protein_sequence "SEQUENCE"
python generate_molecules.py --method intelligent --num_molecules 100 --enable_druglikeness
"""

import os
import sys
import argparse
import importlib.util
from pathlib import Path

def parse_args():
    parser = argparse.ArgumentParser(description='统一分子生成器')
    
    # 生成方法选择
    parser.add_argument('--method', type=str, default='intelligent',
                       choices=['base', 'enhanced', 'intelligent'],
                       help='选择分子生成方法 (default: intelligent)')
    
    # 基本参数
    parser.add_argument('--num_molecules', type=int, default=100,
                       help='生成分子数量 (default: 100)')
    parser.add_argument('--samples_per_molecule', type=int, default=10,
                       help='每个分子的样本数 (default: 10)')
    parser.add_argument('--output_dir', type=str, default='output',
                       help='输出目录 (default: output)')
    
    # 分子参数
    parser.add_argument('--min_atoms', type=int, default=15,
                       help='最小原子数 (default: 15)')
    parser.add_argument('--max_atoms', type=int, default=50,
                       help='最大原子数 (default: 50)')
    parser.add_argument('--min_fitness', type=float, default=0.5,
                       help='最小适应度 (default: 0.5)')
    
    # 生成策略
    parser.add_argument('--generation_method', type=str, default='hybrid',
                       choices=['traditional', 'ppo', 'hybrid'],
                       help='生成策略 (default: hybrid)')
    
    # PPO参数
    parser.add_argument('--ppo_episodes', type=int, default=1000,
                       help='PPO训练轮数 (default: 1000)')
    parser.add_argument('--ppo_lr', type=float, default=3e-4,
                       help='PPO学习率 (default: 3e-4)')
    parser.add_argument('--use_ppo', action='store_true', default=False,
                       help='是否使用PPO强化学习算法')
    parser.add_argument('--use_hybrid', action='store_true', default=True,
                       help='是否使用混合方法（传统+强化学习）')
    parser.add_argument('--use_affinity', action='store_true', default=True,
                       help='是否使用亲和力约束')
    parser.add_argument('--ppo_steps', type=int, default=10,
                       help='PPO每个分子的最大步数')
    parser.add_argument('--ppo_model_path', type=str, default='ppo/model.pt',
                       help='PPO模型保存路径')
    parser.add_argument('--sanitize_mols', action='store_true', default=True,
                       help='是否对生成的分子进行结构修复')
    parser.add_argument('--skip_3d_opt', action='store_true', default=False,
                       help='是否跳过3D结构优化')
    parser.add_argument('--druglikeness_threshold', type=float, default=0.5,
                       help='可成药性分数阈值')
    parser.add_argument('--strict_druglikeness', action='store_true', default=True,
                       help='是否使用严格的可成药性过滤')
    
    # 蛋白质序列 (Enhanced和Intelligent方法)
    parser.add_argument('--protein_sequence', type=str, 
                       default="DIQMTQSPSSLSASVGDRVTITCKASQNVRTVVAWYQQKPGKAPKTLIYLASNRHTGVPSRFSGSGSGTDFTLTISSLQPEDFATYFCLQHWSYPLTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC",
                       help='目标蛋白质序列 (用于亲和力预测)')
    
    # 药物相似性检查 (Intelligent方法)
    parser.add_argument('--enable_druglikeness', action='store_true',
                       help='启用药物相似性检查 (仅Intelligent方法)')
    
    # 参考分子
    parser.add_argument('--reference_smiles', type=str, default=None,
                       help='参考分子SMILES (可选)')
    parser.add_argument('--reference_file', type=str, default='data/chemical_smiles.csv',
                       help='参考分子SMILES文件')
    
    # 其他选项
    parser.add_argument('--seed', type=int, default=42,
                       help='随机种子 (default: 42)')
    parser.add_argument('--verbose', action='store_true',
                       help='详细输出')
    
    return parser.parse_args()

def load_generator_module(method: str):
    """动态加载对应的生成器模块"""
    module_map = {
        'base': 'generate_molecules_base.py',
        'enhanced': 'generate_molecules_Enhanced.py',
        'intelligent': 'generate_molecules_Intelligent.py'
    }
    
    if method not in module_map:
        raise ValueError(f"未知的生成方法: {method}")
    
    module_path = Path(__file__).parent / module_map[method]
    if not module_path.exists():
        raise FileNotFoundError(f"生成器模块不存在: {module_path}")
    
    # 动态导入模块
    spec = importlib.util.spec_from_file_location(f"generator_{method}", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    
    return module

def print_method_info(method: str):
    """打印生成方法信息"""
    method_info = {
        'base': {
            'name': '基础分子生成方法',
            'features': [
                '传统遗传算法优化',
                'PPO强化学习辅助',
                '基本分子属性评估',
                '适合快速原型开发'
            ]
        },
        'enhanced': {
            'name': '增强分子生成方法',
            'features': [
                '集成亲和力预测',
                '混合生成策略',
                '多目标优化',
                '适合药物发现应用'
            ]
        },
        'intelligent': {
            'name': '智能分子生成方法',
            'features': [
                '药物相似性检查',
                '智能决策系统',
                '多层次优化',
                '适合高质量药物设计'
            ]
        }
    }
    
    info = method_info[method]
    print(f"\n=== {info['name']} ===")
    print("主要特性:")
    for feature in info['features']:
        print(f"  • {feature}")
    print()

def main():
    args = parse_args()
    
    print("=" * 60)
    print("统一分子生成器 - Unified Molecule Generator")
    print("=" * 60)
    
    # 打印方法信息
    print_method_info(args.method)
    
    # 打印参数信息
    print("生成参数:")
    print(f"  • 生成方法: {args.method}")
    print(f"  • 分子数量: {args.num_molecules}")
    print(f"  • 生成策略: {args.generation_method}")
    print(f"  • 输出目录: {args.output_dir}")
    
    if args.protein_sequence:
        print(f"  • 蛋白质序列: {args.protein_sequence[:50]}...")
    
    if args.enable_druglikeness:
        print("  • 药物相似性检查: 启用")
    
    print()
    
    # 验证蛋白质序列参数
    if args.method in ['enhanced', 'intelligent'] and not args.protein_sequence:
        print("错误: Enhanced和Intelligent方法需要提供蛋白质序列参数")
        print("请使用 --protein_sequence 参数提供目标蛋白质序列")
        print("示例: python generate_molecules.py --method enhanced --protein_sequence 'MKTVRQERLKSIVRILERSKEPVSGAQLAEELSVSRQVIVQDIAYLRSLGYNIVATPRGYVLAGG'")
        sys.exit(1)
    
    if args.protein_sequence and len(args.protein_sequence.strip()) == 0:
        print("错误: 蛋白质序列不能为空")
        print("请提供有效的蛋白质氨基酸序列")
        sys.exit(1)
    
    try:
        # 动态加载对应的生成器模块
        print(f"正在加载 {args.method} 生成器模块...")
        generator_module = load_generator_module(args.method)
        
        # 调用生成器的主函数
        print("开始分子生成...")
        generator_module.generate_molecules(args)
        
        print("\n分子生成完成!")
        print(f"结果保存在: {args.output_dir}")
        
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()