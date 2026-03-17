import os
import sys
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED


# 更新导入路径
from molecule_generation.molecule_handler import MoleculeHandler
from molecule_generation.molecule_utils import setup_output_dir, load_reference_smiles, save_molecules, is_valid_mol, fix_molecule
from molecule_generation.fitness_evaluator import evaluate_fitness
from molecule_generation.population_manager import initialize_population, select_parents, process_unique_molecules

# 导入PPO相关模块
from ppo.ppo_model import PPOAgent, MoleculeEnvironment

# 导入亲和力计算模块（注释掉旧的导入）
# sys.path.append('HitScreen')
# from drug_protein_affinity_calculator import DrugProteinAffinityCalculator

# 添加 DeepDTA 路径
sys.path.append('DeepDTA')
from affinity_predictor import AffinityPredictor

def parse_args():
    parser = argparse.ArgumentParser(description='分子生成器')
    parser.add_argument('--num_molecules', type=int, default=100, help='要生成的分子数量')
    parser.add_argument('--samples_per_molecule', type=int, default=5, help='每个分子的采样次数')
    parser.add_argument('--output_dir', type=str, default='./output/enhanced_method', help='输出目录')
    parser.add_argument('--seed', type=int, default=42, help='随机种子')
    parser.add_argument('--reference_file', type=str, default='data/chemical_smiles.csv', help='参考分子SMILES文件')
    parser.add_argument('--min_atoms', type=int, default=8, help='分子的最小重原子数量')
    parser.add_argument('--max_atoms', type=int, default=32, help='分子的最大重原子数量')
    parser.add_argument('--min_fitness', type=float, default=1, help='最小适应度阈值')
    parser.add_argument('--use_ppo', action='store_true', default= False, help='是否使用PPO强化学习算法')
    parser.add_argument('--use_hybrid', action='store_true', default=True, help='是否使用混合方法（传统+强化学习）')
    parser.add_argument('--use_affinity', action='store_true', default=True, help='是否使用亲和力约束')
    parser.add_argument('--protein_sequence', type=str, default="DIQMTQSPSSLSASVGDRVTITCKASQNVRTVVAWYQQKPGKAPKTLIYLASNRHTGVPSRFSGSGSGTDFTLTISSLQPEDFATYFCLQHWSYPLTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC", help='目标蛋白质序列')
    parser.add_argument('--ppo_steps', type=int, default=10, help='PPO每个分子的最大步数')
    parser.add_argument('--ppo_episodes', type=int, default=35, help='PPO训练的回合数')
    parser.add_argument('--ppo_model_path', type=str, default='ppo/model.pt', help='PPO模型保存路径')
    parser.add_argument('--sanitize_mols', action='store_true', default=True, help='是否对生成的分子进行结构修复')
    parser.add_argument('--skip_3d_opt', action='store_true', default=False, help='是否跳过3D结构优化')
    return parser.parse_args()

def calculate_enhanced_fitness(mol, reference_smiles, protein_sequence=None, affinity_predictor=None):
    """计算增强版适应度，考虑多个分子性质和与参考分子的相似性，以及亲和力约束"""
    if mol is None:
        return 0.0
        
    try:
        # 基础适应度
        base_fitness = evaluate_fitness(mol, reference_smiles)
        
        # 计算QED药物类似性
        qed_value = QED.qed(mol)
        
        # 计算分子量，偏好较大的分子
        mol_weight = Descriptors.MolWt(mol)
        weight_score = 0.0
        if mol_weight < 100:
            weight_score = mol_weight / 100 * 0.5  # 小分子得分降低
        elif mol_weight < 200:
            weight_score = mol_weight / 200 * 0.8  # 中等分子得分适中
        else:
            weight_score = min(1.0, mol_weight / 300)  # 大分子得分提高
            
        # 计算LogP (油水分配系数)，偏好适中的LogP值
        logp = Descriptors.MolLogP(mol)
        logp_score = 1.0
        if abs(logp) > 5:
            logp_score = max(0.2, 1.0 - (abs(logp) - 5) / 5)
            
        # 计算环的数量，偏好有环结构的分子
        ring_count = Chem.GetSSSR(mol)
        ring_score = min(1.0, len(ring_count) / 2)  # 提高环结构的权重
        
        # 计算芳香性，偏好有芳香环的分子
        aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
        aromatic_score = min(1.0, aromatic_atoms / 6)
        
        # 计算与参考分子的相似性
        similarity_score = 0.0
        if reference_smiles:
            from rdkit import DataStructs
            from rdkit.Chem import AllChem
            
            # 计算当前分子的指纹
            mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            
            # 计算与每个参考分子的相似性，取最大值
            max_sim = 0.0
            for ref_smiles in reference_smiles[:20]:  # 限制计算量，只使用前20个参考分子
                try:
                    ref_mol = Chem.MolFromSmiles(ref_smiles)
                    if ref_mol:
                        ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=1024)
                        sim = DataStructs.TanimotoSimilarity(mol_fp, ref_fp)
                        max_sim = max(max_sim, sim)
                except:
                    continue
            
            similarity_score = max_sim
        
        # 计算亲和力分数
        affinity_score = 0.0
        if protein_sequence and affinity_predictor:
            try:
                smiles = Chem.MolToSmiles(mol)
                affinity_value = affinity_predictor.predict_affinity(smiles, protein_sequence)
                # 假设pKd范围为4-12，标准化到0-1
                normalized_affinity = max(0.0, min(1.0, (affinity_value - 4.0) / 8.0))
                affinity_score = normalized_affinity
                
                # 对高亲和力给予额外奖励
                if normalized_affinity > 0.8:
                    affinity_score += 0.2
                    
            except Exception as e:
                print(f"计算亲和力时出错: {e}")
                affinity_score = 0.0
        
        # 组合所有得分，调整权重以包含亲和力
        if protein_sequence and affinity_predictor:
            # 有亲和力约束时的权重分配
            enhanced_fitness = (
                base_fitness * 0.1 +
                qed_value * 0.15 +
                weight_score * 0.1 +
                logp_score * 0.05 +
                ring_score * 0.05 +
                aromatic_score * 0.05 +
                similarity_score * 0.2 +
                affinity_score * 0.3  # 亲和力权重最高
            )
        else:
            # 无亲和力约束时的原始权重
            enhanced_fitness = (
                base_fitness * 0.2 +
                qed_value * 0.2 +
                weight_score * 0.2 +  # 提高分子大小权重
                logp_score * 0.1 +
                ring_score * 0.1 +
                aromatic_score * 0.1 +
                similarity_score * 0.3  # 提高与参考分子相似性的权重
            )
        
        return enhanced_fitness
    except:
        return 0.1

def save_top_molecules(molecules, fitness_scores, output_dir, group_id):
    """
    保存一组分子中排名前5的分子
    
    参数:
    molecules: 分子列表
    fitness_scores: 对应的适应度分数列表
    output_dir: 输出目录
    group_id: 分子组的ID
    """
    if not molecules:
        return
        
    # 创建组目录
    group_dir = os.path.join(output_dir, f'group_{group_id}')
    os.makedirs(group_dir, exist_ok=True)
    
    # 创建图像目录
    images_dir = os.path.join(group_dir, 'images')
    os.makedirs(images_dir, exist_ok=True)
    
    # 将分子和适应度打包并排序
    mol_fitness_pairs = list(zip(molecules, fitness_scores))
    mol_fitness_pairs.sort(key=lambda x: x[1], reverse=True)  # 按适应度降序排序
    
    # 取前5个或全部（如果不足5个）
    top_pairs = mol_fitness_pairs[:min(5, len(mol_fitness_pairs))]
    
    # 保存SMILES和适应度
    smiles_file = os.path.join(group_dir, 'top_molecules.smi')
    with open(smiles_file, 'w') as f:
        for i, (mol, fitness) in enumerate(top_pairs):
            try:
                smiles = Chem.MolToSmiles(mol)
                f.write(f"{smiles}\t{fitness:.4f}\n")
                
                # 保存分子图像
                img_file = os.path.join(images_dir, f'molecule_{i+1}.png')
                Chem.Draw.MolToFile(mol, img_file)
            except Exception as e:
                print(f"保存分子时出错: {e}")
    
    print(f"组 {group_id} 的前 {len(top_pairs)} 个分子已保存到 {group_dir}")
    return [mol for mol, _ in top_pairs]

def generate_molecules_with_ppo(args, reference_smiles, molecule_handler, images_dir, initial_molecules=None, protein_sequence=None, affinity_predictor=None):
    """使用PPO强化学习算法生成分子"""
    if initial_molecules:
        print("使用PPO强化学习算法优化传统方法生成的分子...")
    else:
        print("使用PPO强化学习算法生成分子...")
    
    # 初始化PPO代理
    agent = PPOAgent(state_dim=128, action_dim=10, hidden_dim=128)
    
    # 设置目标蛋白质序列（默认使用抗体序列，可以根据需要修改）
    if protein_sequence is None:
        protein_sequence = "DIQMTQSPSSLSASVGDRVTITCKASQNVRTVVAWYQQKPGKAPKTLIYLASNRHTGVPSRFSGSGSGTDFTLTISSLQPEDFATYFCLQHWSYPLTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVD"
    
    # 初始化分子环境，集成亲和力计算
    env = MoleculeEnvironment(
        molecule_handler=molecule_handler,
        max_steps=args.ppo_steps,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        reference_smiles=reference_smiles,
        sanitize_mols=args.sanitize_mols,
        skip_3d_opt=args.skip_3d_opt,
        protein_sequence=protein_sequence,  # 添加蛋白质序列
        affinity_weight=0.5  # 增加亲和力权重
    )
    
    # 如果提供了初始分子，使用它们；否则初始化种群
    if initial_molecules:
        population = initial_molecules[:args.num_molecules]  # 使用传统方法生成的分子
        print(f"使用 {len(population)} 个传统方法生成的分子作为初始输入")
    else:
        # 初始化种群，优先使用参考分子
        population = initialize_population(size=args.num_molecules, reference_smiles=reference_smiles)
        
        # 确保种群大小足够
        if len(population) < args.num_molecules:
            # 复制现有分子以达到所需大小
            while len(population) < args.num_molecules:
                population.append(random.choice(population))
    
    # 生成分子
    final_molecules = []
    all_fitness_history = []
    
    # 创建组目录
    groups_dir = os.path.join(args.output_dir, 'groups')
    os.makedirs(groups_dir, exist_ok=True)
    
    for i in tqdm(range(len(population)), desc="优化分子" if initial_molecules else "生成分子"):
        # 如果有初始分子，使用对应的分子作为起点；否则从种群中随机选择
        if initial_molecules and i < len(population):
            seed_mol = population[i]
            print(f"优化分子 {i+1}: {Chem.MolToSmiles(seed_mol) if seed_mol else 'None'}")
        else:
            # 从种群中随机选择一个分子作为起点
            if i < len(reference_smiles) and random.random() < 0.7:
                try:
                    seed_mol = Chem.MolFromSmiles(reference_smiles[i])
                    if not (seed_mol and is_valid_mol(seed_mol)):
                        seed_mol = random.choice(population)
                except:
                    seed_mol = random.choice(population)
            else:
                seed_mol = random.choice(population)
        
        # 使用PPO生成分子
        candidates = []
        fitness_scores = []
        fitness_history = []
        
        # 运行多个PPO回合
        for episode in range(args.ppo_episodes):
            # 重置环境
            current_mol = env.reset(seed_mol=seed_mol)
            
            # 编码初始状态
            state = agent.encode_state(current_mol)
            
            episode_rewards = []
            done = False
            step = 0
            
            # 执行PPO步骤
            while not done and step < args.ppo_steps:
                # 选择动作
                action, log_prob, value = agent.get_action(state)
                
                # 执行动作
                next_mol, reward, done = env.step(action)
                
                # 编码下一个状态
                next_state = agent.encode_state(next_mol)
                
                # 存储转换
                agent.store_transition(state, action, reward, next_state, log_prob, value, done)
                
                # 更新状态
                state = next_state
                current_mol = next_mol
                episode_rewards.append(reward)
                step += 1
                
                # 计算当前分子的适应度
                if current_mol is not None and is_valid_mol(current_mol):
                    fitness = calculate_enhanced_fitness(current_mol, reference_smiles, protein_sequence, affinity_predictor)
                    fitness_history.append(fitness)
                    
                    # 添加到候选列表
                    if args.min_atoms <= current_mol.GetNumHeavyAtoms() <= args.max_atoms:
                        # 创建分子的副本以避免引用问题
                        mol_copy = Chem.Mol(current_mol)
                        candidates.append(mol_copy)
                        fitness_scores.append(fitness)
            
            # 更新PPO代理
            if len(agent.states) > 0:
                agent.update()
            
            # 打印回合信息
            if episode_rewards:
                avg_reward = sum(episode_rewards) / len(episode_rewards)
                best_fitness = max(fitness_scores) if fitness_scores else 0
                print(f"分子 {i+1}, 回合 {episode+1}: 平均奖励 = {avg_reward:.4f}, 最佳适应度 = {best_fitness:.4f}")
        
        # 如果没有找到有效分子，使用种子分子
        if not candidates:
            if seed_mol and is_valid_mol(seed_mol) and args.min_atoms <= seed_mol.GetNumHeavyAtoms() <= args.max_atoms:
                candidates.append(seed_mol)
                fitness = calculate_enhanced_fitness(seed_mol, reference_smiles, protein_sequence, affinity_predictor)
                fitness_scores.append(fitness)
                print(f"使用种子分子 {i+1}, 适应度: {fitness:.4f}")
            else:
                # 尝试使用参考分子
                for smiles in reference_smiles[:10]:
                    try:
                        ref_mol = Chem.MolFromSmiles(smiles)
                        if ref_mol and is_valid_mol(ref_mol) and args.min_atoms <= ref_mol.GetNumHeavyAtoms() <= args.max_atoms:
                            candidates.append(ref_mol)
                            fitness = calculate_enhanced_fitness(ref_mol, reference_smiles, protein_sequence, affinity_predictor)
                            fitness_scores.append(fitness)
                            print(f"使用参考分子, 适应度: {fitness:.4f}")
                            break
                    except:
                        continue
        
        # 保存这一组的前5个分子
        if candidates:
            # 保存前5个分子
            top_mols = save_top_molecules(candidates, fitness_scores, groups_dir, i+1)
            
            # 添加最佳分子到最终列表
            if top_mols:
                final_molecules.append(top_mols[0])  # 添加最佳分子
                all_fitness_history.extend(fitness_history)
                
                # 实时保存和评估最佳分子
                best_mol = top_mols[0]
                smiles = Chem.MolToSmiles(best_mol)
                best_fitness = max(fitness_scores) if fitness_scores else 0
                print(f"生成分子 {i+1}, SMILES: {smiles}, 适应度: {best_fitness:.4f}, 重原子数: {best_mol.GetNumHeavyAtoms()}")
                
                # 保存分子图像到主图像目录
                img_file = os.path.join(images_dir, f'molecule_{i+1}.png')
                Chem.Draw.MolToFile(best_mol, img_file)
    
    # 保存PPO模型
    os.makedirs(os.path.dirname(args.ppo_model_path), exist_ok=True)
    agent.save_model(args.ppo_model_path)
    print(f"PPO模型已保存到 {args.ppo_model_path}")
    
    # 处理最终分子集合，确保分子唯一且有效
    best_molecules = process_unique_molecules(
        final_molecules, 
        reference_smiles=reference_smiles, 
        target_size=args.num_molecules,
        max_atoms=args.max_atoms,
        min_atoms=args.min_atoms
    )
    
    # 输出结果
    print(f"\n生成了 {len(best_molecules)} 个分子")
    
    # 保存分子SMILES
    smiles_file = os.path.join(args.output_dir, 'molecules.smi')
    with open(smiles_file, 'w') as f:
        for mol in best_molecules:
            if mol:
                try:
                    smiles = Chem.MolToSmiles(mol)
                    f.write(f"{smiles}\n")
                except:
                    pass
    
    print(f"分子SMILES已保存到 {smiles_file}")
    
    return best_molecules

def generate_molecules_traditional(args, reference_smiles, molecule_handler, images_dir, protein_sequence=None, affinity_predictor=None):
    """使用传统方法生成分子，支持亲和力约束"""
    print("使用传统方法生成分子...")
    
    # 初始化种群，优先使用参考分子
    population = []
    
    # 创建组目录
    groups_dir = os.path.join(args.output_dir, 'groups')
    os.makedirs(groups_dir, exist_ok=True)
    
    # 首先尝试使用参考分子
    if reference_smiles:
        for smiles in reference_smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol and is_valid_mol(mol) and args.min_atoms <= mol.GetNumHeavyAtoms() <= args.max_atoms:
                    # 添加3D构象
                    try:
                        mol = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(mol, randomSeed=42)
                        mol = Chem.RemoveHs(mol)
                    except:
                        pass
                    population.append(mol)
                    if len(population) >= args.num_molecules * 2:
                        break
            except:
                continue
    
    # 如果参考分子不足，添加高质量的起始分子
    if len(population) < args.num_molecules:
        high_quality_smiles = [
            'c1ccccc1',  # 苯
            'CC(=O)O',   # 乙酸
            'CCO',       # 乙醇
            'c1ccccc1O', # 苯酚
            'C1CCCCC1',  # 环己烷
            'CC(=O)N',   # 乙酰胺
            'CCN',       # 乙胺
            'CC=O',      # 乙醛
            'CCOC(=O)C', # 乙酸乙酯
            'c1cccnc1',  # 吡啶
            'c1ccncc1',  # 吡嗪
            'c1ccc2ccccc2c1', # 萘
            'c1ccc(cc1)C(=O)O', # 苯甲酸
            'c1ccc(cc1)C(=O)N', # 苯甲酰胺
            'c1ccc(cc1)CCO',    # 苯乙醇
            'c1ccc(cc1)Oc2ccccc2',  # 二苯醚
            'c1ccc(cc1)Cc2ccccc2',  # 二苯甲烷
            'c1ccc(cc1)C(=O)c2ccccc2', # 二苯甲酮
            'c1ccc(cc1)c2ccccc2',    # 联苯
            'c1ccc2c(c1)ccc3c2cccc3', # 蒽
            'c1ccc2c(c1)c3ccccc3cc2', # 菲
            'c1ccc2c(c1)ccc3ccc4ccccc4c3c2', # 苯并[a]蒽
            'c1ccc2c(c1)ccc3c2ccc4c3cccc4', # 四环芳烃
        ]
        
        for smiles in high_quality_smiles:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol and is_valid_mol(mol) and args.min_atoms <= mol.GetNumHeavyAtoms() <= args.max_atoms:
                    # 添加3D构象
                    try:
                        mol = Chem.AddHs(mol)
                        AllChem.EmbedMolecule(mol, randomSeed=42)
                        mol = Chem.RemoveHs(mol)
                    except:
                        pass
                    population.append(mol)
            except:
                continue
    
    # 确保种群大小足够
    if len(population) < args.num_molecules:
        # 复制现有分子以达到所需大小
        while len(population) < args.num_molecules:
            population.append(random.choice(population))
    
    # 生成分子
    final_molecules = []
    
    for i in tqdm(range(args.num_molecules), desc="生成分子"):
        # 为每个目标分子生成多个候选分子
        candidates = []
        fitness_scores = []
        
        # 从种群中随机选择一个分子作为起点，优先选择参考分子
        if i < len(reference_smiles) and random.random() < 0.7:
            try:
                seed_mol = Chem.MolFromSmiles(reference_smiles[i])
                if not (seed_mol and is_valid_mol(seed_mol)):
                    seed_mol = random.choice(population)
            except:
                seed_mol = random.choice(population)
        else:
            seed_mol = random.choice(population)
        
        # 生成多个候选分子
        for j in range(args.samples_per_molecule):
            try:
                # 生成一个新分子
                new_mol = molecule_handler.mutate_molecule(seed_mol, max_atoms=args.max_atoms)
                
                # 验证生成的分子是否有效且符合大小要求
                if (new_mol is not None and is_valid_mol(new_mol) and 
                    args.min_atoms <= new_mol.GetNumHeavyAtoms() <= args.max_atoms):
                    # 确保分子不是单原子
                    smiles = Chem.MolToSmiles(new_mol)
                    if len(smiles) > 1:
                        # 计算适应度，包含亲和力约束
                        fitness = calculate_enhanced_fitness(new_mol, reference_smiles, protein_sequence, affinity_predictor)
                        
                        # 实时评估分子
                        affinity_info = ""
                        if protein_sequence and affinity_predictor:
                            try:
                                affinity_value = affinity_predictor.predict_affinity(smiles, protein_sequence)
                                affinity_info = f", 亲和力: {affinity_value:.2f}"
                            except:
                                pass
                        
                        print(f"分子 {i+1}, 候选 {j+1}, 适应度: {fitness:.4f}{affinity_info}, SMILES: {smiles}")
                        
                        # 只保留适应度高于阈值的分子
                        if fitness >= args.min_fitness:
                            # 创建分子的副本以避免引用问题
                            mol_copy = Chem.Mol(new_mol)
                            candidates.append(mol_copy)
                            fitness_scores.append(fitness)
            except Exception as e:
                continue
        
        # 如果没有生成有效的候选分子，尝试使用参考分子
        if not candidates:
            try:
                # 尝试使用参考分子
                if i < len(reference_smiles):
                    ref_mol = Chem.MolFromSmiles(reference_smiles[i])
                    if ref_mol and is_valid_mol(ref_mol) and args.min_atoms <= ref_mol.GetNumHeavyAtoms() <= args.max_atoms:
                        candidates.append(ref_mol)
                        fitness = calculate_enhanced_fitness(ref_mol, reference_smiles, protein_sequence, affinity_predictor)
                        fitness_scores.append(fitness)
                        print(f"使用参考分子 {i+1}, 适应度: {fitness:.4f}")
                
                # 如果没有合适的参考分子，使用种子分子
                if not candidates and seed_mol and is_valid_mol(seed_mol) and args.min_atoms <= seed_mol.GetNumHeavyAtoms() <= args.max_atoms:
                    candidates.append(seed_mol)
                    fitness = calculate_enhanced_fitness(seed_mol, reference_smiles)
                    fitness_scores.append(fitness)
                    print(f"使用种子分子 {i+1}, 适应度: {fitness:.4f}")
                    
                # 如果种子分子也不合适，使用高质量分子
                if not candidates:
                    for smiles in high_quality_smiles:
                        mol = Chem.MolFromSmiles(smiles)
                        if mol and is_valid_mol(mol) and args.min_atoms <= mol.GetNumHeavyAtoms() <= args.max_atoms:
                            candidates.append(mol)
                            fitness = calculate_enhanced_fitness(mol, reference_smiles)
                            fitness_scores.append(fitness)
                            print(f"使用高质量分子 {i+1}, 适应度: {fitness:.4f}")
                            break
            except:
                # 如果所有尝试都失败，跳过这个分子
                continue
        
        # 保存这一组的前5个分子
        if candidates:
            # 保存前5个分子
            top_mols = save_top_molecules(candidates, fitness_scores, groups_dir, i+1)
            
            # 添加最佳分子到最终列表
            if top_mols:
                final_molecules.append(top_mols[0])  # 添加最佳分子
                
                # 实时保存和评估最佳分子
                best_mol = top_mols[0]
                best_idx = fitness_scores.index(max(fitness_scores))
                best_fitness = fitness_scores[best_idx]
                
                print(f"生成分子 {i+1}，适应度: {best_fitness:.4f}, 重原子数: {best_mol.GetNumHeavyAtoms()}")
                
                # 保存分子图像到主图像目录
                img_file = os.path.join(images_dir, f'molecule_{i+1}.png')
                Chem.Draw.MolToFile(best_mol, img_file)
    
    return final_molecules

def generate_molecules_hybrid(args, reference_smiles, molecule_handler, images_dir, protein_sequence=None):
    """混合方法：先用传统方法生成分子，再用强化学习优化"""
    print("使用混合方法生成分子...")
    print("第一阶段：使用传统遗传算法生成初始分子")
    
    # 初始化亲和力预测器
    affinity_predictor = None
    if protein_sequence:
        try:
            affinity_predictor = AffinityPredictor(
                model_path='DeepDTA/deepdta_retrain-prk12-ldk8.pt',
                ligand_dict_path='DeepDTA/ligand_dict-prk12-ldk8.json',
                protein_dict_path='DeepDTA/protein_dict-prk12-ldk8.json'
            )
            affinity_predictor._load_dictionaries()
            affinity_predictor._load_model()
            print("亲和力预测器初始化成功")
        except Exception as e:
            print(f"亲和力预测器初始化失败: {e}")
            affinity_predictor = None
    
    # 第一阶段：使用传统方法生成100个分子
    traditional_args = argparse.Namespace(**vars(args))
    traditional_args.num_molecules = 100  # 生成100个分子
    
    # 创建传统方法的输出目录
    traditional_dir = os.path.join(args.output_dir, 'traditional_stage')
    os.makedirs(traditional_dir, exist_ok=True)
    traditional_images_dir = os.path.join(traditional_dir, 'images')
    os.makedirs(traditional_images_dir, exist_ok=True)
    
    # 使用传统方法生成分子
    traditional_molecules = generate_molecules_traditional(
        traditional_args, 
        reference_smiles, 
        molecule_handler, 
        traditional_images_dir,
        protein_sequence=protein_sequence,
        affinity_predictor=affinity_predictor
    )
    
    print(f"传统方法生成了 {len(traditional_molecules)} 个分子")
    
    # 第二阶段：使用强化学习优化这些分子
    print("\n第二阶段：使用PPO强化学习优化分子")
    
    # 创建强化学习的输出目录
    rl_dir = os.path.join(args.output_dir, 'rl_stage')
    os.makedirs(rl_dir, exist_ok=True)
    rl_images_dir = os.path.join(rl_dir, 'images')
    os.makedirs(rl_images_dir, exist_ok=True)
    
    # 调整强化学习参数以适应优化任务
    rl_args = argparse.Namespace(**vars(args))
    rl_args.num_molecules = min(len(traditional_molecules), args.num_molecules)
    rl_args.ppo_episodes = max(5, args.ppo_episodes // 2)  # 减少回合数，因为有好的起点
    rl_args.output_dir = rl_dir
    
    # 使用强化学习优化传统方法生成的分子
    optimized_molecules = generate_molecules_with_ppo(
        rl_args,
        reference_smiles,
        molecule_handler,
        rl_images_dir,
        initial_molecules=traditional_molecules,
        protein_sequence=protein_sequence,
        affinity_predictor=affinity_predictor
    )
    
    print(f"强化学习优化后得到 {len(optimized_molecules)} 个分子")
    
    # 第三阶段：评估和选择最佳分子
    print("\n第三阶段：评估和选择最佳分子")
    
    # 计算所有分子的适应度
    all_molecules = []
    all_fitness_scores = []
    
    # 评估传统方法生成的分子
    for mol in traditional_molecules:
        if mol and is_valid_mol(mol):
            fitness = calculate_enhanced_fitness(mol, reference_smiles, protein_sequence, affinity_predictor)
            all_molecules.append(mol)
            all_fitness_scores.append(fitness)
    
    # 评估强化学习优化的分子
    for mol in optimized_molecules:
        if mol and is_valid_mol(mol):
            fitness = calculate_enhanced_fitness(mol, reference_smiles, protein_sequence, affinity_predictor)
            all_molecules.append(mol)
            all_fitness_scores.append(fitness)
    
    # 选择最佳分子
    if all_molecules:
        # 按适应度排序
        sorted_pairs = sorted(zip(all_molecules, all_fitness_scores), key=lambda x: x[1], reverse=True)
        
        # 选择前N个最佳分子
        best_molecules = []
        seen_smiles = set()
        
        for mol, fitness in sorted_pairs:
            if len(best_molecules) >= args.num_molecules:
                break
                
            try:
                smiles = Chem.MolToSmiles(mol)
                if smiles not in seen_smiles:
                    best_molecules.append(mol)
                    seen_smiles.add(smiles)
                    print(f"选择分子: {smiles}, 适应度: {fitness:.4f}")
            except:
                continue
        
        # 保存最终结果
        final_images_dir = os.path.join(args.output_dir, 'final_molecules')
        os.makedirs(final_images_dir, exist_ok=True)
        
        for i, mol in enumerate(best_molecules):
            img_file = os.path.join(final_images_dir, f'final_molecule_{i+1}.png')
            Chem.Draw.MolToFile(mol, img_file)
        
        print(f"\n混合方法最终生成了 {len(best_molecules)} 个高质量分子")
        return best_molecules
    
    return []

def generate_molecules(args):
    """生成分子的主函数"""
    # 设置随机种子
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # 创建输出目录
    images_dir = setup_output_dir(args.output_dir)
    
    # 加载参考SMILES
    reference_file = os.path.join(os.path.dirname(__file__), args.reference_file)
    reference_smiles = load_reference_smiles(reference_file)
    
    # 初始化分子处理器
    molecule_handler = MoleculeHandler()
    
    # 设置蛋白质序列（如果需要亲和力约束）
    protein_sequence = getattr(args, 'protein_sequence', None)
    if not protein_sequence and hasattr(args, 'use_affinity') and args.use_affinity:
        # 使用默认的蛋白质序列
        protein_sequence = "DIQMTQSPSSLSASVGDRVTITCKASQNVRTVVAWYQQKPGKAPKTLIYLASNRHTGVPSRFSGSGSGTDFTLTISSLQPEDFATYFCLQHWSYPLTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVD"
    
    # 根据选择的方法生成分子
    if hasattr(args, 'use_hybrid') and args.use_hybrid:
        # 使用混合方法
        best_molecules = generate_molecules_hybrid(args, reference_smiles, molecule_handler, images_dir, protein_sequence)
    elif args.use_ppo:
        # 使用PPO强化学习算法生成分子
        best_molecules = generate_molecules_with_ppo(args, reference_smiles, molecule_handler, images_dir, 
                                                   protein_sequence=protein_sequence)
    else:
        # 使用传统方法生成分子
        # 初始化亲和力预测器
        affinity_predictor = None
        if protein_sequence:
            try:
                affinity_predictor = AffinityPredictor(
                    model_path='DeepDTA/deepdta_retrain-prk12-ldk8.pt',
                    ligand_dict_path='DeepDTA/ligand_dict-prk12-ldk8.json',
                    protein_dict_path='DeepDTA/protein_dict-prk12-ldk8.json'
                )
                affinity_predictor._load_dictionaries()
                affinity_predictor._load_model()
                print("亲和力预测器初始化成功")
            except Exception as e:
                print(f"亲和力预测器初始化失败: {e}")
                affinity_predictor = None
        
        best_molecules = generate_molecules_traditional(args, reference_smiles, molecule_handler, images_dir,
                                                      protein_sequence=protein_sequence, affinity_predictor=affinity_predictor)
    
    # 处理最终分子集合，确保分子唯一且有效
    best_molecules = process_unique_molecules(
        best_molecules, 
        reference_smiles=reference_smiles, 
        target_size=args.num_molecules,
        max_atoms=args.max_atoms,
        min_atoms=args.min_atoms
    )
    
    # 输出结果
    print(f"\n生成了 {len(best_molecules)} 个分子")
    
    # 保存分子图像和SMILES
    save_molecules(best_molecules, args.output_dir, images_dir, reference_smiles)

def calculate_drug_protein_affinity(smiles, protein_sequence="DIQMTQSPSSLSASVGDRVTITCKASQNVRTVVAWYQQKPGKAPKTLIYLASNRHTGVPSRFSGSGSGTDFTLTISSLQPEDFATYFCLQHWSYPLTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"):
    """
    计算药物-蛋白质亲和力
    
    Args:
        smiles (str): 药物分子的SMILES字符串
        protein_sequence (str): 蛋白质氨基酸序列
        
    Returns:
        float: 预测的亲和力分数，如果计算失败返回None
    """
    try:
        # 使用 DeepDTA 的 AffinityPredictor
        predictor = AffinityPredictor(
            model_path='DeepDTA/deepdta_retrain-prk12-ldk8.pt',
            ligand_dict_path='DeepDTA/ligand_dict-prk12-ldk8.json',
            protein_dict_path='DeepDTA/protein_dict-prk12-ldk8.json'
        )
        
        # 加载字典和模型
        predictor._load_dictionaries()
        predictor._load_model()
        
        # 预测亲和力
        affinity_score = predictor.predict_affinity(smiles, protein_sequence)
        
        return affinity_score
        
    except Exception as e:
        print(f"计算亲和力时发生错误: {e}")
        return None


if __name__ == "__main__":
    args = parse_args()
    generate_molecules(args)