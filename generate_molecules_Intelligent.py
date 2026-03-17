import os
import sys
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED, Draw


# 更新导入路径
from molecule_generation.molecule_handler import MoleculeHandler
from molecule_generation.molecule_utils import setup_output_dir, load_reference_smiles, save_molecules, is_valid_mol, fix_molecule
from molecule_generation.fitness_evaluator import evaluate_fitness
from molecule_generation.population_manager import initialize_population, select_parents, process_unique_molecules
from molecule_generation.druglikeness_checker import DruglikenessChecker, check_druglikeness

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
    parser.add_argument('--output_dir', type=str, default='./output/intelligent_method', help='输出目录')
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
    parser.add_argument('--druglikeness_threshold', type=float, default=0.5, help='可成药性分数阈值')
    parser.add_argument('--strict_druglikeness', action='store_true', default=True, help='是否使用严格的可成药性过滤')
    return parser.parse_args()

def calculate_enhanced_fitness(mol, reference_smiles, protein_sequence=None, affinity_predictor=None, druglikeness_checker=None):
    """计算增强版适应度，考虑多个分子性质和与参考分子的相似性，以及亲和力约束和可成药性"""
    if mol is None:
        return 0.0
        
    try:
        # 初始化可成药性检查器
        if druglikeness_checker is None:
            druglikeness_checker = DruglikenessChecker()
        
        # 1. 基础结构有效性检查
        if not is_valid_mol(mol):
            return 0.0
        
        # 2. 可成药性评估
        druglikeness_result = druglikeness_checker.calculate_druglikeness_score(mol)
        druglikeness_score = druglikeness_result['total_score']
        
        # 如果可成药性分数太低，返回一个基础分数而不是极低分数
        if druglikeness_score < 0.2:
            return max(0.3, druglikeness_score * 0.5)  # 提高最低分数
        
        # 3. 基础适应度
        base_fitness = evaluate_fitness(mol, reference_smiles, max_atoms=25, min_atoms=8)
        
        # 4. QED药物类似性
        qed_value = druglikeness_result['qed']
        
        # 5. 分子大小评估（偏好中等大小的分子）
        heavy_atoms = mol.GetNumHeavyAtoms()
        size_score = 1.0
        if heavy_atoms < 10:
            size_score = heavy_atoms / 10 * 0.6  # 小分子得分降低
        elif heavy_atoms > 50:
            size_score = max(0.3, 1.0 - (heavy_atoms - 50) / 20)  # 过大分子得分降低
        
        # 6. 分子复杂性评估
        complexity_score = 1.0
        try:
            # 计算环的数量
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()
            if num_rings == 0:
                complexity_score *= 0.7  # 无环分子得分降低
            elif num_rings > 5:
                complexity_score *= 0.8  # 过多环得分略降低
            
            # 计算芳香性
            aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
            if aromatic_atoms == 0:
                complexity_score *= 0.8  # 无芳香性分子得分降低
            
        except:
            complexity_score = 0.8
        
        # 7. 与参考分子的相似性和差异性平衡
        similarity_score = 0.0
        novelty_score = 1.0
        if reference_smiles:
            from rdkit import DataStructs
            from rdkit.Chem import AllChem
            
            try:
                # 计算当前分子的指纹
                mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                current_smiles = Chem.MolToSmiles(mol)
                
                # 计算与参考分子的相似性
                similarities = []
                for ref_smiles in reference_smiles[:50]:  # 限制计算量
                    try:
                        # 检查是否与参考分子完全相同
                        if current_smiles == ref_smiles:
                            novelty_score = 0.1  # 完全相同的分子新颖性很低
                            similarities.append(1.0)
                            continue
                        
                        ref_mol = Chem.MolFromSmiles(ref_smiles)
                        if ref_mol:
                            ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=1024)
                            sim = DataStructs.TanimotoSimilarity(mol_fp, ref_fp)
                            similarities.append(sim)
                    except:
                        continue
                
                if similarities:
                    max_sim = max(similarities)
                    avg_sim = sum(similarities) / len(similarities)
                    
                    # 相似性分数：希望有一定相似性但不完全相同
                    if max_sim > 0.9:
                        similarity_score = 0.5  # 过于相似
                        novelty_score *= 0.3
                    elif max_sim > 0.7:
                        similarity_score = 0.8  # 适度相似
                        novelty_score *= 0.7
                    elif max_sim > 0.3:
                        similarity_score = 1.0  # 理想相似度
                        novelty_score *= 1.0
                    else:
                        similarity_score = 0.6  # 相似度太低
                        novelty_score *= 1.2  # 但新颖性高
                        
            except:
                similarity_score = 0.5
                novelty_score = 1.0
        
        # 8. 计算亲和力分数
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
                affinity_score = 0.0
        
        # 9. 组合所有得分，调整权重
        if protein_sequence and affinity_predictor:
            # 有亲和力约束时的权重分配
            enhanced_fitness = (
                druglikeness_score * 0.25 +    # 可成药性权重最高
                base_fitness * 0.05 +          # 基础适应度
                qed_value * 0.1 +              # QED
                size_score * 0.05 +            # 分子大小
                complexity_score * 0.05 +      # 分子复杂性
                similarity_score * 0.15 +      # 与参考分子相似性
                novelty_score * 0.1 +          # 新颖性
                affinity_score * 0.25          # 亲和力权重高
            )
        else:
            # 无亲和力约束时的权重分配
            enhanced_fitness = (
                druglikeness_score * 0.3 +     # 可成药性权重最高
                base_fitness * 0.1 +           # 基础适应度
                qed_value * 0.15 +             # QED
                size_score * 0.1 +             # 分子大小
                complexity_score * 0.1 +       # 分子复杂性
                similarity_score * 0.15 +      # 与参考分子相似性
                novelty_score * 0.1            # 新颖性
            )
        
        # 10. 应用可成药性惩罚（降低惩罚力度）
        if not druglikeness_result['is_druglike']:
            enhanced_fitness *= 0.8  # 降低惩罚力度
        
        # 11. 应用PAINS惩罚（降低惩罚力度）
        if druglikeness_result['pains']['is_pains']:
            enhanced_fitness *= 0.6  # 降低惩罚力度
        
        return max(0.0, min(1.0, enhanced_fitness))
        
    except Exception as e:
        print(f"计算增强适应度时出错: {e}")
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
    """使用PPO强化学习算法生成分子 - 优化版本"""
    if initial_molecules:
        print("使用PPO强化学习算法优化传统方法生成的分子...")
    else:
        print("使用PPO强化学习算法生成分子...")
    
    # 初始化可成药性检查器
    druglikeness_checker = DruglikenessChecker()
    
    # 初始化PPO代理 - 使用改进的参数
    from ppo.improved_ppo_model import ImprovedPPOAgent
    agent = ImprovedPPOAgent(
        state_dim=256,  # 增加状态维度
        action_dim=10, 
        hidden_dim=256,  # 增加隐藏层维度
        num_layers=3,    # 增加网络深度
        lr=0.0001,       # 降低学习率以提高稳定性
        gamma=0.95,      # 调整折扣因子
        eps_clip=0.15,   # 调整裁剪参数
        K_epochs=8,      # 增加训练轮数
        entropy_coef=0.02  # 增加熵系数以促进探索
    )
    
    # 设置目标蛋白质序列（默认使用抗体序列，可以根据需要修改）
    if protein_sequence is None:
        protein_sequence = "DIQMTQSPSSLSASVGDRVTITCKASQNVRTVVAWYQQKPGKAPKTLIYLASNRHTGVPSRFSGSGSGTDFTLTISSLQPEDFATYFCLQHWSYPLTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVD"
    
    # 初始化分子环境，集成亲和力计算和可成药性检查
    env = MoleculeEnvironment(
        molecule_handler=molecule_handler,
        max_steps=args.ppo_steps,
        min_atoms=args.min_atoms,
        max_atoms=args.max_atoms,
        reference_smiles=reference_smiles,
        sanitize_mols=args.sanitize_mols,
        skip_3d_opt=args.skip_3d_opt,
        protein_sequence=protein_sequence,
        affinity_weight=0.6,  # 增加亲和力权重
        druglikeness_checker=druglikeness_checker  # 直接传入可成药性检查器
    )
    
    # 如果提供了初始分子，使用它们；否则初始化种群
    if initial_molecules:
        # 过滤初始分子，只保留高质量的分子
        filtered_molecules = []
        for mol_info in initial_molecules:
            # 处理分子信息字典格式
            if isinstance(mol_info, dict) and 'smiles' in mol_info:
                try:
                    mol = Chem.MolFromSmiles(mol_info['smiles'])
                    if mol and is_valid_mol(mol):
                        druglikeness_result = druglikeness_checker.calculate_druglikeness_score(mol)
                        # 只保留可成药性分数较高的分子
                        if druglikeness_result['total_score'] >= 0.4:
                            filtered_molecules.append(mol)
                except:
                    continue
            elif mol_info and is_valid_mol(mol_info):
                # 直接是分子对象
                druglikeness_result = druglikeness_checker.calculate_druglikeness_score(mol_info)
                if druglikeness_result['total_score'] >= 0.4:
                    filtered_molecules.append(mol_info)
        
        population = filtered_molecules[:args.num_molecules] if filtered_molecules else []
        print(f"使用 {len(population)} 个高质量传统方法生成的分子作为初始输入")
        
        # 如果过滤后分子不足，补充一些高质量的参考分子
        if len(population) < args.num_molecules // 2:
            for smiles in reference_smiles[:args.num_molecules]:
                try:
                    ref_mol = Chem.MolFromSmiles(smiles)
                    if ref_mol and is_valid_mol(ref_mol):
                        druglikeness_result = druglikeness_checker.calculate_druglikeness_score(ref_mol)
                        if druglikeness_result['total_score'] >= 0.5:
                            population.append(ref_mol)
                            if len(population) >= args.num_molecules:
                                break
                except:
                    continue
    else:
        # 初始化种群，优先使用参考分子
        population = []
        for smiles in reference_smiles[:args.num_molecules]:
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol and is_valid_mol(mol):
                    population.append(mol)
            except:
                continue
        
        # 确保种群大小足够
        if len(population) < args.num_molecules:
            # 复制现有分子以达到所需大小
            while len(population) < args.num_molecules and population:
                population.append(random.choice(population))
    
    # 生成分子
    final_molecules = []
    all_fitness_history = []
    
    # 创建组目录
    groups_dir = os.path.join(args.output_dir, 'groups')
    os.makedirs(groups_dir, exist_ok=True)
    
    # 添加早停机制和自适应学习
    patience = 5  # 早停耐心值
    best_global_fitness = 0.0
    no_improvement_count = 0
    
    for i in tqdm(range(min(len(population), args.num_molecules)), desc="优化分子" if initial_molecules else "生成分子"):
        # 如果有初始分子，使用对应的分子作为起点；否则从种群中随机选择
        if i < len(population):
            seed_mol = population[i]
            if seed_mol:
                print(f"优化分子 {i+1}: {Chem.MolToSmiles(seed_mol)}")
        else:
            # 从种群中随机选择一个分子作为起点
            if i < len(reference_smiles) and random.random() < 0.7:
                try:
                    seed_mol = Chem.MolFromSmiles(reference_smiles[i])
                    if not (seed_mol and is_valid_mol(seed_mol)):
                        seed_mol = random.choice(population) if population else None
                except:
                    seed_mol = random.choice(population) if population else None
            else:
                seed_mol = random.choice(population) if population else None
        
        if seed_mol is None:
            continue
            
        # 使用PPO生成分子
        candidates = []
        fitness_scores = []
        fitness_history = []
        
        # 自适应回合数
        max_episodes = args.ppo_episodes
        if initial_molecules:
            # 对于优化阶段，增加回合数
            max_episodes = min(args.ppo_episodes * 2, 50)
        
        best_episode_fitness = 0.0
        episode_no_improvement = 0
        
        # 运行多个PPO回合
        for episode in range(max_episodes):
            # 重置环境
            current_mol = env.reset(seed_mol=seed_mol)
            
            # 编码初始状态
            state = agent.encode_state(current_mol)
            
            episode_rewards = []
            episode_candidates = []
            episode_fitness_scores = []
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
                    # 使用改进的适应度计算函数
                    fitness = calculate_enhanced_fitness(
                        current_mol, 
                        reference_smiles, 
                        protein_sequence, 
                        affinity_predictor,
                        druglikeness_checker
                    )
                    fitness_history.append(fitness)
                    
                    # 严格的质量检查
                    druglikeness_result = druglikeness_checker.calculate_druglikeness_score(current_mol)
                    
                    # 只保留高质量分子
                    if (args.min_atoms <= current_mol.GetNumHeavyAtoms() <= args.max_atoms and
                        druglikeness_result['total_score'] >= 0.3 and  # 可成药性阈值
                        not druglikeness_result['pains']['is_pains'] and  # 无PAINS
                        fitness >= 0.2):  # 最低适应度阈值
                        
                        # 创建分子的副本以避免引用问题
                        mol_copy = Chem.Mol(current_mol)
                        episode_candidates.append(mol_copy)
                        episode_fitness_scores.append(fitness)
            
            # 更新PPO代理
            if len(agent.states) > 0:
                agent.update()
            
            # 选择本回合最佳分子
            if episode_fitness_scores:
                best_idx = np.argmax(episode_fitness_scores)
                best_fitness = episode_fitness_scores[best_idx]
                
                # 检查是否有改进
                if best_fitness > best_episode_fitness:
                    best_episode_fitness = best_fitness
                    episode_no_improvement = 0
                    
                    # 添加到总候选列表
                    candidates.extend(episode_candidates)
                    fitness_scores.extend(episode_fitness_scores)
                else:
                    episode_no_improvement += 1
                
                # 打印回合信息
                avg_reward = sum(episode_rewards) / len(episode_rewards) if episode_rewards else 0
                print(f"分子 {i+1}, 回合 {episode+1}: 平均奖励 = {avg_reward:.4f}, 最佳适应度 = {best_fitness:.4f}")
                
                # 早停检查
                if episode_no_improvement >= patience and episode > 10:
                    print(f"分子 {i+1} 在回合 {episode+1} 早停，无改进回合数: {episode_no_improvement}")
                    break
            else:
                episode_no_improvement += 1
        
        # 如果没有找到有效分子，使用种子分子
        if not candidates:
            if seed_mol and is_valid_mol(seed_mol) and args.min_atoms <= seed_mol.GetNumHeavyAtoms() <= args.max_atoms:
                # 检查种子分子的可成药性
                druglikeness_result = druglikeness_checker.calculate_druglikeness_score(seed_mol)
                if druglikeness_result['total_score'] >= 0.2:  # 降低阈值
                    candidates.append(seed_mol)
                    fitness = calculate_enhanced_fitness(seed_mol, reference_smiles, protein_sequence, affinity_predictor, druglikeness_checker)
                    fitness_scores.append(fitness)
                    print(f"使用种子分子 {i+1}, 适应度: {fitness:.4f}")
            
            # 如果仍然没有候选分子，尝试使用参考分子
            if not candidates:
                for smiles in reference_smiles[:10]:
                    try:
                        ref_mol = Chem.MolFromSmiles(smiles)
                        if ref_mol and is_valid_mol(ref_mol) and args.min_atoms <= ref_mol.GetNumHeavyAtoms() <= args.max_atoms:
                            druglikeness_result = druglikeness_checker.calculate_druglikeness_score(ref_mol)
                            if druglikeness_result['total_score'] >= 0.3:
                                candidates.append(ref_mol)
                                fitness = calculate_enhanced_fitness(ref_mol, reference_smiles, protein_sequence, affinity_predictor, druglikeness_checker)
                                fitness_scores.append(fitness)
                                print(f"使用参考分子作为分子 {i+1}, 适应度: {fitness:.4f}")
                                break
                    except:
                        continue
        
        # 选择最佳候选分子
        if candidates and fitness_scores:
            # 去重并选择最佳分子
            unique_candidates = []
            unique_fitness = []
            seen_smiles = set()
            
            for mol, fitness in zip(candidates, fitness_scores):
                try:
                    smiles = Chem.MolToSmiles(mol)
                    if smiles not in seen_smiles:
                        seen_smiles.add(smiles)
                        unique_candidates.append(mol)
                        unique_fitness.append(fitness)
                except:
                    continue
            
            if unique_candidates:
                # 选择适应度最高的分子
                best_idx = np.argmax(unique_fitness)
                best_mol = unique_candidates[best_idx]
                best_fitness = unique_fitness[best_idx]
                
                # 创建分子信息字典
                mol_info = {
                    'smiles': Chem.MolToSmiles(best_mol),
                    'fitness': best_fitness,
                    'mol_weight': Descriptors.MolWt(best_mol),
                    'logp': Descriptors.MolLogP(best_mol),
                    'qed': QED.qed(best_mol),
                    'heavy_atoms': best_mol.GetNumHeavyAtoms(),
                    'druglikeness_score': druglikeness_checker.calculate_druglikeness_score(best_mol)['total_score']
                }
                
                # 添加亲和力信息
                if protein_sequence and affinity_predictor:
                    try:
                        affinity = affinity_predictor.predict_affinity(mol_info['smiles'], protein_sequence)
                        mol_info['affinity'] = affinity
                    except:
                        mol_info['affinity'] = None
                
                final_molecules.append(mol_info)
                all_fitness_history.append(best_fitness)
                
                # 更新全局最佳适应度
                if best_fitness > best_global_fitness:
                    best_global_fitness = best_fitness
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                
                print(f"分子 {i+1} 完成，最佳适应度: {best_fitness:.4f}")
                
                # 保存当前组的分子
                save_top_molecules(unique_candidates, unique_fitness, groups_dir, i+1)
        
        # 全局早停检查
        if no_improvement_count >= patience * 2 and i > 10:
            print(f"全局早停在分子 {i+1}，无改进次数: {no_improvement_count}")
            break
    
    # 最终结果处理
    if final_molecules:
        print(f"\nPPO生成完成，共生成 {len(final_molecules)} 个分子")
        print(f"平均适应度: {np.mean(all_fitness_history):.4f}")
        print(f"最佳适应度: {max(all_fitness_history):.4f}")
        
        # 按适应度排序
        final_molecules.sort(key=lambda x: x['fitness'], reverse=True)
        
        return final_molecules
    else:
        print("PPO未能生成有效分子")
        return []

def generate_molecules_traditional(num_molecules, samples_per_molecule, output_dir, reference_smiles, 
                                 molecule_handler, min_atoms=8, max_atoms=32, min_fitness=1, 
                                 protein_sequence=None, affinity_predictor=None, args=None):
    """传统方法生成分子，增强可成药性和分子质量控制"""
    print("开始传统分子生成...")
    
    # 初始化可成药性检查器
    druglikeness_checker = DruglikenessChecker()
    
    # 初始化种群，包含参考分子和高质量默认分子
    population = []
    
    # 首先添加参考分子
    if reference_smiles:
        for smiles in reference_smiles[:20]:  # 限制数量
            try:
                mol = Chem.MolFromSmiles(smiles)
                if mol and is_valid_mol(mol):
                    population.append(smiles)
            except:
                continue
    
    # 添加一些高质量的药物类似分子作为种子
    drug_like_seeds = [
        # 经典药物分子
        "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # 布洛芬
        "CC1=CC=C(C=C1)C(=O)C2=CC=C(C=C2)N(C)C",  # 他莫昔芬类似物
        "COC1=CC=C(C=C1)CCN2CCC(CC2)C3=NOC4=C3C=CC(=C4)F",  # 利培酮类似物
        "CC1=C(C=C(C=C1)NC(=O)C2=CC=C(C=C2)CN3CCN(CC3)C)C",  # 伊马替尼类似物
        "CN1CCN(CC1)C2=C(C=C3C(=C2)N=CN=C3NC4=CC(=C(C=C4)F)Cl)OC",  # 吉非替尼类似物
        
        # 常见药物骨架
        "CC(=O)NC1=CC=C(C=C1)O",  # 对乙酰氨基酚
        "CC1=CC=CC=C1C(=O)O",  # 甲苯甲酸
        "NC1=CC=C(C=C1)S(=O)(=O)N",  # 磺胺
        "CC1=CC=C(C=C1)C(C)C",  # 异丙基甲苯
        "COC1=CC=C(C=C1)CCN",  # 甲氧基苯乙胺
        
        # 杂环化合物
        "C1=CC=NC=C1",  # 吡啶
        "C1=CN=CC=C1",  # 嘧啶
        "C1=CC=C2C(=C1)C=CC=N2",  # 喹啉
        "C1=CC=C2C(=C1)NC=N2",  # 苯并咪唑
        "C1=CC=C2C(=C1)C=CS2",  # 苯并噻吩
        
        # 含氮杂环
        "C1CCNCC1",  # 哌啶
        "C1COCCN1",  # 吗啉
        "C1CN=CC=C1",  # 哌嗪
        "C1=CC=C2C(=C1)C=CN2",  # 吲哚
        "C1=CC2=C(C=C1)NC=N2",  # 苯并咪唑
        
        # 含氧杂环
        "C1COCCO1",  # 二氧六环
        "C1=CC=C2C(=C1)C=CO2",  # 苯并呋喃
        "COC1=CC=CC=C1",  # 苯甲醚
        "C1=CC=C(C=C1)OC",  # 甲氧基苯
        
        # 含硫化合物
        "C1=CC=C(C=C1)S",  # 苯硫酚
        "CSC",  # 二甲基硫
        "C1=CC=C2C(=C1)C=CS2",  # 苯并噻吩
        
        # 简单起始分子
        "CCO",  # 乙醇
        "CC(=O)O",  # 醋酸
        "CCN",  # 乙胺
        "C1CCCCC1",  # 环己烷
        "c1ccccc1",  # 苯
        "CC(C)O",  # 异丙醇
        "CCCC",  # 丁烷
        "CC(C)C",  # 异丁烷
        "CCC(C)C",  # 异戊烷
        "CCCCO",  # 丁醇
        
        # 功能基团载体
        "CC(=O)C",  # 丙酮
        "CC(=O)N",  # 乙酰胺
        "CCS",  # 乙硫醇
        "CCCl",  # 氯乙烷
        "CCF",  # 氟乙烷
        "CC#C",  # 丙炔
        "C=CC",  # 丙烯
        
        # 双环化合物
        "C1CC2CCCC2C1",  # 十氢萘
        "C1=CC2=C(C=C1)C=CC=C2",  # 萘
        "C1CCC2CCCCC2C1",  # 双环癸烷
        
        # 含多个杂原子的化合物
        "NCCN",  # 乙二胺
        "OCCCO",  # 丙二醇
        "NC(=O)N",  # 尿素
        "OC(=O)O",  # 碳酸
        "NC(=S)N",  # 硫脲
    ]
    
    for seed_smiles in drug_like_seeds:
        try:
            # 保持原始SMILES字符串用于比较
            original_smiles = seed_smiles
            seed_mol = Chem.MolFromSmiles(seed_smiles)
            if seed_mol and is_valid_mol(seed_mol):
                # 降低可成药性要求，确保有足够的种子分子
                druglikeness_result = druglikeness_checker.calculate_druglikeness_score(seed_mol)
                if druglikeness_result['total_score'] >= 0.2:  # 降低阈值
                    population.append(original_smiles)
                else:
                    # 即使不满足可成药性，也添加一些简单分子作为起点
                    if original_smiles in ["CCO", "CC(=O)O", "CCN", "C1CCCCC1", "c1ccccc1"]:
                        population.append(original_smiles)
        except:
            # 如果出错，仍然添加简单分子
            if seed_smiles in ["CCO", "CC(=O)O", "CCN", "C1CCCCC1"]:
                population.append(seed_smiles)
    
    # 去重
    population = list(set(population))
    print(f"初始种群大小: {len(population)}")
    
    generated_molecules = []
    unique_smiles = set()
    
    # 设置可成药性阈值
    druglikeness_threshold = getattr(args, 'druglikeness_threshold', 0.5) if args else 0.5
    strict_druglikeness = getattr(args, 'strict_druglikeness', True) if args else True
    
    # 生成分子的主循环
    for i in tqdm(range(num_molecules), desc="生成分子"):
        best_mol = None
        best_fitness = 0
        best_smiles = None
        
        print(f"\n开始生成第 {i+1} 个分子...")
        
        # 增强多样性策略：动态调整种子选择
        diversity_factor = min(1.0, i / (num_molecules * 0.3))  # 前30%使用多样性策略
        
        # 对每个分子进行多次采样
        for j in range(samples_per_molecule):
            try:
                # 多样性种子选择策略
                if population:
                    if diversity_factor > 0.5 and len(population) > 10:
                        # 高多样性阶段：优先选择不同类型的种子
                        if j % 3 == 0:
                            # 选择复杂分子（原子数较多）
                            complex_seeds = [s for s in population if Chem.MolFromSmiles(s) and 
                                           Chem.MolFromSmiles(s).GetNumHeavyAtoms() > (min_atoms + max_atoms) // 2]
                            seed_smiles = random.choice(complex_seeds) if complex_seeds else random.choice(population)
                        elif j % 3 == 1:
                            # 选择简单分子（原子数较少）
                            simple_seeds = [s for s in population if Chem.MolFromSmiles(s) and 
                                          Chem.MolFromSmiles(s).GetNumHeavyAtoms() <= (min_atoms + max_atoms) // 2]
                            seed_smiles = random.choice(simple_seeds) if simple_seeds else random.choice(population)
                        else:
                            # 随机选择
                            seed_smiles = random.choice(population)
                    else:
                        # 标准随机选择
                        seed_smiles = random.choice(population)
                    
                    seed_mol = Chem.MolFromSmiles(seed_smiles)
                    if seed_mol is None:
                        continue
                else:
                    # 如果种群为空，使用默认分子
                    seed_mol = Chem.MolFromSmiles("CCO")  # 乙醇作为最简单的起始分子
                
                # 多样性变异策略
                mutation_intensity = 1.0
                if diversity_factor > 0.3:
                    # 高多样性阶段使用更强的变异
                    mutation_intensity = 1.5 + random.random() * 0.5
                
                # 生成新分子（标准变异）
                new_mol = molecule_handler.mutate_molecule(seed_mol, max_atoms, min_atoms)
                
                if new_mol is None:
                    continue
                
                # 基础有效性检查
                if not is_valid_mol(new_mol):
                    # 尝试修复分子
                    if args and getattr(args, 'sanitize_mols', True):
                        new_mol = fix_molecule(new_mol)
                        if new_mol is None or not is_valid_mol(new_mol):
                            continue
                    else:
                        continue
                
                # 检查分子大小
                heavy_atoms = new_mol.GetNumHeavyAtoms()
                if heavy_atoms < min_atoms or heavy_atoms > max_atoms:
                    print(f"  分子大小不符合要求: {heavy_atoms} 原子")
                    continue
                
                # 可成药性预筛选
                druglikeness_result = druglikeness_checker.calculate_druglikeness_score(new_mol)
                
                # 动态调整可成药性阈值
                dynamic_threshold = 0.2
                if diversity_factor > 0.5:
                    # 高多样性阶段进一步降低阈值
                    dynamic_threshold = 0.1
                
                # 严格可成药性过滤（进一步降低严格性）
                if strict_druglikeness:
                    if druglikeness_result['total_score'] < dynamic_threshold:
                        print(f"  可成药性分数过低: {druglikeness_result['total_score']:.3f}")
                        continue
                
                # 检查分子唯一性
                new_smiles = Chem.MolToSmiles(new_mol)
                if new_smiles in unique_smiles:
                    continue
                
                # 增强的与参考分子差异性检查
                is_too_similar = False
                if reference_smiles:
                    from rdkit import DataStructs
                    from rdkit.Chem import AllChem
                    
                    try:
                        mol_fp = AllChem.GetMorganFingerprintAsBitVect(new_mol, 2, nBits=1024)
                        
                        # 动态调整相似性阈值
                        similarity_threshold = 0.95
                        if diversity_factor > 0.3:
                            # 高多样性阶段使用更严格的差异性要求
                            similarity_threshold = 0.85
                        
                        for ref_smiles in reference_smiles[:30]:  # 检查更多参考分子
                            if new_smiles == ref_smiles:
                                is_too_similar = True
                                break
                            
                            ref_mol = Chem.MolFromSmiles(ref_smiles)
                            if ref_mol:
                                ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=1024)
                                sim = DataStructs.TanimotoSimilarity(mol_fp, ref_fp)
                                if sim > similarity_threshold:
                                    is_too_similar = True
                                    print(f"  与参考分子过于相似: {sim:.3f}")
                                    break
                        
                        # 检查与已生成分子的差异性
                        if not is_too_similar and len(unique_smiles) > 0:
                            for existing_smiles in list(unique_smiles)[-20:]:  # 检查最近生成的20个分子
                                existing_mol = Chem.MolFromSmiles(existing_smiles)
                                if existing_mol:
                                    existing_fp = AllChem.GetMorganFingerprintAsBitVect(existing_mol, 2, nBits=1024)
                                    sim = DataStructs.TanimotoSimilarity(mol_fp, existing_fp)
                                    if sim > 0.90:  # 与已生成分子的相似性阈值
                                        is_too_similar = True
                                        print(f"  与已生成分子过于相似: {sim:.3f}")
                                        break
                                        
                    except Exception as e:
                        print(f"相似性计算出错: {e}")
                        pass
                
                if is_too_similar:
                    continue
                
                # 计算增强适应度
                fitness = calculate_enhanced_fitness(
                    new_mol, reference_smiles, protein_sequence, 
                    affinity_predictor, druglikeness_checker
                )
                
                print(f"  样本 {j+1}: 适应度 = {fitness:.4f}")
                
                # 动态适应度阈值策略（降低阈值以支持更多复杂分子）
                base_threshold = 0.05  # 降低基础阈值
                if diversity_factor > 0.5:
                    # 高多样性阶段进一步降低适应度阈值
                    dynamic_fitness_threshold = base_threshold * 0.3  # 进一步降低
                else:
                    # 后期适度提高质量要求，但仍保持较低阈值
                    dynamic_fitness_threshold = base_threshold * (0.8 + diversity_factor * 0.5)
                
                # 检查适应度阈值
                if fitness < dynamic_fitness_threshold:
                    print(f"  适应度过低: {fitness:.4f} < {dynamic_fitness_threshold:.4f}")
                    continue
                
                # 质量加权评分：结合多个指标
                quality_score = fitness
                
                # 分子复杂度奖励（鼓励更复杂的分子）
                complexity_bonus = 0
                if heavy_atoms >= min_atoms + 5:  # 奖励较复杂的分子
                    complexity_bonus = 0.1
                elif heavy_atoms >= min_atoms + 10:  # 更高奖励给更复杂的分子
                    complexity_bonus = 0.15
                
                # 只对过于简单的分子进行轻微惩罚
                complexity_penalty = 0
                if heavy_atoms < min_atoms + 2:
                    complexity_penalty = 0.05  # 减少对简单分子的惩罚
                
                quality_score += complexity_bonus - complexity_penalty
                
                # 可成药性奖励
                druglikeness_bonus = 0
                if druglikeness_result['total_score'] > 0.5:
                    druglikeness_bonus = 0.1
                elif druglikeness_result['total_score'] > 0.3:
                    druglikeness_bonus = 0.05
                
                quality_score += druglikeness_bonus
                
                # 新颖性奖励（与参考分子差异大的奖励）
                novelty_bonus = 0
                if reference_smiles:
                    try:
                        mol_fp = AllChem.GetMorganFingerprintAsBitVect(new_mol, 2, nBits=1024)
                        min_similarity = 1.0
                        for ref_smiles in reference_smiles[:10]:
                            ref_mol = Chem.MolFromSmiles(ref_smiles)
                            if ref_mol:
                                ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=1024)
                                sim = DataStructs.TanimotoSimilarity(mol_fp, ref_fp)
                                min_similarity = min(min_similarity, sim)
                        
                        # 相似性越低，新颖性奖励越高
                        if min_similarity < 0.5:
                            novelty_bonus = 0.15
                        elif min_similarity < 0.7:
                            novelty_bonus = 0.1
                        elif min_similarity < 0.85:
                            novelty_bonus = 0.05
                    except:
                        pass
                
                quality_score += novelty_bonus
                
                print(f"  质量评分: {quality_score:.4f} (基础:{fitness:.4f}, 复杂度:-{complexity_penalty:.3f}, 可成药性:+{druglikeness_bonus:.3f}, 新颖性:+{novelty_bonus:.3f})")
                
                # 更新最佳分子（使用质量评分）
                if quality_score > best_fitness:
                    best_fitness = quality_score
                    best_mol = new_mol
                    best_smiles = new_smiles
                    
            except Exception as e:
                print(f"生成分子时出错: {e}")
                continue
        
        # 保存最佳分子
        if best_mol is not None and best_smiles is not None:
            unique_smiles.add(best_smiles)
            
            # 计算详细的分子信息
            mol_info = {
                'smiles': best_smiles,
                'fitness': best_fitness,
                'mol_weight': Descriptors.MolWt(best_mol),
                'logp': Descriptors.MolLogP(best_mol),
                'qed': QED.qed(best_mol),
                'heavy_atoms': best_mol.GetNumHeavyAtoms(),
                'druglikeness_score': druglikeness_checker.calculate_druglikeness_score(best_mol)['total_score']
            }
            
            # 添加亲和力信息
            if protein_sequence and affinity_predictor:
                try:
                    affinity = affinity_predictor.predict_affinity(best_smiles, protein_sequence)
                    mol_info['affinity'] = affinity
                except:
                    mol_info['affinity'] = None
            
            generated_molecules.append(mol_info)
            
            # 将高质量分子添加到种群中用于后续生成
            if best_fitness > min_fitness * 1.2:  # 只添加高质量分子
                population.append(best_smiles)
                # 限制种群大小
                if len(population) > 200:
                    population = population[-150:]  # 保留最新的150个
    
    print(f"传统方法生成了 {len(generated_molecules)} 个分子")
    
    # 按适应度排序
    generated_molecules.sort(key=lambda x: x['fitness'], reverse=True)
    
    # 创建分类输出文件夹结构
    import os
    from datetime import datetime
    
    # 创建主输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(output_dir, f"traditional_molecules_{timestamp}")
    os.makedirs(main_output_dir, exist_ok=True)
    
    # 创建子目录
    high_quality_dir = os.path.join(main_output_dir, "high_quality")  # 适应度 > 0.7
    medium_quality_dir = os.path.join(main_output_dir, "medium_quality")  # 0.3 < 适应度 <= 0.7
    low_quality_dir = os.path.join(main_output_dir, "low_quality")  # 适应度 <= 0.3
    novel_molecules_dir = os.path.join(main_output_dir, "novel_molecules")  # 新颖分子
    druglike_dir = os.path.join(main_output_dir, "druglike")  # 高可成药性分子
    
    for dir_path in [high_quality_dir, medium_quality_dir, low_quality_dir, novel_molecules_dir, druglike_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # 分类保存分子
    high_quality_mols = []
    medium_quality_mols = []
    low_quality_mols = []
    novel_mols = []
    druglike_mols = []
    
    for mol_info in generated_molecules:
        fitness = mol_info['fitness']
        druglikeness = mol_info['druglikeness_score']
        
        # 按适应度分类
        if fitness > 0.7:
            high_quality_mols.append(mol_info)
        elif fitness > 0.3:
            medium_quality_mols.append(mol_info)
        else:
            low_quality_mols.append(mol_info)
        
        # 高可成药性分子
        if druglikeness > 0.6:
            druglike_mols.append(mol_info)
        
        # 新颖分子（与参考分子差异大）
        if reference_smiles:
            try:
                mol = Chem.MolFromSmiles(mol_info['smiles'])
                if mol:
                    mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                    max_similarity = 0.0
                    for ref_smiles in reference_smiles[:20]:
                        ref_mol = Chem.MolFromSmiles(ref_smiles)
                        if ref_mol:
                            ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=1024)
                            sim = DataStructs.TanimotoSimilarity(mol_fp, ref_fp)
                            max_similarity = max(max_similarity, sim)
                    
                    if max_similarity < 0.6:  # 新颖性阈值
                        novel_mols.append(mol_info)
            except:
                pass
    
    # 保存分类结果
    def save_molecules_to_category(molecules, category_dir, category_name):
        if not molecules:
            return
        
        # 保存SMILES文件
        smiles_file = os.path.join(category_dir, f"{category_name}_molecules.smi")
        with open(smiles_file, 'w', encoding='utf-8') as f:
            f.write("SMILES\tFitness\tMW\tLogP\tQED\tHeavyAtoms\tDruglikeness\tAffinity\n")
            for mol_info in molecules:
                affinity_str = f"{mol_info.get('affinity', 'N/A')}"
                f.write(f"{mol_info['smiles']}\t{mol_info['fitness']:.4f}\t"
                       f"{mol_info['mol_weight']:.2f}\t{mol_info['logp']:.2f}\t"
                       f"{mol_info['qed']:.3f}\t{mol_info['heavy_atoms']}\t"
                       f"{mol_info['druglikeness_score']:.3f}\t{affinity_str}\n")
        
        # 保存详细信息JSON
        import json
        json_file = os.path.join(category_dir, f"{category_name}_details.json")
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(molecules, f, indent=2, ensure_ascii=False)
        
        # 生成分子图像
        images_subdir = os.path.join(category_dir, "images")
        os.makedirs(images_subdir, exist_ok=True)
        
        for i, mol_info in enumerate(molecules[:20]):  # 只为前20个分子生成图像
            try:
                mol = Chem.MolFromSmiles(mol_info['smiles'])
                if mol:
                    img_path = os.path.join(images_subdir, f"{category_name}_{i+1:03d}.png")
                    Draw.MolToFile(mol, img_path, size=(300, 300))
            except Exception as e:
                print(f"生成图像失败 {mol_info['smiles']}: {e}")
        
        print(f"保存了 {len(molecules)} 个{category_name}分子到 {category_dir}")
    
    # 执行分类保存
    save_molecules_to_category(high_quality_mols, high_quality_dir, "high_quality")
    save_molecules_to_category(medium_quality_mols, medium_quality_dir, "medium_quality")
    save_molecules_to_category(low_quality_mols, low_quality_dir, "low_quality")
    save_molecules_to_category(novel_mols, novel_molecules_dir, "novel")
    save_molecules_to_category(druglike_mols, druglike_dir, "druglike")
    
    # 保存总体统计信息
    stats_file = os.path.join(main_output_dir, "generation_statistics.txt")
    with open(stats_file, 'w', encoding='utf-8') as f:
        f.write(f"传统分子生成统计报告\n")
        f.write(f"生成时间: {timestamp}\n")
        f.write(f"总生成分子数: {len(generated_molecules)}\n")
        f.write(f"高质量分子数 (适应度>0.7): {len(high_quality_mols)}\n")
        f.write(f"中等质量分子数 (0.3<适应度<=0.7): {len(medium_quality_mols)}\n")
        f.write(f"低质量分子数 (适应度<=0.3): {len(low_quality_mols)}\n")
        f.write(f"新颖分子数 (与参考分子相似性<0.6): {len(novel_mols)}\n")
        f.write(f"高可成药性分子数 (可成药性>0.6): {len(druglike_mols)}\n")
        f.write(f"\n适应度分布:\n")
        if generated_molecules:
            f.write(f"最高适应度: {generated_molecules[0]['fitness']:.4f}\n")
            f.write(f"最低适应度: {generated_molecules[-1]['fitness']:.4f}\n")
            avg_fitness = sum(mol['fitness'] for mol in generated_molecules) / len(generated_molecules)
            f.write(f"平均适应度: {avg_fitness:.4f}\n")
    
    print(f"所有结果已保存到: {main_output_dir}")
    print(f"生成统计: 高质量({len(high_quality_mols)}) | 中等质量({len(medium_quality_mols)}) | 低质量({len(low_quality_mols)}) | 新颖({len(novel_mols)}) | 可成药({len(druglike_mols)})")
    
    return generated_molecules

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
        num_molecules=traditional_args.num_molecules,
        samples_per_molecule=traditional_args.samples_per_molecule,
        output_dir=traditional_dir,
        reference_smiles=reference_smiles,
        molecule_handler=molecule_handler,
        min_atoms=traditional_args.min_atoms,
        max_atoms=traditional_args.max_atoms,
        min_fitness=traditional_args.min_fitness,
        protein_sequence=protein_sequence,
        affinity_predictor=affinity_predictor,
        args=traditional_args
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
        protein_sequence = "DIQMTQSPSSLSASVGDRVTITCKASQNVRTVVAWYQQKPGKAPKTLIYLASNRHTGVPSRFSGSGSGTDFTLTISSLQPEDFATYFCLQHWSYPLTFGQGTKVEIKRTVAAPSVFIFPPSDEQLKSGTASVVCLLNNFYPREAKVQWKVDNALQSGNSQESVTEQDSKDSTYSLSSTLTLSKADYEKHKVYACEVTHQGLSSPVTKSFNRGEC"
    
    # 根据选择的方法生成分子
    if hasattr(args, 'use_hybrid') and args.use_hybrid:
        # 使用混合方法
        best_molecules = generate_molecules_hybrid(args, reference_smiles, molecule_handler, images_dir, protein_sequence)
    elif args.use_ppo:
        # 使用PPO强化学习算法生成分子
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
        
        best_molecules = generate_molecules_with_ppo(args, reference_smiles, molecule_handler, images_dir, 
                                                   protein_sequence=protein_sequence, affinity_predictor=affinity_predictor)
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
        
        best_molecules = generate_molecules_traditional(
            num_molecules=args.num_molecules,
            samples_per_molecule=args.samples_per_molecule,
            output_dir=args.output_dir,
            reference_smiles=reference_smiles,
            molecule_handler=molecule_handler,
            min_atoms=args.min_atoms,
            max_atoms=args.max_atoms,
            min_fitness=args.min_fitness,
            protein_sequence=protein_sequence,
            affinity_predictor=affinity_predictor,
            args=args
        )
    
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
    save_molecules(best_molecules, args.output_dir, images_dir, reference_smiles, max_atoms=args.max_atoms, min_atoms=args.min_atoms)

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