import random
from rdkit import Chem
from rdkit.Chem import AllChem
from .molecule_utils import is_valid_mol

def initialize_population(size=20, reference_smiles=None):
    """初始化分子种群，优先使用参考SMILES"""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from .molecule_utils import is_valid_mol
    
    population = []
    
    # 如果有参考SMILES，优先使用
    if reference_smiles and len(reference_smiles) > 0:
        # 确保初始种群多样性
        used_smiles = set()
        
        # 先尝试使用参考SMILES
        for smiles in reference_smiles:
            if len(population) >= size:
                break
                
            if smiles not in used_smiles:
                try:
                    # 使用sanitize选项确保分子有效
                    mol = Chem.MolFromSmiles(smiles, sanitize=True)
                    if mol and is_valid_mol(mol):
                        # 添加氢原子并生成3D构象
                        mol = Chem.AddHs(mol)
                        try:
                            # 使用更稳健的3D构象生成方法
                            AllChem.EmbedMolecule(mol, randomSeed=random.randint(1, 1000), 
                                                maxAttempts=50, useRandomCoords=True)
                            # 尝试优化，但如果失败也不要中断
                            try:
                                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
                            except:
                                pass
                            
                            # 再次验证分子有效性
                            if is_valid_mol(mol):
                                population.append(mol)
                                used_smiles.add(smiles)
                        except:
                            # 如果3D构象生成失败，仍然添加分子
                            if is_valid_mol(mol):
                                population.append(mol)
                                used_smiles.add(smiles)
                except Exception as e:
                    pass
    
    # 如果参考SMILES不足，使用默认的起始分子
    if len(population) < size:
        # 一些简单的起始分子SMILES
        starter_smiles = [
            'C', 'CC', 'CCC', 'CCCC', 'CCO', 'CCN', 'C1CCCCC1',
            'CC(=O)O', 'CCO', 'CCN', 'CS', 'C=C'
        ]
        
        # 确保初始种群多样性
        used_smiles = set()
        
        # 先尝试生成不重复的分子
        for _ in range(min(size - len(population), len(starter_smiles))):
            # 随机选择一个未使用的SMILES
            available_smiles = [s for s in starter_smiles if s not in used_smiles]
            if not available_smiles:
                break
                
            smiles = random.choice(available_smiles)
            used_smiles.add(smiles)
            
            mol = Chem.MolFromSmiles(smiles, sanitize=True)
            if mol and is_valid_mol(mol):
                # 添加氢原子并生成3D构象
                mol = Chem.AddHs(mol)
                try:
                    AllChem.EmbedMolecule(mol, randomSeed=random.randint(1, 1000))
                    AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
                    
                    # 验证分子有效性
                    if is_valid_mol(mol):
                        population.append(mol)
                except:
                    # 如果3D构象生成失败，仍然添加分子
                    if is_valid_mol(mol):
                        population.append(mol)
    
    # 如果还需要更多分子，允许重复选择
    while len(population) < size:
        smiles = random.choice(starter_smiles)
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
        if mol and is_valid_mol(mol):
            # 添加氢原子并生成3D构象
            mol = Chem.AddHs(mol)
            try:
                AllChem.EmbedMolecule(mol, randomSeed=random.randint(1, 1000))
                AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
                
                # 验证分子有效性
                if is_valid_mol(mol):
                    population.append(mol)
            except:
                # 如果3D构象生成失败，仍然添加分子
                if is_valid_mol(mol):
                    population.append(mol)
    
    return population

def select_parents(population, fitness_scores, num_parents, tournament_size=None):
    """使用锦标赛选择法选择父代"""
    if tournament_size is None:
        tournament_size = max(2, len(population) // 5)
        
    parents = []
    for _ in range(num_parents):
        # 锦标赛选择，增加多样性
        candidates = random.sample(list(range(len(population))), tournament_size)
        winner = max(candidates, key=lambda i: fitness_scores[i])
        parents.append(population[winner])
    
    return parents

def process_unique_molecules(molecules, reference_smiles=None, target_size=20, max_atoms=15, min_atoms=5):
    """
    处理分子集合，确保分子唯一且有效
    
    参数:
    - molecules: 分子列表（可以是分子对象或分子信息字典）
    - reference_smiles: 参考SMILES列表
    - target_size: 目标分子数量
    - max_atoms: 分子的最大重原子数量
    - min_atoms: 分子的最小重原子数量
    
    返回:
    - 处理后的分子列表
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from .molecule_utils import is_valid_mol
    
    # 过滤无效分子和不符合大小要求的分子
    valid_molecules = []
    for mol_info in molecules:
        # 处理分子信息字典格式
        if isinstance(mol_info, dict) and 'smiles' in mol_info:
            try:
                mol = Chem.MolFromSmiles(mol_info['smiles'])
                if mol is not None and is_valid_mol(mol) and min_atoms <= mol.GetNumHeavyAtoms() <= max_atoms:
                    valid_molecules.append(mol)
            except:
                continue
        # 处理直接的分子对象
        elif mol_info is not None and is_valid_mol(mol_info) and min_atoms <= mol_info.GetNumHeavyAtoms() <= max_atoms:
            valid_molecules.append(mol_info)
    
    # 使用SMILES字符串去重
    unique_smiles = set()
    unique_molecules = []
    
    for mol in valid_molecules:
        try:
            smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
            if smiles not in unique_smiles:
                unique_smiles.add(smiles)
                unique_molecules.append(mol)
        except:
            continue
    
    # 如果分子数量不足，可以从参考SMILES中添加
    if len(unique_molecules) < target_size and reference_smiles:
        random.shuffle(reference_smiles)
        for smiles in reference_smiles:
            if len(unique_molecules) >= target_size:
                break
                
            if smiles not in unique_smiles:
                try:
                    mol = Chem.MolFromSmiles(smiles)
                    if mol and is_valid_mol(mol) and min_atoms <= mol.GetNumHeavyAtoms() <= max_atoms:
                        unique_smiles.add(smiles)
                        unique_molecules.append(mol)
                except:
                    continue
    
    return unique_molecules
    from .molecule_handler import MoleculeHandler
    molecule_handler = MoleculeHandler()
    
    # 确保我们有足够的分子，并且它们是不同的
    unique_mols = {}
    for mol in population:
        if mol is None:
            continue
        try:
            # 额外验证分子有效性
            if not is_valid_mol(mol):
                continue
                
            # 尝试通过SMILES重建分子，确保结构有效
            smiles = Chem.MolToSmiles(mol)
            
            # 确保生成的分子与参考分子不同
            if reference_smiles and smiles in reference_smiles:
                continue
                
            rebuilt_mol = Chem.MolFromSmiles(smiles, sanitize=True)
            
            if rebuilt_mol is not None:
                unique_mols[smiles] = rebuilt_mol
        except:
            pass
    
    # 如果没有足够的唯一分子，尝试变异现有分子
    attempts = 0
    while len(unique_mols) < target_size and len(unique_mols) > 0 and attempts < 100:
        # 随机选择一个现有分子进行变异
        parent = random.choice(list(unique_mols.values()))
        try:
            child = molecule_handler.mutate_molecule(parent)
            
            if child is not None and is_valid_mol(child):
                smiles = Chem.MolToSmiles(child)
                
                # 确保生成的分子与参考分子不同
                if reference_smiles and smiles in reference_smiles:
                    continue
                    
                if smiles not in unique_mols:
                    # 通过SMILES重建分子，确保结构有效
                    rebuilt_mol = Chem.MolFromSmiles(smiles)
                    if rebuilt_mol is not None:
                        unique_mols[smiles] = rebuilt_mol
        except:
            pass
        attempts += 1
    
    # 将唯一分子转换为列表
    result_molecules = list(unique_mols.values())
    
    # 如果还是不够，添加一些默认分子
    default_smiles_list = [
        'CC', 'CCC', 'CCCC', 'c1ccccc1', 'CCO', 'CCN', 'c1ccccc1O', 'c1ccccc1N', 
        'CC(=O)N', 'C=C', 'C1CCCCC1', 'c1ccncc1', 'CC(=O)O', 'CS', 'c1cc(O)ccc1',
        'c1cc(N)ccc1', 'c1cc(C)ccc1', 'C1CCNCC1', 'C=O', 'C1CCOC1', 'c1ccoc1',
        'c1ccsc1', 'C1CCOCC1', 'CCCl', 'CCBr', 'CCF', 'CCS', 'CCCC=O'
    ]
    
    # 确保有足够多的默认分子可供选择
    while len(result_molecules) < target_size:
        for smiles in default_smiles_list:
            if len(result_molecules) >= target_size:
                break
            
            # 确保默认分子与参考分子不同
            if reference_smiles and smiles in reference_smiles:
                continue
                
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                # 检查是否已经存在相同的分子
                mol_smiles = Chem.MolToSmiles(mol)
                existing_smiles = [Chem.MolToSmiles(m) for m in result_molecules if m]
                if mol_smiles not in existing_smiles:
                    result_molecules.append(mol)
    
    # 如果分子数量超过要求，随机选择所需数量
    if len(result_molecules) > target_size:
        result_molecules = random.sample(result_molecules, target_size)
    
    return result_molecules