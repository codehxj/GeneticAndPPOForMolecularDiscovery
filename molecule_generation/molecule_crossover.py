import random
from rdkit import Chem
from rdkit import RDLogger
from .molecule_utils import get_max_valence

# 禁用RDKit日志，避免错误输出
RDLogger.DisableLog('rdApp.*')

def crossover_molecules(mol1, mol2, is_valid_mol_func, fix_molecule_func):
    """对两个分子进行交叉操作"""
    if mol1 is None or mol2 is None:
        return None
        
    try:
        # 确保分子有效
        if not is_valid_mol_func(mol1) or not is_valid_mol_func(mol2):
            return None
            
        # 获取分子的原子数量
        num_atoms1 = mol1.GetNumAtoms()
        num_atoms2 = mol2.GetNumAtoms()
        
        # 如果分子太小，不进行交叉
        if num_atoms1 < 3 or num_atoms2 < 3:
            return mol1 if random.random() < 0.5 else mol2
            
        # 严格限制分子大小
        max_atoms = 25  # 进一步降低药物分子的理想上限
        
        # 如果任一分子超过限制，返回较小的一个或随机选择一个小分子
        if num_atoms1 > max_atoms or num_atoms2 > max_atoms:
            # 如果两个都超过限制，创建一个新的小分子
            if num_atoms1 > max_atoms and num_atoms2 > max_atoms:
                # 创建一个随机的小分子
                scaffolds = ["c1ccccc1", "c1ccncc1", "C1CCCCC1", "c1ccoc1", "c1ccsc1"]
                smiles = random.choice(scaffolds)
                new_mol = Chem.MolFromSmiles(smiles)
                if new_mol:
                    return new_mol
                else:
                    # 如果创建失败，返回较小的父代
                    return mol1 if num_atoms1 <= num_atoms2 else mol2
            else:
                # 返回较小的分子
                return mol1 if num_atoms1 <= num_atoms2 else mol2
        
        # 限制交叉点，避免生成过大的分子
        # 计算每个分子可以贡献的最大原子数
        max_contrib1 = min(num_atoms1, int(max_atoms * 0.5))
        max_contrib2 = min(num_atoms2, int(max_atoms * 0.5))
        
        # 选择交叉点
        cut_point1 = random.randint(1, max_contrib1)
        cut_point2 = random.randint(1, max_contrib2)
        
        # 创建子图
        frag1 = Chem.RWMol(Chem.Mol())
        frag2 = Chem.RWMol(Chem.Mol())
        
        # 添加原子和键到片段1
        frag1, atom_map1 = _create_fragment(mol1, 0, cut_point1)
        
        # 添加原子和键到片段2
        frag2, atom_map2 = _create_fragment(mol2, cut_point2, num_atoms2)
        
        # 检查片段是否为空
        if frag1.GetNumAtoms() == 0 or frag2.GetNumAtoms() == 0:
            return mol1 if random.random() < 0.5 else mol2
        
        # 检查合并后的分子大小是否会超过限制
        if frag1.GetNumAtoms() + frag2.GetNumAtoms() > max_atoms:
            # 如果会超过限制，返回较小的父代分子
            return mol1 if num_atoms1 <= num_atoms2 else mol2
        
        # 合并片段
        combined = Chem.CombineMols(frag1.GetMol(), frag2.GetMol())
        
        # 添加一个连接键
        rwmol = Chem.RWMol(combined)
        
        # 初始化环信息和原子价态
        try:
            mol = rwmol.GetMol()
            Chem.GetSSSR(mol)  # 使用GetSSSR代替FastFindRings
            rwmol = Chem.RWMol(mol)  # 重新创建可写分子
            for atom in rwmol.GetAtoms():
                atom.UpdatePropertyCache(strict=False)
        except:
            pass
        
        if frag1.GetNumAtoms() > 0 and frag2.GetNumAtoms() > 0:
            # 连接两个片段
            connect_idx1, connect_idx2 = _find_connection_points(rwmol, frag1.GetNumAtoms())
            
            # 如果找到合适的连接点，添加单键
            if connect_idx1 >= 0 and connect_idx2 >= 0:
                rwmol.AddBond(connect_idx1, connect_idx2, Chem.BondType.SINGLE)
                
                # 重新初始化环信息和原子价态
                try:
                    mol = rwmol.GetMol()
                    Chem.GetSSSR(mol)
                    rwmol = Chem.RWMol(mol)
                    for atom in rwmol.GetAtoms():
                        atom.UpdatePropertyCache(strict=False)
                except:
                    pass
                
        # 转换为分子对象
        try:
            new_mol = rwmol.GetMol()
            
            # 确保环信息和原子价态已初始化
            Chem.GetSSSR(new_mol)
            for atom in new_mol.GetAtoms():
                atom.UpdatePropertyCache(strict=False)
                
            # 检查分子大小是否超过限制
            if new_mol.GetNumHeavyAtoms() > max_atoms:
                # 如果超过限制，返回较小的父代分子
                return mol1 if num_atoms1 <= num_atoms2 else mol2
                
            # 验证并修复分子
            if not is_valid_mol_func(new_mol):
                # 尝试修复分子
                fixed_mol = fix_molecule_func(new_mol)
                if fixed_mol is not None and is_valid_mol_func(fixed_mol):
                    # 再次检查修复后的分子大小
                    if fixed_mol.GetNumHeavyAtoms() <= max_atoms:
                        return fixed_mol
                
                # 如果修复失败或分子过大，尝试使用SMILES重建
                try:
                    smiles = Chem.MolToSmiles(new_mol, isomericSmiles=False)
                    if smiles:
                        rebuilt_mol = Chem.MolFromSmiles(smiles)
                        if rebuilt_mol is not None and is_valid_mol_func(rebuilt_mol):
                            # 检查重建后的分子大小
                            if rebuilt_mol.GetNumHeavyAtoms() <= max_atoms:
                                return rebuilt_mol
                except:
                    pass
                
                # 如果所有修复尝试都失败，返回父代之一
                return mol1 if num_atoms1 <= num_atoms2 else mol2
                
            return new_mol
        except:
            return mol1 if random.random() < 0.5 else mol2
    except Exception as e:
        return mol1 if random.random() < 0.5 else mol2

def _create_fragment(mol, start_idx, end_idx):
    """创建分子片段"""
    frag = Chem.RWMol(Chem.Mol())
    atom_map = {}
    
    # 添加原子
    for i in range(start_idx, end_idx):
        atom = mol.GetAtomWithIdx(i)
        # 跳过孤立的氢原子
        if atom.GetAtomicNum() == 1 and atom.GetDegree() == 0:
            continue
        new_idx = frag.AddAtom(Chem.Atom(atom.GetAtomicNum()))
        atom_map[i] = new_idx
    
    # 添加键
    for i in range(start_idx, end_idx):
        if i not in atom_map:
            continue
        atom = mol.GetAtomWithIdx(i)
        for bond in atom.GetBonds():
            j = bond.GetOtherAtomIdx(i)
            if start_idx <= j < end_idx and j > i and j in atom_map:
                # 检查添加键后是否会导致价态问题
                begin_atom = frag.GetAtomWithIdx(atom_map[i])
                end_atom = frag.GetAtomWithIdx(atom_map[j])
                
                # 确保价态已计算
                try:
                    begin_atom.UpdatePropertyCache(strict=False)
                    end_atom.UpdatePropertyCache(strict=False)
                    
                    # 检查添加键后是否会超出最大价态
                    begin_max_valence = get_max_valence(begin_atom.GetAtomicNum())
                    end_max_valence = get_max_valence(end_atom.GetAtomicNum())
                    
                    # 如果添加键会导致价态超出，跳过
                    if begin_atom.GetExplicitValence() + 1 > begin_max_valence or \
                       end_atom.GetExplicitValence() + 1 > end_max_valence:
                        continue
                except:
                    pass
                
                # 检查键类型，优先使用单键
                bond_type = bond.GetBondType()
                if bond_type != Chem.BondType.SINGLE:
                    # 使用单键以避免价态问题
                    bond_type = Chem.BondType.SINGLE
                
                frag.AddBond(atom_map[i], atom_map[j], bond_type)
    
    return frag, atom_map

def _find_connection_points(rwmol, frag1_size):
    """寻找两个片段之间的连接点"""
    connect_idx1 = -1
    connect_idx2 = -1
    
    # 在片段1中寻找合适的连接点
    valid_points1 = []
    for i in range(frag1_size):
        atom = rwmol.GetAtomWithIdx(i)
        # 避免使用氢原子或已经饱和的原子
        try:
            atom.UpdatePropertyCache(strict=False)
            if atom.GetAtomicNum() != 1 and atom.GetExplicitValence() < get_max_valence(atom.GetAtomicNum()):
                valid_points1.append(i)
        except:
            continue
    
    # 在片段2中寻找合适的连接点
    valid_points2 = []
    for i in range(rwmol.GetNumAtoms() - frag1_size):
        atom = rwmol.GetAtomWithIdx(i + frag1_size)
        # 避免使用氢原子或已经饱和的原子
        try:
            atom.UpdatePropertyCache(strict=False)
            if atom.GetAtomicNum() != 1 and atom.GetExplicitValence() < get_max_valence(atom.GetAtomicNum()):
                valid_points2.append(i + frag1_size)
        except:
            continue
    
    # 如果有多个可能的连接点，随机选择一个
    if valid_points1 and valid_points2:
        connect_idx1 = random.choice(valid_points1)
        connect_idx2 = random.choice(valid_points2)
    
    return connect_idx1, connect_idx2