import random
from rdkit import Chem
from rdkit.Chem import AllChem
from .molecule_utils import is_valid_mol, fix_molecule
# 修改导入语句，适应新的molecule_mutation模块
from .molecule_mutation import replace_random_functional_group, generate_molecule_with_functional_groups, identify_functional_groups
# 仍然导入crossover_molecules
from .molecule_crossover import crossover_molecules

class MoleculeHandler:
    """处理分子的变异和交叉操作"""
    
    def __init__(self):
        """初始化分子处理器"""
        # 定义常用官能团类型
        self.functional_group_types = [
            "羟基", "羧基", "醛基", "酮基", "氨基", "硝基", "氰基", 
            "卤素", "烷基", "烯基", "炔基", "醚基", "酯基", "酰胺基"
        ]
        
        # 定义常用原子类型（移除P以避免[PH]异常结构）
        self.atom_types = ['C', 'N', 'O', 'F', 'Cl', 'Br', 'S']
        
        # 定义常用键类型
        self.bond_types = [
            Chem.rdchem.BondType.SINGLE,
            Chem.rdchem.BondType.DOUBLE,
            Chem.rdchem.BondType.TRIPLE,
            Chem.rdchem.BondType.AROMATIC
        ]
        
    def _is_valid_mol(self, mol):
        """检查分子是否有效"""
        return is_valid_mol(mol)
    
    def crossover_molecules(self, mol1, mol2):
        """对两个分子进行交叉操作"""
        # 由于我们现在专注于官能团替换，交叉操作可能不再适用
        # 但我们保留此方法以保持API兼容性
        try:
            # 尝试使用原有的交叉操作
            result = crossover_molecules(mol1, mol2, self._is_valid_mol, fix_molecule)
            if result and self._is_valid_mol(result):
                return result
                
            # 如果交叉失败，返回其中一个父代分子
            return mol1 if random.random() < 0.5 else mol2
        except:
            return mol1 if random.random() < 0.5 else mol2
    
    def mutate_molecule(self, mol, max_atoms=25, min_atoms=8):
        """
        对分子进行变异操作，使用官能团替换，确保分子大小在指定范围内
        
        参数:
        - mol: 要变异的分子
        - max_atoms: 分子的最大重原子数量
        - min_atoms: 分子的最小重原子数量
        
        返回:
        - 变异后的分子或原始分子
        """
        if mol is None:
            return None
            
        try:
            # 确保分子有效
            if not self._is_valid_mol(mol):
                return mol
                
            current_atoms = mol.GetNumHeavyAtoms()
            
            # 如果分子太大，尝试简化
            if current_atoms > max_atoms:
                # 尝试移除一些原子或官能团
                simplified_mol = self._simplify_molecule(mol, max_atoms)
                if simplified_mol and simplified_mol.GetNumHeavyAtoms() <= max_atoms:
                    return simplified_mol
                return mol
                
            # 如果分子太小，尝试扩展
            if current_atoms < min_atoms:
                # 尝试添加官能团或原子
                expanded_mol = self._expand_molecule(mol, min_atoms, max_atoms)
                if expanded_mol and min_atoms <= expanded_mol.GetNumHeavyAtoms() <= max_atoms:
                    return expanded_mol
                    
            # 识别分子中的官能团
            functional_groups = identify_functional_groups(mol)
            
            # 如果没有找到官能团，尝试生成一个新分子
            if not functional_groups:
                try:
                    smiles = Chem.MolToSmiles(mol)
                    new_mol = generate_molecule_with_functional_groups(smiles, n_replacements=1)
                    if new_mol and self._is_valid_mol(new_mol):
                        new_atoms = new_mol.GetNumHeavyAtoms()
                        if min_atoms <= new_atoms <= max_atoms:
                            return new_mol
                except:
                    pass
                return mol
                
            # 使用官能团替换进行变异
            new_mol = replace_random_functional_group(mol)
            
            # 如果替换失败，返回原始分子
            if new_mol is None:
                return mol
                
            # 检查分子大小
            new_atoms = new_mol.GetNumHeavyAtoms()
            if new_atoms > max_atoms or new_atoms < min_atoms:
                return mol
                
            # 验证并修复分子
            if not self._is_valid_mol(new_mol):
                new_mol = fix_molecule(new_mol)
                
            # 如果修复后仍然无效或超过大小限制，返回原始分子
            if new_mol is None or not self._is_valid_mol(new_mol) or new_mol.GetNumHeavyAtoms() > max_atoms:
                return mol
                
            return new_mol
        except Exception as e:
            # 如果出现任何错误，返回原始分子
            return mol
    
    def generate_new_molecule(self, max_atoms=10):
        """
        生成一个全新的分子
        
        参数:
        - max_atoms: 分子的最大重原子数量
        
        返回:
        - 新生成的分子
        """
        try:
            # 使用基本分子作为种子
            seed_smiles = [
                "c1ccccc1",  # 苯
                "CC(=O)O",   # 乙酸
                "CCO",       # 乙醇
                "c1ccccc1O", # 苯酚
                "C1CCCCC1"   # 环己烷
            ]
            
            # 随机选择一个种子分子
            smiles = random.choice(seed_smiles)
            
            # 生成新分子
            mol = generate_molecule_with_functional_groups(smiles, n_replacements=1)
            
            # 验证分子
            if mol and self._is_valid_mol(mol) and mol.GetNumHeavyAtoms() <= max_atoms:
                return mol
                
            # 如果生成失败，返回种子分子
            return Chem.MolFromSmiles(smiles)
        except:
            # 如果出现错误，返回一个简单分子
            return Chem.MolFromSmiles("C")
    
    def add_bond(self, mol, atom1_idx, atom2_idx, bond_type=Chem.rdchem.BondType.SINGLE):
        """
        在两个原子之间添加化学键
        
        参数:
        mol: RDKit分子对象
        atom1_idx: 第一个原子的索引
        atom2_idx: 第二个原子的索引
        bond_type: 键类型，默认为单键
        
        返回:
        修改后的分子，如果操作失败则返回None
        """
        if mol is None:
            return None
            
        try:
            # 创建可编辑的分子对象
            rwmol = Chem.RWMol(mol)
            
            # 检查原子索引是否有效
            if atom1_idx < 0 or atom1_idx >= rwmol.GetNumAtoms() or atom2_idx < 0 or atom2_idx >= rwmol.GetNumAtoms():
                return None
                
            # 检查是否已经存在键
            if rwmol.GetBondBetweenAtoms(atom1_idx, atom2_idx) is not None:
                return None
                
            # 添加键
            rwmol.AddBond(atom1_idx, atom2_idx, bond_type)
            
            # 转换回普通分子对象
            new_mol = rwmol.GetMol()
            
            # 验证分子有效性
            if not is_valid_mol(new_mol):
                return None
                
            return new_mol
        except Exception as e:
            print(f"添加键时出错: {e}")
            return None
    
    def remove_bond(self, mol, atom1_idx, atom2_idx):
        """
        移除两个原子之间的化学键
        
        参数:
        mol: RDKit分子对象
        atom1_idx: 第一个原子的索引
        atom2_idx: 第二个原子的索引
        
        返回:
        修改后的分子，如果操作失败则返回None
        """
        if mol is None:
            return None
            
        try:
            # 创建可编辑的分子对象
            rwmol = Chem.RWMol(mol)
            
            # 检查原子索引是否有效
            if atom1_idx < 0 or atom1_idx >= rwmol.GetNumAtoms() or atom2_idx < 0 or atom2_idx >= rwmol.GetNumAtoms():
                return None
                
            # 获取键
            bond = rwmol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
            if bond is None:
                return None
                
            # 移除键
            rwmol.RemoveBond(atom1_idx, atom2_idx)
            
            # 转换回普通分子对象
            new_mol = rwmol.GetMol()
            
            # 验证分子有效性
            if not is_valid_mol(new_mol):
                return None
                
            return new_mol
        except Exception as e:
            print(f"移除键时出错: {e}")
            return None
    
    def optimize_3d(self, mol):
        """
        优化分子的3D结构
        
        参数:
        mol: RDKit分子对象
        
        返回:
        优化后的分子，如果操作失败则返回原始分子
        """
        if mol is None:
            return None
            
        try:
            # 创建分子的副本
            new_mol = Chem.Mol(mol)
            
            # 尝试修复分子结构
            try:
                Chem.SanitizeMol(new_mol)
            except:
                # 如果无法修复，尝试更宽松的方法
                try:
                    Chem.SanitizeMol(new_mol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS|
                                                         Chem.SanitizeFlags.SANITIZE_KEKULIZE|
                                                         Chem.SanitizeFlags.SANITIZE_SETAROMATICITY|
                                                         Chem.SanitizeFlags.SANITIZE_SETCONJUGATION|
                                                         Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION|
                                                         Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                                   catchErrors=True)
                except:
                    return mol
            
            # 检查是否有不合理的价态
            for atom in new_mol.GetAtoms():
                if atom.GetSymbol() == 'O' and atom.GetExplicitValence() > 2:
                    return mol  # 氧原子价态不应超过2
                if atom.GetSymbol() == 'N' and atom.GetExplicitValence() > 3:
                    return mol  # 氮原子价态不应超过3
            
            # 添加氢原子
            try:
                new_mol = Chem.AddHs(new_mol)
            except:
                return mol
            
            # 生成3D构象
            try:
                # 使用更安全的参数
                AllChem.EmbedMolecule(new_mol, randomSeed=42, maxAttempts=10, useRandomCoords=True)
            except:
                try:
                    # 如果标准方法失败，尝试使用距离几何法
                    AllChem.EmbedMolecule(new_mol, randomSeed=42, useRandomCoords=True, 
                                         useBasicKnowledge=True, enforceChirality=False)
                except:
                    # 如果仍然失败，返回原始分子
                    return mol
            
            # 使用MMFF优化
            try:
                # 检查是否可以使用MMFF
                props = AllChem.MMFFGetMoleculeProperties(new_mol)
                if props:
                    AllChem.MMFFOptimizeMolecule(new_mol, maxIters=200, mmffVariant='MMFF94s')
                else:
                    # 如果MMFF不适用，尝试UFF
                    AllChem.UFFOptimizeMolecule(new_mol, maxIters=200)
            except:
                # 如果优化失败，仍然保留3D坐标
                pass
            
            # 移除氢原子
            try:
                new_mol = Chem.RemoveHs(new_mol)
            except:
                return mol
            
            return new_mol
        except Exception as e:
            print(f"变异分子时出错: {e}")
            return mol
    
    def _simplify_molecule(self, mol, max_atoms):
        """简化分子，移除一些原子或官能团"""
        if mol is None:
            return None
            
        try:
            # 创建可编辑的分子
            rwmol = Chem.RWMol(mol)
            
            # 获取当前原子数
            current_atoms = rwmol.GetNumHeavyAtoms()
            atoms_to_remove = current_atoms - max_atoms
            
            if atoms_to_remove <= 0:
                return mol
                
            # 优先移除末端原子（度数为1的原子）
            atoms_removed = 0
            for i in range(rwmol.GetNumAtoms() - 1, -1, -1):
                if atoms_removed >= atoms_to_remove:
                    break
                    
                atom = rwmol.GetAtomWithIdx(i)
                if atom.GetDegree() == 1 and atom.GetAtomicNum() != 1:  # 不移除氢原子
                    rwmol.RemoveAtom(i)
                    atoms_removed += 1
            
            # 如果还需要移除更多原子，随机移除
            while atoms_removed < atoms_to_remove and rwmol.GetNumAtoms() > 1:
                atom_idx = random.randint(0, rwmol.GetNumAtoms() - 1)
                atom = rwmol.GetAtomWithIdx(atom_idx)
                if atom.GetAtomicNum() != 1:  # 不移除氢原子
                    rwmol.RemoveAtom(atom_idx)
                    atoms_removed += 1
            
            # 获取简化后的分子
            simplified_mol = rwmol.GetMol()
            
            # 验证分子
            if simplified_mol and self._is_valid_mol(simplified_mol):
                return simplified_mol
            else:
                return mol
                
        except Exception as e:
            print(f"简化分子时出错: {e}")
            return mol
    
    def _expand_molecule(self, mol, min_atoms, max_atoms):
        """扩展分子，添加原子或官能团"""
        if mol is None:
            return None
            
        try:
            current_atoms = mol.GetNumHeavyAtoms()
            atoms_to_add = min_atoms - current_atoms
            
            if atoms_to_add <= 0:
                return mol
                
            # 创建可编辑的分子
            rwmol = Chem.RWMol(mol)
            
            # 添加简单的原子和键
            for _ in range(min(atoms_to_add, max_atoms - current_atoms)):
                # 随机选择一个现有原子作为连接点
                if rwmol.GetNumAtoms() > 0:
                    connect_idx = random.randint(0, rwmol.GetNumAtoms() - 1)
                    
                    # 添加一个碳原子
                    new_atom_idx = rwmol.AddAtom(Chem.Atom(6))  # 碳原子
                    
                    # 连接到现有原子
                    rwmol.AddBond(connect_idx, new_atom_idx, Chem.BondType.SINGLE)
                else:
                    # 如果没有原子，添加第一个原子
                    rwmol.AddAtom(Chem.Atom(6))  # 碳原子
            
            # 获取扩展后的分子
            expanded_mol = rwmol.GetMol()
            
            # 验证分子
            if expanded_mol and self._is_valid_mol(expanded_mol):
                return expanded_mol
            else:
                return mol
                
        except Exception as e:
            print(f"扩展分子时出错: {e}")
            return mol  # 如果优化失败，返回原始分子
    
    def mutate_atom(self, mol, atom_idx):
        """
        变异指定原子的类型
        
        参数:
        mol: RDKit分子对象
        atom_idx: 要变异的原子索引
        
        返回:
        修改后的分子，如果操作失败则返回None
        """
        if mol is None:
            return None
            
        try:
            # 创建可编辑的分子对象
            rwmol = Chem.RWMol(mol)
            
            # 检查原子索引是否有效
            if atom_idx < 0 or atom_idx >= rwmol.GetNumAtoms():
                return None
                
            # 获取当前原子
            atom = rwmol.GetAtomWithIdx(atom_idx)
            current_symbol = atom.GetSymbol()
            
            # 选择一个不同的原子类型
            new_symbols = [s for s in self.atom_types if s != current_symbol]
            if not new_symbols:
                return None
                
            new_symbol = random.choice(new_symbols)
            
            # 更改原子类型
            atom.SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(new_symbol))
            
            # 转换回普通分子对象
            new_mol = rwmol.GetMol()
            
            # 验证分子有效性
            if not is_valid_mol(new_mol):
                return None
                
            return new_mol
        except Exception as e:
            print(f"变异原子时出错: {e}")
            return None
    
    def add_atom(self, mol, atom_type=None):
        """
        向分子添加一个新原子并连接到现有原子
        
        参数:
        mol: RDKit分子对象
        atom_type: 要添加的原子类型，如果为None则随机选择
        
        返回:
        修改后的分子，如果操作失败则返回None
        """
        if mol is None or mol.GetNumAtoms() == 0:
            return None
            
        try:
            # 创建可编辑的分子对象
            rwmol = Chem.RWMol(mol)
            
            # 选择一个随机原子作为连接点
            connect_idx = random.randint(0, rwmol.GetNumAtoms() - 1)
            
            # 选择原子类型
            if atom_type is None:
                atom_type = random.choice(self.atom_types)
                
            # 添加新原子
            new_atom_idx = rwmol.AddAtom(Chem.Atom(atom_type))
            
            # 添加键连接新原子和选定的原子
            rwmol.AddBond(connect_idx, new_atom_idx, Chem.rdchem.BondType.SINGLE)
            
            # 转换回普通分子对象
            new_mol = rwmol.GetMol()
            
            # 验证分子有效性
            if not is_valid_mol(new_mol):
                return None
                
            return new_mol
        except Exception as e:
            print(f"添加原子时出错: {e}")
            return None
    
    def remove_atom(self, mol, atom_idx):
        """
        从分子中移除指定原子
        
        参数:
        mol: RDKit分子对象
        atom_idx: 要移除的原子索引
        
        返回:
        修改后的分子，如果操作失败则返回None
        """
        if mol is None:
            return None
            
        try:
            # 创建可编辑的分子对象
            rwmol = Chem.RWMol(mol)
            
            # 检查原子索引是否有效
            if atom_idx < 0 or atom_idx >= rwmol.GetNumAtoms():
                return None
                
            # 移除原子
            rwmol.RemoveAtom(atom_idx)
            
            # 转换回普通分子对象
            new_mol = rwmol.GetMol()
            
            # 验证分子有效性
            if not is_valid_mol(new_mol) or new_mol.GetNumAtoms() == 0:
                return None
                
            return new_mol
        except Exception as e:
            print(f"移除原子时出错: {e}")
            return None
    
    def change_bond_type(self, mol, atom1_idx, atom2_idx):
        """
        改变两个原子之间的键类型
        
        参数:
        mol: RDKit分子对象
        atom1_idx: 第一个原子的索引
        atom2_idx: 第二个原子的索引
        
        返回:
        修改后的分子，如果操作失败则返回None
        """
        if mol is None:
            return None
            
        try:
            # 创建可编辑的分子对象
            rwmol = Chem.RWMol(mol)
            
            # 检查原子索引是否有效
            if atom1_idx < 0 or atom1_idx >= rwmol.GetNumAtoms() or atom2_idx < 0 or atom2_idx >= rwmol.GetNumAtoms():
                return None
                
            # 获取键
            bond = rwmol.GetBondBetweenAtoms(atom1_idx, atom2_idx)
            if bond is None:
                return None
                
            # 获取当前键类型
            current_type = bond.GetBondType()
            
            # 选择一个不同的键类型
            new_types = [t for t in self.bond_types if t != current_type]
            if not new_types:
                return None
                
            new_type = random.choice(new_types)
            
            # 移除旧键并添加新键
            rwmol.RemoveBond(atom1_idx, atom2_idx)
            rwmol.AddBond(atom1_idx, atom2_idx, new_type)
            
            # 转换回普通分子对象
            new_mol = rwmol.GetMol()
            
            # 验证分子有效性
            if not is_valid_mol(new_mol):
                return None
                
            return new_mol
        except Exception as e:
            print(f"改变键类型时出错: {e}")
            return None
    
    def add_ring(self, mol):
        """
        向分子添加一个环结构
        
        参数:
        mol: RDKit分子对象
        
        返回:
        修改后的分子，如果操作失败则返回None
        """
        if mol is None or mol.GetNumAtoms() < 2:
            return None
            
        try:
            # 创建可编辑的分子对象
            rwmol = Chem.RWMol(mol)
            
            # 选择两个随机原子作为连接点
            num_atoms = rwmol.GetNumAtoms()
            atom1_idx = random.randint(0, num_atoms - 1)
            
            # 确保选择不同的原子
            atom2_idx = atom1_idx
            while atom2_idx == atom1_idx:
                atom2_idx = random.randint(0, num_atoms - 1)
                
            # 检查两个原子之间是否已经有键
            if rwmol.GetBondBetweenAtoms(atom1_idx, atom2_idx) is not None:
                return None
                
            # 决定环的大小 (3-6)
            ring_size = random.randint(3, 6)
            
            # 添加原子形成环
            last_idx = atom1_idx
            new_atoms = []
            
            for i in range(ring_size - 2):  # 减2是因为我们已经有两个原子
                new_atom_idx = rwmol.AddAtom(Chem.Atom('C'))
                rwmol.AddBond(last_idx, new_atom_idx, Chem.rdchem.BondType.SINGLE)
                new_atoms.append(new_atom_idx)
                last_idx = new_atom_idx
                
            # 连接最后一个新原子到第二个选定的原子
            rwmol.AddBond(last_idx, atom2_idx, Chem.rdchem.BondType.SINGLE)
            
            # 转换回普通分子对象
            new_mol = rwmol.GetMol()
            
            # 验证分子有效性
            if not is_valid_mol(new_mol):
                return None
                
            return new_mol
        except Exception as e:
            print(f"添加环时出错: {e}")
            return None
    
    def add_functional_group(self, mol, group_type=None):
        """
        向分子添加一个官能团
        
        参数:
        mol: RDKit分子对象
        group_type: 要添加的官能团类型，如果为None则随机选择
        
        返回:
        修改后的分子，如果操作失败则返回None
        """
        if mol is None or mol.GetNumAtoms() == 0:
            return None
            
        try:
            # 官能团SMILES字典
            functional_groups = {
                "羟基": "O",
                "羧基": "C(=O)O",
                "醛基": "C=O",
                "酮基": "C(=O)C",
                "氨基": "N",
                "硝基": "N(=O)=O",
                "氰基": "C#N",
                "卤素": random.choice(["F", "Cl", "Br"]),
                "烷基": random.choice(["C", "CC", "CCC"]),
                "烯基": "C=C",
                "炔基": "C#C",
                "醚基": "OC",
                "酯基": "C(=O)OC",
                "酰胺基": "C(=O)N"
            }
            
            # 选择一个随机原子作为连接点
            atom_idx = random.randint(0, mol.GetNumAtoms() - 1)
            
            # 选择官能团类型
            if group_type is None or group_type not in functional_groups:
                group_type = random.choice(list(functional_groups.keys()))
                
            group_smiles = functional_groups[group_type]
            
            # 创建官能团分子
            group_mol = Chem.MolFromSmiles(group_smiles)
            if group_mol is None:
                return None
                
            # 合并分子和官能团
            combo = Chem.CombineMols(mol, group_mol)
            
            # 创建可编辑的合并分子
            rwmol = Chem.RWMol(combo)
            
            # 添加键连接分子和官能团
            rwmol.AddBond(atom_idx, mol.GetNumAtoms(), Chem.rdchem.BondType.SINGLE)
            
            # 转换回普通分子对象
            new_mol = rwmol.GetMol()
            
            # 验证分子有效性
            if not is_valid_mol(new_mol):
                return None
                
            return new_mol
        except Exception as e:
            print(f"添加官能团时出错: {e}")
            return None
