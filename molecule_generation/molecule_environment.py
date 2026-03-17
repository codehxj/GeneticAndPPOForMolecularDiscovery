import random
import numpy as np
import sys
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, QED
from .molecule_utils import is_valid_mol, fix_molecule

# 添加 DeepDTA 路径用于亲和力计算
sys.path.append('DeepDTA')
from affinity_predictor import AffinityPredictor

class MoleculeEnvironment:
    """
    分子环境类，用于PPO强化学习算法
    提供分子修改的动作空间和奖励计算
    """
    
    def __init__(self, molecule_handler, max_steps=10, min_atoms=5, max_atoms=50, reference_smiles=None, 
                 sanitize_mols=True, skip_3d_opt=False, protein_sequence=None, affinity_weight=0.3, druglikeness_checker=None):
        """
        初始化分子环境
        
        参数:
        molecule_handler: 分子处理器对象
        max_steps: 每个回合的最大步数
        min_atoms: 分子的最小重原子数量
        max_atoms: 分子的最大重原子数量
        reference_smiles: 参考分子的SMILES字符串列表
        sanitize_mols: 是否对生成的分子进行结构修复
        skip_3d_opt: 是否跳过3D结构优化
        protein_sequence: 目标蛋白质序列，用于亲和力计算
        affinity_weight: 亲和力在奖励函数中的权重
        druglikeness_checker: 可成药性检查器对象
        """
        self.molecule_handler = molecule_handler
        self.max_steps = max_steps
        self.min_atoms = min_atoms
        self.max_atoms = max_atoms
        self.reference_smiles = reference_smiles
        self.current_mol = None
        self.steps_taken = 0
        self.action_space = 10  # 动作空间大小
        self.sanitize_mols = sanitize_mols
        self.skip_3d_opt = skip_3d_opt
        self.protein_sequence = protein_sequence
        self.affinity_weight = affinity_weight
        self.druglikeness_checker = druglikeness_checker
        
        # 初始化亲和力预测器（如果提供了蛋白质序列）
        self.affinity_predictor = None
        if self.protein_sequence:
            try:
                self.affinity_predictor = AffinityPredictor(
                    model_path='DeepDTA/deepdta_retrain-prk12-ldk8.pt',
                    ligand_dict_path='DeepDTA/ligand_dict-prk12-ldk8.json',
                    protein_dict_path='DeepDTA/protein_dict-prk12-ldk8.json'
                )
                self.affinity_predictor._load_dictionaries()
                self.affinity_predictor._load_model()
                print("亲和力预测器初始化成功")
            except Exception as e:
                print(f"亲和力预测器初始化失败: {e}")
                self.affinity_predictor = None
        
        # 定义动作类型
        self.action_types = [
            'add_atom',           # 添加原子
            'remove_atom',        # 移除原子
            'add_bond',           # 添加键
            'remove_bond',        # 移除键
            'change_bond_type',   # 改变键类型
            'mutate_atom',        # 变异原子类型
            'add_ring',           # 添加环
            'add_functional_group', # 添加官能团
            'optimize_3d' if not skip_3d_opt else 'no_op',  # 优化3D结构或不操作
            'no_op'               # 不执行操作
        ]
    
    def reset(self, seed_mol=None):
        """
        重置环境状态
        
        参数:
        seed_mol: 种子分子，如果为None则随机生成
        
        返回:
        初始分子
        """
        self.steps_taken = 0
        
        # 如果提供了种子分子，使用它
        if seed_mol is not None and is_valid_mol(seed_mol):
            self.current_mol = Chem.Mol(seed_mol)
        else:
            # 否则生成一个新分子
            self.current_mol = self.molecule_handler.generate_new_molecule(max_atoms=self.max_atoms)
        
        return self.current_mol
    
    def step(self, action):
        """
        执行一个动作并返回新状态、奖励和是否结束
        
        参数:
        action: 动作索引 (0-9)
        
        返回:
        (新分子, 奖励, 是否结束)
        """
        self.steps_taken += 1
        done = self.steps_taken >= self.max_steps
        
        # 如果当前没有有效分子，返回负奖励
        if self.current_mol is None or not is_valid_mol(self.current_mol):
            return None, -1.0, True
        
        # 获取当前分子的副本
        mol = Chem.Mol(self.current_mol)
        
        # 根据动作索引选择动作类型
        action_type = self.action_types[action % len(self.action_types)]
        
        # 执行选定的动作
        new_mol = self._execute_action(mol, action_type)
        
        # 如果动作执行失败，保持当前分子不变
        if new_mol is None:
            new_mol = mol
            reward = -0.1  # 轻微惩罚无效动作
        else:
            # 计算奖励
            reward = self._calculate_reward(new_mol)
            
            # 更新当前分子
            self.current_mol = new_mol
        
        return new_mol, reward, done
    
    def _execute_action(self, mol, action_type):
        """
        执行指定类型的动作
        
        参数:
        mol: 当前分子
        action_type: 动作类型
        
        返回:
        修改后的分子，如果操作失败则返回None
        """
        try:
            num_atoms = mol.GetNumAtoms()
            
            # 如果分子没有原子，只能添加原子
            if num_atoms == 0 and action_type != 'add_atom':
                return self.molecule_handler.add_atom(mol)
            
            # 根据动作类型执行相应操作
            if action_type == 'add_atom':
                new_mol = self.molecule_handler.add_atom(mol)
                
            elif action_type == 'remove_atom':
                if num_atoms <= self.min_atoms:
                    return mol  # 不允许移除原子，如果分子已经达到最小大小
                atom_idx = random.randint(0, num_atoms - 1)
                new_mol = self.molecule_handler.remove_atom(mol, atom_idx)
                
            elif action_type == 'add_bond':
                if num_atoms < 2:
                    return mol  # 需要至少两个原子才能添加键
                # 随机选择两个不同的原子
                atom1_idx = random.randint(0, num_atoms - 1)
                atom2_idx = random.randint(0, num_atoms - 1)
                # 尝试最多5次找到不同的原子
                attempts = 0
                while atom2_idx == atom1_idx and attempts < 5:
                    atom2_idx = random.randint(0, num_atoms - 1)
                    attempts += 1
                if atom2_idx == atom1_idx:
                    return mol  # 如果找不到不同的原子，返回原始分子
                # 随机选择键类型
                bond_type = random.choice(self.molecule_handler.bond_types)
                new_mol = self.molecule_handler.add_bond(mol, atom1_idx, atom2_idx, bond_type)
                
            elif action_type == 'remove_bond':
                if num_atoms < 2:
                    return mol  # 需要至少两个原子才能移除键
                # 获取所有键
                bonds = []
                for bond in mol.GetBonds():
                    bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                if not bonds:
                    return mol  # 没有键可以移除
                # 随机选择一个键
                atom1_idx, atom2_idx = random.choice(bonds)
                new_mol = self.molecule_handler.remove_bond(mol, atom1_idx, atom2_idx)
                
            elif action_type == 'change_bond_type':
                if num_atoms < 2:
                    return mol  # 需要至少两个原子才能改变键类型
                # 获取所有键
                bonds = []
                for bond in mol.GetBonds():
                    bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                if not bonds:
                    return mol  # 没有键可以改变
                # 随机选择一个键
                atom1_idx, atom2_idx = random.choice(bonds)
                new_mol = self.molecule_handler.change_bond_type(mol, atom1_idx, atom2_idx)
                
            elif action_type == 'mutate_atom':
                if num_atoms == 0:
                    return mol  # 没有原子可以变异
                atom_idx = random.randint(0, num_atoms - 1)
                new_mol = self.molecule_handler.mutate_atom(mol, atom_idx)
                
            elif action_type == 'add_ring':
                if num_atoms < 2:
                    return mol  # 需要至少两个原子才能添加环
                new_mol = self.molecule_handler.add_ring(mol)
                
            elif action_type == 'add_functional_group':
                if num_atoms == 0:
                    return mol  # 没有原子可以添加官能团
                if mol.GetNumHeavyAtoms() >= self.max_atoms - 2:
                    return mol  # 分子太大，不添加官能团
                new_mol = self.molecule_handler.add_functional_group(mol)
                
            elif action_type == 'optimize_3d':
                if self.skip_3d_opt:
                    return mol  # 如果设置了跳过3D优化，直接返回原始分子
                new_mol = self.molecule_handler.optimize_3d(mol)
                
            elif action_type == 'no_op':
                return mol  # 不执行任何操作
                
            else:
                return mol  # 未知动作类型
            
            # 如果操作失败，返回原始分子
            if new_mol is None:
                return mol
                
            # 如果需要，对分子进行结构修复
            if self.sanitize_mols and new_mol is not None:
                try:
                    # 尝试修复分子结构
                    new_mol = fix_molecule(new_mol)
                    if new_mol is None:
                        return mol
                except:
                    return mol
                    
            return new_mol
                
        except Exception as e:
            print(f"执行动作时出错: {e}")
            return mol  # 出错时返回原始分子，而不是None
    
    def _calculate_reward(self, mol):
        """
        计算分子的奖励值 - 优化版本，重点关注亲和力和成药性
        
        参数:
        mol: 分子对象
        
        返回:
        奖励值
        """
        if mol is None or not is_valid_mol(mol):
            return -2.0  # 增加无效分子的惩罚
        
        try:
            # 使用可成药性检查器进行全面评估
            if hasattr(self, 'druglikeness_checker') and self.druglikeness_checker:
                druglikeness_result = self.druglikeness_checker.calculate_druglikeness_score(mol)
                
                # 严格的可成药性筛选
                if druglikeness_result['total_score'] < 0.2:
                    return -1.5  # 严重惩罚低可成药性分子
                
                # PAINS检查
                if druglikeness_result['pains']['is_pains']:
                    return -1.2  # 惩罚PAINS分子
                
                # 结构问题检查
                if druglikeness_result['structural_alerts']['total_alerts'] > 3:
                    return -1.0  # 惩罚结构问题较多的分子
            else:
                # 回退到基础的药物类似性检查
                if not self._is_drug_like_molecule(mol):
                    return -1.2
            
            # 检查分子大小
            num_atoms = mol.GetNumHeavyAtoms()
            if num_atoms < self.min_atoms:
                return -1.0  # 增加分子太小的惩罚
            if num_atoms > self.max_atoms:
                return -1.0  # 增加分子太大的惩罚
            
            # 基础奖励
            reward = 0.0
            
            # 1. 计算药物-蛋白质亲和力（最高优先级）
            affinity_reward = 0.0
            if self.affinity_predictor and self.protein_sequence:
                try:
                    smiles = Chem.MolToSmiles(mol)
                    affinity_score = self.affinity_predictor.predict_affinity(smiles, self.protein_sequence)
                    
                    # 根据训练数据分析，调整亲和力阈值
                    # 训练数据显示：平均值约3000，中位数约20，我们的预测范围2-6
                    # 将阈值调整为更合理的范围
                    if affinity_score > 4.0:  # 降低高亲和力阈值
                        # 非线性增强高亲和力
                        affinity_reward = 1.0 + (affinity_score - 4.0) * 0.5
                        # 分层奖励
                        if affinity_score > 5.0:
                            affinity_reward += 0.5  # 额外奖励
                        if affinity_score > 5.5:
                            affinity_reward += 0.5  # 更高奖励
                    else:
                        # 线性奖励低亲和力
                        affinity_reward = affinity_score / 4.0
                    
                    # 归一化到合理范围
                    affinity_reward = min(affinity_reward, 3.0)
                    reward += affinity_reward * 0.5  # 增加权重到0.5
                        
                except Exception as e:
                    print(f"计算亲和力奖励时出错: {e}")
                    affinity_reward = 0.0
            
            # 2. 使用可成药性检查器的综合评分
            druglikeness_reward = 0.0
            if hasattr(self, 'druglikeness_checker') and self.druglikeness_checker:
                try:
                    druglikeness_result = self.druglikeness_checker.calculate_druglikeness_score(mol)
                    
                    # 基础可成药性奖励
                    base_score = druglikeness_result['total_score']
                    druglikeness_reward = base_score * 2.0  # 放大奖励
                    
                    # QED奖励（从可成药性结果中获取）
                    qed_value = druglikeness_result.get('qed', 0.0)
                    if qed_value > 0.8:
                        druglikeness_reward += 0.5  # 优秀的药物类似性
                    elif qed_value > 0.6:
                        druglikeness_reward += 0.3  # 良好的药物类似性
                    elif qed_value > 0.4:
                        druglikeness_reward += 0.1  # 中等的药物类似性
                    
                    # Lipinski规则奖励
                    lipinski_violations = druglikeness_result['lipinski']['violations']
                    if lipinski_violations == 0:
                        druglikeness_reward += 0.3
                    elif lipinski_violations == 1:
                        druglikeness_reward += 0.1
                    
                    # Veber规则奖励
                    veber_violations = druglikeness_result['veber']['violations']
                    if veber_violations == 0:
                        druglikeness_reward += 0.2
                    
                    # 结构质量奖励
                    structural_score = druglikeness_result['structural_quality']['score']
                    druglikeness_reward += structural_score * 0.3
                    
                    reward += druglikeness_reward * 0.3  # 可成药性权重
                    
                except Exception as e:
                    print(f"计算可成药性奖励时出错: {e}")
                    # 回退到基础QED计算
                    try:
                        qed_value = QED.qed(mol)
                        qed_reward = qed_value ** 1.5
                        if qed_value > 0.8:
                            qed_reward += 0.3
                        elif qed_value > 0.6:
                            qed_reward += 0.15
                        elif qed_value > 0.4:
                            qed_reward += 0.05
                        reward += qed_reward * 0.25
                    except:
                        pass
            else:
                # 回退到基础QED计算
                try:
                    qed_value = QED.qed(mol)
                    qed_reward = qed_value ** 1.5
                    if qed_value > 0.8:
                        qed_reward += 0.3
                    elif qed_value > 0.6:
                        qed_reward += 0.15
                    elif qed_value > 0.4:
                        qed_reward += 0.05
                    reward += qed_reward * 0.25
                except:
                    pass
            
            # 3. 计算分子复杂度和多样性奖励
            complexity_reward = 0.0
            try:
                # 环结构奖励（优化版本 - 提高复杂度要求）
                ring_info = mol.GetRingInfo()
                num_rings = ring_info.NumRings()
                
                # 偏好3-6个环的结构（提高复杂度）
                if 3 <= num_rings <= 6:
                    complexity_reward += 0.4  # 增加奖励
                elif num_rings == 2 or num_rings == 7:
                    complexity_reward += 0.2
                elif num_rings == 1:
                    complexity_reward += 0.05  # 降低单环奖励
                elif num_rings > 7:
                    complexity_reward -= 0.1  # 减少过多环的惩罚
                
                # 芳香性奖励（提高芳香性要求）
                aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
                aromatic_ratio = aromatic_atoms / num_atoms if num_atoms > 0 else 0
                if 0.3 <= aromatic_ratio <= 0.7:  # 提高芳香性比例要求
                    complexity_reward += 0.3  # 增加芳香性奖励
                elif 0.1 <= aromatic_ratio < 0.3 or 0.7 < aromatic_ratio <= 0.9:
                    complexity_reward += 0.1
                
                # 旋转键数量（柔性）- 提高复杂度要求
                rotatable_bonds = Descriptors.NumRotatableBonds(mol)
                if 5 <= rotatable_bonds <= 15:  # 提高柔性要求
                    complexity_reward += 0.15
                elif 3 <= rotatable_bonds < 5 or 15 < rotatable_bonds <= 20:
                    complexity_reward += 0.05
                elif rotatable_bonds > 20:  # 过度柔性
                    complexity_reward -= 0.1
                
                reward += complexity_reward * 0.15  # 增加复杂度奖励权重
                
            except Exception as e:
                print(f"计算复杂度奖励时出错: {e}")
            
            # 4. 与参考分子的相似性/新颖性平衡
            similarity_reward = 0.0
            if self.reference_smiles:
                try:
                    from rdkit import DataStructs
                    from rdkit.Chem import rdMolDescriptors
                    
                    mol_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
                    similarities = []
                    
                    for ref_smiles in self.reference_smiles[:20]:  # 限制比较数量
                        try:
                            ref_mol = Chem.MolFromSmiles(ref_smiles)
                            if ref_mol:
                                ref_fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=2048)
                                similarity = DataStructs.TanimotoSimilarity(mol_fp, ref_fp)
                                similarities.append(similarity)
                        except:
                            continue
                    
                    if similarities:
                        max_similarity = max(similarities)
                        avg_similarity = sum(similarities) / len(similarities)
                        
                        # 平衡相似性和新颖性
                        if 0.3 <= max_similarity <= 0.7:  # 适度相似
                            similarity_reward = 0.3
                        elif 0.1 <= max_similarity < 0.3:  # 新颖但相关
                            similarity_reward = 0.2
                        elif max_similarity > 0.9:  # 过于相似
                            similarity_reward = -0.3
                        elif max_similarity < 0.1:  # 过于新颖
                            similarity_reward = -0.1
                        
                        reward += similarity_reward * 0.1
                        
                except Exception as e:
                    print(f"计算相似性奖励时出错: {e}")
            
            # 5. 分子稳定性和合成可行性奖励
            stability_reward = 0.0
            try:
                # 检查原子价态
                valid_valences = True
                for atom in mol.GetAtoms():
                    if atom.GetTotalValence() > Chem.GetPeriodicTable().GetValenceList(atom.GetAtomicNum())[-1]:
                        valid_valences = False
                        break
                
                if valid_valences:
                    stability_reward += 0.2
                
                # 检查是否有不合理的键
                for bond in mol.GetBonds():
                    bond_type = bond.GetBondType()
                    atom1 = bond.GetBeginAtom()
                    atom2 = bond.GetEndAtom()
                    
                    # 简单的键合理性检查
                    if (atom1.GetAtomicNum() == 1 or atom2.GetAtomicNum() == 1) and bond_type != Chem.BondType.SINGLE:
                        stability_reward -= 0.1  # 氢原子不应有多重键
                        break
                
                reward += stability_reward * 0.05
                
            except Exception as e:
                print(f"计算稳定性奖励时出错: {e}")
            
            # 最终奖励归一化和调整
            reward = max(-2.0, min(reward, 5.0))  # 限制奖励范围
            
            return reward
            
        except Exception as e:
            print(f"计算奖励时发生错误: {e}")
            return -1.0
    
    def _is_drug_like_molecule(self, mol):
        """
        检查分子是否符合基本的成药性要求
        
        参数:
        mol: 分子对象
        
        返回:
        bool: 是否符合成药性要求
        """
        if mol is None:
            return False
        
        try:
            # 1. 基本有效性检查
            if not is_valid_mol(mol):
                return False
            
            # 2. 检查分子是否可以生成SMILES
            try:
                smiles = Chem.MolToSmiles(mol)
                if not smiles or len(smiles) < 3:
                    return False
                
                # 检查SMILES是否可以重新解析
                test_mol = Chem.MolFromSmiles(smiles)
                if test_mol is None:
                    return False
            except:
                return False
            
            # 3. 检查分子大小
            num_heavy_atoms = mol.GetNumHeavyAtoms()
            if num_heavy_atoms < 5 or num_heavy_atoms > 50:
                return False
            
            # 4. 检查基本的Lipinski规则
            try:
                mw = Descriptors.MolWt(mol)
                logp = Descriptors.MolLogP(mol)
                hbd = Descriptors.NumHDonors(mol)
                hba = Descriptors.NumHAcceptors(mol)
                
                # 严格的Lipinski规则检查
                if mw > 600 or mw < 100:  # 分子量范围
                    return False
                if logp > 5 or logp < -2:  # 放宽LogP范围，从6/-3改为5/-2
                    return False
                if hbd > 6:  # 放宽氢键供体数，从8改为6
                    return False
                if hba > 10:  # 放宽氢键受体数，从12改为10
                    return False
            except:
                return False
            
            # 5. 检查TPSA
            try:
                tpsa = Descriptors.TPSA(mol)
                if tpsa > 200 or tpsa < 10:
                    return False
            except:
                return False
            
            # 6. 检查可旋转键数量（放宽限制以支持更复杂分子）
            try:
                rotatable_bonds = Descriptors.NumRotatableBonds(mol)
                if rotatable_bonds > 20:  # 提高可旋转键上限
                    return False
            except:
                return False
            
            # 7. 检查环结构（放宽限制以支持更复杂分子）
            try:
                ring_info = mol.GetRingInfo()
                num_rings = ring_info.NumRings()
                if num_rings > 8:  # 提高环数量上限
                    return False
                
                # 检查是否有过大的环
                for ring in ring_info.AtomRings():
                    if len(ring) > 10:  # 提高单个环的大小限制
                        return False
            except:
                return False
            
            # 8. 检查原子类型多样性
            try:
                atom_types = set()
                for atom in mol.GetAtoms():
                    atom_types.add(atom.GetAtomicNum())
                
                # 确保有合理的原子类型多样性
                if len(atom_types) < 2:  # 至少要有两种不同的原子类型
                    return False
                
                # 检查是否包含不常见的原子（移除磷原子15以避免[PH]异常结构）
                common_atoms = {1, 6, 7, 8, 9, 16, 17, 35, 53}  # H, C, N, O, F, S, Cl, Br, I
                for atom_num in atom_types:
                    if atom_num not in common_atoms:
                        return False
            except:
                return False
            
            # 9. 检查分子连通性
            try:
                # 确保分子是连通的（没有孤立的片段）
                frags = Chem.GetMolFrags(mol, asMols=True)
                if len(frags) > 1:
                    return False
            except:
                return False
            
            # 10. 检查价态合理性
            try:
                for atom in mol.GetAtoms():
                    # 检查原子的价态是否合理
                    valence = atom.GetTotalValence()
                    atomic_num = atom.GetAtomicNum()
                    
                    # 常见原子的最大价态（移除磷原子15以避免[PH]异常结构）
                    max_valences = {1: 1, 6: 4, 7: 5, 8: 2, 9: 1, 16: 6, 17: 7, 35: 7, 53: 7}
                    
                    if atomic_num in max_valences and valence > max_valences[atomic_num]:
                        return False
            except:
                return False
            
            return True
            
        except Exception as e:
            print(f"检查成药性时出错: {e}")
            return False