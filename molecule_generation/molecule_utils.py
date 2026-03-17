import os
import random
from rdkit import Chem
from rdkit.Chem import Draw, AllChem, SanitizeMol, SanitizeFlags
from rdkit.Chem import Descriptors, QED
from rdkit.Chem import rdMolDescriptors
import numpy as np
from rdkit import DataStructs

def setup_output_dir(output_dir):
    """创建输出目录"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 创建子目录
    images_dir = os.path.join(output_dir, 'images')
    if not os.path.exists(images_dir):
        os.makedirs(images_dir)
    
    return images_dir

def load_reference_smiles(file_path):
    """从CSV文件加载参考SMILES"""
    try:
        import pandas as pd
        df = pd.read_csv(file_path)
        if 'SMILES' in df.columns:
            # 过滤掉无效的SMILES
            valid_smiles = []
            for smiles in df['SMILES'].dropna():
                mol = Chem.MolFromSmiles(smiles)
                if mol is not None:
                    valid_smiles.append(smiles)
            
            print(f"从 {file_path} 加载了 {len(valid_smiles)} 个有效SMILES")
            return valid_smiles
        else:
            print(f"警告: {file_path} 中没有找到SMILES列")
            return []
    except Exception as e:
        print(f"加载参考SMILES时出错: {e}")
        return []

def save_molecules(molecules, output_dir, images_dir, reference_smiles=None, max_atoms=25, min_atoms=8):
    """保存分子到文件和图像"""
    # 使用默认目标属性
    target_props = default_target_properties
    
    smiles_file = os.path.join(output_dir, 'molecules.smi')
    with open(smiles_file, 'w') as f:
        for i, mol in enumerate(molecules):
            if mol:
                try:
                    # 确保分子有效
                    Chem.SanitizeMol(mol)
                    
                    # 移除氢原子以便可视化
                    mol_no_h = Chem.RemoveAllHs(mol)
                    
                    # 清理分子，移除孤立的原子和片段
                    frags = Chem.GetMolFrags(mol_no_h, asMols=True, sanitizeFrags=True)
                    if len(frags) > 1:
                        # 如果有多个片段，选择最大的片段
                        largest_frag = max(frags, key=lambda x: x.GetNumAtoms())
                        mol_no_h = largest_frag
                    
                    smiles = Chem.MolToSmiles(mol_no_h, isomericSmiles=True, canonical=True)
                    
                    # 保存SMILES
                    f.write(f"{smiles}\n")
                    
                    # 保存图像
                    img_file = os.path.join(images_dir, f'molecule_{i+1}.png')
                    Draw.MolToFile(mol_no_h, img_file)
                    
                    # 计算适应度
                    print(f"\n评估分子 {i+1}:")
                    fitness = evaluate_fitness(mol_no_h, reference_smiles, target_props, max_atoms=max_atoms, min_atoms=min_atoms)
                    
                    # 打印信息
                    print(f"分子 {i+1}: 适应度 = {fitness:.4f}, SMILES = {smiles}")
                except Exception as e:
                    print(f"处理分子 {i+1} 时出错: {e}")
    
    print(f"\n分子SMILES已保存到 {smiles_file}")
    print(f"分子图像已保存到 {images_dir} 目录")

def evaluate_fitness(mol, reference_smiles=None, target_properties=None, max_atoms=25, min_atoms=8):
    """评估分子的适应度"""
    if mol is None:
        return 0.0
    
    try:
        # 确保分子有效
        if not is_valid_mol(mol):
            return 0.0
        
        # 移除所有氢原子
        mol = Chem.RemoveAllHs(mol)
        
        # 清理分子，移除孤立的原子和片段
        frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
        if len(frags) > 1:
            # 如果有多个片段，选择最大的片段
            largest_frag = max(frags, key=lambda x: x.GetNumAtoms())
            mol = largest_frag
        
        # 获取分子大小
        heavy_atoms = mol.GetNumHeavyAtoms()
        
        # 获取SMILES字符串用于调试
        smiles = Chem.MolToSmiles(mol)
        
        # 检查分子大小，使用传入的参数
        if heavy_atoms > max_atoms:  # 使用传入的最大原子数限制
            print(f"分子过大 ({heavy_atoms} 重原子): {smiles[:50]}...")
            return 0.0  # 直接拒绝过大的分子
        elif heavy_atoms < min_atoms:  # 使用传入的最小原子数限制
            print(f"分子过小 ({heavy_atoms} 重原子): {smiles}")
            return 0.1  # 给予很低的适应度
        
        # 计算分子属性
        properties = calculate_molecular_properties(mol)
        if properties is None:
            return 0.0
        
        # 获取关键属性
        qed = properties.get('qed', 0.0)
        sa_score = properties.get('sa_score', 10.0)
        mol_weight = properties.get('molecular_weight', 0.0)
        logp = properties.get('logp', 0.0)
        tpsa = properties.get('tpsa', 0.0)
        hbd = properties.get('hbd', 0)
        hba = properties.get('hba', 0)
        rotatable_bonds = properties.get('rotatable_bonds', 0)
        
        # 打印关键属性用于调试
        print(f"分子属性: MW={mol_weight:.1f}, LogP={logp:.1f}, TPSA={tpsa:.1f}, QED={qed:.2f}, SA={sa_score:.1f}")
        
        # 药物性规则检查
        # Lipinski规则：MW<500, logP<5, HBD<5, HBA<10
        lipinski_violations = 0
        if mol_weight > 500: lipinski_violations += 1
        if logp > 5: lipinski_violations += 1
        if hbd > 5: lipinski_violations += 1
        if hba > 10: lipinski_violations += 1
        
        # Veber规则：rotatable_bonds <= 10, TPSA <= 140
        veber_violations = 0
        if rotatable_bonds > 10: veber_violations += 1
        if tpsa > 140: veber_violations += 1
        
        # 计算基础适应度
        base_fitness = 0.0
        
        # QED贡献 (0-0.6) - 大幅增加QED权重
        qed_contribution = min(0.6, qed * 0.8)
        base_fitness += qed_contribution
        
        # SA分数贡献 (0-0.3)
        # SA分数范围通常为1-10，值越低越好
        # 只有SA < 5.0的分子才有贡献
        if sa_score < 5.0:
            sa_contribution = max(0.0, min(0.3, (5.0 - sa_score) / 4.0 * 0.3))
        else:
            sa_contribution = 0.0
        base_fitness += sa_contribution
        
        # 药物规则贡献 (0-0.2)
        rule_violations = lipinski_violations + veber_violations
        if rule_violations == 0:
            rule_contribution = 0.2  # 完全符合规则
        elif rule_violations == 1:
            rule_contribution = 0.1  # 违反1条规则
        else:
            rule_contribution = 0.0  # 违反多条规则
        base_fitness += rule_contribution
        
        # 分子大小适中性贡献 (0-0.1)
        size_contribution = 0.0
        if 10 <= heavy_atoms <= 20:  # 理想的药物分子大小
            size_contribution = 0.1
        elif 8 <= heavy_atoms < 10 or 20 < heavy_atoms <= 25:
            size_contribution = 0.05
        base_fitness += size_contribution
        
        # 打印各部分贡献
        print(f"适应度组成: QED={qed_contribution:.2f}, SA={sa_contribution:.2f}, 规则={rule_contribution:.2f}, 大小={size_contribution:.2f}")
        
        # 确定最终适应度
        # 药物类分子必须满足: QED > 0.5, SA < 4.0, 违反药物规则不超过1条
        if qed >= 0.5 and sa_score < 4.0 and rule_violations <= 1:
            # 有效药物分子，适应度至少为0.8
            fitness = max(0.8, base_fitness)
            print(f"有效药物分子，适应度={fitness:.4f}")
        elif qed >= 0.4 and sa_score < 5.0 and rule_violations <= 2:
            # 潜在药物分子，适应度在0.6-0.8之间
            fitness = min(0.79, max(0.6, base_fitness))
            print(f"潜在药物分子，适应度={fitness:.4f}")
        else:
            # 无效药物分子，适应度上限为0.5
            fitness = min(0.5, base_fitness)
            print(f"无效药物分子，适应度={fitness:.4f}")
        
        return fitness
    
    except Exception as e:
        print(f"评估适应度时出错: {str(e)}")
        return 0.0

# 确保calculate_molecular_properties函数正确实现
def calculate_molecular_properties(mol):
    """计算分子的各种属性"""
    if mol is None:
        return None
        
    try:
        # 确保分子有效
        mol = Chem.AddHs(mol)  # 添加氢原子以获得更准确的属性
        
        # 计算基本属性
        properties = {}
        
        # 分子量
        properties['molecular_weight'] = Descriptors.MolWt(mol)
        
        # LogP (油水分配系数)
        properties['logp'] = Descriptors.MolLogP(mol)
        
        # 拓扑极性表面积 (TPSA)
        properties['tpsa'] = Descriptors.TPSA(mol)
        
        # 氢键供体数量
        properties['hbd'] = Descriptors.NumHDonors(mol)
        
        # 氢键受体数量
        properties['hba'] = Descriptors.NumHAcceptors(mol)
        
        # 可旋转键数量
        properties['rotatable_bonds'] = Descriptors.NumRotatableBonds(mol)
        
        # 芳香环数量
        properties['aromatic_rings'] = Descriptors.NumAromaticRings(mol)
        
        # 重原子数量
        properties['heavy_atoms'] = mol.GetNumHeavyAtoms()
        
        # QED药物类似性
        try:
            properties['qed'] = QED.qed(mol)
        except:
            properties['qed'] = 0.0
        
        # 合成可行性评分 (SA Score)
        try:
            from rdkit.Chem import sascorer
            properties['sa_score'] = sascorer.calculateScore(mol)
        except:
            properties['sa_score'] = 10.0  # 默认为最差值
        
        # 移除氢原子以恢复原始分子
        mol = Chem.RemoveHs(mol)
        
        return properties
    except Exception as e:
        print(f"计算分子属性时出错: {str(e)}")
        return None

def compare_property_values(value1, value2):
    """比较两个属性值的相似度"""
    if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
        # 数值型属性
        max_val = max(abs(value1), abs(value2))
        if max_val < 1e-6:  # 两个值都接近零
            return 1.0
        diff = abs(value1 - value2) / max_val
        return max(0.0, 1.0 - min(diff, 1.0))
    elif isinstance(value1, str) and isinstance(value2, str):
        # 字符串属性
        if value1 == value2:
            return 1.0
        else:
            return 0.0
    elif isinstance(value1, bool) and isinstance(value2, bool):
        # 布尔属性
        return 1.0 if value1 == value2 else 0.0
    else:
        # 其他类型
        return 0.0


def is_valid_mol(mol):
    """检查分子是否有效"""
    if mol is None:
        return False
        
    try:
        # 检查分子是否有原子
        if mol.GetNumAtoms() == 0:
            return False
            
        # 尝试计算分子的指纹，这会触发多种验证
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
        
        # 检查分子是否可以被规范化
        smiles = Chem.MolToSmiles(mol)
        mol2 = Chem.MolFromSmiles(smiles)
        
        return mol2 is not None
    except:
        return False

# 保留原有函数作为别名，确保向后兼容
def is_valid_molecule(mol):
    """检查分子是否有效（别名）"""
    return is_valid_mol(mol)

def fix_molecule(mol):
    """修复分子结构中的问题"""
    if mol is None:
        return None
        
    try:
        # 创建可写分子
        rwmol = Chem.RWMol(mol)
        
        # 尝试修复价态问题
        for atom in rwmol.GetAtoms():
            atom.UpdatePropertyCache(strict=False)
            
        # 尝试修复键的问题
        for bond in rwmol.GetBonds():
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            begin_atom = rwmol.GetAtomWithIdx(begin_idx)
            end_atom = rwmol.GetAtomWithIdx(end_idx)
            
            # 检查原子的价态是否超出
            begin_valence = begin_atom.GetExplicitValence()
            end_valence = end_atom.GetExplicitValence()
            
            begin_max_valence = get_max_valence(begin_atom.GetAtomicNum())
            end_max_valence = get_max_valence(end_atom.GetAtomicNum())
            
            # 如果价态超出，将键降级为单键
            if (begin_valence > begin_max_valence or end_valence > end_max_valence) and bond.GetBondType() != Chem.BondType.SINGLE:
                rwmol.RemoveBond(begin_idx, end_idx)
                rwmol.AddBond(begin_idx, end_idx, Chem.BondType.SINGLE)
        
        # 如果分子为空，返回None
        if rwmol.GetNumAtoms() == 0:
            return None
            
        # 尝试获取修复后的分子
        try:
            fixed_mol = rwmol.GetMol()
            Chem.SanitizeMol(fixed_mol)
            return fixed_mol
        except:
            # 如果仍然无效，尝试更激进的修复
            try:
                # 移除所有非单键
                for bond in rwmol.GetBonds():
                    begin_idx = bond.GetBeginAtomIdx()
                    end_idx = bond.GetEndAtomIdx()
                    if bond.GetBondType() != Chem.BondType.SINGLE:
                        rwmol.RemoveBond(begin_idx, end_idx)
                        rwmol.AddBond(begin_idx, end_idx, Chem.BondType.SINGLE)
                        
                # 尝试获取修复后的分子
                fixed_mol = rwmol.GetMol()
                Chem.SanitizeMol(fixed_mol)
                return fixed_mol
            except:
                return None
    except:
        return None

def get_max_valence(atomic_num):
    """获取原子的最大价态"""
    if atomic_num == 6:  # 碳
        return 4
    elif atomic_num == 7:  # 氮
        return 3
    elif atomic_num == 8:  # 氧
        return 2
    elif atomic_num in [1, 9, 17]:  # 氢、氟、氯
        return 1
    elif atomic_num == 16:  # 硫
        return 6
    elif atomic_num == 15:  # 磷
        return 5
    elif atomic_num == 5:   # 硼
        return 3
    elif atomic_num == 14:  # 硅
        return 4
    elif atomic_num == 34:  # 硒
        return 6
    elif atomic_num == 35:  # 溴
        return 1
    elif atomic_num == 53:  # 碘
        return 1
    else:
        return 0  # 对于未知原子，返回0

# 默认目标属性
default_target_properties = {
    'molecular_weight': 400.0,
    'logp': 2.5,
    'tpsa': 90.0,
    'hbd': 2,
    'hba': 5,
    'rotatable_bonds': 5,
    'aromatic_rings': 2,
    'qed': 0.7,
    'sa_score': 3.5
}