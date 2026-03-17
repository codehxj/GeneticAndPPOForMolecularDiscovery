import random
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

def evaluate_fitness(mol, reference_smiles=None, target_properties=None, max_atoms=25, min_atoms=8):
    """评估分子的适应度，考虑与参考分子的相似性"""
    if mol is None:
        return 0.0
    
    try:
        # 首先检查分子是否有效
        if not Chem.SanitizeMol(Chem.Mol(mol), catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE:
            return 0.1  # 给予无效分子一个很低但非零的适应度
        
        # 计算一些基本属性作为适应度
        num_atoms = mol.GetNumAtoms()
        num_rings = Chem.rdMolDescriptors.CalcNumRings(mol)
        num_hetero = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
        
        # 计算更多分子描述符以增加多样性评估
        logp = Chem.Crippen.MolLogP(mol)
        tpsa = Chem.rdMolDescriptors.CalcTPSA(mol)
        num_rotatable_bonds = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
        
        # 更复杂的适应度函数，考虑多个因素
        fitness = 0.3 + 0.1 * num_rings + 0.05 * num_hetero
        
        # 奖励适度的脂溶性
        if -2 < logp < 5:
            fitness += 0.1
            
        # 奖励适度的极性表面积
        if 20 < tpsa < 140:
            fitness += 0.1
            
        # 奖励适度的可旋转键数量
        if 0 < num_rotatable_bonds < 10:
            fitness += 0.05
        
        # 限制分子大小
        if num_atoms > max_atoms:
            fitness *= 0.5
        elif num_atoms < min_atoms:
            fitness *= 0.7
        
        # 如果有参考分子，考虑与参考分子的相似性
        if reference_smiles and len(reference_smiles) > 0:
            try:
                # 计算当前分子的SMILES
                current_smiles = Chem.MolToSmiles(mol)
                
                # 如果分子与参考分子完全相同，降低适应度
                if current_smiles in reference_smiles:
                    fitness *= 0.5
                else:
                    # 计算与参考分子的平均相似度
                    current_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
                    
                    similarities = []
                    for ref_smiles in random.sample(reference_smiles, min(5, len(reference_smiles))):
                        ref_mol = Chem.MolFromSmiles(ref_smiles)
                        if ref_mol:
                            ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2)
                            similarity = DataStructs.TanimotoSimilarity(current_fp, ref_fp)
                            similarities.append(similarity)
                    
                    if similarities:
                        avg_similarity = sum(similarities) / len(similarities)
                        # 奖励适度的相似性，但不要太相似
                        if 0.3 < avg_similarity < 0.7:
                            fitness += 0.1
                        elif avg_similarity > 0.9:
                            fitness *= 0.7  # 惩罚过于相似的分子
            except:
                pass
        
        # 添加随机因素以避免过早收敛
        fitness += random.uniform(0, 0.05)
        
        return min(fitness, 1.0)  # 将适应度限制在[0,1]范围内
    except:
        return 0.0