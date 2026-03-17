import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem, QED
from rdkit.Chem import rdMolDescriptors
from .druglikeness_checker import DruglikenessChecker

class MoleculeEvaluator:
    """评估分子的各种性质，集成可成药性检查"""
    
    def __init__(self):
        self.druglikeness_checker = DruglikenessChecker()
    
    def calculate_properties(self, mol):
        """计算分子的各种药物化学性质"""
        try:
            # 计算基本性质
            logp = Descriptors.MolLogP(mol)
            tpsa = Descriptors.TPSA(mol)
            qed = QED.qed(mol)
            mw = Descriptors.MolWt(mol)
            hba = rdMolDescriptors.CalcNumHBA(mol)
            hbd = rdMolDescriptors.CalcNumHBD(mol)
            rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
            heavy_atoms = mol.GetNumHeavyAtoms()
            
            # 计算可成药性相关属性
            druglikeness_result = self.druglikeness_checker.calculate_druglikeness_score(mol)
            
            # 计算分子复杂性
            ring_info = mol.GetRingInfo()
            num_rings = ring_info.NumRings()
            aromatic_atoms = sum(1 for atom in mol.GetAtoms() if atom.GetIsAromatic())
            
            # 计算分子的柔性（可旋转键密度）
            flexibility = rotatable_bonds / max(heavy_atoms, 1)
            
            # 计算分子的极性表面积密度
            tpsa_density = tpsa / max(mw, 1)
            
            return {
                'logp': logp,
                'tpsa': tpsa,
                'qed': qed,
                'mw': mw,
                'hba': hba,
                'hbd': hbd,
                'rotatable_bonds': rotatable_bonds,
                'heavy_atoms': heavy_atoms,
                'num_rings': num_rings,
                'aromatic_atoms': aromatic_atoms,
                'flexibility': flexibility,
                'tpsa_density': tpsa_density,
                'druglikeness_score': druglikeness_result['total_score'],
                'is_druglike': druglikeness_result['is_druglike'],
                'lipinski_violations': druglikeness_result['lipinski']['violations'],
                'veber_violations': druglikeness_result['veber']['violations'],
                'is_pains': druglikeness_result['pains']['is_pains'],
                'structural_alerts': len(druglikeness_result.get('structural', {}).get('problems', []))
            }
        except Exception as e:
            print(f"属性计算失败: {e}")
            return {
                'logp': 0.0,
                'tpsa': 0.0,
                'qed': 0.0,
                'mw': 0.0,
                'hba': 0.0,
                'hbd': 0.0,
                'rotatable_bonds': 0.0,
                'heavy_atoms': 0.0,
                'num_rings': 0.0,
                'aromatic_atoms': 0.0,
                'flexibility': 0.0,
                'tpsa_density': 0.0,
                'druglikeness_score': 0.0,
                'is_druglike': False,
                'lipinski_violations': 5,
                'veber_violations': 2,
                'is_pains': True,
                'structural_alerts': 10
            }
    
    def calculate_fitness(self, mol, target_properties=None, reference_smiles=None, protein_sequence=None, affinity_predictor=None):
        """计算分子的综合适应度分数"""
        # 默认目标属性（基于药物类似分子的理想范围）
        if target_properties is None:
            target_properties = {
                'logp': 2.5,  # 理想的logP值（1-3范围）
                'tpsa': 90.0,  # 理想的TPSA值（60-140范围）
                'qed': 0.7,  # 理想的QED值（>0.5）
                'mw': 400.0,  # 理想的分子量（300-500范围）
                'hba': 6,  # 理想的氢键受体数（≤10）
                'hbd': 2,  # 理想的氢键供体数（≤5）
                'rotatable_bonds': 6,  # 理想的可旋转键数（≤10）
                'heavy_atoms': 25,  # 理想的重原子数（15-35范围）
                'num_rings': 3,  # 理想的环数（2-4）
                'flexibility': 0.25,  # 理想的柔性（0.1-0.3）
                'druglikeness_score': 0.8  # 理想的可成药性分数
            }
        
        # 计算当前分子的属性
        properties = self.calculate_properties(mol)
        
        # 计算各属性的得分
        scores = {}
        
        # 1. 基础药物化学属性评分
        # LogP评分（理想范围：1-3）
        logp = properties['logp']
        if 1.0 <= logp <= 3.0:
            scores['logp'] = 1.0
        elif 0.5 <= logp < 1.0 or 3.0 < logp <= 4.0:
            scores['logp'] = 0.8
        elif 0.0 <= logp < 0.5 or 4.0 < logp <= 5.0:
            scores['logp'] = 0.5
        else:
            scores['logp'] = 0.2
        
        # TPSA评分（理想范围：60-140）
        tpsa = properties['tpsa']
        if 60.0 <= tpsa <= 140.0:
            scores['tpsa'] = 1.0
        elif 40.0 <= tpsa < 60.0 or 140.0 < tpsa <= 160.0:
            scores['tpsa'] = 0.7
        else:
            scores['tpsa'] = 0.3
        
        # 分子量评分（理想范围：300-500）
        mw = properties['mw']
        if 300.0 <= mw <= 500.0:
            scores['mw'] = 1.0
        elif 250.0 <= mw < 300.0 or 500.0 < mw <= 600.0:
            scores['mw'] = 0.7
        elif 200.0 <= mw < 250.0 or 600.0 < mw <= 700.0:
            scores['mw'] = 0.4
        else:
            scores['mw'] = 0.1
        
        # QED评分
        qed = properties['qed']
        scores['qed'] = qed  # QED本身就是0-1的分数
        
        # 2. 可成药性评分
        scores['druglikeness'] = properties['druglikeness_score']
        
        # 3. 结构质量评分
        # Lipinski规则违反惩罚
        lipinski_violations = properties['lipinski_violations']
        scores['lipinski'] = max(0.0, 1.0 - lipinski_violations * 0.25)
        
        # Veber规则违反惩罚
        veber_violations = properties['veber_violations']
        scores['veber'] = max(0.0, 1.0 - veber_violations * 0.5)
        
        # PAINS惩罚
        scores['pains'] = 0.0 if properties['is_pains'] else 1.0
        
        # 结构警报惩罚
        structural_alerts = properties['structural_alerts']
        scores['structural'] = max(0.0, 1.0 - structural_alerts * 0.1)
        
        # 4. 分子复杂性评分（提高复杂度要求）
        # 环数评分（理想：3-6个环，提高复杂度）
        num_rings = properties['num_rings']
        if 3 <= num_rings <= 6:
            scores['rings'] = 1.0
        elif num_rings == 2 or num_rings == 7:
            scores['rings'] = 0.8
        elif num_rings == 1 or num_rings == 8:
            scores['rings'] = 0.5
        elif num_rings == 0:
            scores['rings'] = 0.2  # 降低无环分子评分
        else:
            scores['rings'] = 0.3
        
        # 芳香性评分（提高芳香性要求）
        aromatic_ratio = properties['aromatic_atoms'] / max(properties['heavy_atoms'], 1)
        if 0.4 <= aromatic_ratio <= 0.8:  # 提高芳香性比例要求
            scores['aromatic'] = 1.0
        elif 0.2 <= aromatic_ratio < 0.4 or 0.8 < aromatic_ratio <= 0.9:
            scores['aromatic'] = 0.8
        elif 0.1 <= aromatic_ratio < 0.2:
            scores['aromatic'] = 0.5
        else:
            scores['aromatic'] = 0.2  # 降低无芳香性分子评分
        
        # 柔性评分（理想：0.15-0.4，提高柔性要求）
        flexibility = properties['flexibility']
        if 0.15 <= flexibility <= 0.4:
            scores['flexibility'] = 1.0
        elif 0.1 <= flexibility < 0.15 or 0.4 < flexibility <= 0.5:
            scores['flexibility'] = 0.8
        elif 0.05 <= flexibility < 0.1 or 0.5 < flexibility <= 0.6:
            scores['flexibility'] = 0.5
        else:
            scores['flexibility'] = 0.2
        
        # 5. 新颖性和相似性评分
        scores['novelty'] = 1.0  # 默认值
        if reference_smiles:
            try:
                from rdkit import DataStructs
                from rdkit.Chem import AllChem
                
                mol_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
                current_smiles = Chem.MolToSmiles(mol)
                
                similarities = []
                for ref_smiles in reference_smiles[:50]:  # 限制计算量
                    if current_smiles == ref_smiles:
                        scores['novelty'] = 0.1  # 完全相同
                        break
                    
                    try:
                        ref_mol = Chem.MolFromSmiles(ref_smiles)
                        if ref_mol:
                            ref_fp = AllChem.GetMorganFingerprintAsBitVect(ref_mol, 2, nBits=1024)
                            sim = DataStructs.TanimotoSimilarity(mol_fp, ref_fp)
                            similarities.append(sim)
                    except:
                        continue
                
                if similarities:
                    max_sim = max(similarities)
                    if max_sim > 0.9:
                        scores['novelty'] = 0.3
                    elif max_sim > 0.8:
                        scores['novelty'] = 0.6
                    elif max_sim > 0.7:
                        scores['novelty'] = 0.8
                    else:
                        scores['novelty'] = 1.0
            except:
                scores['novelty'] = 0.8
        
        # 6. 亲和力评分（如果提供）
        scores['affinity'] = 0.5  # 默认中等分数
        if protein_sequence and affinity_predictor:
            try:
                smiles = Chem.MolToSmiles(mol)
                affinity_value = affinity_predictor.predict_affinity(smiles, protein_sequence)
                # 假设pKd范围为4-12，标准化到0-1
                normalized_affinity = max(0.0, min(1.0, (affinity_value - 4.0) / 8.0))
                scores['affinity'] = normalized_affinity
                
                # 对高亲和力给予额外奖励
                if normalized_affinity > 0.8:
                    scores['affinity'] += 0.2
            except:
                scores['affinity'] = 0.5
        
        # 7. 组合得分（调整权重以强调可成药性）
        if protein_sequence and affinity_predictor:
            # 有亲和力约束时的权重
            weights = {
                'logp': 0.08,
                'tpsa': 0.08,
                'mw': 0.05,
                'qed': 0.10,
                'druglikeness': 0.20,  # 可成药性权重最高
                'lipinski': 0.08,
                'veber': 0.05,
                'pains': 0.10,  # PAINS惩罚权重高
                'structural': 0.05,
                'rings': 0.03,
                'aromatic': 0.03,
                'flexibility': 0.03,
                'novelty': 0.07,
                'affinity': 0.15  # 亲和力权重较高
            }
        else:
            # 无亲和力约束时的权重
            weights = {
                'logp': 0.10,
                'tpsa': 0.10,
                'mw': 0.08,
                'qed': 0.12,
                'druglikeness': 0.25,  # 可成药性权重最高
                'lipinski': 0.10,
                'veber': 0.08,
                'pains': 0.12,  # PAINS惩罚权重高
                'structural': 0.08,
                'rings': 0.05,
                'aromatic': 0.05,
                'flexibility': 0.05,
                'novelty': 0.10,
                'affinity': 0.0  # 无亲和力时权重为0
            }
        
        # 计算加权适应度
        fitness = sum(scores.get(prop, 0.0) * weight for prop, weight in weights.items())
        
        # 应用严格的可成药性惩罚
        if not properties['is_druglike']:
            fitness *= 0.6  # 不符合可成药性的分子得分大幅降低
        
        if properties['is_pains']:
            fitness *= 0.3  # PAINS分子得分大幅降低
        
        # 确保适应度在合理范围内
        fitness = max(0.0, min(1.0, fitness))
        
        return fitness, properties, scores
    
    def evaluate_molecule_quality(self, mol):
        """评估分子的整体质量"""
        properties = self.calculate_properties(mol)
        
        quality_metrics = {
            'is_valid': True,
            'is_druglike': properties['is_druglike'],
            'has_pains': properties['is_pains'],
            'lipinski_compliant': properties['lipinski_violations'] == 0,
            'veber_compliant': properties['veber_violations'] == 0,
            'qed_score': properties['qed'],
            'druglikeness_score': properties['druglikeness_score'],
            'structural_quality': 1.0 - properties['structural_alerts'] * 0.1
        }
        
        # 计算整体质量分数
        quality_score = (
            quality_metrics['qed_score'] * 0.3 +
            quality_metrics['druglikeness_score'] * 0.4 +
            quality_metrics['structural_quality'] * 0.3
        )
        
        # 应用惩罚
        if not quality_metrics['is_druglike']:
            quality_score *= 0.7
        if quality_metrics['has_pains']:
            quality_score *= 0.5
        if not quality_metrics['lipinski_compliant']:
            quality_score *= 0.8
        
        quality_metrics['overall_quality'] = quality_score
        
        return quality_metrics