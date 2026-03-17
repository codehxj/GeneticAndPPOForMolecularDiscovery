"""
可成药性检查模块
包含Lipinski规则、PAINS过滤、毒性预测等功能
"""

import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, QED, Crippen, Lipinski
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams
import warnings
warnings.filterwarnings('ignore')

class DruglikenessChecker:
    """可成药性检查器"""
    
    def __init__(self):
        """初始化可成药性检查器"""
        # 初始化PAINS过滤器
        try:
            params = FilterCatalogParams()
            params.AddCatalog(FilterCatalogParams.FilterCatalogs.PAINS)
            self.pains_catalog = FilterCatalog(params)
        except:
            self.pains_catalog = None
            print("警告：PAINS过滤器初始化失败")
        
        # 定义可成药性规则的理想范围
        self.ideal_ranges = {
            'molecular_weight': (150, 500),  # 分子量
            'logp': (-0.4, 5.6),            # 脂水分配系数
            'hbd': (0, 5),                  # 氢键供体
            'hba': (2, 10),                 # 氢键受体
            'tpsa': (20, 130),              # 拓扑极性表面积
            'rotatable_bonds': (0, 10),     # 可旋转键
            'aromatic_rings': (1, 4),       # 芳香环数量
            'heavy_atoms': (10, 70),        # 重原子数
            'formal_charge': (-2, 2),       # 形式电荷
        }
        
        # 严格过滤阈值
        self.strict_thresholds = {
            'min_druglikeness_score': 0.4,  # 最低可成药性分数
            'max_pains_alerts': 0,  # 最大PAINS警报数
            'max_structural_alerts': 2,  # 最大结构警报数
            'min_qed': 0.3,  # 最低QED分数
            'max_lipinski_violations': 1,  # 最大Lipinski违规数
            'max_veber_violations': 1,  # 最大Veber违规数
        }
        
        # 毒性相关的SMARTS模式（扩展版本）
        self.toxicity_patterns = {
            'reactive_groups': [
                '[N+](=O)[O-]',  # 硝基
                'C(=O)Cl',  # 酰氯
                'S(=O)(=O)Cl',  # 磺酰氯
                'C#N',  # 腈基
                '[N,O,S][N+](=O)[O-]',  # 硝酸酯
                'C=C-C=O',  # α,β-不饱和羰基
                'c1ccc([N+](=O)[O-])cc1',  # 硝基苯
                'C(=O)N([H])N([H])',  # 肼基
                'N=N',  # 偶氮基
                'C(=S)',  # 硫代羰基
                '[Cl,Br,I]C(=O)',  # 卤代酰基
                'P(=O)(O)(O)',  # 磷酸基
                'S(=O)(=O)N',  # 磺胺基
                'C1=CC=C(C=C1)O',  # 酚基（某些情况下）
                'c1ccc(cc1)N',  # 苯胺（某些情况下）
            ],
            'mutagenic_groups': [
                'c1ccc2c(c1)ccc3c2ccc4c3cccc4',  # 多环芳烃
                'c1ccc(cc1)N=Nc2ccccc2',  # 偶氮苯
                'C1=CC=C(C=C1)N([H])N([H])',  # 苯肼
                'c1ccc(cc1)[N+](=O)[O-]',  # 硝基苯
                'C(=O)N([H])N([H])',  # 肼基化合物
                'N=C=S',  # 异硫氰酸酯
                'C(=O)C(=O)',  # 二酮
                'C=C-C(=O)-C=C',  # 共轭二烯酮
            ],
            'hepatotoxic_groups': [
                'c1ccc(cc1)N',  # 苯胺
                'C(=O)N',  # 酰胺（某些情况）
                'c1ccc(cc1)S',  # 苯硫醚
                'C(F)(F)F',  # 三氟甲基
                'c1ccc(cc1)Cl',  # 氯苯
                'C(=O)C(=O)N',  # 酰胺酮
                'c1ccc2c(c1)cccc2',  # 萘
                'C1=CC=C(C=C1)C(=O)',  # 苯甲酰基
            ],
            'cardiotoxic_groups': [
                'c1ccc2c(c1)nc3ccccc3n2',  # 喹喔啉
                'c1cnc2ccccc2c1',  # 喹啉
                'c1ccc2c(c1)ncc3ccccc32',  # 吖啶
                'C1=CC=C(C=C1)N=C=S',  # 异硫氰酸苯酯
                'c1ccc(cc1)S(=O)(=O)',  # 苯磺酰基
            ]
        }
        
        # 定义有问题的结构模式（SMARTS格式）
        self.problematic_patterns = [
            # 反应性基团
            '[N,O,S][CH2][N,O,S]',  # 不稳定的亚甲基桥
            '[#6]=[#6]-[#6]=[#6]',  # 共轭双键系统（可能不稳定）
            '[N,O,S]-[N,O,S]',      # 直接相连的杂原子
            '[C,c]#[C,c]',          # 三键（可能反应性强）
            # 毒性基团
            '[N+](=O)[O-]',         # 硝基
            'C(=O)Cl',              # 酰氯
            'S(=O)(=O)Cl',          # 磺酰氯
            '[As,Hg,Pb,Cd]',        # 重金属
            # 不稳定结构
            '[C,c][O,o][O,o][C,c]', # 过氧化物
            'N=N',                  # 偶氮基团
        ]
        
        # 编译SMARTS模式
        self.compiled_patterns = {}
        for category, patterns in self.toxicity_patterns.items():
            self.compiled_patterns[category] = []
            for pattern in patterns:
                try:
                    mol_pattern = Chem.MolFromSmarts(pattern)
                    if mol_pattern:
                        self.compiled_patterns[category].append(mol_pattern)
                except:
                    continue
    
    def check_lipinski_rule(self, mol):
        """
        检查Lipinski五规则
        
        参数:
        mol: RDKit分子对象
        
        返回:
        dict: 包含各项指标和违规数量的字典
        """
        if mol is None:
            return {'violations': 5, 'details': {}}
        
        try:
            # 计算各项指标
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            hbd = Descriptors.NumHDonors(mol)
            hba = Descriptors.NumHAcceptors(mol)
            
            # 检查违规
            violations = 0
            details = {
                'molecular_weight': mw,
                'logp': logp,
                'hbd': hbd,
                'hba': hba,
                'mw_violation': mw > 500,
                'logp_violation': logp > 5,
                'hbd_violation': hbd > 5,
                'hba_violation': hba > 10
            }
            
            if details['mw_violation']: violations += 1
            if details['logp_violation']: violations += 1
            if details['hbd_violation']: violations += 1
            if details['hba_violation']: violations += 1
            
            return {'violations': violations, 'details': details}
        except:
            return {'violations': 5, 'details': {}}
    
    def check_veber_rule(self, mol):
        """
        检查Veber规则（口服生物利用度）
        
        参数:
        mol: RDKit分子对象
        
        返回:
        dict: 包含各项指标和违规数量的字典
        """
        if mol is None:
            return {'violations': 2, 'details': {}}
        
        try:
            # 计算各项指标
            tpsa = Descriptors.TPSA(mol)
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            
            # 检查违规
            violations = 0
            details = {
                'tpsa': tpsa,
                'rotatable_bonds': rotatable_bonds,
                'tpsa_violation': tpsa > 140,
                'rotatable_bonds_violation': rotatable_bonds > 10
            }
            
            if details['tpsa_violation']: violations += 1
            if details['rotatable_bonds_violation']: violations += 1
            
            return {'violations': violations, 'details': details}
        except:
            return {'violations': 2, 'details': {}}
    
    def check_pains_filter(self, mol):
        """
        检查PAINS（Pan Assay Interference Compounds）过滤器
        
        参数:
        mol: RDKit分子对象
        
        返回:
        dict: 包含是否通过PAINS过滤的信息
        """
        if mol is None or self.pains_catalog is None:
            return {'is_pains': True, 'matches': []}
        
        try:
            # 检查PAINS匹配
            matches = []
            for i in range(self.pains_catalog.GetNumEntries()):
                entry = self.pains_catalog.GetEntry(i)
                if mol.HasSubstructMatch(entry.GetPattern()):
                    matches.append(entry.GetDescription())
            
            return {
                'is_pains': len(matches) > 0,
                'matches': matches
            }
        except:
            return {'is_pains': True, 'matches': []}
    
    def check_toxicity_alerts(self, mol):
        """
        检查毒性警报
        
        参数:
        mol: RDKit分子对象
        
        返回:
        dict: 包含毒性警报信息的字典
        """
        if mol is None:
            return {'total_alerts': 10, 'alerts_by_category': {}, 'is_toxic': True}
        
        alerts_by_category = {}
        total_alerts = 0
        
        try:
            for category, patterns in self.compiled_patterns.items():
                alerts = []
                for pattern in patterns:
                    if mol.HasSubstructMatch(pattern):
                        alerts.append(Chem.MolToSmarts(pattern))
                
                alerts_by_category[category] = alerts
                total_alerts += len(alerts)
            
            # 判断是否有毒性风险
            is_toxic = (
                total_alerts > 3 or  # 总警报数过多
                len(alerts_by_category.get('reactive_groups', [])) > 1 or  # 多个反应性基团
                len(alerts_by_category.get('mutagenic_groups', [])) > 0 or  # 任何致突变基团
                len(alerts_by_category.get('hepatotoxic_groups', [])) > 2 or  # 多个肝毒性基团
                len(alerts_by_category.get('cardiotoxic_groups', [])) > 1  # 多个心脏毒性基团
            )
            
            return {
                'total_alerts': total_alerts,
                'alerts_by_category': alerts_by_category,
                'is_toxic': is_toxic
            }
        except:
            return {'total_alerts': 10, 'alerts_by_category': {}, 'is_toxic': True}
    
    def check_advanced_druglikeness(self, mol):
        """
        检查高级可成药性规则（Ghose, Muegge, Egan等）
        
        参数:
        mol: RDKit分子对象
        
        返回:
        dict: 包含高级可成药性检查结果的字典
        """
        if mol is None:
            return {'violations': 10, 'details': {}}
        
        try:
            violations = 0
            details = {}
            
            # Ghose规则
            mw = Descriptors.MolWt(mol)
            logp = Descriptors.MolLogP(mol)
            atoms = mol.GetNumAtoms()
            molar_refractivity = Crippen.MolMR(mol)
            
            details['ghose'] = {
                'mw_violation': not (160 <= mw <= 480),
                'logp_violation': not (-0.4 <= logp <= 5.6),
                'atoms_violation': not (20 <= atoms <= 70),
                'mr_violation': not (40 <= molar_refractivity <= 130)
            }
            violations += sum(details['ghose'].values())
            
            # Muegge规则
            tpsa = Descriptors.TPSA(mol)
            rings = Descriptors.RingCount(mol)
            carbons = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() == 6)
            heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            
            details['muegge'] = {
                'mw_violation': not (200 <= mw <= 600),
                'logp_violation': not (-2 <= logp <= 5),
                'tpsa_violation': not (tpsa <= 150),
                'rings_violation': not (rings <= 7),
                'carbons_violation': not (carbons >= 5),
                'heteroatoms_violation': not (heteroatoms >= 1),
                'rotatable_violation': not (rotatable_bonds <= 15)
            }
            violations += sum(details['muegge'].values())
            
            # Egan规则（口服生物利用度）
            details['egan'] = {
                'tpsa_violation': not (tpsa <= 131.6),
                'logp_violation': not (logp <= 5.88)
            }
            violations += sum(details['egan'].values())
            
            # 额外的结构检查
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            aliphatic_rings = rings - aromatic_rings
            stereocenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            formal_charge = Chem.rdmolops.GetFormalCharge(mol)
            
            details['structural'] = {
                'aromatic_rings': aromatic_rings,
                'aliphatic_rings': aliphatic_rings,
                'stereocenters': stereocenters,
                'formal_charge': formal_charge,
                'excessive_aromatic': aromatic_rings > 4,
                'excessive_rings': rings > 6,
                'high_charge': abs(formal_charge) > 2,
                'too_flexible': rotatable_bonds > 10
            }
            violations += sum([
                details['structural']['excessive_aromatic'],
                details['structural']['excessive_rings'],
                details['structural']['high_charge'],
                details['structural']['too_flexible']
            ])
            
            # 分子复杂度和柔性评估
            complexity_score = self._calculate_complexity(mol)
            flexibility_score = self._calculate_flexibility(mol)
            
            details['complexity'] = {
                'complexity_score': complexity_score,
                'flexibility_score': flexibility_score,
                'too_complex': complexity_score > 0.8,
                'too_rigid': flexibility_score < 0.2
            }
            violations += sum([
                details['complexity']['too_complex'],
                details['complexity']['too_rigid']
            ])
            
            return {'violations': violations, 'details': details}
        except:
            return {'violations': 10, 'details': {}}
    
    def _calculate_complexity(self, mol):
        """计算分子复杂度"""
        try:
            # 基于环系统、立体中心、杂原子等计算复杂度
            rings = Descriptors.RingCount(mol)
            aromatic_rings = Descriptors.NumAromaticRings(mol)
            stereocenters = len(Chem.FindMolChiralCenters(mol, includeUnassigned=True))
            heteroatoms = sum(1 for atom in mol.GetAtoms() if atom.GetAtomicNum() not in [1, 6])
            
            # 归一化复杂度分数
            complexity = (rings * 0.3 + aromatic_rings * 0.2 + stereocenters * 0.3 + heteroatoms * 0.2) / 20
            return min(1.0, complexity)
        except:
            return 1.0
    
    def _calculate_flexibility(self, mol):
        """计算分子柔性"""
        try:
            rotatable_bonds = Descriptors.NumRotatableBonds(mol)
            total_bonds = mol.GetNumBonds()
            
            if total_bonds == 0:
                return 0.0
            
            flexibility = rotatable_bonds / total_bonds
            return min(1.0, flexibility)
        except:
            return 0.0

    def check_structural_problems(self, mol):
        """
        检查结构问题
        
        参数:
        mol: RDKit分子对象
        
        返回:
        dict: 包含结构问题的信息
        """
        if mol is None:
            return {'problems': ['Invalid molecule'], 'score': 0.0}
        
        problems = []
        
        try:
            # 检查基本结构有效性
            try:
                Chem.SanitizeMol(mol)
            except:
                problems.append('Sanitization failed')
            
            # 检查原子价态
            for atom in mol.GetAtoms():
                try:
                    valence = atom.GetTotalValence()
                    expected_valence = Chem.GetPeriodicTable().GetValenceList(atom.GetAtomicNum())
                    if valence not in expected_valence:
                        problems.append(f'Invalid valence for atom {atom.GetSymbol()}')
                        break
                except:
                    problems.append('Valence calculation failed')
                    break
            
            # 检查有问题的结构模式
            for pattern in self.problematic_patterns:
                try:
                    if mol.HasSubstructMatch(Chem.MolFromSmarts(pattern)):
                        problems.append(f'Contains problematic pattern: {pattern}')
                except:
                    continue
            
            # 检查分子连通性
            if len(Chem.GetMolFrags(mol)) > 1:
                problems.append('Disconnected fragments')
            
            # 计算结构质量分数
            score = max(0.0, 1.0 - len(problems) * 0.2)
            
            return {'problems': problems, 'score': score}
        except:
            return {'problems': ['Structure analysis failed'], 'score': 0.0}
    
    def calculate_druglikeness_score(self, mol):
        """
        计算综合可成药性分数（增强版本）
        
        参数:
        mol: RDKit分子对象
        
        返回:
        dict: 包含各项检查结果和综合分数的字典
        """
        if mol is None:
            return {
                'total_score': 0.0,
                'lipinski': {'violations': 5, 'details': {}},
                'veber': {'violations': 2, 'details': {}},
                'pains': {'is_pains': True, 'matches': []},
                'structural': {'problems': ['Invalid molecule'], 'score': 0.0},
                'toxicity': {'total_alerts': 10, 'is_toxic': True},
                'advanced': {'violations': 10, 'details': {}},
                'qed': 0.0,
                'is_druglike': False,
                'passes_strict_filter': False
            }
        
        try:
            # 执行各项检查
            lipinski_result = self.check_lipinski_rule(mol)
            veber_result = self.check_veber_rule(mol)
            pains_result = self.check_pains_filter(mol)
            structural_result = self.check_structural_problems(mol)
            toxicity_result = self.check_toxicity_alerts(mol)
            advanced_result = self.check_advanced_druglikeness(mol)
            
            # 计算QED
            try:
                qed_score = QED.qed(mol)
            except:
                qed_score = 0.0
            
            # 计算各项分数
            # Lipinski规则分数 (0-1)
            lipinski_score = max(0.0, 1.0 - lipinski_result['violations'] * 0.15)
            
            # Veber规则分数 (0-1)
            veber_score = max(0.0, 1.0 - veber_result['violations'] * 0.25)
            
            # PAINS过滤分数 (0-1)
            pains_score = 0.0 if pains_result['is_pains'] else 1.0
            
            # 结构质量分数
            structural_score = structural_result['score']
            
            # 毒性分数 (0-1)
            toxicity_score = 0.0 if toxicity_result['is_toxic'] else max(0.0, 1.0 - toxicity_result['total_alerts'] * 0.1)
            
            # 高级可成药性分数 (0-1)
            advanced_score = max(0.0, 1.0 - advanced_result['violations'] * 0.05)
            
            # 加权综合分数计算
            weights = {
                'qed': 0.25,           # QED权重
                'lipinski': 0.20,      # Lipinski规则权重
                'veber': 0.15,         # Veber规则权重
                'pains': 0.15,         # PAINS过滤权重
                'structural': 0.10,    # 结构质量权重
                'toxicity': 0.10,      # 毒性检查权重
                'advanced': 0.05       # 高级规则权重
            }
            
            total_score = (
                qed_score * weights['qed'] +
                lipinski_score * weights['lipinski'] +
                veber_score * weights['veber'] +
                pains_score * weights['pains'] +
                structural_score * weights['structural'] +
                toxicity_score * weights['toxicity'] +
                advanced_score * weights['advanced']
            )
            
            # 严格过滤检查
            passes_strict_filter = (
                total_score >= self.strict_thresholds['min_druglikeness_score'] and
                not pains_result['is_pains'] and
                toxicity_result['total_alerts'] <= self.strict_thresholds['max_structural_alerts'] and
                qed_score >= self.strict_thresholds['min_qed'] and
                lipinski_result['violations'] <= self.strict_thresholds['max_lipinski_violations'] and
                veber_result['violations'] <= self.strict_thresholds['max_veber_violations'] and
                not toxicity_result['is_toxic']
            )
            
            # 基本可成药性判断
            is_druglike = (
                total_score >= 0.3 and
                lipinski_result['violations'] <= 2 and
                not pains_result['is_pains'] and
                qed_score >= 0.2 and
                structural_score >= 0.5
            )
            
            return {
                'total_score': min(1.0, max(0.0, total_score)),
                'lipinski': lipinski_result,
                'veber': veber_result,
                'pains': pains_result,
                'structural': structural_result,
                'toxicity': toxicity_result,
                'advanced': advanced_result,
                'qed': qed_score,
                'is_druglike': is_druglike,
                'passes_strict_filter': passes_strict_filter,
                'component_scores': {
                    'qed_score': qed_score,
                    'lipinski_score': lipinski_score,
                    'veber_score': veber_score,
                    'pains_score': pains_score,
                    'structural_score': structural_score,
                    'toxicity_score': toxicity_score,
                    'advanced_score': advanced_score
                }
            }
        except Exception as e:
            print(f"可成药性计算错误: {e}")
            return {
                'total_score': 0.0,
                'lipinski': {'violations': 5, 'details': {}},
                'veber': {'violations': 2, 'details': {}},
                'pains': {'is_pains': True, 'matches': []},
                'structural': {'problems': ['Calculation failed'], 'score': 0.0},
                'toxicity': {'total_alerts': 10, 'is_toxic': True},
                'advanced': {'violations': 10, 'details': {}},
                'qed': 0.0,
                'is_druglike': False,
                'passes_strict_filter': False
            }
    
    def filter_molecules(self, molecules, min_score=0.5, require_druglike=True, use_strict_filter=False):
        """
        过滤分子列表，只保留可成药的分子（增强版本）
        
        参数:
        molecules: 分子列表
        min_score: 最小可成药性分数阈值
        require_druglike: 是否要求严格的可成药性
        use_strict_filter: 是否使用严格过滤器
        
        返回:
        list: 过滤后的分子列表
        """
        filtered_molecules = []
        
        for mol in molecules:
            if mol is None:
                continue
            
            try:
                result = self.calculate_druglikeness_score(mol)
                
                # 根据过滤模式选择标准
                if use_strict_filter:
                    # 使用严格过滤标准
                    if result['passes_strict_filter']:
                        filtered_molecules.append(mol)
                elif require_druglike:
                    # 使用基本可成药性标准
                    if result['is_druglike'] and result['total_score'] >= min_score:
                        filtered_molecules.append(mol)
                else:
                    # 仅使用分数阈值
                    if result['total_score'] >= min_score:
                        filtered_molecules.append(mol)
                        
            except Exception as e:
                print(f"过滤分子时出错: {e}")
                continue
        
        return filtered_molecules
    
    def get_druglikeness_summary(self, mol):
        """
        获取分子可成药性摘要信息
        
        参数:
        mol: RDKit分子对象
        
        返回:
        dict: 包含摘要信息的字典
        """
        if mol is None:
            return {'summary': '无效分子', 'recommendations': []}
        
        try:
            result = self.calculate_druglikeness_score(mol)
            
            # 生成摘要
            summary_parts = []
            recommendations = []
            
            # 总分评估
            total_score = result['total_score']
            if total_score >= 0.7:
                summary_parts.append("优秀的可成药性")
            elif total_score >= 0.5:
                summary_parts.append("良好的可成药性")
            elif total_score >= 0.3:
                summary_parts.append("中等的可成药性")
            else:
                summary_parts.append("较差的可成药性")
            
            # 具体问题分析
            if result['pains']['is_pains']:
                summary_parts.append("含有PAINS结构")
                recommendations.append("移除或修改PAINS结构")
            
            if result['toxicity']['is_toxic']:
                summary_parts.append("存在毒性风险")
                recommendations.append("减少毒性基团")
            
            if result['lipinski']['violations'] > 1:
                summary_parts.append("违反Lipinski规则")
                recommendations.append("优化分子量、LogP或氢键特性")
            
            if result['qed'] < 0.3:
                summary_parts.append("QED分数较低")
                recommendations.append("改善整体药物相似性")
            
            # 严格过滤状态
            if result['passes_strict_filter']:
                summary_parts.append("通过严格过滤")
            
            return {
                'summary': ', '.join(summary_parts),
                'total_score': total_score,
                'recommendations': recommendations,
                'detailed_results': result
            }
            
        except Exception as e:
            return {
                'summary': f'分析失败: {e}',
                'recommendations': ['检查分子结构有效性'],
                'detailed_results': {}
            }
    
    def get_improvement_suggestions(self, mol):
        """
        为分子提供改进建议
        
        参数:
        mol: RDKit分子对象
        
        返回:
        list: 改进建议列表
        """
        if mol is None:
            return ['Invalid molecule - cannot provide suggestions']
        
        suggestions = []
        druglikeness = self.calculate_druglikeness_score(mol)
        
        # 基于Lipinski规则的建议
        lipinski = druglikeness['lipinski']['details']
        if lipinski.get('mw_violation', False):
            suggestions.append('Reduce molecular weight (currently > 500 Da)')
        if lipinski.get('logp_violation', False):
            suggestions.append('Reduce lipophilicity (LogP currently > 5)')
        if lipinski.get('hbd_violation', False):
            suggestions.append('Reduce hydrogen bond donors (currently > 5)')
        if lipinski.get('hba_violation', False):
            suggestions.append('Reduce hydrogen bond acceptors (currently > 10)')
        
        # 基于Veber规则的建议
        veber = druglikeness['veber']['details']
        if veber.get('tpsa_violation', False):
            suggestions.append('Reduce topological polar surface area (currently > 140 Ų)')
        if veber.get('rotatable_bonds_violation', False):
            suggestions.append('Reduce rotatable bonds (currently > 10)')
        
        # 基于PAINS的建议
        if druglikeness['pains']['is_pains']:
            suggestions.append('Remove PAINS substructures to avoid assay interference')
        
        # 基于结构问题的建议
        structural_problems = druglikeness['structural']['problems']
        for problem in structural_problems:
            suggestions.append(f'Fix structural issue: {problem}')
        
        # 基于QED的建议
        if druglikeness['qed'] < 0.5:
            suggestions.append('Improve overall drug-likeness (QED score is low)')
        
        return suggestions if suggestions else ['Molecule appears to be drug-like']

# 创建全局实例
druglikeness_checker = DruglikenessChecker()

def check_druglikeness(mol):
    """便捷函数：检查单个分子的可成药性"""
    return druglikeness_checker.calculate_druglikeness_score(mol)

def filter_druglike_molecules(molecules, min_score=0.5):
    """便捷函数：过滤可成药分子"""
    return druglikeness_checker.filter_molecules(molecules, min_score)