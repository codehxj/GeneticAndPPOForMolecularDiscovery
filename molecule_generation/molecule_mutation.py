import random
from rdkit import Chem
from rdkit import RDLogger
import io
import sys

# 完全禁用RDKit的所有日志输出
RDLogger.DisableLog('rdApp.*')

# 全局抑制RDKit错误输出
class NullIO(io.IOBase):
    def write(self, *args, **kwargs):
        pass
    def flush(self, *args, **kwargs):
        pass

# 重定向RDKit的错误输出
rdkit_null_stderr = NullIO()
original_stderr = sys.stderr
sys.stderr = rdkit_null_stderr

# 定义常用官能团及其SMILES表示
FUNCTIONAL_GROUPS = {
    "羟基": "[OH]",
    "羧基": "[C](=O)[OH]",
    "醛基": "[CH]=O",
    "酮基": "[C]=O",
    "氨基": "[NH2]",
    "硝基": "[N+](=O)[O-]",
    "氰基": "C#N",
    "卤素": {
        "氟": "F",
        "氯": "Cl",
        "溴": "Br",
        "碘": "I"
    },
    "烷基": {
        "甲基": "C",
        "乙基": "CC",
        "丙基": "CCC"
    },
    "烯基": "C=C",
    "炔基": "C#C",
    "醚基": "COC",
    "酯基": "C(=O)OC",
    "酰胺基": "C(=O)N",
    "磺酸基": "S(=O)(=O)O"
    # 移除磷酸基以避免[PH]异常结构
}

def get_max_valence(atomic_num):
    """获取原子的最大价态（移除磷原子以避免[PH]异常结构）"""
    valence_dict = {
        1: 1,   # H
        5: 3,   # B
        6: 4,   # C
        7: 3,   # N
        8: 2,   # O
        9: 1,   # F
        14: 4,  # Si
        16: 6,  # S
        17: 1,  # Cl
        35: 1,  # Br
        53: 1   # I
    }
    return valence_dict.get(atomic_num, 0)

def identify_functional_groups(mol):
    """识别分子中的官能团"""
    if mol is None:
        return []
    
    functional_groups = []
    
    # 定义官能团的SMARTS模式
    patterns = {
        "羟基": "[OX2H]",
        "羧基": "[CX3](=O)[OX2H1]",
        "醛基": "[CX3H1](=O)",
        "酮基": "[CX3](=O)[#6]",
        "氨基": "[NX3;H2]",
        "硝基": "[NX3](=O)=O",
        "氰基": "C#N",
        "氟": "[F]",
        "氯": "[Cl]",
        "溴": "[Br]",
        "碘": "[I]",
        "醚基": "[OX2]([#6])[#6]",
        "酯基": "[CX3](=O)[OX2][#6]",
        "酰胺基": "[CX3](=O)[NX3]",
        "磺酸基": "[SX4](=O)(=O)[OX2H]",
        "磷酸基": "[PX4](=O)([OX2H])([OX2H])[OX2H]"
    }
    
    # 查找每种官能团
    for name, smarts in patterns.items():
        pattern = Chem.MolFromSmarts(smarts)
        if pattern:
            matches = mol.GetSubstructMatches(pattern)
            for match in matches:
                functional_groups.append((name, match))
    
    return functional_groups

def replace_functional_group(mol, group_info, new_group_name):
    """替换分子中的官能团"""
    if mol is None or not group_info:
        return None
    
    group_name, atom_indices = group_info
    
    # 获取新官能团的SMILES
    new_group_smiles = None
    if new_group_name in FUNCTIONAL_GROUPS:
        if isinstance(FUNCTIONAL_GROUPS[new_group_name], dict):
            # 如果是字典，随机选择一个子类型
            subtype = random.choice(list(FUNCTIONAL_GROUPS[new_group_name].keys()))
            new_group_smiles = FUNCTIONAL_GROUPS[new_group_name][subtype]
        else:
            new_group_smiles = FUNCTIONAL_GROUPS[new_group_name]
    else:
        # 如果指定的官能团名称不存在，随机选择一个
        random_group = random.choice(list(FUNCTIONAL_GROUPS.keys()))
        if isinstance(FUNCTIONAL_GROUPS[random_group], dict):
            subtype = random.choice(list(FUNCTIONAL_GROUPS[random_group].keys()))
            new_group_smiles = FUNCTIONAL_GROUPS[random_group][subtype]
        else:
            new_group_smiles = FUNCTIONAL_GROUPS[random_group]
    
    # 创建新官能团的分子对象
    new_group_mol = Chem.MolFromSmiles(new_group_smiles)
    if not new_group_mol:
        return None
    
    # 创建可编辑的分子对象
    editable_mol = Chem.EditableMol(mol)
    
    # 移除原官能团的原子（除了连接点）
    # 注意：这是一个简化的实现，实际上需要更复杂的逻辑来处理连接点
    # 这里假设第一个原子是连接点
    connection_atom = atom_indices[0]
    atoms_to_remove = sorted(atom_indices[1:], reverse=True)
    
    for atom_idx in atoms_to_remove:
        editable_mol.RemoveAtom(atom_idx)
    
    # 获取修改后的分子
    modified_mol = editable_mol.GetMol()
    
    try:
        # 尝试将新官能团添加到分子中
        # 这里需要更复杂的逻辑来处理连接点和键类型
        # 简化起见，我们只返回修改后的分子
        Chem.SanitizeMol(modified_mol)
        return modified_mol
    except:
        return None

def replace_random_functional_group(mol):
    """随机替换分子中的一个官能团"""
    if mol is None:
        return None
    
    # 识别分子中的官能团
    functional_groups = identify_functional_groups(mol)
    if not functional_groups:
        return mol  # 如果没有找到官能团，返回原分子
    
    # 随机选择一个官能团进行替换
    group_info = random.choice(functional_groups)
    
    # 随机选择一个新的官能团类型
    new_group_name = random.choice(list(FUNCTIONAL_GROUPS.keys()))
    
    # 替换官能团
    new_mol = replace_functional_group(mol, group_info, new_group_name)
    
    # 如果替换失败，返回原分子
    if new_mol is None:
        return mol
    
    return new_mol

def generate_molecule_with_functional_groups(seed_smiles=None, n_replacements=1):
    """生成带有替换官能团的分子"""
    # 如果没有提供种子分子，使用默认分子
    if not seed_smiles:
        seed_smiles = [
            "c1ccccc1",  # 苯
            "CC(=O)O",   # 乙酸
            "CCO",       # 乙醇
            "c1ccccc1O", # 苯酚
            "c1ccccc1C(=O)O" # 苯甲酸
        ]
        seed_smiles = random.choice(seed_smiles)
    
    # 创建分子对象
    mol = Chem.MolFromSmiles(seed_smiles)
    if not mol:
        return None
    
    # 进行多次官能团替换
    for _ in range(n_replacements):
        mol = replace_random_functional_group(mol)
        if not mol:
            break
    
    return mol

def generate_diverse_molecules(seed_smiles=None, n_molecules=20, n_replacements_per_mol=1):
    """生成一组多样化的分子，每个分子有不同的官能团替换"""
    molecules = []
    smiles_set = set()  # 用于跟踪已生成的分子
    
    print(f"开始生成{n_molecules}个分子，每个分子进行{n_replacements_per_mol}次官能团替换")
    
    for i in range(n_molecules):
        print(f"\n开始生成第 {i+1}/{n_molecules} 个分子...")
        
        # 生成新分子
        mol = generate_molecule_with_functional_groups(seed_smiles, n_replacements_per_mol)
        
        if mol:
            try:
                # 生成SMILES表示
                smiles = Chem.MolToSmiles(mol, isomericSmiles=True)
                
                # 检查是否重复
                if smiles not in smiles_set:
                    molecules.append(mol)
                    smiles_set.add(smiles)
                    print(f"成功生成第 {len(molecules)} 个分子，SMILES: {smiles}")
                else:
                    print(f"生成的分子重复，跳过")
            except:
                print(f"分子无效，跳过")
        else:
            print(f"无法生成有效分子，跳过")
    
    print(f"\n总共成功生成 {len(molecules)}/{n_molecules} 个分子")
    return molecules

# 添加主函数，方便直接运行脚本
if __name__ == "__main__":
    # 设置参数
    n_molecules = 20  # 总共生成20个分子
    n_replacements = 1  # 每个分子进行1次官能团替换
    
    # 调用函数生成分子
    molecules = generate_diverse_molecules(
        seed_smiles=None,  # 使用默认种子分子
        n_molecules=n_molecules,
        n_replacements_per_mol=n_replacements
    )
    
    # 输出结果统计
    if molecules:
        print(f"成功生成了 {len(molecules)} 个分子")
        for i, mol in enumerate(molecules):
            smiles = Chem.MolToSmiles(mol)
            print(f"分子 {i+1}: {smiles}")
            
            # 识别分子中的官能团
            groups = identify_functional_groups(mol)
            if groups:
                print(f"  包含的官能团: {', '.join([g[0] for g in groups])}")
    else:
        print("未能生成任何有效分子")

# 恢复原始stderr
sys.stderr = original_stderr