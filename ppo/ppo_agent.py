import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import random
from rdkit import Chem
from rdkit.Chem import AllChem

class PPOAgent:
    """PPO强化学习代理，用于优化分子修饰策略"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, clip_epsilon=0.2, device="cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        
        # 策略网络
        self.policy = PolicyNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
    
    def select_action(self, state):
        """根据当前状态选择动作"""
        state = torch.FloatTensor(state).to(self.device)
        with torch.no_grad():
            action_probs, value = self.policy(state)
        
        # 从概率分布中采样动作
        dist = torch.distributions.Categorical(action_probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        
        return action.item(), log_prob.item(), value.item()
    
    def update(self, states, actions, log_probs, rewards, values):
        """更新策略网络"""
        # 转换为张量
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        old_log_probs = torch.FloatTensor(log_probs).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        values = torch.FloatTensor(values).to(self.device)
        
        # 计算优势函数
        advantages = rewards - values
        
        # 策略网络前向传播
        action_probs, current_values = self.policy(states)
        dist = torch.distributions.Categorical(action_probs)
        current_log_probs = dist.log_prob(actions)
        
        # 计算策略损失
        ratios = torch.exp(current_log_probs - old_log_probs)
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # 计算价值损失
        value_loss = ((current_values - rewards) ** 2).mean()
        
        # 计算总损失
        loss = policy_loss + 0.5 * value_loss
        
        # 优化
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()


class PolicyNetwork(nn.Module):
    """PPO的策略网络，包括动作概率和价值估计"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        
        # 共享特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 策略头
        self.policy_head = nn.Sequential(
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # 价值头
        self.value_head = nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        """前向传播"""
        features = self.feature_layer(x)
        action_probs = self.policy_head(features)
        value = self.value_head(features)
        return action_probs, value.squeeze(-1)


class MoleculeActionSpace:
    """定义分子优化的动作空间"""
    
    def __init__(self, mol_handler):
        self.mol_handler = mol_handler
        
        # 定义动作类型
        self.action_types = [
            'adjust_gui_strength',  # 调整DiffGui引导强度
            'adjust_property_weight',  # 调整性质引导权重
            'adjust_bond_weight',  # 调整键引导权重
            'adjust_pharmacophore_weight',  # 调整药效团引导权重
            'adjust_repulsion_weight',  # 调整排斥引导权重
            'add_functional_group',  # 添加官能团
            'modify_bond_type'  # 修改键类型
        ]
        
        # 定义官能团列表
        self.functional_groups = [
            'methyl',  # -CH3
            'hydroxyl',  # -OH
            'amino',  # -NH2
            'carboxyl',  # -COOH
            'fluoro',  # -F
            'chloro',  # -Cl
            'bromo'  # -Br
        ]
    
    def get_action_dim(self):
        """获取动作空间维度"""
        return len(self.action_types) + len(self.functional_groups) + 3  # 3种键类型
    
    def get_state_dim(self):
        """获取状态空间维度"""
        # 分子属性 + 当前引导权重 + 分子指纹
        return 10 + 5 + 32  # 简化为固定维度
    
    def get_state_representation(self, mol, guidance_weights):
        """获取分子和当前引导权重的状态表示"""
        # 导入评估器
        from molecule_evaluator import MoleculeEvaluator
        
        # 计算分子属性
        evaluator = MoleculeEvaluator()
        properties = evaluator.calculate_properties(mol)
        
        # 提取当前引导权重
        gui_strength = guidance_weights.get('gui_strength', 3.0)
        property_weight = guidance_weights.get('property_weight', 1.0)
        bond_weight = guidance_weights.get('bond_weight', 1.0)
        pharmacophore_weight = guidance_weights.get('pharmacophore_weight', 1.0)
        repulsion_weight = guidance_weights.get('repulsion_weight', 1.0)
        
        # 计算分子指纹（简化处理）
        fingerprint = np.zeros(32)  # 假设32维指纹
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=32)
            fingerprint = np.array(fp)
        except:
            pass
        
        # 组合状态表示
        state = np.concatenate([
            [properties['logp'], properties['tpsa'], properties['qed'], properties['mw'],
             properties['hba'], properties['hbd'], properties['rotatable_bonds'],
             properties['sa'], properties['pharmacophore'], properties['affinity']],
            [gui_strength, property_weight, bond_weight, pharmacophore_weight, repulsion_weight],
            fingerprint
        ])
        
        return state
    
    def execute_action(self, action_idx, mol, guidance_weights):
        """执行选定的动作"""
        # 解析动作类型
        if action_idx < len(self.action_types):
            # 调整引导权重
            action_type = self.action_types[action_idx]
            
            if action_type == 'adjust_gui_strength':
                # 随机调整引导强度
                delta = random.uniform(-0.5, 0.5)
                guidance_weights['gui_strength'] = max(0.5, min(5.0, guidance_weights.get('gui_strength', 3.0) + delta))
                return mol, guidance_weights
                
            elif action_type == 'adjust_property_weight':
                # 调整性质引导权重
                delta = random.uniform(-0.2, 0.2)
                guidance_weights['property_weight'] = max(0.1, min(2.0, guidance_weights.get('property_weight', 1.0) + delta))
                return mol, guidance_weights
                
            elif action_type == 'adjust_bond_weight':
                # 调整键引导权重
                delta = random.uniform(-0.2, 0.2)
                guidance_weights['bond_weight'] = max(0.1, min(2.0, guidance_weights.get('bond_weight', 1.0) + delta))
                return mol, guidance_weights
                
            elif action_type == 'adjust_pharmacophore_weight':
                # 调整药效团引导权重
                delta = random.uniform(-0.2, 0.2)
                guidance_weights['pharmacophore_weight'] = max(0.1, min(2.0, guidance_weights.get('pharmacophore_weight', 1.0) + delta))
                return mol, guidance_weights
            
            elif action_type == 'adjust_repulsion_weight':
                # 调整排斥引导权重
                delta = random.uniform(-0.2, 0.2)
                guidance_weights['repulsion_weight'] = max(0.1, min(2.0, guidance_weights.get('repulsion_weight', 1.0) + delta))
                return mol, guidance_weights
            
            elif action_type == 'add_functional_group':
                # 添加官能团
                return self.mol_handler.mutate_molecule(mol, 'add_group'), guidance_weights
            
            elif action_type == 'modify_bond_type':
                # 修改键类型
                return self.mol_handler.mutate_molecule(mol, 'change_bond'), guidance_weights
                
        elif action_idx < len(self.action_types) + len(self.functional_groups):
            # 添加特定官能团
            group_idx = action_idx - len(self.action_types)
            group_type = self.functional_groups[group_idx]
            
            # 创建分子的副本
            new_mol = Chem.Mol(mol)
            
            try:
                # 根据官能团类型添加不同的官能团
                atom_idx = random.randint(0, new_mol.GetNumAtoms()-1)
                editable_mol = Chem.EditableMol(new_mol)
                
                if group_type == 'methyl':
                    # 添加甲基(-CH3)
                    new_atom_idx = editable_mol.AddAtom(Chem.Atom('C'))
                    editable_mol.AddBond(atom_idx, new_atom_idx, Chem.BondType.SINGLE)
                elif group_type == 'hydroxyl':
                    # 添加羟基(-OH)
                    new_atom_idx = editable_mol.AddAtom(Chem.Atom('O'))
                    editable_mol.AddBond(atom_idx, new_atom_idx, Chem.BondType.SINGLE)
                elif group_type == 'amino':
                    # 添加氨基(-NH2)
                    new_atom_idx = editable_mol.AddAtom(Chem.Atom('N'))
                    editable_mol.AddBond(atom_idx, new_atom_idx, Chem.BondType.SINGLE)
                elif group_type == 'carboxyl':
                    # 添加羧基(-COOH)
                    c_atom_idx = editable_mol.AddAtom(Chem.Atom('C'))
                    o1_atom_idx = editable_mol.AddAtom(Chem.Atom('O'))
                    o2_atom_idx = editable_mol.AddAtom(Chem.Atom('O'))
                    editable_mol.AddBond(atom_idx, c_atom_idx, Chem.BondType.SINGLE)
                    editable_mol.AddBond(c_atom_idx, o1_atom_idx, Chem.BondType.DOUBLE)
                    editable_mol.AddBond(c_atom_idx, o2_atom_idx, Chem.BondType.SINGLE)
                elif group_type == 'fluoro':
                    # 添加氟(-F)
                    new_atom_idx = editable_mol.AddAtom(Chem.Atom('F'))
                    editable_mol.AddBond(atom_idx, new_atom_idx, Chem.BondType.SINGLE)
                elif group_type == 'chloro':
                    # 添加氯(-Cl)
                    new_atom_idx = editable_mol.AddAtom(Chem.Atom('Cl'))
                    editable_mol.AddBond(atom_idx, new_atom_idx, Chem.BondType.SINGLE)
                elif group_type == 'bromo':
                    # 添加溴(-Br)
                    new_atom_idx = editable_mol.AddAtom(Chem.Atom('Br'))
                    editable_mol.AddBond(atom_idx, new_atom_idx, Chem.BondType.SINGLE)
                
                new_mol = editable_mol.GetMol()
                
                # 尝试清理和标准化分子
                new_mol = Chem.RemoveHs(new_mol)
                Chem.SanitizeMol(new_mol)
                return new_mol, guidance_weights
            except Exception as e:
                print(f"添加官能团失败: {e}")
                return mol, guidance_weights  # 返回原始分子
        else:
            # 修改特定键类型
            bond_idx = action_idx - len(self.action_types) - len(self.functional_groups)
            bond_types = [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]
            
            # 创建分子的副本
            new_mol = Chem.Mol(mol)
            
            try:
                # 确保分子有键可以修改
                if new_mol.GetNumBonds() > 0:
                    # 随机选择一个键
                    bond_idx_to_modify = random.randint(0, new_mol.GetNumBonds()-1)
                    bond = new_mol.GetBondWithIdx(bond_idx_to_modify)
                    begin_atom = bond.GetBeginAtomIdx()
                    end_atom = bond.GetEndAtomIdx()
                    
                    # 使用指定的键类型
                    new_bond_type = bond_types[bond_idx % len(bond_types)]
                    
                    # 修改键类型
                    editable_mol = Chem.EditableMol(new_mol)
                    editable_mol.RemoveBond(begin_atom, end_atom)
                    editable_mol.AddBond(begin_atom, end_atom, new_bond_type)
                    new_mol = editable_mol.GetMol()
                    
                    # 尝试清理和标准化分子
                    new_mol = Chem.RemoveHs(new_mol)
                    Chem.SanitizeMol(new_mol)
                    return new_mol, guidance_weights
                else:
                    return mol, guidance_weights  # 如果没有键可以修改，返回原始分子
            except Exception as e:
                print(f"修改键类型失败: {e}")
                return mol, guidance_weights  # 如果修改键类型失败，返回原始分子

class ActorCritic(nn.Module):
    """Actor-Critic网络"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        
        # 共享特征提取层
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Actor网络 (策略)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )
        
        # 动作标准差
        self.action_log_std = nn.Parameter(torch.zeros(1, action_dim))
        
        # Critic网络 (价值函数)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """前向传播"""
        features = self.feature_extractor(state)
        
        # 计算动作均值
        action_mean = self.actor(features)
        action_log_std = self.action_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)
        
        # 计算状态价值
        value = self.critic(features)
        
        return action_mean, action_std, value
    
    def get_action(self, state):
        """获取动作"""
        action_mean, action_std, _ = self.forward(state)
        
        # 创建正态分布
        dist = Normal(action_mean, action_std)
        
        # 采样动作
        action = dist.sample()
        
        # 计算动作的对数概率
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        return action, log_prob
    
    def evaluate(self, state, action):
        """评估动作"""
        action_mean, action_std, value = self.forward(state)
        
        # 创建正态分布
        dist = Normal(action_mean, action_std)
        
        # 计算动作的对数概率
        log_prob = dist.log_prob(action).sum(dim=-1)
        
        # 计算熵
        entropy = dist.entropy().sum(dim=-1)
        
        return log_prob, entropy, value

class PPOAgent:
    """PPO代理"""
    
    def __init__(self, state_dim, action_dim, lr=3e-4, gamma=0.99, 
                 clip_ratio=0.2, value_coef=0.5, entropy_coef=0.01, 
                 max_grad_norm=0.5, device='cuda'):
        """
        初始化PPO代理
        
        参数:
        - state_dim: 状态维度
        - action_dim: 动作维度
        - lr: 学习率
        - gamma: 折扣因子
        - clip_ratio: PPO裁剪比例
        - value_coef: 价值损失系数
        - entropy_coef: 熵损失系数
        - max_grad_norm: 梯度裁剪阈值
        - device: 计算设备
        """
        self.gamma = gamma
        self.clip_ratio = clip_ratio
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        
        # 创建Actor-Critic网络
        self.ac = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=lr)
        
        # 经验缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []
    
    def select_action(self, state):
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob = self.ac.get_action(state)
        
        return action.cpu().numpy()[0], log_prob.cpu().item()
    
    def store_transition(self, state, action, log_prob, reward, next_state, done):
        """存储经验"""
        self.states.append(state)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
    
    def update(self, batch_size=64, epochs=10):
        """更新策略"""
        # 如果经验不足，不进行更新
        if len(self.states) < batch_size:
            return
        
        # 转换为张量
        states = torch.FloatTensor(np.array(self.states)).to(self.device)
        actions = torch.FloatTensor(np.array(self.actions)).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(self.log_probs)).to(self.device)
        rewards = torch.FloatTensor(np.array(self.rewards)).to(self.device)
        next_states = torch.FloatTensor(np.array(self.next_states)).to(self.device)
        dones = torch.FloatTensor(np.array(self.dones)).to(self.device)
        
        # 计算回报
        with torch.no_grad():
            _, _, next_values = self.ac(next_states)
            next_values = next_values.squeeze(-1)
            returns = rewards + self.gamma * next_values * (1 - dones)
        
        # 多次更新
        for _ in range(epochs):
            # 计算新的动作概率和价值
            log_probs, entropy, values = self.ac.evaluate(states, actions)
            values = values.squeeze(-1)
            
            # 计算比率
            ratio = torch.exp(log_probs - old_log_probs)
            
            # 计算裁剪后的目标函数
            surr1 = ratio * (returns - values.detach())
            surr2 = torch.clamp(ratio, 1.0 - self.clip_ratio, 1.0 + self.clip_ratio) * (returns - values.detach())
            policy_loss = -torch.min(surr1, surr2).mean()
            
            # 计算价值损失
            value_loss = ((values - returns) ** 2).mean()
            
            # 计算熵损失
            entropy_loss = -entropy.mean()
            
            # 计算总损失
            loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
            
            # 更新参数
            self.optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.ac.parameters(), self.max_grad_norm)
            self.optimizer.step()
        
        # 清空经验缓冲区
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.next_states = []
        self.dones = []