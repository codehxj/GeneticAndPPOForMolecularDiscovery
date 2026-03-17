import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

class AttentionLayer(nn.Module):
    """注意力机制层"""
    
    def __init__(self, hidden_dim, num_heads=4):
        super(AttentionLayer, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.head_dim = hidden_dim // num_heads
        
        assert hidden_dim % num_heads == 0, "hidden_dim必须能被num_heads整除"
        
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        self.output = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        # 计算Q, K, V
        Q = self.query(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.key(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.value(x).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(self.head_dim)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # 应用注意力
        context = torch.matmul(attention_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        # 输出投影
        output = self.output(context)
        return output.squeeze(1) if output.dim() == 3 and output.size(1) == 1 else output

class ResidualBlock(nn.Module):
    """残差块"""
    
    def __init__(self, hidden_dim, dropout_rate=0.1):
        super(ResidualBlock, self).__init__()
        self.linear1 = nn.Linear(hidden_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout_rate)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)
        x = self.layer_norm(x + residual)
        return x

class ImprovedPPONetwork(nn.Module):
    """改进的PPO神经网络模型"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=3, num_heads=8, dropout_rate=0.1):
        """
        初始化改进的PPO网络
        
        参数:
        state_dim: 状态维度
        action_dim: 动作维度
        hidden_dim: 隐藏层维度（增加到256）
        num_layers: 网络层数
        num_heads: 注意力头数
        dropout_rate: Dropout率
        """
        super(ImprovedPPONetwork, self).__init__()
        
        # 输入投影层
        self.input_projection = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 多层特征提取
        self.feature_layers = nn.ModuleList()
        for i in range(num_layers):
            layer = nn.Sequential(
                ResidualBlock(hidden_dim, dropout_rate),
                AttentionLayer(hidden_dim, num_heads)
            )
            self.feature_layers.append(layer)
        
        # 特征融合层
        self.feature_fusion = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # 策略网络 (Actor) - 更深的网络
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # 价值网络 (Critic) - 更深的网络
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # 初始化权重
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """权重初始化"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                torch.nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.constant_(module.bias, 0)
            torch.nn.init.constant_(module.weight, 1.0)
    
    def forward(self, state):
        """
        前向传播
        
        参数:
        state: 状态张量
        
        返回:
        action_probs: 动作概率
        state_value: 状态价值
        """
        # 输入投影
        x = self.input_projection(state)
        
        # 多层特征提取
        for layer in self.feature_layers:
            residual_out = layer[0](x)  # ResidualBlock
            if residual_out.dim() == 2:
                residual_out = residual_out.unsqueeze(1)  # 为注意力层添加序列维度
            attention_out = layer[1](residual_out)  # AttentionLayer
            x = attention_out
        
        # 特征融合
        features = self.feature_fusion(x)
        
        # 计算动作概率和状态价值
        action_probs = self.actor(features)
        state_value = self.critic(features)
        
        return action_probs, state_value

class ImprovedPPOAgent:
    """改进的PPO代理类"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=256, num_layers=3, num_heads=8, 
                 lr=0.0003, gamma=0.99, eps_clip=0.2, K_epochs=10, entropy_coef=0.01):
        """
        初始化改进的PPO代理
        
        参数:
        state_dim: 状态维度
        action_dim: 动作维度
        hidden_dim: 隐藏层维度
        num_layers: 网络层数
        num_heads: 注意力头数
        lr: 学习率
        gamma: 折扣因子
        eps_clip: PPO裁剪参数
        K_epochs: 每次更新的训练轮数
        entropy_coef: 熵正则化系数
        """
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.entropy_coef = entropy_coef
        
        # 使用改进的网络
        self.policy = ImprovedPPONetwork(state_dim, action_dim, hidden_dim, num_layers, num_heads)
        self.optimizer = optim.AdamW(self.policy.parameters(), lr=lr, weight_decay=0.01)
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=1000)
        
        self.policy_old = ImprovedPPONetwork(state_dim, action_dim, hidden_dim, num_layers, num_heads)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 存储经验
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.values = []
        self.dones = []
        
        # 分子指纹编码器
        self.morgan_radius = 3  # 增加指纹半径
        self.morgan_nbits = state_dim
        
        # 训练统计
        self.training_stats = {
            'actor_losses': [],
            'critic_losses': [],
            'entropy_losses': [],
            'total_losses': []
        }
    
    def encode_state(self, mol):
        """
        将分子编码为状态向量（改进版本）
        
        参数:
        mol: RDKit分子对象
        
        返回:
        状态向量
        """
        if mol is None:
            return torch.zeros(self.morgan_nbits)
        
        try:
            # 使用更大半径的Morgan指纹
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.morgan_radius, nBits=self.morgan_nbits)
            state = torch.tensor([float(b) for b in fp], dtype=torch.float32)
            
            # 添加分子描述符作为额外特征
            if self.morgan_nbits > 128:
                # 计算额外的分子描述符
                mw = Chem.Descriptors.MolWt(mol) / 500.0  # 归一化分子量
                logp = Chem.Descriptors.MolLogP(mol) / 5.0  # 归一化LogP
                tpsa = Chem.Descriptors.TPSA(mol) / 200.0  # 归一化TPSA
                
                # 将描述符添加到状态向量的末尾
                extra_features = torch.tensor([mw, logp, tpsa], dtype=torch.float32)
                state = torch.cat([state[:-3], extra_features])
            
            return state
        except:
            return torch.zeros(self.morgan_nbits)
    
    def get_action(self, state):
        """
        根据状态选择动作（改进版本）
        
        参数:
        state: 状态向量
        
        返回:
        action: 选择的动作
        log_prob: 动作的对数概率
        value: 状态价值
        """
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)
            action_probs, state_value = self.policy_old(state)
            
            # 添加噪声以增加探索
            action_probs = action_probs + torch.randn_like(action_probs) * 0.01
            action_probs = F.softmax(action_probs, dim=-1)
            
            # 从动作概率分布中采样
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), state_value.item()
    
    def store_transition(self, state, action, reward, next_state, log_prob, value, done):
        """存储转换"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def update(self):
        """更新策略网络（改进版本）"""
        if len(self.states) == 0:
            return
            
        # 计算回报
        returns = []
        discounted_reward = 0
        
        for reward, done in zip(reversed(self.rewards), reversed(self.dones)):
            if done:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            returns.insert(0, discounted_reward)
        
        try:
            # 转换为张量
            if isinstance(self.states[0], torch.Tensor):
                states = torch.stack(self.states)
            else:
                states = torch.FloatTensor(self.states)
                
            actions = torch.LongTensor(self.actions)
            returns = torch.FloatTensor(returns)
            old_log_probs = torch.FloatTensor(self.log_probs)
            
            # 归一化回报
            if returns.std() > 1e-5:
                returns = (returns - returns.mean()) / (returns.std() + 1e-5)
            
            # 优化策略
            for epoch in range(self.K_epochs):
                # 计算当前策略的动作概率和状态价值
                action_probs, state_values = self.policy(states)
                dist = torch.distributions.Categorical(action_probs)
                log_probs = dist.log_prob(actions)
                entropy = dist.entropy()
                
                # 计算比率
                ratios = torch.exp(log_probs - old_log_probs.detach())
                
                # 计算优势
                advantages = returns - state_values.detach().squeeze()
                
                # 计算PPO损失
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(state_values.squeeze(), returns)
                entropy_loss = -self.entropy_coef * entropy.mean()
                
                # 总损失
                total_loss = actor_loss + 0.5 * critic_loss + entropy_loss
                
                # 优化
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=0.5)
                self.optimizer.step()
                
                # 记录损失
                self.training_stats['actor_losses'].append(actor_loss.item())
                self.training_stats['critic_losses'].append(critic_loss.item())
                self.training_stats['entropy_losses'].append(entropy_loss.item())
                self.training_stats['total_losses'].append(total_loss.item())
            
            # 更新学习率
            self.scheduler.step()
            
            # 更新旧策略
            self.policy_old.load_state_dict(self.policy.state_dict())
        
        except Exception as e:
            print(f"PPO更新时出错: {e}")
        
        # 清空经验
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.log_probs = []
        self.values = []
        self.dones = []
    
    def save_model(self, path):
        """保存模型"""
        torch.save({
            'policy': self.policy.state_dict(),
            'policy_old': self.policy_old.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'training_stats': self.training_stats
        }, path)
    
    def load_model(self, path):
        """加载模型"""
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location='cpu')
            self.policy.load_state_dict(checkpoint['policy'])
            self.policy_old.load_state_dict(checkpoint['policy_old'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            if 'scheduler' in checkpoint:
                self.scheduler.load_state_dict(checkpoint['scheduler'])
            if 'training_stats' in checkpoint:
                self.training_stats = checkpoint['training_stats']
    
    def get_training_stats(self):
        """获取训练统计信息"""
        return self.training_stats