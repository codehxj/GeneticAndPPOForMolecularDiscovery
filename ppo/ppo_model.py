import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem

# 导入分子环境
from molecule_generation.molecule_environment import MoleculeEnvironment

class PPONetwork(nn.Module):
    """PPO神经网络模型"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        """
        初始化PPO网络
        
        参数:
        state_dim: 状态维度
        action_dim: 动作维度
        hidden_dim: 隐藏层维度
        """
        super(PPONetwork, self).__init__()
        
        # 共享特征提取层
        self.feature_layer = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 策略网络 (Actor)
        self.actor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
        
        # 价值网络 (Critic)
        self.critic = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, state):
        """
        前向传播
        
        参数:
        state: 状态张量
        
        返回:
        action_probs: 动作概率
        state_value: 状态价值
        """
        features = self.feature_layer(state)
        action_probs = self.actor(features)
        state_value = self.critic(features)
        return action_probs, state_value

class PPOAgent:
    """PPO代理类"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128, lr=0.001, gamma=0.99, eps_clip=0.2, K_epochs=4):
        """
        初始化PPO代理
        
        参数:
        state_dim: 状态维度
        action_dim: 动作维度
        hidden_dim: 隐藏层维度
        lr: 学习率
        gamma: 折扣因子
        eps_clip: PPO裁剪参数
        K_epochs: 每次更新的训练轮数
        """
        self.lr = lr
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.policy = PPONetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        
        self.policy_old = PPONetwork(state_dim, action_dim, hidden_dim)
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
        self.morgan_radius = 2
        self.morgan_nbits = state_dim
    
    def encode_state(self, mol):
        """
        将分子编码为状态向量
        
        参数:
        mol: RDKit分子对象
        
        返回:
        状态向量
        """
        if mol is None:
            return torch.zeros(self.morgan_nbits)
        
        try:
            # 使用Morgan指纹编码分子
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, self.morgan_radius, nBits=self.morgan_nbits)
            state = torch.tensor([float(b) for b in fp], dtype=torch.float32)
            return state
        except:
            return torch.zeros(self.morgan_nbits)
    
    def get_action(self, state):
        """
        根据状态选择动作
        
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
            
            # 从动作概率分布中采样
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
            return action.item(), log_prob.item(), state_value.item()
    
    def store_transition(self, state, action, reward, next_state, log_prob, value, done):
        """
        存储转换
        
        参数:
        state: 当前状态
        action: 执行的动作
        reward: 获得的奖励
        next_state: 下一个状态
        log_prob: 动作的对数概率
        value: 状态价值
        done: 是否结束
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)
    
    def update(self):
        """更新策略网络"""
        # 如果没有足够的数据，直接返回
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
        
        # 转换为张量，确保所有状态具有相同的维度
        try:
            # 检查状态是否已经是张量
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
            for _ in range(self.K_epochs):
                # 计算当前策略的动作概率和状态价值
                action_probs, state_values = self.policy(states)
                dist = torch.distributions.Categorical(action_probs)
                log_probs = dist.log_prob(actions)
                
                # 计算比率
                ratios = torch.exp(log_probs - old_log_probs.detach())
                
                # 计算优势
                advantages = returns - state_values.detach().squeeze()
                
                # 计算PPO损失
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.mse_loss(state_values.squeeze(), returns)
                
                # 总损失
                loss = actor_loss + 0.5 * critic_loss
                
                # 优化
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
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
        """
        保存模型
        
        参数:
        path: 保存路径
        """
        torch.save({
            'policy': self.policy.state_dict(),
            'policy_old': self.policy_old.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }, path)
    
    def load_model(self, path):
        """
        加载模型
        
        参数:
        path: 加载路径
        """
        if os.path.exists(path):
            checkpoint = torch.load(path)
            self.policy.load_state_dict(checkpoint['policy'])
            self.policy_old.load_state_dict(checkpoint['policy_old'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])