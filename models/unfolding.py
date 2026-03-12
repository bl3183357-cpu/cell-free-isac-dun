import torch
import torch.nn as nn
from .mlp import PowerNormalizationLayer 

class WMMSE_UnfoldingLayer(nn.Module):
    """
    深度展开网络的一个单层 (Single Layer of Deep Unfolding)
    代表传统 WMMSE 算法中的一次迭代，但波束更新步骤由 AI 接管。
    """
    def __init__(self, num_tx, num_users, hidden_dim=256):
        super().__init__()
        self.num_tx = num_tx
        self.K = num_users
        
        # 提取物理特征的维度：
        # H: (K, num_tx) -> 实部虚部 2*K*num_tx
        # a: (num_tx) -> 实部虚部 2*num_tx
        # W_prev (上一层的波束): (num_tx, K) -> 实部虚部 2*num_tx*K
        # U (接收机权重): (K) -> 实部虚部 2*K
        # w (MSE权重): (K) -> 实数 K
        input_dim = (2 * self.K * self.num_tx) + (2 * self.num_tx) + \
                    (2 * self.num_tx * self.K) + (2 * self.K) + self.K
                    
        output_dim = 2 * self.num_tx * self.K # 输出更新后的波束 W 的实部和虚部
        
        # AI 核心：一个轻量级的残差多层感知机 (Residual MLP)
        # 它负责学习如何在高维空间中更新波束，避开复杂的矩阵求逆
        self.ai_updater = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 可学习的步长参数 (Learnable Step Size)，这是深度展开的精髓之一
        # 传统算法步长固定，AI 可以自适应调整步长加速收敛
        self.step_size = nn.Parameter(torch.tensor(0.1))

    def forward(self, H, a, W_prev, noise_var=1e-3):
        batch_size = H.shape[0]
        
        # ==========================================
        # 步骤 1: 物理公式计算 (保留 WMMSE 的领域知识)
        # ==========================================
        # 计算接收信号: H * W_prev -> (Batch, K, K)
        HW = torch.bmm(H, W_prev)
        
        # 提取期望信号的信道增益 (对角线元素): h_k^H * w_k
        # 形状: (Batch, K)
        signal_gain = torch.diagonal(HW, dim1=-2, dim2=-1)
        
        # 计算每个用户的总接收功率 (信号 + 干扰): sum(|h_k^H * w_j|^2)
        # 形状: (Batch, K)
        total_rx_power = torch.sum(torch.abs(HW)**2, dim=-1)
        
        # 1.1 计算最优接收机权重 U (WMMSE 公式)
        # U_k = (h_k^H * w_k) / (总接收功率 + 噪声)
        # 形状: (Batch, K)
        U = signal_gain / (total_rx_power + noise_var)
        
        # 1.2 计算均方误差 (MSE)
        # E_k = 1 - U_k^* * (h_k^H * w_k)
        # 形状: (Batch, K)
        E = 1.0 - torch.real(torch.conj(U) * signal_gain)
        
        # 1.3 计算 MSE 权重 w (WMMSE 公式)
        # w_k = 1 / E_k
        # 形状: (Batch, K)
        w = 1.0 / torch.clamp(E, min=1e-6) # 防止除以 0
        
        # ==========================================
        # 步骤 2: 特征拼接 (将物理特征送入 AI)
        # ==========================================
        # 将所有复数张量展平为实数向量
        H_flat = torch.cat([torch.real(H).reshape(batch_size, -1), torch.imag(H).reshape(batch_size, -1)], dim=-1)
        a_flat = torch.cat([torch.real(a).reshape(batch_size, -1), torch.imag(a).reshape(batch_size, -1)], dim=-1)
        W_prev_flat = torch.cat([torch.real(W_prev).reshape(batch_size, -1), torch.imag(W_prev).reshape(batch_size, -1)], dim=-1)
        U_flat = torch.cat([torch.real(U).reshape(batch_size, -1), torch.imag(U).reshape(batch_size, -1)], dim=-1)
        w_flat = w.reshape(batch_size, -1) # w 已经是实数
        
        # 拼接所有特征
        features = torch.cat([H_flat, a_flat, W_prev_flat, U_flat, w_flat], dim=-1)
        
        # ==========================================
        # 步骤 3: AI 波束更新 (替代矩阵求逆)
        # ==========================================
        # AI 输出波束的更新方向 (Delta W)
        delta_W_flat = self.ai_updater(features)
        
        # 重塑为复数矩阵
        delta_W_flat = delta_W_flat.reshape(batch_size, self.num_tx, self.K, 2)
        delta_W = torch.complex(delta_W_flat[..., 0], delta_W_flat[..., 1])
        
        # 残差连接 (Residual Connection): W_new = W_prev + step_size * Delta_W
        # 这保证了展开网络即使在最差情况下，也不会比上一层更差
        W_new = W_prev + self.step_size * delta_W
        
        return W_new

class ISAC_DeepUnfoldingNet(nn.Module):
    """
    完整的 Cell-Free ISAC 深度展开网络
    包含多层展开结构和最终的功率归一化。
    """
    def __init__(self, num_ap, antennas_per_ap, num_users, num_layers=4, hidden_dim=256, p_max=1.0):
        super().__init__()
        self.M = num_ap
        self.N = antennas_per_ap
        self.K = num_users
        self.num_tx = self.M * self.N
        self.num_layers = num_layers
        
        # 初始化展开层列表 (ModuleList)
        self.unfolding_layers = nn.ModuleList([
            WMMSE_UnfoldingLayer(self.num_tx, self.K, hidden_dim) 
            for _ in range(num_layers)
        ])
        
        # 物理约束层 (极其重要)
        self.power_norm = PowerNormalizationLayer(self.M, self.N, p_max)
        
        # 初始化一个极简单的线性层，用于生成第 0 层的初始波束 W_0
        # 好的初始值对展开网络至关重要
        self.init_layer = nn.Linear(2 * self.num_tx * self.K, 2 * self.num_tx * self.K)

    def forward(self, H, a):
        batch_size = H.shape[0]
        
        # ==========================================
        # 步骤 0: 生成初始波束 W_0 (类似于最大比传输 MRT)
        # ==========================================
        # 我们用信道的共轭转置 (H^H) 作为初始特征，因为它是最经典的通信波束
        # H 形状: (Batch, K, num_tx) -> H^H 形状: (Batch, num_tx, K)
        H_H = torch.conj(torch.transpose(H, 1, 2))
        H_H_flat = torch.cat([
            torch.real(H_H).reshape(batch_size, -1), 
            torch.imag(H_H).reshape(batch_size, -1)
        ], dim=-1)
        
        W_0_flat = self.init_layer(H_H_flat)
        W_0_flat = W_0_flat.reshape(batch_size, self.num_tx, self.K, 2)
        W = torch.complex(W_0_flat[..., 0], W_0_flat[..., 1])
        
        # 必须对初始波束进行功率归一化，否则第一层的物理公式会爆炸
        W = self.power_norm(W)
        
        # ==========================================
        # 步骤 1: 逐层展开 (Layer-by-Layer Unfolding)
        # ==========================================
        # 每一层的输出 W 作为下一层的输入 W_prev
        for i in range(self.num_layers):
            W = self.unfolding_layers[i](H, a, W)
            # 每一层结束后强制满足物理约束，保证迭代轨迹的合法性
            W = self.power_norm(W)
            
        return W
