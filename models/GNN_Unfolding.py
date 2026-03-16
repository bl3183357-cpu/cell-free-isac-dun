import torch
import torch.nn as nn
import torch.nn.functional as F
from .mlp import PowerNormalizationLayer 
from utils.baseline import get_zf_beamformer, get_heuristic_isac_beamformer



class BipartiteGNN_WMMSE_Layer(nn.Module):
    """
    结合二分图神经网络 (Bipartite GNN) 的 WMMSE 深度展开单层。
    完全消除了对天线数 (num_tx) 和用户数 (K) 的硬编码依赖。
    """
    def __init__(self, hidden_dim=64):
        super().__init__()
        
        # 输入特征维度计算:
        # H (实+虚=2) + W_prev (实+虚=2) + U (实+虚=2) + w (实=1) + a (实+虚=2) = 9
        input_dim = 9
        
        # 边特征提取 MLP (作用于每一对 天线-用户 之间)
        self.edge_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # 节点状态更新与波束残差输出 MLP
        # 输入: 边特征(hidden_dim) + 用户聚合特征(hidden_dim) + 天线聚合特征(hidden_dim)
        self.update_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2) # 输出 Delta W 的实部和虚部
        )
        
        # 可学习的自适应步长
        self.step_size = nn.Parameter(torch.tensor(0.1))

    def forward(self, H, a, W_prev, noise_var=1e-3):
        """
        H: (Batch, K, num_tx) Complex
        a: (Batch, num_tx) Complex - 感知导向矢量
        W_prev: (Batch, num_tx, K) Complex
        """
        B, K, N = H.shape
        
        # ==========================================\n        # 步骤 1: WMMSE 物理公式计算 (保持不变，提取领域知识)
        # ==========================================
        HW = torch.bmm(H, W_prev) # (B, K, K)
        signal_gain = torch.diagonal(HW, dim1=-2, dim2=-1) # (B, K)
        total_rx_power = torch.sum(torch.abs(HW)**2, dim=-1) # (B, K)
        
        U = signal_gain / (total_rx_power + noise_var) # (B, K)
        E = 1.0 - torch.real(torch.conj(U) * signal_gain) # (B, K)
        w = 1.0 / torch.clamp(E, min=1e-6) # (B, K)
        
        # ==========================================\n        # 步骤 2: 构建二分图的边特征矩阵 (Edge Features)
        # 目标形状统一对齐为: (Batch, K, N, Feature_Dim)
        # ==========================================
        # 1. 信道 H: (B, K, N, 2)
        H_feat = torch.stack([torch.real(H), torch.imag(H)], dim=-1)
        
        # 2. 波束 W_prev: 转置为 (B, K, N) 后提取实虚部 -> (B, K, N, 2)
        W_T = torch.transpose(W_prev, 1, 2)
        W_feat = torch.stack([torch.real(W_T), torch.imag(W_T)], dim=-1)
        
        # 3. 接收机权重 U (用户节点特征): 广播到所有天线 -> (B, K, N, 2)
        U_feat = torch.stack([torch.real(U), torch.imag(U)], dim=-1)
        U_feat = U_feat.unsqueeze(2).expand(B, K, N, 2)
        
        # 4. MSE权重 w (用户节点特征): 广播到所有天线 -> (B, K, N, 1)
        w_feat = w.unsqueeze(-1).unsqueeze(-1).expand(B, K, N, 1)
        
        # 5. 感知矢量 a (天线节点特征): 广播到所有用户 -> (B, K, N, 2)
        a_feat = torch.stack([torch.real(a), torch.imag(a)], dim=-1)
        a_feat = a_feat.unsqueeze(1).expand(B, K, N, 2)
        
        # 拼接形成初始边特征图 Z: (B, K, N, 9)
        Z = torch.cat([H_feat, W_feat, U_feat, w_feat, a_feat], dim=-1)
        
        # ==========================================\n        # 步骤 3: GNN 消息传递与聚合 (Message Passing)
        # ==========================================
        # 3.1 提取高维边特征 (B, K, N, hidden_dim)
        E_feat = self.edge_mlp(Z) 
        
        # 3.2 聚合到用户节点 (按天线维度 N 求平均) -> (B, K, 1, hidden_dim)
        user_node_feat = torch.mean(E_feat, dim=2, keepdim=True)
        
        # 3.3 聚合到天线节点 (按用户维度 K 求平均) -> (B, 1, N, hidden_dim)
        ant_node_feat = torch.mean(E_feat, dim=1, keepdim=True)
        
        # 3.4 节点状态融合与广播
        # 将用户特征和天线特征广播回边上，与边特征拼接
        combined_feat = torch.cat([
            E_feat, 
            user_node_feat.expand(B, K, N, -1), 
            ant_node_feat.expand(B, K, N, -1)
        ], dim=-1) # 形状: (B, K, N, hidden_dim * 3)
        
        # ==========================================\n        # 步骤 4: 输出波束残差并更新
        # ==========================================
        delta_W_T_real = self.update_mlp(combined_feat) # (B, K, N, 2)
        delta_W_T = torch.complex(delta_W_T_real[..., 0], delta_W_T_real[..., 1]) # (B, K, N)
        
        # 转置回原始波束形状 (B, N, K)
        delta_W = torch.transpose(delta_W_T, 1, 2)
        
        # 残差连接
        W_new = W_prev + self.step_size * delta_W
        
        return W_new

class ISAC_GNN_UnfoldingNet(nn.Module):
    """
    基于 GNN 的 Cell-Free ISAC 深度展开网络
    """
    def __init__(self, num_ap, antennas_per_ap, num_layers=4, hidden_dim=64, p_max=1.0, init_method='mrt'):
        super().__init__()
        self.num_layers = num_layers
        self.init_method = init_method

        self.unfolding_layers = nn.ModuleList([
            BipartiteGNN_WMMSE_Layer(hidden_dim) 
            for _ in range(num_layers)
        ])
        
        self.power_norm = PowerNormalizationLayer(num_ap, antennas_per_ap, p_max)
        
        self.mrt_scale = nn.Parameter(torch.tensor(0.1))

    def forward(self, H, a):
        # ==========================================
        # 步骤 0: 根据配置选择初始化/基线算法
        # ==========================================
        if self.init_method == 'mrt':
            W_0 = torch.conj(torch.transpose(H, 1, 2))
        elif self.init_method == 'zf':
            W_0 = get_zf_beamformer(H) 
        elif self.init_method == 'heuristic':
            W_0 = get_heuristic_isac_beamformer(H, a) 
        else:
            raise ValueError(f"Unknown init_method: {self.init_method}")

        
        # 初始功率归一化
        W = self.power_norm(W_0) 
        
        # ==========================================
        # 步骤 1: GNN 逐层展开
        # ==========================================
        for i in range(self.num_layers):
            W = self.unfolding_layers[i](H, a, W)
            W = self.power_norm(W) 
            
        return W
