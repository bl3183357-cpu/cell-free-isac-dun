import torch
import torch.nn as nn

class PowerNormalizationLayer(nn.Module):
    """
    物理约束层：单 AP 功率归一化 (Power Normalization)
    这是无监督物理层 AI 最致命、最核心的一层。没有任何可训练参数，只做张量投影。
    确保每个 AP (Access Point) 的发射功率不超过最大功率 P_max。
    """
    def __init__(self, num_ap, antennas_per_ap, p_max=1.0):
        super().__init__()
        self.M = num_ap
        self.N = antennas_per_ap
        self.p_max = p_max

    def forward(self, W):
        """
        参数:
            W: 神经网络输出的原始波束矩阵, 形状 (Batch, M*N, K)
        返回:
            W_norm: 满足功率约束的波束矩阵, 形状 (Batch, M*N, K)
        """
        batch_size, num_tx, K = W.shape
        
        # 将 W 重塑为 (Batch, M, N, K)，方便按 AP 提取波束
        W_reshaped = W.view(batch_size, self.M, self.N, K)
        
        # 计算每个 AP 的当前发射功率: Tr(W_m * W_m^H)
        # 也就是对每个 AP 的所有天线 (N) 和所有用户 (K) 的波束幅度平方求和
        # 形状: (Batch, M)
        power_per_ap = torch.sum(torch.abs(W_reshaped)**2, dim=(2, 3))
        
        # 计算缩放因子: 如果功率超标，则缩放；如果不超标，则保持不变 (乘以 1)
        # 形状: (Batch, M)
        scaling_factor = torch.sqrt(self.p_max / torch.clamp(power_per_ap, min=self.p_max))
        
        # 将缩放因子扩展维度，使其能与 W_reshaped 相乘
        # 形状: (Batch, M, 1, 1)
        scaling_factor = scaling_factor.unsqueeze(-1).unsqueeze(-1)
        
        # 应用缩放
        W_norm_reshaped = W_reshaped * scaling_factor
        
        # 恢复原始形状 (Batch, M*N, K)
        W_norm = W_norm_reshaped.view(batch_size, num_tx, K)
        
        return W_norm

class ISAC_MLP(nn.Module):
    """
    基础的通感一体化多层感知机 (Baseline MLP)
    输入: 通信信道 H 和感知导向矢量 a
    输出: 满足功率约束的波束赋形矩阵 W
    """
    def __init__(self, num_ap, antennas_per_ap, num_users, hidden_dim=512, p_max=1.0):
        super().__init__()
        self.M = num_ap
        self.N = antennas_per_ap
        self.K = num_users
        self.num_tx = self.M * self.N
        
        # 1. 计算输入维度
        # H 是复数 (Batch, K, num_tx) -> 实部和虚部分开，展平: K * num_tx * 2
        # a 是复数 (Batch, num_tx) -> 实部和虚部分开，展平: num_tx * 2
        input_dim = (self.K * self.num_tx * 2) + (self.num_tx * 2)
        
        # 2. 计算输出维度
        # W 是复数 (Batch, num_tx, K) -> 实部和虚部分开，展平: num_tx * K * 2
        self.output_dim = self.num_tx * self.K * 2
        
        # 3. 构建多层感知机 (MLP)
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            
            nn.Linear(hidden_dim, self.output_dim)
        )
        
        # 4. 物理约束层
        self.power_norm = PowerNormalizationLayer(self.M, self.N, p_max)

    def forward(self, H, a):
        batch_size = H.shape[0]
        
        # ==========================================
        # 步骤 1: 特征提取与预处理 (复数转实数)
        # ==========================================
        # 将 H 拆分为实部和虚部并展平: (Batch, K*num_tx*2)
        H_real = torch.real(H).view(batch_size, -1)
        H_imag = torch.imag(H).view(batch_size, -1)
        
        # 将 a 拆分为实部和虚部并展平: (Batch, num_tx*2)
        a_real = torch.real(a).view(batch_size, -1)
        a_imag = torch.imag(a).view(batch_size, -1)
        
        # 拼接所有输入特征: (Batch, input_dim)
        x = torch.cat([H_real, H_imag, a_real, a_imag], dim=-1)
        
        # ==========================================
        # 步骤 2: 神经网络前向传播
        # ==========================================
        # 输出形状: (Batch, output_dim)
        out = self.net(x)
        
        # ==========================================
        # 步骤 3: 后处理与物理约束 (实数转复数 -> 功率投影)
        # ==========================================
        # 将输出重塑为实部和虚部: (Batch, num_tx, K)
        out = out.view(batch_size, self.num_tx, self.K, 2)
        W_real = out[..., 0]
        W_imag = out[..., 1]
        
        # 合成为复数波束矩阵 W
        W_raw = torch.complex(W_real, W_imag)
        
        # 经过功率归一化层，确保物理合法性
        W_final = self.power_norm(W_raw)
        
        return W_final
