import torch
import math

def get_zf_beamformer(H, noise_var=1e-9):
    """
    计算迫零 (Zero-Forcing) 波束赋形矩阵 (支持 Batch 处理)
    
    参数:
    H: 信道矩阵, shape (B, K, N)
    noise_var: 正则化参数, 防止矩阵求逆时出现 NaN
    
    返回:
    W_zf: 迫零波束矩阵, shape (B, N, K)
    """
    B, K, N = H.shape
    
    # 1. 计算 H 的共轭转置 H^H -> shape: (B, N, K)
    H_H = torch.conj(torch.transpose(H, 1, 2))
    
    # 2. 计算 H * H^H -> shape: (B, K, K)
    H_HH = torch.matmul(H, H_H)
    
    # 3. 生成正则化项 (noise_var * I)
    # 注意：必须保持与 H 相同的设备 (CPU/GPU) 和数据类型 (Complex)
    eye = torch.eye(K, dtype=H.dtype, device=H.device).unsqueeze(0).expand(B, -1, -1)
    
    # 4. 矩阵求逆 (H * H^H + noise_var * I)^-1
    H_HH_inv = torch.linalg.inv(H_HH + noise_var * eye)
    
    # 5. 计算最终的 ZF 波束 W = H^H * (H * H^H)^-1 -> shape: (B, N, K)
    W_zf = torch.matmul(H_H, H_HH_inv)
    
    return W_zf

def get_heuristic_isac_beamformer(H, a, rho=0.5):
    """
    计算启发式 ISAC 波束赋形矩阵 (Warm-start 初始化)
    """
    B, K, N = H.shape
    
    # 1. 获取纯通信波束 (ZF)
    W_zf = get_zf_beamformer(H) # shape: (B, N, K)
    
    # ==========================================
    # 如果 a 是 2D 的 (B, N)，给它增加一个维度变成 (B, N, 1)
    if a.dim() == 2:
        a = a.unsqueeze(-1)
        
    # 现在 a 的形状是 (B, N, 1)，可以安全地 expand 到 (B, N, K) 了
    W_sense = a.expand(-1, -1, K) / math.sqrt(K)
    
    # 3. 提取纯方向 (全局 Frobenius 范数归一化)
    norm_zf = torch.norm(W_zf, dim=(1, 2), keepdim=True) + 1e-12
    W_zf_dir = W_zf / norm_zf
    
    norm_sense = torch.norm(W_sense, dim=(1, 2), keepdim=True) + 1e-12
    W_sense_dir = W_sense / norm_sense
    
    # 4. 按能量比例混合
    W_isac = math.sqrt(rho) * W_zf_dir + math.sqrt(1.0 - rho) * W_sense_dir
    
    return W_isac

