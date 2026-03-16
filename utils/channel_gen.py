import torch
import numpy as np

def generate_cell_free_channel(
    batch_size, num_users, num_aps, antennas_per_ap, 
    area_size=1000.0,      # 仿真区域大小 1000m x 1000m
    ap_height=15.0,        # AP 高度 (米)
    user_height=1.5,       # 用户高度 (米)
    carrier_freq_ghz=1.9,  # 载波频率 (GHz)
    shadow_fading_std=8.0, # 阴影衰落标准差 (dB)
    device='cuda'
):
    """
    生成符合物理几何拓扑的 Cell-Free 瑞利衰落信道
    """
    total_antennas = num_aps * antennas_per_ap
    
    # ==========================================
    # 1. 物理空间拓扑生成 (Spatial Topology)
    # ==========================================
    # 随机生成 AP 和 User 的二维坐标 (x, y)
    # 形状: (batch_size, num_aps/num_users, 2)
    ap_coords = torch.rand(batch_size, num_aps, 2, device=device) * area_size
    user_coords = torch.rand(batch_size, num_users, 2, device=device) * area_size
    
    # 扩展维度以计算两两之间的距离矩阵
    # ap_coords_exp: (batch_size, 1, num_aps, 2)
    # user_coords_exp: (batch_size, num_users, 1, 2)
    ap_coords_exp = ap_coords.unsqueeze(1)
    user_coords_exp = user_coords.unsqueeze(2)
    
    # 计算二维距离平方
    dist_2d_sq = torch.sum((ap_coords_exp - user_coords_exp)**2, dim=-1)
    
    # 计算三维距离 (加入高度差)
    height_diff_sq = (ap_height - user_height)**2
    dist_3d = torch.sqrt(dist_2d_sq + height_diff_sq) # 形状: (batch_size, num_users, num_aps)
    
    # ==========================================
    # 2. 大尺度衰落计算 (Large-scale Fading)
    # ==========================================
    # A. 路径损耗 (Path Loss) - 采用类似 3GPP UMi 的简化模型
    # 公式: PL(dB) = 35.3 * log10(d) + 37.6 + 21 * log10(fc)
    # 注意：为了避免距离过近导致对数发散，通常设置一个最小参考距离 (如 10m)
    min_dist = 10.0
    dist_3d_clipped = torch.clamp(dist_3d, min=min_dist)
    
    pl_db = 35.3 * torch.log10(dist_3d_clipped) + 37.6 + 21.0 * np.log10(carrier_freq_ghz)
    
    # B. 阴影衰落 (Shadow Fading) - 对数正态分布
    shadow_fading_db = torch.randn_like(dist_3d) * shadow_fading_std
    
    # C. 综合大尺度衰落 Beta (dB)
    # Beta_dB = -PL + Shadow_Fading
    beta_db = -pl_db + shadow_fading_db
    beta_linear = 10 ** (beta_db / 10.0) # 转换为线性值
    
    # ==========================================
    # 3. 扩展 Beta 并生成小尺度衰落
    # ==========================================
    # 将 Beta 扩展到每个 AP 的所有天线上
    # 形状: (batch_size, num_users, num_aps, antennas_per_ap)
    beta_expanded = beta_linear.unsqueeze(-1).expand(-1, -1, -1, antennas_per_ap)
    beta_flat = beta_expanded.reshape(batch_size, num_users, total_antennas)
    
    # 生成小尺度衰落 (Small-scale fading) g ~ CN(0, 1)
    std_dev = np.sqrt(0.5)
    g_real = torch.randn(batch_size, num_users, total_antennas, device=device) * std_dev
    g_imag = torch.randn(batch_size, num_users, total_antennas, device=device) * std_dev
    G = torch.complex(g_real, g_imag)
    
    # ==========================================
    # 4. 组合信道
    # ==========================================
    # H = sqrt(Beta) * G
    H = torch.sqrt(beta_flat) * G
    
    # 返回信道 H，同时可以返回坐标和 beta 供后续验证或算法使用, 
    # beta_linear, ap_coords, user_coords
    return H

def generate_cell_free_steering_vector(
    batch_size,
    num_aps,
    antennas_per_ap,
    area_size=1000.0,
    ap_height=15.0,
    target_height=1.5,
    carrier_freq_ghz=1.9,
    device='cuda',
    ap_coords=None,
    target_coords=None,
    include_pathloss=False,
    pathloss_exp=2.0
):
    """
    基于几何拓扑生成 Cell-Free 架构下的分布式感知导向矢量 a
    风格与 generate_cell_free_channel 类似

    返回:
        a: (batch_size, total_antennas) complex
        ap_coords: (batch_size, num_aps, 2)
        target_coords: (batch_size, 1, 2)
    """
    total_antennas = num_aps * antennas_per_ap

    # 波长
    c = 3e8
    fc = carrier_freq_ghz * 1e9
    wavelength = c / fc

    # ==========================================
    # 1. 生成空间拓扑
    # ==========================================
    if ap_coords is None:
        ap_coords = torch.rand(batch_size, num_aps, 2, device=device) * area_size

    if target_coords is None:
        target_coords = torch.rand(batch_size, 1, 2, device=device) * area_size

    # 目标坐标扩展到各 AP
    # ap_coords:     (B, M, 2)
    # target_coords: (B, 1, 2)
    delta = target_coords - ap_coords   # (B, M, 2)

    dx = delta[..., 0]
    dy = delta[..., 1]

    # ==========================================
    # 2. 几何量：距离、角度
    # ==========================================
    dist_2d_sq = dx**2 + dy**2
    height_diff_sq = (ap_height - target_height)**2
    dist_3d = torch.sqrt(dist_2d_sq + height_diff_sq)   # (B, M)

    # 每个 AP 看目标的方位角
    theta = torch.atan2(dy, dx)   # (B, M)

    # ==========================================
    # 3. 每个 AP 内部的局部 ULA 导向矢量
    # ==========================================
    # 半波长间距 d = lambda / 2
    # 相位增量 = 2pi * d/lambda * sin(theta) = pi * sin(theta)
    local_indices = torch.arange(antennas_per_ap, device=device).float()  # (N_A,)

    # (B, M, N_A)
    local_phase = np.pi * local_indices.view(1, 1, -1) * torch.sin(theta).unsqueeze(-1)

    # ==========================================
    # 4. AP 级传播相位
    # ==========================================
    # (B, M, 1)
    ap_phase = (-2.0 * np.pi / wavelength) * dist_3d.unsqueeze(-1)

    total_phase = ap_phase + local_phase   # (B, M, N_A)

    # ==========================================
    # 5. 是否加入幅度衰减
    # ==========================================
    if include_pathloss:
        amp = 1.0 / torch.clamp(dist_3d, min=1.0).pow(pathloss_exp / 2.0)  # (B, M)
        amp = amp.unsqueeze(-1).expand(-1, -1, antennas_per_ap)            # (B, M, N_A)
    else:
        amp = torch.ones(batch_size, num_aps, antennas_per_ap, device=device)

    # ==========================================
    # 6. 生成复数导向矢量并展平
    # ==========================================
    a_local = torch.polar(amp, total_phase)      # (B, M, N_A)
    a = a_local.reshape(batch_size, total_antennas)  # (B, M*N_A)
    #可选返回ap_coords, target_coords
    return a