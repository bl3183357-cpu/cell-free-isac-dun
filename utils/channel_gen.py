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

def generate_cell_free_steering_vector(batch_size, num_aps, antennas_per_ap, target_angles=None, device='cuda'):
    """
    生成 Cell-Free 架构下的分布式感知导向矢量
    假设每个 AP 是一个局部的 ULA，但 AP 之间没有连续的相位关系
    """
    total_antennas = num_aps * antennas_per_ap
    
    if target_angles is None:
        # 假设目标相对于每个 AP 的角度是不同的 (因为 AP 分布在不同位置)
        # 形状: (batch_size, num_aps, 1)
        target_angles = torch.rand(batch_size, num_aps, 1, device=device) * np.pi
    
    # 局部天线索引: [0, 1, ..., N_A - 1]
    local_indices = torch.arange(antennas_per_ap, device=device).float()
    
    # 计算每个 AP 内部的局部相位: pi * n * cos(theta_m)
    # 形状: (batch_size, num_aps, antennas_per_ap)
    local_phase = np.pi * local_indices.view(1, 1, -1) * torch.cos(target_angles)
    
    # 模拟不同 AP 之间由于距离目标不同而产生的随机初始相位偏移
    # 形状: (batch_size, num_aps, 1)
    ap_phase_offset = torch.rand(batch_size, num_aps, 1, device=device) * 2 * np.pi
    
    # 总相位 = AP初始相位 + 局部 ULA 相位
    total_phase = ap_phase_offset + local_phase
    
    # 展平为总天线维度: (batch_size, total_antennas)
    total_phase_flat = total_phase.reshape(batch_size, total_antennas)
    
    # 生成复数导向矢量
    ones_amplitude = torch.ones_like(total_phase_flat)
    a = torch.polar(ones_amplitude, total_phase_flat)
    
    # 注意：在真实的 ISAC 中，这里还需要乘以目标到各个 AP 的大尺度衰落(雷达方程)，
    # 这里仅生成了纯方向矢量(幅度为1)。
    
    return a
