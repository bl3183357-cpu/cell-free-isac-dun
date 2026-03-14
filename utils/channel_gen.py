import torch
import numpy as np

def generate_cell_free_channel(batch_size, num_users, num_aps, antennas_per_ap, device='cuda'):
    """
    生成符合物理规律的 Cell-Free 瑞利衰落信道
    包含大尺度衰落 (基于距离的路径损耗) 和 小尺度衰落
    """
    total_antennas = num_aps * antennas_per_ap
    
    # 1. 生成小尺度衰落 (Small-scale fading) g ~ CN(0, 1)
    std_dev = np.sqrt(0.5)
    g_real = torch.randn(batch_size, num_users, total_antennas, device=device) * std_dev
    g_imag = torch.randn(batch_size, num_users, total_antennas, device=device) * std_dev
    G = torch.complex(g_real, g_imag)
    
    # 2. 生成大尺度衰落系数 Beta (Large-scale fading)
    # 在实际仿真中，这应该由随机撒点(AP坐标和用户坐标)计算距离得出。
    # 这里我们用一个简化的对数正态分布或均匀分布模拟不同AP到用户的巨大增益差异
    # 形状: (batch_size, num_users, num_aps)
    # 假设路径损耗在 -60dB 到 -100dB 之间波动 (数值仅为示例)
    beta_db = -60.0 - 40.0 * torch.rand(batch_size, num_users, num_aps, device=device)
    beta_linear = 10 ** (beta_db / 10.0)
    
    # 3. 将 Beta 扩展到每个 AP 的所有天线上
    # 同一个 AP 上的天线共享相同的大尺度衰落
    # 形状: (batch_size, num_users, num_aps, antennas_per_ap)
    beta_expanded = beta_linear.unsqueeze(-1).expand(-1, -1, -1, antennas_per_ap)
    
    # 重塑为与 G 相同的形状: (batch_size, num_users, total_antennas)
    beta_flat = beta_expanded.reshape(batch_size, num_users, total_antennas)
    
    # 4. 组合信道: H = sqrt(Beta) * G
    H = torch.sqrt(beta_flat) * G
    
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
