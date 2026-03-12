import torch
import numpy as np

def generate_rayleigh_channel(batch_size, num_users, num_tx_antennas, device='cuda'):
    """
    生成通信用户的瑞利衰落信道 (Rayleigh Fading Channel)
    数学模型: H ~ CN(0, 1)，即实部和虚部均服从 N(0, 0.5) 的高斯分布
    
    参数:
        batch_size: 批次大小 (一次并行计算多少组信道)
        num_users: 通信用户数 K
        num_tx_antennas: Cell-Free 系统总发射天线数 (M个AP * 每个AP N根天线)
        device: 运行设备 ('cuda' 或 'cpu')
    返回:
        H: 形状为 (batch_size, num_users, num_tx_antennas) 的复数张量
    """
    # 生成实部和虚部，方差为 0.5 (标准差为 sqrt(0.5))
    std_dev = np.sqrt(0.5)
    real_part = torch.randn(batch_size, num_users, num_tx_antennas, device=device) * std_dev
    imag_part = torch.randn(batch_size, num_users, num_tx_antennas, device=device) * std_dev
    
    # 合成为复数张量
    H = torch.complex(real_part, imag_part)
    return H

def generate_steering_vector(batch_size, num_tx_antennas, angles=None, device='cuda'):
    """
    生成雷达感知目标的均匀线阵(ULA)导向矢量 (Steering Vector)
    数学公式: a(theta) = [1, e^{j*pi*cos(theta)}, ..., e^{j*pi*(N-1)*cos(theta)}]
    
    参数:
        batch_size: 批次大小
        num_tx_antennas: 总发射天线数
        angles: 目标的角度(弧度制)。如果为 None，则随机生成 [0, pi] 的角度
        device: 运行设备
    返回:
        a: 形状为 (batch_size, num_tx_antennas) 的复数张量
    """
    if angles is None:
        # 随机生成目标角度，范围 [0, π]
        angles = torch.rand(batch_size, 1, device=device) * np.pi
        
    # 天线索引: [0, 1, 2, ..., N-1]
    indices = torch.arange(num_tx_antennas, device=device).float()
    
    # 计算相位: pi * n * cos(theta) (假设天线间距 d = lambda / 2)
    phase = np.pi * indices * torch.cos(angles)
    
    # 使用 torch.polar(abs, angle) 生成复数张量: r*e^{j*theta}，这里幅度 r=1
    ones_amplitude = torch.ones_like(phase)
    a = torch.polar(ones_amplitude, phase)
    
    return a
