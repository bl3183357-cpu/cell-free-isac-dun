import torch

def compute_isac_loss(H, W, a, noise_var=1e-3, alpha=0.5, rate_norm=10.0, sense_norm=1.0):
    """
    计算 Cell-Free ISAC 系统的无监督损失函数
    
    参数:
        H: 通信信道矩阵, 形状 (Batch, K, N_tx)
        W: 神经网络输出的波束赋形矩阵, 形状 (Batch, N_tx, K)
        a: 感知目标的导向矢量, 形状 (Batch, N_tx)
        noise_var: 接收机噪声方差 (sigma^2)
        alpha: 权重因子 (0~1)。1表示纯通信，0表示纯感知
        rate_norm: 通信速率的归一化常数 (经验值)
        sense_norm: 感知功率的归一化常数 (经验值)
        
    返回:
        loss: 标量，用于反向传播的最终损失
        mean_sum_rate: 标量，当前批次的平均通信速率 (仅用于监控)
        mean_sense_power: 标量，当前批次的平均感知功率 (仅用于监控)
    """
    
    # ==========================================
    # 1. 通信指标：计算多用户 SINR 与 香农容量 (Sum-Rate)
    # ==========================================
    # 计算信道与波束的内积: H * W -> 形状 (Batch, K, K)
    # 结果矩阵的第 (b, i, j) 个元素代表：第 b 个样本中，第 i 个用户接收到的发给第 j 个用户的信号
    HW = torch.bmm(H, W) 
    
    # 期望信号功率：对角线元素的模平方 (用户 i 接收发给自己的信号)
    # torch.diagonal 提取对角线，形状变为 (Batch, K)
    signal_power = torch.abs(torch.diagonal(HW, dim1=-2, dim2=-1)) ** 2
    
    # 接收到的总功率：所有元素的模平方按行求和，形状 (Batch, K)
    total_power = torch.sum(torch.abs(HW) ** 2, dim=-1)
    
    # 干扰功率：总功率 - 期望信号功率
    interference_power = total_power - signal_power
    
    # 计算 SINR: 信号 / (干扰 + 噪声)
    sinr = signal_power / (interference_power + noise_var)
    
    # 计算香农容量 (Sum-Rate): R = sum( log2(1 + SINR) )
    # 形状 (Batch,)
    sum_rate = torch.sum(torch.log2(1 + sinr), dim=-1)
    
    # ==========================================
    # 2. 感知指标：计算目标方向的波束图增益 (Beampattern Gain)
    # ==========================================
    # 将导向矢量取共轭并增加维度，准备与 W 相乘
    # a 形状: (Batch, N_tx) -> a_H 形状: (Batch, 1, N_tx)
    a_H = a.conj().unsqueeze(1)
    
    # 计算感知方向的信号: a^H * W -> 形状 (Batch, 1, K)
    a_H_W = torch.bmm(a_H, W)
    
    # 计算感知目标方向上的总辐射功率 (所有波束在该方向的能量叠加)
    # 形状 (Batch,)
    sensing_power = torch.sum(torch.abs(a_H_W) ** 2, dim=-1).squeeze(-1)
    
    # ==========================================
    # 3. 终极 Loss 计算 (带有归一化与负号)
    # ==========================================
    # 注意：PyTorch 优化器是找最小值，所以我们要最大化速率和感知，必须加负号！
    # 归一化极其重要，否则数量级大的指标会“吞噬”数量级小的指标
    loss_comm = sum_rate / rate_norm
    loss_sense = sensing_power / sense_norm
    
    # 综合 Loss (对 Batch 取平均)
    loss = - torch.mean(alpha * loss_comm + (1 - alpha) * loss_sense)
    
    return loss, torch.mean(sum_rate), torch.mean(sensing_power)
