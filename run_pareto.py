# 暂时不能用
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

# 修正导入路径和函数名
from utils.channel_gen import generate_rayleigh_channel, generate_steering_vector
from models.unfolding import ISAC_DeepUnfoldingNet
from models.GNN_Unfolding import ISAC_GNN_UnfoldingNet  
# ==========================================
# 1. 实验超参数设置 (与 main.py 保持一致)
# ==========================================
M = 4            # AP 数量
N = 4            # 每个 AP 的天线数
K = 4            # 通信用户数
P_MAX = 1.0      # 每个 AP 的最大发射功率 (瓦特)
NUM_SAMPLES = 2000 # 训练样本数
BATCH_SIZE = 64
EPOCHS = 150     # 为了加快速度，Pareto实验每个点跑 150 个 Epoch
LR = 1e-3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pareto 前沿测试的 Alpha 列表 (从 0.1 到 0.9)
# Alpha 越大，越偏向感知；Alpha 越小，越偏向通信
ALPHA_LIST = [0.1, 0.3, 0.5, 0.7, 0.9]

# ==========================================
# 2. 核心训练函数 (封装成可重复调用的形式)
# ==========================================
def train_for_pareto_point(alpha, H_train, a_train):
    """
    针对特定的 alpha 值训练模型，并返回最终的通信速率和感知功率
    """
    print(f"\n--- 开始训练 Alpha = {alpha:.1f} ---")
    
    # 实例化模型 
    model = ISAC_DeepUnfoldingNet(
        num_ap=M, antennas_per_ap=N, num_users=K, 
        num_layers=8, hidden_dim=256, p_max=P_MAX
    ).to(DEVICE)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    num_batches = NUM_SAMPLES // BATCH_SIZE
    
    # 用于记录最后 10 个 Epoch 的平均性能，使结果更稳定
    final_rates = []
    final_senses = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_rate = 0.0
        epoch_sense = 0.0
        
        # 随机打乱数据
        indices = torch.randperm(NUM_SAMPLES)
        H_shuffled = H_train[indices]
        a_shuffled = a_train[indices]
        
        for b in range(num_batches):
            H_batch = H_shuffled[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
            a_batch = a_shuffled[b*BATCH_SIZE : (b+1)*BATCH_SIZE]
            
            optimizer.zero_grad()
            
            # 前向传播
            W = model(H_batch, a_batch)
            
            # --- 计算通信速率 (Sum-Rate) ---
            HW = torch.bmm(H_batch, W)
            signal_power = torch.abs(torch.diagonal(HW, dim1=-2, dim2=-1))**2
            total_interference_plus_noise = torch.sum(torch.abs(HW)**2, dim=-1) - signal_power + 1e-3
            sinr = signal_power / total_interference_plus_noise
            rate = torch.log2(1.0 + sinr)
            sum_rate = torch.sum(rate, dim=-1)
            mean_sum_rate = torch.mean(sum_rate)
            
            # --- 计算感知波束图增益 (Sensing Beampattern Gain) ---
            W_flat = W.reshape(BATCH_SIZE, M * N, K)
            W_cov = torch.bmm(W_flat, torch.conj(torch.transpose(W_flat, 1, 2)))
            a_H = torch.conj(a_batch).unsqueeze(1)
            a_col = a_batch.unsqueeze(2)
            sensing_power = torch.real(torch.bmm(torch.bmm(a_H, W_cov), a_col)).squeeze()
            mean_sensing_power = torch.mean(sensing_power)
            
            # --- 计算无监督 Loss ---
            # 动态调整归一化权重 (根据 alpha 动态平衡)
            rate_norm = 1.0 / (1.0 - alpha + 1e-6)
            sense_norm = 10.0 / (alpha + 1e-6)
            
            loss_comm = -mean_sum_rate / rate_norm
            loss_sense = -mean_sensing_power / sense_norm
            
            loss = (1 - alpha) * loss_comm + alpha * loss_sense
            
            # 反向传播
            loss.backward()
            optimizer.step()
            
            epoch_rate += mean_sum_rate.item()
            epoch_sense += mean_sensing_power.item()
            
        avg_rate = epoch_rate / num_batches
        avg_sense = epoch_sense / num_batches
        
        # 记录最后 10 个 Epoch 的数据
        if epoch >= EPOCHS - 10:
            final_rates.append(avg_rate)
            final_senses.append(avg_sense)
            
        # 减少打印频率，只在最后打印一次
        if epoch == EPOCHS - 1:
            print(f"Epoch {epoch+1}/{EPOCHS} | Rate: {avg_rate:.2f} bps/Hz | Sense: {avg_sense:.2f} W")

    # 返回最后 10 个 Epoch 的平均值
    return np.mean(final_rates), np.mean(final_senses)

# ==========================================
# 3. 主程序：运行实验并绘图
# ==========================================
def main():
    print("生成固定的训练数据集...")
    H_train = generate_rayleigh_channel(NUM_SAMPLES, K, M * N, device=DEVICE)
    a_train = generate_steering_vector(NUM_SAMPLES, M * N, device=DEVICE)
    
    pareto_rates = []
    pareto_senses = []
    
    # 遍历所有的 Alpha 值
    for alpha in ALPHA_LIST:
        rate, sense = train_for_pareto_point(alpha, H_train, a_train)
        pareto_rates.append(rate)
        pareto_senses.append(sense)
        print(f"==> Alpha={alpha:.1f} 最终结果: Rate={rate:.2f}, Sense={sense:.2f}\n")
        
    # ==========================================
    # 4. 绘制 Pareto 前沿图
    # ==========================================
    plt.figure(figsize=(8, 6), dpi=150)
    
    # 画出折线图和散点
    plt.plot(pareto_senses, pareto_rates, 'b-o', linewidth=2, markersize=8, label='DUN (Proposed)')
    
    # 为每个点添加 Alpha 标签
    for i, alpha in enumerate(ALPHA_LIST):
        plt.annotate(f'$\\alpha={alpha}$', 
                     (pareto_senses[i], pareto_rates[i]), 
                     textcoords="offset points", 
                     xytext=(0,10), 
                     ha='center',
                     fontsize=10)
        
    plt.title('ISAC Pareto Frontier (Trade-off Curve)', fontsize=14)
    plt.xlabel('Sensing Beampattern Gain (Watts)', fontsize=12)
    plt.ylabel('Communication Sum-Rate (bps/Hz)', fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=12)
    
    plt.tight_layout()
    plt.savefig('pareto_frontier.png')
    print("Pareto 前沿图已保存为 pareto_frontier.png！")

if __name__ == "__main__":
    main()
