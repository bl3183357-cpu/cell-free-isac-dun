import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm 
import numpy as np

import argparse
import os

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from utils.channel_gen import generate_cell_free_channel, generate_cell_free_steering_vector
from utils.loss_fn import compute_isac_loss
from models.mlp import ISAC_MLP
from models.unfolding import ISAC_DeepUnfoldingNet
from models.GNN_Unfolding import ISAC_GNN_UnfoldingNet

import datetime 

def parse_args():
    parser = argparse.ArgumentParser(description="ISAC GNN Training")
    # 核心实验参数
    parser.add_argument('--algo', type=str, default='gnn_mrt', 
                        choices=['gnn_mrt', 'gnn_zf', 'gnn_heuristic', 'zf_only', 'mrt_only', 'heuristic_only'])
    parser.add_argument('--alpha', type=float, default=0.5, help='Trade-off between Rate and Sensing')
    parser.add_argument('--epochs', type=int, default=100)
    return parser.parse_args()

def train_isac_model():
    # ==========================================
    # 1. 系统参数与超参数设置
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" 启动训练! 当前设备: {device}")
    
    K = 3
    NUM_APS = 16             # 接入点 (AP) 的数量
    ANTENNAS_PER_AP = 4 
    P_MAX = 1.0
    NOISE_VAR = 1e-13
    
    # 【修改点 1】：定义固定的数据集大小
    TOTAL_SAMPLES = 20000  
    BATCH_SIZE = 1024
    EPOCHS = 1000
    LR = 1e-3
    ALPHA, RATE_NORM, SENSE_NORM = 0.8, 8.0, 1024

    # ==========================================
    # 2. 生成固定数据集并使用 sklearn 划分
    # ==========================================
    print(f" 正在生成 {TOTAL_SAMPLES} 个固定样本数据集...")
    # 注意：为了防止显存溢出，大规模数据集先在 CPU 上生成
    H_all = generate_cell_free_channel(
        batch_size=TOTAL_SAMPLES, 
        num_users=K, 
        num_aps=NUM_APS, 
        antennas_per_ap=ANTENNAS_PER_AP, 
        device='cpu'
)

    a_all = generate_cell_free_steering_vector(
        batch_size=TOTAL_SAMPLES, 
        num_aps=NUM_APS, 
        antennas_per_ap=ANTENNAS_PER_AP, 
        device='cpu'
    )

    # 将 PyTorch Tensor 转为 Numpy Array 交给 sklearn 划分
    # test_size=0.2 表示 80% 用于训练 (4万)，20% 用于验证 (1万)
    H_train_np, H_val_np, a_train_np, a_val_np = train_test_split(
        H_all.numpy(), a_all.numpy(), test_size=0.2, random_state=42
    )

    # 将划分好的 Numpy 数据转回 PyTorch Tensor
    H_train = torch.tensor(H_train_np)
    a_train = torch.tensor(a_train_np)
    H_val = torch.tensor(H_val_np)
    a_val = torch.tensor(a_val_np)

    # 构建 PyTorch 的 DataLoader，负责自动分批次 (Batching) 和打乱 (Shuffle)
    train_dataset = TensorDataset(H_train, a_train)
    val_dataset = TensorDataset(H_val, a_val)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    print(f" 数据集划分完成: 训练集 {len(train_dataset)} 样本, 验证集 {len(val_dataset)} 样本")

    # ==========================================
    # 3. 初始化模型与优化器
    # ==========================================
    args = parse_args()
    
    # 根据命令行参数配置模型
    if args.algo == 'gnn_mrt':
        model = ISAC_GNN_UnfoldingNet(NUM_APS, ANTENNAS_PER_AP, num_layers=4, init_method='mrt').to(device)
    elif args.algo == 'gnn_zf':
        model = ISAC_GNN_UnfoldingNet(NUM_APS, ANTENNAS_PER_AP, num_layers=4, init_method='zf').to(device)
    elif args.algo == 'gnn_heuristic':
        model = ISAC_GNN_UnfoldingNet(NUM_APS, ANTENNAS_PER_AP, num_layers=4, init_method='heuristic').to(device)
    elif args.algo == 'mrt_only':
        model = ISAC_GNN_UnfoldingNet(NUM_APS, ANTENNAS_PER_AP, num_layers=0, init_method='mrt').to(device)
    elif args.algo == 'heuristic_only':
        model = ISAC_GNN_UnfoldingNet(NUM_APS, ANTENNAS_PER_AP, num_layers=0, init_method='heuristic').to(device)
    elif args.algo == 'zf_only':
        model = ISAC_GNN_UnfoldingNet(NUM_APS, ANTENNAS_PER_AP, num_layers=0, init_method='zf').to(device)
        args.epochs = 1  # 传统算法不需要迭代训练，跑 1 个 epoch 评估即可
    else:
        raise ValueError(f"❌ 未知的算法配置: args.algo='{args.algo}'，请检查拼写！")

    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)

    # 【修改点 2】：分别记录训练集和验证集的指标
    history = {
        'train_loss': [], 'val_loss': [], 
        'val_sum_rate': [], 'val_sense_power': []
    }

    # ==========================================
    # 4. 开始训练循环 (Training Loop)
    # ==========================================
    pbar = tqdm(range(EPOCHS), desc="Training Progress")
    
    for epoch in pbar:
        # ------------------------------------------
        # 阶段 A: 训练阶段 (Training Phase)
        # ------------------------------------------
        

        model.train() # 启用 Dropout/BatchNorm
        epoch_train_loss = 0.0
        for H_batch, a_batch in train_loader:
            # 将当前批次的数据移动到 GPU
            H_batch = H_batch.to(device)
            a_batch = a_batch.to(device)
            
            optimizer.zero_grad()
            W = model(H_batch, a_batch)
            loss, _, _ = compute_isac_loss(
                H_batch, W, a_batch, NOISE_VAR, ALPHA, RATE_NORM, SENSE_NORM
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            epoch_train_loss += loss.item() * H_batch.size(0)
            
        avg_train_loss = epoch_train_loss / len(train_dataset)
        if epoch % 50 == 0:
            with torch.no_grad():
                HW = torch.bmm(H_batch, W)

                signal_power = torch.abs(torch.diagonal(HW, dim1=-2, dim2=-1)) ** 2
                total_power = torch.sum(torch.abs(HW) ** 2, dim=-1)
                interference_power = total_power - signal_power
                sinr = signal_power / (interference_power + NOISE_VAR)

                # ===== Channel statistics =====
                print("\n===== CHANNEL =====")
                print("H mean |h|^2:", torch.mean(torch.abs(H_batch)**2).item())
                print("H Fro norm:", torch.mean(torch.norm(H_batch, dim=(1,2))).item())

                # ===== Beamformer statistics =====
                print("\n===== BEAMFORMER =====")
                print("W mean |w|^2:", torch.mean(torch.abs(W)**2).item())
                print("W total power:", torch.mean(torch.sum(torch.abs(W)**2, dim=(1,2))).item())

                # ===== Signal statistics =====
                print("\n===== RECEIVED POWER =====")
                print("signal power:", signal_power.mean().item())
                print("interf power:", interference_power.mean().item())
                print("noise:", NOISE_VAR)

                # ===== SINR =====
                print("\n===== SINR =====")
                print("mean SINR:", sinr.mean().item())
                print("max SINR:", sinr.max().item())
                print("min SINR:", sinr.min().item())

                # ===== Rate =====
                sum_rate = torch.sum(torch.log2(1 + sinr), dim=-1)
                print("\n===== RATE =====")
                print("mean sum-rate:", sum_rate.mean().item())
        # ------------------------------------------
        # 阶段 B: 验证阶段 (Validation Phase) - 严谨的性能评估
        # ------------------------------------------
        model.eval() # 关闭 Dropout/BatchNorm，冻结权重
        epoch_val_loss = 0.0
        epoch_val_rate = 0.0
        epoch_val_sense = 0.0
        
        with torch.no_grad(): # 不计算梯度，节省显存并加速
            for H_batch, a_batch in val_loader:
                H_batch = H_batch.to(device)
                a_batch = a_batch.to(device)
                
                W = model(H_batch, a_batch)
                loss, mean_rate, mean_sense = compute_isac_loss(
                    H_batch, W, a_batch, NOISE_VAR, ALPHA, RATE_NORM, SENSE_NORM
                )
                
                # 累加验证集指标
                epoch_val_loss += loss.item() * H_batch.size(0)
                epoch_val_rate += mean_rate.item() * H_batch.size(0)
                epoch_val_sense += mean_sense.item() * H_batch.size(0)
                
        # 计算验证集上的平均指标
        avg_val_loss = epoch_val_loss / len(val_dataset)
        avg_val_rate = epoch_val_rate / len(val_dataset)
        avg_val_sense = epoch_val_sense / len(val_dataset)

        # ------------------------------------------
        # 阶段 C: 记录与调度
        # ------------------------------------------
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_sum_rate'].append(avg_val_rate)
        history['val_sense_power'].append(avg_val_sense)
        
        scheduler.step(avg_val_loss) # 根据严格的验证集 Loss 来衰减学习率

        pbar.set_postfix({
            'T_Loss': f"{avg_train_loss:.4f}", 
            'V_Loss': f"{avg_val_loss:.4f}", 
            'V_Rate': f"{avg_val_rate:.2f}", 
            'V_Sense': f"{avg_val_sense:.2f}"
        })

    print("\n 训练完成！")
    
    train_params = {
        'K': K,
        'AP': NUM_APS,
        'Ant/AP': ANTENNAS_PER_AP,
        'Samples': TOTAL_SAMPLES,
        'Batch': BATCH_SIZE,
        'Epochs': EPOCHS,
        'LR': LR,
        'Alpha': ALPHA,
        'Rate_Norm': RATE_NORM,
        'Sense_Norm': SENSE_NORM
    }
    
    # 调用更新后的绘图函数
    plot_training_history(history, train_params)
    
    save_dir = f"./results/{args.algo}/alpha_{args.alpha}/"
    os.makedirs(save_dir, exist_ok=True)
    
    # 训练逻辑...
    # 保存模型权重和日志到 save_dir
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pth"))


def plot_training_history(history, params):
    """
    绘制训练曲线并保存
    :param history: 包含 loss, rate, sense_power 的字典
    :param params: 包含训练超参数的字典
    """
    epochs = range(len(history['train_loss']))
    
    # 稍微增加图片高度，为底部的参数文本留出空间
    fig = plt.figure(figsize=(16, 6))
    
    plt.subplot(1, 3, 1)
    plt.plot(epochs, history['train_loss'], 'r--', label='Train Loss', alpha=0.5)
    plt.plot(epochs, history['val_loss'], 'r-', label='Val Loss')
    plt.title('Unsupervised Loss')
    plt.grid(True); plt.legend()
    
    plt.subplot(1, 3, 2)
    plt.plot(epochs, history['val_sum_rate'], 'b-', label='Val Sum-Rate')
    plt.title('Validation Sum-Rate')
    plt.grid(True); plt.legend()
    
    plt.subplot(1, 3, 3)
    plt.plot(epochs, history['val_sense_power'], 'g-', label='Val Sensing Power')
    plt.title('Validation Sensing Gain')
    plt.grid(True); plt.legend()
    
    # ==========================================
    # 将参数字典转换为格式化字符串
    # ==========================================
    # 每 5 个参数换一行，避免文本太长超出边界
    param_items = [f"{k}: {v}" for k, v in params.items()]
    param_str = " | ".join(param_items[:5]) + "\n" + " | ".join(param_items[5:])
    
    # 在图片底部添加文本框
    plt.figtext(0.5, 0.02, f"Training Parameters:\n{param_str}", 
                ha="center", va="bottom", fontsize=10, 
                bbox={"facecolor":"#f0f0f0", "alpha":0.8, "pad":5, "edgecolor":"gray"})
    
    plt.tight_layout(rect=[0, 0.15, 1, 1]) 
    
    # ==========================================
    # 生成带时间戳的唯一文件名，防止覆盖
    # ==========================================
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"training_curves_{timestamp}.png"
    
    plt.savefig(filename, dpi=300)
    print(f"\n 绘图完成！训练曲线已保存为: {filename}")
    
    plt.close() # 释放内存，防止在循环中多次调用时内存泄漏

if __name__ == '__main__':
    train_isac_model()