import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from utils.channel_gen import generate_cell_free_channel, generate_cell_free_steering_vector
from utils.loss_fn import compute_isac_loss
from models.GNN_Unfolding import ISAC_GNN_UnfoldingNet
from models.mlp import PowerNormalizationLayer  

def per_ap_power_normalize(W, num_ap=16, antennas_per_ap=4, p_max=1.0):
    """
    对任意波束矩阵进行单 AP 功率归一化，确保每个 AP 满功率发射
    """
    batch_size, num_tx, K = W.shape
    
    # 重塑为 (Batch, AP数量, 每个AP的天线数, 用户数)
    W_reshaped = W.view(batch_size, num_ap, antennas_per_ap, K)
    
    # 计算每个 AP 的当前总功率
    power_per_ap = torch.sum(torch.abs(W_reshaped)**2, dim=(2, 3))
    
    # 计算缩放因子：强制将每个 AP 的功率拉伸/压缩到 p_max
    # 注意：基线算法通常需要满功率发射才能公平对比，所以这里直接除以当前功率
    scaling_factor = torch.sqrt(p_max / (power_per_ap + 1e-12))
    scaling_factor = scaling_factor.unsqueeze(-1).unsqueeze(-1)
    
    # 应用缩放并恢复形状
    W_norm = (W_reshaped * scaling_factor).view(batch_size, num_tx, K)
    
    return W_norm


def get_heuristic_isac_beamformer(H, a, K, rho=0.5, noise_var=1e-13, num_ap=16, antennas_per_ap=4):
    """
    传统 ISAC 算法 1: 启发式加权 (Heuristic Trade-off)
    使用方向矢量进行能量比例混合
    rho: 通信能量占比 (0~1)
    """
    W_zf = get_zf_beamformer(H, noise_var)
    W_sense = get_sensing_beamformer(a, K)
    
    # 提取方向 (总功率为1)
    W_zf_dir = W_zf / (torch.norm(W_zf, dim=(1,2), keepdim=True) + 1e-12)
    W_sense_dir = W_sense / (torch.norm(W_sense, dim=(1,2), keepdim=True) + 1e-12)
    
    # 混合 (此时总功率约为1)
    # 注意：这里最好用 torch.sqrt 而不是 np.sqrt，保持张量操作
    W_isac_raw = torch.sqrt(torch.tensor(rho)) * W_zf_dir + torch.sqrt(torch.tensor(1 - rho)) * W_sense_dir
    
    # 致命修复：将总功率为 1 的混合波束，按单 AP 约束放大到满功率 (总功率 16)！
    return per_ap_power_normalize(W_isac_raw, num_ap, antennas_per_ap)


def get_nsp_isac_beamformer(H, a, K, rho=0.8, noise_var=1e-13):
    """
    传统 ISAC 算法 2: 零空间投影 (Null-Space Projection)
    """
    B, num_K, N = H.shape
    
    W_zf = get_zf_beamformer(H, noise_var)
    
    # 计算零空间投影矩阵 P
    H_H = torch.conj(torch.transpose(H, 1, 2))
    H_H_H = torch.bmm(H, H_H)
    reg = (noise_var * 10) * torch.eye(num_K, dtype=H.dtype, device=H.device).unsqueeze(0)
    inv = torch.linalg.inv(H_H_H + reg)
    Proj_H = torch.bmm(torch.bmm(H_H, inv), H)
    
    I = torch.eye(N, dtype=H.dtype, device=H.device).unsqueeze(0).expand(B, -1, -1)
    P_null = I - Proj_H
    
    # 投影感知波束
    a_proj = torch.bmm(P_null, a.unsqueeze(-1))
    W_sense_nsp = a_proj.expand(-1, -1, K)
    
    # 提取纯方向矩阵
    W_zf_dir = W_zf / (torch.norm(W_zf, dim=(1,2), keepdim=True) + 1e-12)
    W_sense_dir = W_sense_nsp / (torch.norm(W_sense_nsp, dim=(1,2), keepdim=True) + 1e-12)
    
    # 按能量比例混合
    W_isac_nsp = np.sqrt(rho) * W_zf_dir + np.sqrt(1 - rho) * W_sense_dir
    return W_isac_nsp


def get_mrt_beamformer(H, num_ap=16, antennas_per_ap=4):
    W_raw = torch.conj(torch.transpose(H, 1, 2))
    return per_ap_power_normalize(W_raw, num_ap, antennas_per_ap)

def get_zf_beamformer(H, noise_var=1e-13):
    """传统 ZF 波束赋形: W = H^H (H H^H)^-1"""
    H_H = torch.conj(torch.transpose(H, 1, 2)) # (B, N, K)
    H_H_H = torch.bmm(H, H_H) # (B, K, K)
    
    # 加入微小正则化防止矩阵奇异 (类似 RZF)
    reg = (noise_var * 10) * torch.eye(H.size(1), dtype=H.dtype, device=H.device).unsqueeze(0)
    inv = torch.linalg.inv(H_H_H + reg)
    
    W_zf = torch.bmm(H_H, inv)
    return W_zf

def get_sensing_beamformer(a, K):
    """纯感知波束: 将导向矢量 a 广播给所有 K 个数据流"""
    # a 形状: (B, N) -> (B, N, 1) -> (B, N, K)
    return a.unsqueeze(-1).expand(-1, -1, K)

def evaluate_models():
    # ==========================================
    # 1. 参数设置 (需与训练时保持绝对一致)
    # ==========================================
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 启动评估! 当前设备: {device}")
    
    K = 3
    NUM_APS = 16
    ANTENNAS_PER_AP = 4 
    P_MAX = 1.0
    NOISE_VAR = 1e-13
    
    TEST_SAMPLES = 2000  # 测试集大小
    BATCH_SIZE = 500
    ALPHA, RATE_NORM, SENSE_NORM = 0.7, 20.0, 1024

    # ==========================================
    # 2. 生成全新的测试数据
    # ==========================================
    print(f"📊 正在生成 {TEST_SAMPLES} 个测试样本...")
    H_test = generate_cell_free_channel(TEST_SAMPLES, K, NUM_APS, ANTENNAS_PER_AP, device=device)
    a_test = generate_cell_free_steering_vector(TEST_SAMPLES, NUM_APS, ANTENNAS_PER_AP, device=device)
    
    # ==========================================
    # 3. 加载训练好的 GNN 模型与归一化层
    # ==========================================
    model = ISAC_GNN_UnfoldingNet(
        num_ap=NUM_APS, antennas_per_ap=ANTENNAS_PER_AP, num_layers=8, hidden_dim=64, p_max=P_MAX
    ).to(device)
    
    try:
        model.load_state_dict(torch.load("isac_gnn_weights.pth", map_location=device))
        print("✅ 成功加载模型权重: isac_gnn_weights.pth")
    except FileNotFoundError:
        print("❌ 未找到 isac_gnn_weights.pth，请先运行 main.py 进行训练！")
        return
    
    model.eval()
    
    # 实例化功率归一化层，用于传统算法的公平对比
    power_norm = PowerNormalizationLayer(NUM_APS, ANTENNAS_PER_AP, P_MAX).to(device)

    # ==========================================
    # 4. 评估循环
    # ==========================================
    results = {
        "GNN Model": {"rate": [], "sense": []},
        "ZF (Comm Only)": {"rate": [], "sense": []},
        "MRT (Comm Only)": {"rate": [], "sense": []},
        "Sensing Only": {"rate": [], "sense": []},
        "ISAC: Heuristic": {"rate": [], "sense": []},
        "ISAC: NSP": {"rate": [], "sense": []}
    }
    
    num_batches = TEST_SAMPLES // BATCH_SIZE
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Evaluating"):
            start_idx = i * BATCH_SIZE
            end_idx = start_idx + BATCH_SIZE
            
            H_batch = H_test[start_idx:end_idx]
            a_batch = a_test[start_idx:end_idx]
            
            # --- 方法 A: 训练好的 GNN 模型 ---
            W_gnn = model(H_batch, a_batch)
            _, rate_gnn, sense_gnn = compute_isac_loss(H_batch, W_gnn, a_batch, NOISE_VAR, ALPHA, RATE_NORM, SENSE_NORM)
            results["GNN Model"]["rate"].append(rate_gnn.item())
            results["GNN Model"]["sense"].append(sense_gnn.item())
            
            # --- 方法 B: ZF (迫零) ---
            W_zf = power_norm(get_zf_beamformer(H_batch, NOISE_VAR))
            _, rate_zf, sense_zf = compute_isac_loss(H_batch, W_zf, a_batch, NOISE_VAR, ALPHA, RATE_NORM, SENSE_NORM)
            results["ZF (Comm Only)"]["rate"].append(rate_zf.item())
            results["ZF (Comm Only)"]["sense"].append(sense_zf.item())
            
            # --- 方法 C: MRT (最大比传输) ---
            W_mrt = power_norm(get_mrt_beamformer(H_batch))
            _, rate_mrt, sense_mrt = compute_isac_loss(H_batch, W_mrt, a_batch, NOISE_VAR, ALPHA, RATE_NORM, SENSE_NORM)
            results["MRT (Comm Only)"]["rate"].append(rate_mrt.item())
            results["MRT (Comm Only)"]["sense"].append(sense_mrt.item())
            
            # --- 方法 D: 纯感知 ---
            W_sense = power_norm(get_sensing_beamformer(a_batch, K))
            _, rate_sense, sense_sense = compute_isac_loss(H_batch, W_sense, a_batch, NOISE_VAR, ALPHA, RATE_NORM, SENSE_NORM)
            results["Sensing Only"]["rate"].append(rate_sense.item())
            results["Sensing Only"]["sense"].append(sense_sense.item())

            # --- 方法 E: ISAC Heuristic ---
            W_heu = power_norm(get_heuristic_isac_beamformer(H_batch, a_batch, K, rho=0.8, noise_var=NOISE_VAR))
            _, rate_heu, sense_heu = compute_isac_loss(H_batch, W_heu, a_batch, NOISE_VAR, ALPHA, RATE_NORM, SENSE_NORM)
            results["ISAC: Heuristic"]["rate"].append(rate_heu.item())
            results["ISAC: Heuristic"]["sense"].append(sense_heu.item())
            
            # --- 方法 F: ISAC NSP ---
            W_nsp = power_norm(get_nsp_isac_beamformer(H_batch, a_batch, K, noise_var=NOISE_VAR))
            _, rate_nsp, sense_nsp = compute_isac_loss(H_batch, W_nsp, a_batch, NOISE_VAR, ALPHA, RATE_NORM, SENSE_NORM)
            results["ISAC: NSP"]["rate"].append(rate_nsp.item())
            results["ISAC: NSP"]["sense"].append(sense_nsp.item())

    # ==========================================
    # 5. 打印结果与可视化
    # ==========================================
    print("\n" + "="*50)
    print(f"{'Algorithm':<18} | {'Sum-Rate (bps/Hz)':<15} | {'Sensing Power':<15}")
    print("-" * 50)
    
    plot_data_rate = []
    plot_data_sense = []
    labels = []
    
    for algo, metrics in results.items():
        avg_rate = np.mean(metrics["rate"])
        avg_sense = np.mean(metrics["sense"])
        print(f"{algo:<18} | {avg_rate:<15.4f} | {avg_sense:<15.4f}")
        
        labels.append(algo)
        plot_data_rate.append(avg_rate)
        plot_data_sense.append(avg_sense)
    print("="*50)

    # 绘制柱状图对比
    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color1 = 'tab:blue'
    ax1.set_xlabel('Algorithms', fontweight='bold')
    ax1.set_ylabel('Sum-Rate (bps/Hz)', color=color1, fontweight='bold')
    bars1 = ax1.bar(x - width/2, plot_data_rate, width, label='Sum-Rate', color=color1, alpha=0.8)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)

    ax2 = ax1.twinx()  
    color2 = 'tab:green'
    ax2.set_ylabel('Sensing Power', color=color2, fontweight='bold')
    bars2 = ax2.bar(x + width/2, plot_data_sense, width, label='Sensing Power', color=color2, alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color2)

    plt.title('Performance Comparison: GNN vs Traditional Algorithms', fontweight='bold')
    fig.tight_layout()
    plt.savefig("evaluation_comparison.png", dpi=300)
    print("\n 📈 对比图表已保存为 evaluation_comparison.png")

if __name__ == '__main__':
    evaluate_models()
