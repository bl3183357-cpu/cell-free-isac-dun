import torch
import torch.optim as optim
import argparse
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader

from utils.channel_gen import generate_cell_free_channel, generate_cell_free_steering_vector
from utils.loss_fn import compute_isac_loss
from models.GNN_Unfolding import ISAC_GNN_UnfoldingNet

# ==========================================
# 1. 参数解析
# ==========================================
def parse_args():
    parser = argparse.ArgumentParser(description="ISAC GNN Training with Lightning")
    parser.add_argument('--algo', type=str, default='gnn_mrt', 
                        choices=['gnn_mrt', 'gnn_zf', 'gnn_heuristic', 'zf_only', 'mrt_only', 'heuristic_only'])
    parser.add_argument('--alpha', type=float, default=0.5, help='Trade-off between Rate and Sensing')
    parser.add_argument('--epochs', type=int, default=100)
    return parser.parse_args()

# ==========================================
# 2. 定义 Lightning 系统 
# ==========================================
class ISAC_System(pl.LightningModule):
    def __init__(self, args, num_aps, antennas_per_ap, noise_var, rate_norm, sense_norm):
        super().__init__()
        # 自动保存超参数到 Wandb 和 Checkpoint 中
        self.save_hyperparameters() 
        self.args = args
        self.noise_var = noise_var
        self.rate_norm = rate_norm
        self.sense_norm = sense_norm
        
        # 根据参数初始化模型
        init_method = args.algo.replace('gnn_', '').replace('_only', '')
        num_layers = 1 if 'only' in args.algo else 8
        self.model = ISAC_GNN_UnfoldingNet(num_aps, antennas_per_ap, num_layers=num_layers, init_method=init_method)

    def forward(self, H, a):
        return self.model(H, a)

    def training_step(self, batch, batch_idx):
        H_batch, a_batch = batch
        W = self(H_batch, a_batch)
        
        # 计算 Loss
        loss, _, _ = compute_isac_loss(
            H_batch, W, a_batch, self.noise_var, self.args.alpha, self.rate_norm, self.sense_norm
        )
        
        # 记录训练 Loss (on_epoch=True 会自动计算整个 epoch 的平均值)
        self.log("Loss/Train", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        H_batch, a_batch = batch
        W = self(H_batch, a_batch)
        
        loss, mean_rate, mean_sense = compute_isac_loss(
            H_batch, W, a_batch, self.noise_var, self.args.alpha, self.rate_norm, self.sense_norm
        )
        
        # 记录验证集指标 (sync_dist=True 用于未来可能的多卡训练同步)
        self.log("Loss/Validation", loss, prog_bar=True, sync_dist=True)
        self.log("Metrics/Sum_Rate", mean_rate, sync_dist=True)
        self.log("Metrics/Sensing_Power", mean_sense, sync_dist=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        # Lightning 会自动根据 monitor 的指标调用 scheduler.step()
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "Loss/Validation" 
            }
        }

# ==========================================
# 3. 主训练流程 
# ==========================================
def train_isac_model():
    args = parse_args()
    
    # 系统常量定义
    K = 3
    NUM_APS = 16
    ANTENNAS_PER_AP = 4 
    NOISE_VAR = 1e-13
    TOTAL_SAMPLES = 20000  
    BATCH_SIZE = 1024
    RATE_NORM, SENSE_NORM = 8.0, 1024

    # ------------------------------------------
    # 数据准备 
    # ------------------------------------------
    print(f" 正在生成 {TOTAL_SAMPLES} 个固定样本数据集...")
    H_all = generate_cell_free_channel(
        batch_size=TOTAL_SAMPLES, num_users=K, num_aps=NUM_APS, 
        antennas_per_ap=ANTENNAS_PER_AP, device='cpu'
    )
    a_all = generate_cell_free_steering_vector(
        batch_size=TOTAL_SAMPLES, num_aps=NUM_APS, 
        antennas_per_ap=ANTENNAS_PER_AP, device='cpu'
    )

    H_train_np, H_val_np, a_train_np, a_val_np = train_test_split(
        H_all.numpy(), a_all.numpy(), test_size=0.2, random_state=42
    )

    train_dataset = TensorDataset(torch.tensor(H_train_np), torch.tensor(a_train_np))
    val_dataset = TensorDataset(torch.tensor(H_val_np), torch.tensor(a_val_np))
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # ------------------------------------------
    # 初始化 Lightning 系统与回调
    # ------------------------------------------
    system = ISAC_System(args, NUM_APS, ANTENNAS_PER_AP, NOISE_VAR, RATE_NORM, SENSE_NORM)
    
    wandb_logger = WandbLogger(project="ISAC_CellFree_GNN", name=f"{args.algo}_alpha_{args.alpha}")
    
    # 自动保存验证集 Loss 最低的那个模型权重
    checkpoint_callback = ModelCheckpoint(
        dirpath=f"./results/{args.algo}/alpha_{args.alpha}/",
        filename="best-model-{epoch:02d}-{Loss/Validation:.4f}",
        monitor="Loss/Validation", 
        mode="min",
        save_top_k=1
    )
    lr_monitor = LearningRateMonitor(logging_interval='epoch')

    # ------------------------------------------
    # 启动 Trainer
    # ------------------------------------------
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        max_epochs=args.epochs if 'only' not in args.algo else 1, 
        accelerator=accelerator, 
        devices="auto",           
        logger=wandb_logger,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=1.0,    
    )
    
    # 开始训练！
    trainer.fit(system, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    train_isac_model()
