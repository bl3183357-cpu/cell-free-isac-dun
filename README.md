# Cell-Free ISAC Deep Unfolding Network (DUN)

本项目实现了一个面向**Cell-Free Integrated Sensing and Communication (ISAC)** 的深度展开网络（Deep Unfolding Network, DUN）框架，旨在联合优化通信速率与感知波束图增益（Sensing Beampattern Gain）。该代码库包含训练脚本、Pareto 曲线实验，以及相关模型与数据生成工具。

---

## ✅ 主要功能

- 基于 PyTorch 的无监督 ISAC 训练框架（通信 + 感知联合优化）
- 提供 **GNN 结构的深度展开网络**（`ISAC_GNN_UnfoldingNet`）和 **传统 DUN**（`ISAC_DeepUnfoldingNet`）
- 支持训练与验证曲线可视化（保存在 `training_curves_val.png`）
- 支持画出不同权衡系数 α 下的 **Pareto 前沿**（保存为 `pareto_frontier.png`）

---

## 🧩 目录结构（核心文件）

- `main.py`：训练主流程（生成数据、训练、验证、保存权重与曲线）
- `run_pareto.py`：针对不同 α 权衡值跑 Pareto 实验，并生成 Pareto 前沿图
- `models/`：包含网络结构实现
  - `GNN_Unfolding.py`：基于图神经网络的展开模块
  - `unfolding.py`：传统深度展开网络实现
  - `mlp.py`：用于对比的 MLP 基线
- `utils/`：工具函数
  - `channel_gen.py`：射线衰落通道与方向矢量生成
  - `loss_fn.py`：ISAC 目标函数实现

---

## 🧪 环境依赖

> ```bash
> pixi install
> ```

---

## ▶️ 快速开始

### 1) 训练模型

```bash
pixi run start
```

运行后，会生成如下文件：

- `isac_gnn_weights.pth`：训练得到的模型权重
- `training_curves_val.png`：训练/验证 Loss、通讯速率、感知增益曲线图

### 2) 生成 Pareto 前沿图

```bash
pixi run python run_pareto.py
```

运行结束后会生成：

- `pareto_frontier.png`：不同 α 下的通信速率 vs 感知增益贸易曲线

---

## 🔧 关键超参数（可在代码中修改）

- `M, N, K`：AP 数量、每个 AP 天线数、通信用户数
- `P_MAX`：每个 AP 的最大发射功率
- `ALPHA`：通信/感知损失加权系数（越大越偏向感知）
- `EPOCHS` / `BATCH_SIZE`：训练轮数与批大小

---

## 📌 注意事项

- 本项目默认在 GPU 可用时使用 CUDA，若没有 GPU 会自动退回到 CPU。
- 训练数据通过 `utils/channel_gen.py` 生成，是一个**固定的随机样本集**（可在脚本中修改 `TOTAL_SAMPLES`、`NUM_SAMPLES` 以扩展规模）。

---

