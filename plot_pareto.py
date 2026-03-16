import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

rate_file = 'wandb_export_2026-03-17T00_19_07.148+08_00.csv'
sense_file = 'wandb_export_2026-03-17T00_19_23.742+08_00.csv'

df_rate = pd.read_csv(rate_file)
df_sense = pd.read_csv(sense_file)

# 2. 定义要提取的 alpha 列表
alphas = [0.1, 0.3, 0.5, 0.6, 0.7, 0.8, 0.9]

pareto_data = []

# 3. 提取收敛值 (取最后 10 个 Step 的平均值以平滑震荡)
for alpha in alphas:
    rate_col = f'gnn_heuristic_alpha_{alpha} - Metrics/Sum_Rate'
    sense_col = f'gnn_heuristic_alpha_{alpha} - Metrics/Sensing_Power'
    
    if rate_col in df_rate.columns and sense_col in df_sense.columns:
        # 取最后 10 行求平均
        final_rate = df_rate[rate_col].tail(10).mean()
        final_sense = df_sense[sense_col].tail(10).mean()
        pareto_data.append({'alpha': alpha, 'Rate': final_rate, 'Sensing': final_sense})

# 转换为 DataFrame 并按 Sensing Power 排序，方便连线
df_pareto = pd.DataFrame(pareto_data).sort_values(by='Sensing')

# 4. 绘制高质量帕累托前沿图
plt.figure(figsize=(8, 6), dpi=150)

# 画线和点
plt.plot(df_pareto['Sensing'], df_pareto['Rate'], marker='o', linestyle='-', 
         linewidth=2, markersize=8, color='#d62728', label='GNN-based ISAC')

# 为每个点标注 alpha 值
for i, row in df_pareto.iterrows():
    plt.annotate(f"$\\alpha$={row['alpha']}", 
                 (row['Sensing'], row['Rate']), 
                 textcoords="offset points", 
                 xytext=(10, -10), 
                 ha='center', fontsize=10)

# 图表美化 (IEEE 风格)
plt.title('Pareto Front: Sum-Rate vs Sensing Power', fontsize=14, fontweight='bold')
plt.xlabel('Sensing Power', fontsize=12)
plt.ylabel('Sum-Rate (bps/Hz)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(loc='lower left', fontsize=12)

plt.tight_layout()
plt.savefig('pareto_front.png', dpi=300)
plt.show()

print("✅ 帕累托前沿数据点：")
print(df_pareto.to_string(index=False))
