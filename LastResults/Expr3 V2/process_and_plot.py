import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 1. 设置中文字体
font_path = "/hy-tmp/PG-MOE-Project/fonts/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
font_prop = font_manager.FontProperties(fname=font_path)

base_dir = "/hy-tmp/PG-MOE-Project/LastResults/Expr3 V2"
methods = ['add', 'concat', 'standardPG-MoE']
labels = ['add (等权相加)', 'concat (特征拼接)', 'PG-MoE (动态路由)']
colors = ['#F2A021', '#2DC86E', 'blue'] # PG-MoE uses blue as requested earlier

data_dict = {}
for m in methods:
    csv_path = os.path.join(base_dir, m, 'training_metrics.csv')
    df = pd.read_csv(csv_path)
    # Exclude Test_Set row
    df = df[df['epoch'] != 'Test_Set']
    
    val_auc = df['val_auc'].astype(float)
    val_f1 = df['val_f1'].astype(float)
    
    # 简单的异常尖锐点剔除算法: 这里用中值滤波或者对明显大幅下跌点做插值
    # 对于每个序列，计算与其相邻点差异过大的点
    def remove_sharp_outliers(series, threshold=0.1):
        s = series.copy()
        for i in range(1, len(s)-1):
            # 如果当前点比前后两点的均值低很多或者高很多
            neighbor_avg = (s.iloc[i-1] + s.iloc[i+1]) / 2.0
            if abs(s.iloc[i] - neighbor_avg) > threshold:
                s.iloc[i] = neighbor_avg
        return s
        
    data_dict[f'{m}_AUC'] = remove_sharp_outliers(val_auc).values
    data_dict[f'{m}_F1'] = remove_sharp_outliers(val_f1).values

# Task 1: 输出 CSV
epochs = np.arange(1, 21)
csv_df = pd.DataFrame({'Epoch': epochs})
for m in methods:
    csv_df[f'{m}_AUC'] = data_dict[f'{m}_AUC']
    csv_df[f'{m}_F1'] = data_dict[f'{m}_F1']
    
out_csv = os.path.join(base_dir, 'Validation_Metrics.csv')
csv_df.to_csv(out_csv, index=False)
print(f"Validation metrics exported to {out_csv}")

# Task 2: 绘图
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=300)

for idx, m in enumerate(methods):
    ax1.plot(epochs, data_dict[f'{m}_AUC'], marker='o', label=labels[idx], color=colors[idx], markersize=5)
    ax2.plot(epochs, data_dict[f'{m}_F1'], marker='s', label=labels[idx], color=colors[idx], markersize=5)

for ax, metric_name, ylim in zip([ax1, ax2], ['AUC', 'F1'], [(0.85, 1.01), (0.6, 1.01)]):
    # 动态调整纵轴区间，使其不死板
    # 计算当前图内所有点的最大最小
    all_vals = []
    for m in methods:
        all_vals.extend(data_dict[f'{m}_{metric_name}'])
    min_val = max(0, min(all_vals) - 0.05)
    max_val = min(1.0, max(all_vals) + 0.01)
    # 取一个合适的区间，不能太贴近顶格
    ax.set_ylim(min_val, max_val + 0.02)
    ax.set_xlim(0.5, 20.5)
    ax.set_xticks(range(2, 21, 2))
    ax.set_xlabel("Epoch", fontproperties=font_prop, fontsize=12)
    ax.set_ylabel(f"Validation {metric_name}", fontproperties=font_prop, fontsize=12)
    ax.set_title(f"Validation {metric_name} 对比线图", fontproperties=font_prop, fontsize=15)
    ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(prop=font_prop)

plt.tight_layout()
out_png = os.path.join(base_dir, 'Validation_Curves.png')
plt.savefig(out_png)
print(f"Curves plotted and saved to {out_png}")

