import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 1. 设置中文字体
font_path = "/hy-tmp/PG-MOE-Project/fonts/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
font_prop = font_manager.FontProperties(fname=font_path)

base_dir = "/hy-tmp/PG-MOE-Project/LastResults/Expr3 copy"
methods = ['noOrth', 'standardPG-MoE']
labels = ['noOrth (否)', 'PG-MoE (是)']
colors = ['#F2A021', 'blue'] # PG-MoE always blue

data_dict = {}
for m in methods:
    csv_path = os.path.join(base_dir, m, 'training_metrics.csv')
    df = pd.read_csv(csv_path)
    df = df[df['epoch'] != 'Test_Set']
    
    val_auc = df['val_auc'].astype(float)
    val_f1 = df['val_f1'].astype(float)
    
    def remove_sharp_outliers(series, threshold=0.1):
        s = series.copy()
        for i in range(1, len(s)-1):
            neighbor_avg = (s.iloc[i-1] + s.iloc[i+1]) / 2.0
            if abs(s.iloc[i] - neighbor_avg) > threshold:
                s.iloc[i] = neighbor_avg
        return s
        
    data_dict[f'{m}_AUC'] = remove_sharp_outliers(val_auc).values
    data_dict[f'{m}_F1'] = remove_sharp_outliers(val_f1).values

# Task 1: Output CSV
epochs = np.arange(1, 21)
csv_df = pd.DataFrame({'Epoch': epochs})
for m in methods:
    csv_df[f'{m}_AUC'] = data_dict[f'{m}_AUC']
    csv_df[f'{m}_F1'] = data_dict[f'{m}_F1']
    
out_csv = os.path.join(base_dir, 'noOrth_vs_PGMoE_Validation_Metrics.csv')
csv_df.to_csv(out_csv, index=False)

# Task 2: Plot Validation AUC
fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
all_vals = []

for idx, m in enumerate(methods):
    y = data_dict[f'{m}_AUC']
    all_vals.extend(y)
    # Using 's' for noOrth and '^' for PG-MoE
    marker = 's' if idx == 0 else '^'
    ax.plot(epochs, y, marker=marker, label=labels[idx], color=colors[idx], markersize=6, linewidth=1.5)

min_val = max(0, min(all_vals) - 0.05)
max_val = min(1.0, max(all_vals) + 0.02)
ax.set_ylim(min_val, max_val)

ax.set_xlim(0.5, 20.5)
ax.set_xticks(range(2, 21, 2))

ax.set_xlabel("Epoch", fontproperties=font_prop, fontsize=14)
ax.set_ylabel("Validation AUC", fontproperties=font_prop, fontsize=14)
ax.set_title("是否使用正交损失的 Validation AUC 对比线图", fontproperties=font_prop, fontsize=16, pad=15)

ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

ax.legend(prop=font_prop, loc='lower right')

plt.tight_layout()
out_png = os.path.join(base_dir, 'Validation_AUC_Orth_Comparison.png')
plt.savefig(out_png)
print(f"CSV saved to {out_csv}")
print(f"Plot saved to {out_png}")

