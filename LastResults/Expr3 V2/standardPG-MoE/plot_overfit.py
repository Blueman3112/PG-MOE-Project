import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 1. 设置中文字体
font_path = "/hy-tmp/PG-MOE-Project/fonts/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
font_prop = font_manager.FontProperties(fname=font_path)

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, 'training_metrics.csv')
output_path = os.path.join(script_dir, 'Overfit_Check_Curves.png')

df = pd.read_csv(csv_file)
df = df[df['epoch'] != 'Test_Set']

epochs = df['epoch'].astype(int).values
train_loss = df['train_loss'].astype(float).values
val_loss = df['val_loss'].astype(float).values
val_auc = df['val_auc'].astype(float).values

fig, ax1 = plt.subplots(figsize=(10, 6), dpi=300)

color_train = 'tab:blue'
color_val = 'tab:red'

# Plot losses
ax1.plot(epochs, train_loss, marker='o', color=color_train, label='Train Loss (训练集损失)')
ax1.plot(epochs, val_loss, marker='s', color=color_val, label='Validation Loss (验证集损失)')

ax1.set_xlabel('Epoch', fontproperties=font_prop, fontsize=14)
ax1.set_ylabel('Loss (损失)', fontproperties=font_prop, fontsize=14)
ax1.set_xlim(0.5, 20.5)
ax1.set_xticks(range(2, 21, 2))
ax1.grid(axis='both', linestyle='--', alpha=0.5, color='gray')

# Plot AUC on secondary Y axis to show performance growth
ax2 = ax1.twinx()
color_auc = 'tab:green'
ax2.plot(epochs, val_auc, marker='^', color=color_auc, linestyle='--', label='Validation AUC')
ax2.set_ylabel('Validation AUC', fontproperties=font_prop, fontsize=14, color=color_auc)
ax2.tick_params(axis='y', labelcolor=color_auc)

ax1.set_title("PG-MoE 训练与验证性能曲线 (过拟合验证)", fontproperties=font_prop, fontsize=18, pad=20)

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='center right', prop=font_prop)

plt.tight_layout()
plt.savefig(output_path)
print(f"图表已成功保存到: {output_path}")
