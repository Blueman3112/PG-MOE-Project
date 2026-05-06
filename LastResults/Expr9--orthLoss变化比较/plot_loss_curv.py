import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 1. 设置中文字体
font_path = "/hy-tmp/PG-MOE-Project/fonts/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
font_prop = font_manager.FontProperties(fname=font_path)

# 获取脚本所在目录作为工作目录
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, '正交损失.csv')
output_path = os.path.join(script_dir, '图5-6_loss_curve.png')

# 2. 读取数据
df = pd.read_csv(csv_file, index_col=0)

# X轴位Epoch
epochs = np.arange(1, 21)

# 获取指定的行数据，由于 withOrth 下的 orth loss 极低（含0.0000），为防止对数坐标轴报错，应用 np.maximum 保底边界 
train_loss_with = df.loc['train_loss_withOrth'].values.astype(float)
train_loss_no = df.loc['train_loss_noOrth'].values.astype(float)
orth_loss_with = np.maximum(df.loc['train_orth_withOrth'].values.astype(float), 1e-5) # 保底 1e-5 避免与横轴完全重合并支持对数显示
orth_loss_no = df.loc['train_orth_noOrth'].values.astype(float)

# 3. 颜色及绘图参数设置
fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# PG-MoE (withOrth) 采用蓝色，noOrth采用红色/橙色，Total Loss使用实线，Orth Loss使用虚线
color_with = 'blue'
color_no = '#E44D35' 

# 绘制线图
ax.plot(epochs, train_loss_with, label='Total Loss (PG-MoE: withOrth)', color=color_with, linestyle='-', marker='o', markersize=5)
ax.plot(epochs, orth_loss_with, label='Orth Loss (PG-MoE: withOrth)', color=color_with, linestyle='--', marker='s', markersize=5)

ax.plot(epochs, train_loss_no, label='Total Loss (noOrth)', color=color_no, linestyle='-', marker='o', markersize=5)
ax.plot(epochs, orth_loss_no, label='Orth Loss (noOrth)', color=color_no, linestyle='--', marker='s', markersize=5)

# 4. 为了让差异达上千倍的值（0.26 vs 0.0001）都能清晰展现，并且 withOrth 的 orth 不会沉底，使用对数坐标系 (Log Scale)
ax.set_yscale('log')

# 5. 设置坐标限和刻度
ax.set_xticks(epochs)
ax.set_xlim(0.5, 20.5)

# 6. 设置文本、Title等
ax.set_xlabel("Epochs", fontproperties=font_prop, fontsize=14)
ax.set_ylabel("Loss Value (Log Scale)", fontproperties=font_prop, fontsize=14)
ax.set_title("训练过程中的 Loss与正交Loss 变化对比", fontproperties=font_prop, fontsize=18, pad=20)

# 7. 全局网格线与边框设定
ax.grid(axis='both', linestyle='--', alpha=0.5, color='gray', zorder=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 8. 调整图例
legend = ax.legend(prop=font_prop, loc='center right', bbox_to_anchor=(1.05, 0.5), framealpha=1.0)

plt.tight_layout()
plt.savefig(output_path)
print(f"对比折线图已成功保存到: {output_path}")
