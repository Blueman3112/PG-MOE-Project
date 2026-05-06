import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager

# 设置中文字体
font_path = "/hy-tmp/PG-MOE-Project/fonts/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
font_prop = font_manager.FontProperties(fname=font_path)

script_dir = "/hy-tmp/PG-MOE-Project/LastResults/Expr3 V2"
csv_file = os.path.join(script_dir, 'Validation_Metrics.csv')
output_path = os.path.join(script_dir, 'Validation_AUC_Only.png')

df = pd.read_csv(csv_file)
epochs = df['Epoch'].values

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

# 定义要绘制的数据列、图例名、颜色和点型
methods = [
    ('add_AUC', 'add (等权相加)', '#F2A021', 'o'), 
    ('concat_AUC', 'concat (特征拼接)', '#2DC86E', 's'),
    ('standardPG-MoE_AUC', 'PG-MoE (动态路由)', 'blue', '^')
]

all_vals = []
for col, label, color, marker in methods:
    y = df[col].values
    all_vals.extend(y)
    ax.plot(epochs, y, marker=marker, color=color, label=label, markersize=6, linewidth=1.5)

# 根据数据动态调整Y轴返回
min_val = max(0, min(all_vals) - 0.05)
max_val = min(1.0, max(all_vals) + 0.02)
ax.set_ylim(min_val, max_val)

ax.set_xlim(0.5, 20.5)
ax.set_xticks(range(2, 21, 2))

# 设置文本、网格与边框
ax.set_xlabel("Epoch", fontproperties=font_prop, fontsize=14)
ax.set_ylabel("Validation AUC", fontproperties=font_prop, fontsize=14)
ax.set_title("三种融合方式的 Validation AUC 曲线对比", fontproperties=font_prop, fontsize=18, pad=15)

ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# 调整图例
ax.legend(prop=font_prop, loc='lower right')

plt.tight_layout()
plt.savefig(output_path)
print(f"单独的 AUC 对比曲线图已成功保存到: {output_path}")

