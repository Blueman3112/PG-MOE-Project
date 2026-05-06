import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.colors as mcolors
from matplotlib import font_manager

# 1. 设置中文字体
font_path = "/hy-tmp/PG-MOE-Project/fonts/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
font_prop = font_manager.FontProperties(fname=font_path)

# 获取脚本所在目录作为工作目录，满足“绘图脚本与使用的csv文件同级目录”要求
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_file = os.path.join(script_dir, '表5-3.csv')
output_path = os.path.join(script_dir, '图5-3_grouped_bar.png')

# 2. 读取数据 (跳过缺失或无需的列)
df = pd.read_csv(csv_file, index_col=0)

# 只提取需要的指标并转换为百分制（乘100）
metrics = ['AUC', 'Acc', 'F1']
df_subset = df[metrics].copy() * 100

# 按照PG-MoE优先的展示顺序排列
method_order = ['PG-MoE', 'spatial_only', 'frequency_only']
df_subset = df_subset.reindex(method_order)

# 3. 颜色及绘图参数设置
colors = ['#E44D35', '#F2A021', '#2DC86E'] # 对应的柱子基础色：红, 橙, 绿
alphas = [1.0, 0.8, 0.7] # 对应柱体的透明度(制造褪色效果)

x = np.arange(len(method_order))
width = 0.22 # 单个柱子的宽度

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

# 4. 绘制分组柱状图
for i, metric in enumerate(metrics):
    vals = df_subset[metric]
    
    for j, method in enumerate(method_order):
        # 让中间的指标(Acc)居中，左侧(AUC)偏移 -width，右侧(F1)偏移 +width
        pos = x[j] + (i - 1) * width
        
        # 带有透明度的色块，以保持边框(edgecolor)的全黑不透明
        rgba_color = mcolors.to_rgba(colors[i], alpha=alphas[j])
        
        val = vals.iloc[j]
        # 跳过空值
        if not np.isnan(val):
            # 将zorder设置高些，保证覆盖在背景虚线网格上方
            bar = ax.bar(pos, val, width, color=rgba_color, edgecolor='black', linewidth=1.5, zorder=3)
            # 头顶数值标签
            ax.text(pos, val + 0.3, f"{val:.2f}", ha='center', va='bottom', 
                    fontsize=11, fontproperties=font_prop, zorder=4)

# 5. 为图例创建带黑色外框的1.0透明度占位方块，放入Legend（隐去位置）
for i, metric in enumerate(metrics):
    ax.bar(-10, 0, width, color=colors[i], edgecolor='black', linewidth=1.5, label=f"Test {metric}")

# 6. 设置坐标限和刻度
ax.set_xlim(-0.5, len(method_order) - 0.5)
ax.set_ylim(70, 105)
ax.set_yticks(np.arange(70, 106, 5))

# 7. 设置文本、Title等
ax.set_ylabel("Scores (%)", fontproperties=font_prop, fontsize=14)
ax.set_title("测试集性能对比 (Test Performance)", fontproperties=font_prop, fontsize=18, pad=20)

x_labels = [
    "PG-MoE\n(全专家)",
    "spatial_only\n(仅空间专家)",
    "frequency_only\n(仅频域专家)"
]
ax.set_xticks(x)
ax.set_xticklabels(x_labels, fontproperties=font_prop, fontsize=14)

# 定制X Label颜色，PG-MoE 蓝色，其它黑色
xtick_colors = ['blue', 'black', 'black']
for tick_label, color in zip(ax.get_xticklabels(), xtick_colors):
    tick_label.set_color(color)

# 8. 全局网格线与边框隐藏
ax.grid(axis='y', linestyle='--', alpha=0.5, color='gray', zorder=0)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.0)
ax.spines['bottom'].set_linewidth(1.0)

# 9. 调整图例
legend = ax.legend(title="Metrics", prop=font_prop, loc='upper right', framealpha=1.0)
legend.get_title().set_fontproperties(font_prop)
legend.get_title().set_fontsize(11)

plt.tight_layout()
plt.savefig(output_path)
print(f"图表已成功保存到: {output_path}")