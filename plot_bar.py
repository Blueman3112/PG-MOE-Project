import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

# 设置中文字体
font_path = "fonts/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf"
font_prop = font_manager.FontProperties(fname=font_path)

# 读取数据
csv_file = 'LastResults/Expr1/图5-2fixed.csv'
df = pd.read_csv(csv_file, index_col=0)

# 只提取 AUC、Acc、F1 这三个指标
metrics = ['AUC', 'Acc', 'F1']
df_subset = df[metrics]

# 提供绘图数据
methods = df_subset.index.tolist()
y = np.arange(len(methods))  # 纵坐标位置
height = 0.25  # 柱子的高度

fig, ax = plt.subplots(figsize=(10, 6))

# 绘制横向柱状图
for i, metric in enumerate(metrics):
    # 计算每个指标柱子的偏移量
    offset = height * (i - 1)  # 让三个柱子居中对齐
    bars = ax.barh(y + offset, df_subset[metric], height, label=metric)
    
    # 为每个柱子添加数值标签
    ax.bar_label(bars, padding=3, fmt='%.4f', fontproperties=font_prop)

# 设置轴、标题和字体
ax.set_yticks(y)
ax.set_yticklabels(methods, fontproperties=font_prop, fontsize=12)
ax.set_xlabel('Score (数值)', fontproperties=font_prop, fontsize=12)
ax.set_ylabel('Method (方法)', fontproperties=font_prop, fontsize=12)
ax.set_title('不同方法在Test集上的性能对比 (AUC/ACC/F1)', fontproperties=font_prop, fontsize=14)

# 限制x轴范围，让数值标签能够显示完整
ax.set_xlim(0, 1.15) 

# 设置图例
ax.legend(prop=font_prop, loc='lower right')

plt.tight_layout()
output_path = 'LastResults/Expr1/图5-2_bar.png'
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"图表已成功保存到: {output_path}")
