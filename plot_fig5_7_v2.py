import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

font_path = '/hy-tmp/PG-MOE-Project/fonts/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf'
if not os.path.exists(font_path):
    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc' 
my_font = fm.FontProperties(fname=font_path)

plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('results/dataset-B_test_report.csv')

# 按照真实/伪造类型统计
df['SampleType'] = df['gt_label'].apply(lambda x: '真实图像 (Real)' if x == 1 else '伪造图像 (Fake)')

# 计算均值和标准差
grouped_mean = df.groupby('SampleType')[['w_s', 'w_f']].mean()
grouped_std = df.groupby('SampleType')[['w_s', 'w_f']].std()

print("Grouped Mean:\n", grouped_mean)

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

x = np.arange(len(grouped_mean))
width = 0.35

rects1 = ax.bar(x - width/2, grouped_mean['w_s'], width, yerr=grouped_std['w_s'], label='空间专家权重 (w_s)', color='#1f77b4', capsize=5, edgecolor='black', alpha=0.85)
rects2 = ax.bar(x + width/2, grouped_mean['w_f'], width, yerr=grouped_std['w_f'], label='频域专家权重 (w_f)', color='#ff7f0e', capsize=5, edgecolor='black', alpha=0.85)

ax.set_ylabel('平均路由权重', fontproperties=my_font, fontsize=14)
ax.set_title('不同样本类型下的专家权重统计', fontproperties=my_font, fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(grouped_mean.index, fontproperties=my_font, fontsize=14)

ax.tick_params(axis='y', labelsize=12)
ax.legend(prop=my_font, loc='upper right', fontsize=12)

# 添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.2f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=12, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

ax.grid(axis='y', linestyle='--', alpha=0.6)
ax.set_ylim(0, 1.1)

plt.tight_layout()
save_path = 'results/Fig5-7_V2.png'
plt.savefig(save_path, bbox_inches='tight')
print(f'Done! Saved Figure 5-7 to {save_path}')
