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

# 读取包含子类别的新CSV
df = pd.read_csv('results/dataset-B_PGMoE_report_new.csv')

# 对子类别名称进行映射，以在中英文结合显示时更好看
def map_category(cat):
    if cat == 'Real':
        return '真实图像\n(Real)'
    elif cat == 'proGAN':
        return 'GAN伪造\n(proGAN)'
    elif cat == 'SDV5':
        return '扩散伪造\n(SDV5)'
    return cat

df['DisplayCategory'] = df['sub_category'].apply(map_category)

# 为了让顺序固定（真实图像在前，或者伪造在前，我们选择：GAN -> 扩散 -> 真实排列）
order = ['GAN伪造\n(proGAN)', '扩散伪造\n(SDV5)', '真实图像\n(Real)']
df['DisplayCategory'] = pd.Categorical(df['DisplayCategory'], categories=order, ordered=True)

# 计算均值和标准差
grouped_mean = df.groupby('DisplayCategory')[['w_s', 'w_f']].mean()
grouped_std = df.groupby('DisplayCategory')[['w_s', 'w_f']].std()

print("Detailed Grouped Mean:\n", grouped_mean)

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

x = np.arange(len(order))
width = 0.35

rects1 = ax.bar(x - width/2, grouped_mean['w_s'], width, yerr=grouped_std['w_s'], label='空间专家权重 (w_s)', color='#1f77b4', capsize=5, edgecolor='black', alpha=0.85)
rects2 = ax.bar(x + width/2, grouped_mean['w_f'], width, yerr=grouped_std['w_f'], label='频域专家权重 (w_f)', color='#ff7f0e', capsize=5, edgecolor='black', alpha=0.85)

ax.set_ylabel('平均路由权重', fontproperties=my_font, fontsize=14)
ax.set_title('不同样本类型下的专家权重统计', fontproperties=my_font, fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(order, fontproperties=my_font, fontsize=14)

ax.tick_params(axis='y', labelsize=12)
ax.legend(prop=my_font, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=12)

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
save_path = 'results/Fig5-7_Detailed_V3.png'
plt.savefig(save_path, bbox_inches='tight')
print(f'Done! Saved detailed Figure 5-7 to {save_path}')
