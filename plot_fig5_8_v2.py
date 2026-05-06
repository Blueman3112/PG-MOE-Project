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

# 我们挑选四个典型的样本类型：
# 1. 频域主导的伪造样本 (w_f > 0.8)
fake_freq = df[(df['gt_label'] == 0) & (df['w_f'] > 0.8)].iloc[0]
# 2. 均衡特性的伪造样本 (w_f 接近 0.5)
fake_bal = df[(df['gt_label'] == 0) & (df['w_f'] > 0.45) & (df['w_f'] < 0.55)].iloc[0]
# 3. 空域主导的伪造样本 (w_s > 0.8)
fake_spa = df[(df['gt_label'] == 0) & (df['w_s'] > 0.8)].iloc[0]
# 4. 典型的真实图像 (w_s > 0.9)
real_spa = df[(df['gt_label'] == 1) & (df['w_s'] > 0.9)].iloc[0]

categories = [
    '频域主导伪造样本\n(Freq-dominant Fake)',
    '混合特征伪造样本\n(Balanced Fake)',
    '空域主导伪造样本\n(Spatial-dominant Fake)',
    '典型真实图像\n(Typical Real)'
]

w_s_values = [fake_freq['w_s'], fake_bal['w_s'], fake_spa['w_s'], real_spa['w_s']]
w_f_values = [fake_freq['w_f'], fake_bal['w_f'], fake_spa['w_f'], real_spa['w_f']]

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

y = np.arange(len(categories))
height = 0.5

# 绘制堆叠水平条形图
bars1 = ax.barh(y, w_s_values, height, label='空间专家权重 (w_s)', color='#1f77b4', edgecolor='black', alpha=0.85)
bars2 = ax.barh(y, w_f_values, height, left=w_s_values, label='频域专家权重 (w_f)', color='#ff7f0e', edgecolor='black', alpha=0.85)

ax.set_xlabel('路由权重分配比例', fontproperties=my_font, fontsize=14)
ax.set_title('典型样本的路由结果分析', fontproperties=my_font, fontsize=16)
ax.set_yticks(y)
ax.set_yticklabels(categories, fontproperties=my_font, fontsize=12)
ax.set_xlim(0, 1.0)

ax.legend(prop=my_font, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2, fontsize=12)

# 添加文本标注
for i, (ws, wf) in enumerate(zip(w_s_values, w_f_values)):
    if ws > 0.05:
        ax.text(ws / 2, i, f'{ws:.2f}', va='center', ha='center', color='white', fontweight='bold', fontsize=11)
    if wf > 0.05:
        ax.text(ws + wf / 2, i, f'{wf:.2f}', va='center', ha='center', color='white', fontweight='bold', fontsize=11)

plt.tight_layout()
save_path = 'results/Fig5-8_V2.png'
plt.savefig(save_path, bbox_inches='tight')
print(f'Done! Saved Figure 5-8 to {save_path}')
