import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import os

font_path = '/hy-tmp/PG-MOE-Project/fonts/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf'
if not os.path.exists(font_path):
    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc' 
my_font = fm.FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False

# 从您提供的 5 份 train_info.txt 提取出的数据
runs = ['Run-1\n(03-17)', 'Run-2\n(05-05)', 'Run-3\n(05-05)', 'Run-4\n(05-05)', 'Run-5\n(05-05)']
auc_scores = [0.9992, 0.9991, 0.9985, 0.9991, 0.9991]
f1_scores =  [0.9881, 0.9871, 0.9879, 0.9892, 0.9884]

x = np.arange(len(runs))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 6), dpi=300)

rects1 = ax.bar(x - width/2, auc_scores, width, label='Test AUC', color='#1f77b4', edgecolor='black', alpha=0.85)
rects2 = ax.bar(x + width/2, f1_scores, width, label='Test F1', color='#ff7f0e', edgecolor='black', alpha=0.85)

ax.set_ylabel('评估指标 (Score)', fontproperties=my_font, fontsize=14)
# 按之前约定的风格不包含“图号”
ax.set_title('PG-MoE 多次实验稳定性对比', fontproperties=my_font, fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(runs, fontproperties=my_font, fontsize=13)

# 纵轴不从0开始，以放大细节（用户建议 0.95~1.00，考虑到数值皆在0.98以上，用0.97~1.005观察更明显）
ax.set_ylim(0.97, 1.005)

ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(prop=my_font, loc='upper right', fontsize=12)

# 添加数值标签，展示精度
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight='bold')

autolabel(rects1)
autolabel(rects2)

ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
save_path = 'results/Fig5-1_Stability.png'
plt.savefig(save_path, bbox_inches='tight')
print(f'Done! Saved Figure 5-1 to {save_path}')
