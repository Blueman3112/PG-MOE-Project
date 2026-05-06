import matplotlib.pyplot as plt
import numpy as np
import matplotlib.font_manager as fm
import os

font_path = '/hy-tmp/PG-MOE-Project/fonts/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf'
if not os.path.exists(font_path):
    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc' 
my_font = fm.FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False

# 据5份 train_info.txt 提取数据
runs = ['Run-1', 'Run-2', 'Run-3', 'Run-4', 'Run-5']
auc_scores = [0.9992, 0.9991, 0.9985, 0.9991, 0.9991]
acc_scores = [0.9879, 0.9869, 0.9877, 0.9890, 0.9882]
f1_scores =  [0.9881, 0.9871, 0.9879, 0.9892, 0.9884]

x = np.arange(len(runs))
width = 0.25

fig, ax = plt.subplots(figsize=(10, 6), dpi=300)

color_auc = '#1f77b4'
color_acc = '#2ca02c'
color_f1 = '#ff7f0e'

rects1 = ax.bar(x - width, auc_scores, width, label='Test AUC', color=color_auc, edgecolor='black', alpha=0.85)
rects2 = ax.bar(x, acc_scores, width, label='Test Accuracy', color=color_acc, edgecolor='black', alpha=0.85)
rects3 = ax.bar(x + width, f1_scores, width, label='Test F1', color=color_f1, edgecolor='black', alpha=0.85)

# 画平均值水平线
mean_auc = np.mean(auc_scores)
mean_acc = np.mean(acc_scores)
mean_f1 = np.mean(f1_scores)

ax.axhline(mean_auc, color=color_auc, linestyle='--', linewidth=1.5, zorder=0, label=f'Avg AUC ({mean_auc:.4f})')
ax.axhline(mean_acc, color=color_acc, linestyle='--', linewidth=1.5, zorder=0, label=f'Avg Acc ({mean_acc:.4f})')
ax.axhline(mean_f1, color=color_f1, linestyle='--', linewidth=1.5, zorder=0, label=f'Avg F1 ({mean_f1:.4f})')

ax.set_ylabel('评估指标 (Score)', fontproperties=my_font, fontsize=14)
ax.set_title('PG-MoE稳定性分析', fontproperties=my_font, fontsize=16)
ax.set_xticks(x)
ax.set_xticklabels(runs, fontproperties=my_font, fontsize=13)

# 调整Y轴范围，使得数据差异清晰且水平带不重叠
ax.set_ylim(0.980, 1.006)

ax.tick_params(axis='both', which='major', labelsize=12)

# 把图例放置于外部上方，以免柱子被遮挡
ax.legend(prop=my_font, loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=3, fontsize=11, frameon=False)

# 添加数值标签
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 4),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=10, fontweight='bold', rotation=90)

autolabel(rects1)
autolabel(rects2)
autolabel(rects3)

ax.grid(axis='y', linestyle=':', alpha=0.6)

plt.tight_layout()
save_path = 'results/Fig5-1_Stability_v2.png'
plt.savefig(save_path, bbox_inches='tight')
print(f'Done! Saved updated Figure 5-1 to {save_path}')
