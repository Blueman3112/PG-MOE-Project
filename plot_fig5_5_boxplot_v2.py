import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

font_path = '/hy-tmp/PG-MOE-Project/fonts/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf'
if not os.path.exists(font_path):
    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc' 
my_font = fm.FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False

df_pg_moe = pd.read_csv('results/dataset-B_test_report.csv')
df_no_orth = pd.read_csv('results/dataset-B_test_report_noOrth.csv')

data = [
    df_pg_moe['cosine_similarity'].dropna(),
    df_no_orth['cosine_similarity'].dropna()
]

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

box = ax.boxplot(data, patch_artist=True, widths=0.4, showfliers=True, flierprops=dict(marker='o', markeredgecolor='gray', alpha=0.3, markersize=3))
colors = ['#1f77b4', '#ff7f0e']
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)
    patch.set_alpha(0.7)
    patch.set_edgecolor('black')

for median in box['medians']:
    median.set(color='black', linewidth=2)

ax.set_xticklabels(['PG-MoE\n(带正交约束)', 'PG-MoE\n(无正交约束)'], fontproperties=my_font, fontsize=14)
ax.set_ylabel('专家特征余弦相似度 (Cosine Similarity)', fontproperties=my_font, fontsize=14)
ax.set_title('不同模型下的专家特征余弦相似度分布', fontproperties=my_font, fontsize=16)
ax.tick_params(axis='y', labelsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.6)

plt.tight_layout()
save_path = 'results/Fig5-5_Boxplot_V2.png'
plt.savefig(save_path, bbox_inches='tight')
print(f'Done! Saved Boxplot to {save_path}')
