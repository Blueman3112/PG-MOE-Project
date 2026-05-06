import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from scipy.stats import gaussian_kde
import os

font_path = '/hy-tmp/PG-MOE-Project/fonts/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf'
if not os.path.exists(font_path):
    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc' 
my_font = fm.FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False

file_pg_moe = 'results/dataset-B_test_report.csv'
file_no_orth = 'results/dataset-B_test_report_noOrth.csv'

df_pg_moe = pd.read_csv(file_pg_moe)
df_no_orth = pd.read_csv(file_no_orth)

fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
x_eval = np.linspace(-1, 1, 500)

kde_pg = gaussian_kde(df_pg_moe['cosine_similarity'].dropna())
y_pg = kde_pg(x_eval)
ax.plot(x_eval, y_pg, color='#1f77b4', linewidth=2.5, label='PG-MoE (带正交约束)')
ax.fill_between(x_eval, y_pg, color='#1f77b4', alpha=0.3)

kde_no_orth = gaussian_kde(df_no_orth['cosine_similarity'].dropna())
y_no_orth = kde_no_orth(x_eval)
ax.plot(x_eval, y_no_orth, color='#ff7f0e', linewidth=2.5, label='PG-MoE (无正交约束)')
ax.fill_between(x_eval, y_no_orth, color='#ff7f0e', alpha=0.3)

ax.set_xlabel('专家特征余弦相似度 (Cosine Similarity)', fontproperties=my_font, fontsize=14)
ax.set_ylabel('样本密度 (Density)', fontproperties=my_font, fontsize=14)
ax.set_title('不同模型下的专家特征余弦相似度分布', fontproperties=my_font, fontsize=16)

ax.tick_params(axis='both', which='major', labelsize=12)
ax.legend(prop=my_font, loc='upper right', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.6)
ax.set_xlim([-1, 1])
ax.set_ylim(bottom=0)

plt.tight_layout()
save_path = 'results/Fig5-5_V2.png'
plt.savefig(save_path, bbox_inches='tight')
print(f'Done! Saved Figure 5-5 V2 to {save_path}')
