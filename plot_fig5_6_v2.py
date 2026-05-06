import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from sklearn.metrics import roc_auc_score
import os

font_path = '/hy-tmp/PG-MOE-Project/fonts/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf'
if not os.path.exists(font_path):
    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc' 
my_font = fm.FontProperties(fname=font_path)
plt.rcParams['axes.unicode_minus'] = False

lambdas = [0.0, 0.01, 0.05, 0.1, 0.2]
files = [
    'results/dataset-B_test_report_Orth-0.csv',
    'results/dataset-B_test_report_Orth-0.01.csv',
    'results/dataset-B_test_report_Orth-0.05.csv',
    'results/dataset-B_test_report_Orth-0.1.csv',
    'results/dataset-B_test_report_Orth-0.2.csv'
]

aucs = []
similarities = []

for lam, f in zip(lambdas, files):
    if not os.path.exists(f):
        print(f"Warning: {f} not found!")
        continue
    df = pd.read_csv(f)
    auc = roc_auc_score(df['gt_label'], df['pred_prob'])
    aucs.append(auc)
    mean_sim = df['cosine_similarity'].mean()
    similarities.append(mean_sim)

print(f"Lambdas: {lambdas}")
print(f"AUCs: {aucs}")
print(f"Similarities: {similarities}")

fig, ax1 = plt.subplots(figsize=(8, 6), dpi=300)

color1 = '#1f77b4' # blue
ax1.set_xlabel(r'正交损失权重 ($\lambda_{orth}$)', fontproperties=my_font, fontsize=14)
ax1.set_ylabel('AUC', color=color1, fontproperties=my_font, fontsize=14)
line1 = ax1.plot(lambdas, aucs, marker='o', markersize=8, color=color1, linewidth=2.5, label='AUC')
ax1.tick_params(axis='y', labelcolor=color1, labelsize=12)
ax1.tick_params(axis='x', labelsize=12)
ax1.set_xticks(lambdas)
ax1.grid(True, linestyle='--', alpha=0.6)

ax2 = ax1.twinx()  
color2 = '#ff7f0e' # orange
ax2.set_ylabel('特征余弦相似度均值', color=color2, fontproperties=my_font, fontsize=14)
line2 = ax2.plot(lambdas, similarities, marker='s', markersize=8, color=color2, linewidth=2.5, label='特征余弦相似度')
ax2.tick_params(axis='y', labelcolor=color2, labelsize=12)

lines = line1 + line2
labels = [l.get_label() for l in lines]
ax1.legend(lines, labels, prop=my_font, loc='center right', fontsize=12)

plt.title('不同正交损失权重下性能与特征相似度的变化', fontproperties=my_font, fontsize=16)

plt.tight_layout()
save_path = 'results/Fig5-6_V2.png'
plt.savefig(save_path, bbox_inches='tight')
print(f"Done! Saved Figure 5-6 to {save_path}")
