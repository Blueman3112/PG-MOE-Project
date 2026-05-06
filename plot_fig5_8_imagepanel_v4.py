import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from PIL import Image
import os

font_path = '/hy-tmp/PG-MOE-Project/fonts/OTF/SimplifiedChinese/SourceHanSansSC-Regular.otf'
if not os.path.exists(font_path):
    font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc' 
my_font = fm.FontProperties(fname=font_path)

plt.rcParams['axes.unicode_minus'] = False

df = pd.read_csv('results/dataset-B_PGMoE_report_new.csv')

# 挑选4个典型样本
try:
    s1 = df[(df['gt_label'] == 0) & (df['sub_category'] == 'proGAN') & (df['w_f'] > 0.7)].iloc[0]
except:
    s1 = df[(df['gt_label'] == 0) & (df['sub_category'] == 'proGAN')].iloc[0]

try:
    s2 = df[(df['gt_label'] == 0) & (df['sub_category'] == 'SDV5') & (df['w_s'] > 0.7)].iloc[0]
except:
    s2 = df[(df['gt_label'] == 0) & (df['sub_category'] == 'SDV5')].iloc[0]

try:
    s3 = df[(df['gt_label'] == 0) & (df['w_f'] > 0.45) & (df['w_f'] < 0.55)].iloc[0]
except:
    s3 = df[(df['gt_label'] == 0)].iloc[0]

try:
    s4 = df[(df['gt_label'] == 1) & (df['w_s'] > 0.9)].iloc[0]
except:
    s4 = df[(df['gt_label'] == 1)].iloc[0]


samples = [s1, s2, s3, s4]
labels = ['频域主导伪造 (proGAN)', '空域主导伪造 (SDV5)', '混合特征伪造 (均衡)', '典型真实图像 (Real)']

# 画布设置
fig = plt.figure(figsize=(12, 8), dpi=300)
gs = fig.add_gridspec(4, 3, width_ratios=[1.2, 1.2, 2.5], wspace=0.1, hspace=0.4)

fig.suptitle('典型样本的路由结果分析', fontproperties=my_font, fontsize=18, y=0.95)

for i, (sample, desc) in enumerate(zip(samples, labels)):
    ax_img = fig.add_subplot(gs[i, 0])
    ax_txt = fig.add_subplot(gs[i, 1])
    ax_bar = fig.add_subplot(gs[i, 2])
    
    # 1. 缩略图
    img_path = sample['image_path']
    try:
        img = Image.open(img_path).convert('RGB')
        # Center crop
        w, h = img.size
        min_dim = min(w, h)
        left = (w - min_dim)/2
        top = (h - min_dim)/2
        right = (w + min_dim)/2
        bottom = (h + min_dim)/2
        img = img.crop((left, top, right, bottom))
        img = img.resize((224, 224))
        ax_img.imshow(img)
    except Exception as e:
        ax_img.text(0.5, 0.5, 'Image not found', ha='center', va='center')
        
    ax_img.axis('off')
    ax_img.set_title(desc, fontproperties=my_font, fontsize=12)
    
    # 2. 文本信息框
    gt = 'Real' if sample['gt_label'] == 1 else f"Fake ({sample['sub_category']})"
    pred = 'Real' if sample['pred_label'] == 1 else 'Fake'
    prob = sample['pred_prob']
    
    text_str = f"GT: {gt}\nPred: {pred}\nProb(fake): {prob:.3f}"
    ax_txt.axis('off')
    ax_txt.text(0.1, 0.5, text_str, va='center', ha='left', fontsize=12, fontfamily='monospace',
                bbox=dict(facecolor='#f8f9fa', edgecolor='#dee2e6', boxstyle='round,pad=0.5'))
    
    # 3. 水平条形图
    ws = sample['w_s']
    wf = sample['w_f']
    
    ax_bar.barh([0], [ws], height=0.4, color='#1f77b4', edgecolor='black', alpha=0.85)
    ax_bar.barh([0], [wf], left=[ws], height=0.4, color='#ff7f0e', edgecolor='black', alpha=0.85)
    
    ax_bar.set_xlim(0, 1)
    ax_bar.set_ylim(-0.4, 0.4)
    ax_bar.set_yticks([])
    if i < 3:
        ax_bar.set_xticks([])
        ax_bar.spines['bottom'].set_visible(False)
    else:
        ax_bar.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_bar.set_xlabel('路由权重分配', fontproperties=my_font, fontsize=12)
        
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    ax_bar.spines['left'].set_visible(False)
    
    # 图内文字，使用 LaTeX 公式渲染下划线为下标
    if ws > 0.05:
        ax_bar.text(ws/2, 0, f'$w_s$: {ws:.2f}', va='center', ha='center', color='white', fontweight='bold', fontsize=11)
    if wf > 0.05:
        ax_bar.text(ws + wf/2, 0, f'$w_f$: {wf:.2f}', va='center', ha='center', color='white', fontweight='bold', fontsize=11)

# 添加整体图例，同样渲染为下标
import matplotlib.patches as mpatches
legend_patches = [
    mpatches.Patch(color='#1f77b4', label=r'空间专家权重 ($w_s$)'),
    mpatches.Patch(color='#ff7f0e', label=r'频域专家权重 ($w_f$)')
]
fig.legend(handles=legend_patches, prop=my_font, loc='upper center', bbox_to_anchor=(0.5, 0.92), ncol=2, fontsize=12, frameon=False)

# 保存
save_path = 'results/Fig5-8_ImagePanel_V4.png'
plt.savefig(save_path, bbox_inches='tight')
print(f'Done! Saved corrected Figure 5-8 Panel to {save_path}')
