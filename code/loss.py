# code/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class OrthogonalLoss(nn.Module):
    """
    自定义的总损失函数，包含 BCE Loss 和 Orthogonal Loss。
    L_total = L_BCE + λ * L_orth
    """
    def __init__(self, lambda_orth=0.1):
        super().__init__()
        self.lambda_orth = lambda_orth
        # BCEWithLogitsLoss 结合了 Sigmoid 和 BCE Loss，数值上更稳定
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, outputs, labels):
        # 从模型输出的字典中获取所需的值
        logits, F_s, F_f = outputs["logits"], outputs["F_s"], outputs["F_f"]
        
        # 1. 计算标准的二元交叉熵损失 (BCE Loss)
        # 将标签转为 float 类型并匹配 logits 的形状 [B, 1]
        labels = labels.float().unsqueeze(1)
        l_bce = self.bce_loss(logits, labels)
        
        # 2. 计算正交损失 (Orthogonal Loss)
        # 首先对特征向量进行 L2 归一化
        F_s_norm = F.normalize(F_s, p=2, dim=1)
        F_f_norm = F.normalize(F_f, p=2, dim=1)
        
        # 计算归一化后向量的点积，即余弦相似度
        cosine_sim = (F_s_norm * F_f_norm).sum(dim=1)
        
        # 正交损失是余弦相似度平方的均值
        l_orth = torch.mean(cosine_sim ** 2)
        
        # 3. 合并总损失
        total_loss = l_bce + self.lambda_orth * l_orth
        
        return total_loss, l_bce, l_orth