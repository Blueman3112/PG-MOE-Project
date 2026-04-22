import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SpatialAdapter, FrequencyAdapter, GatingRouter

class PGMoEFast(nn.Module):
    """
    为了极速实验重构的 PG-MoE：
    移除内部的 CLIP 骨干，改为直接接收预先提取好的 class_token 和 patch_tokens。
    """
    def __init__(self, internal_clip_dim=1024, output_clip_dim=768, fusion_type='moe'):
        super().__init__()
        
        self.fusion_type = fusion_type 
        
        # 专家初始化
        self.spatial_expert = SpatialAdapter(input_dim=internal_clip_dim, output_dim=512)
        self.frequency_expert = FrequencyAdapter(input_dim=internal_clip_dim, output_dim=512)
        
        if self.fusion_type == 'moe':
            # 路由初始化
            self.router = GatingRouter(input_dim=output_clip_dim, num_experts=2)
            # MoE 加权后的特征维度还是 512
            self.classifier = nn.Linear(512, 1)
        elif self.fusion_type == 'concat':
            # 如果是 Concat 对比实验，分类器接收 512 + 512 = 1024 取代 router 加权
            self.classifier = nn.Linear(1024, 1)
        else:
            raise ValueError("fusion_type 必须是 'moe' 或 'concat'")

    def forward(self, patch_tokens, class_token):
        # 预先提取的特征无需经过 CLIP
        # patch_tokens shape: [B, 257, 1024]
        # class_token shape: [B, 768]
        
        F_s = self.spatial_expert(patch_tokens)
        F_f = self.frequency_expert(patch_tokens)
        
        if self.fusion_type == 'moe':
            weights = self.router(class_token)
            w_s, w_f = weights[:, 0].unsqueeze(1), weights[:, 1].unsqueeze(1)
            
            fused_feature = w_s * F_s + w_f * F_f
            logits = self.classifier(fused_feature)
            
            return {
                "logits": logits,
                "F_s": F_s,
                "F_f": F_f,
                "w_s": weights[:, 0],
                "w_f": weights[:, 1]
            }
            
        elif self.fusion_type == 'concat':
            # 简单特征拼接进行对比实验 (Exp 1.1)
            fused_feature = torch.cat([F_s, F_f], dim=1)
            logits = self.classifier(fused_feature)
            
            return {
                "logits": logits,
                "F_s": F_s,
                "F_f": F_f
            }

