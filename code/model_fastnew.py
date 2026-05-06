import torch
import torch.nn as nn
import torch.nn.functional as F

from model import SpatialAdapter, FrequencyAdapter, GatingRouter

class PGMoEFast(nn.Module):
    """
    为了极速实验重构的 PG-MoE (新版)：
    移除内部的 CLIP 骨干，改为直接接收预先提取好的 class_token 和 patch_tokens。
    支持的 fusion_type 扩展为: 'moe', 'concat', 'spatial_only', 'frequency_only', 'add'
    """
    def __init__(self, internal_clip_dim=1024, output_clip_dim=768, fusion_type='moe'):
        super().__init__()
        
        self.fusion_type = fusion_type 
        
        # 根据融合类型，按需初始化专家（严格控制参数量）
        if self.fusion_type in ['moe', 'concat', 'add', 'spatial_only']:
            self.spatial_expert = SpatialAdapter(input_dim=internal_clip_dim, output_dim=512)
            
        if self.fusion_type in ['moe', 'concat', 'add', 'frequency_only']:
            self.frequency_expert = FrequencyAdapter(input_dim=internal_clip_dim, output_dim=512)
        
        # 根据融合类型初始化分类器和路由
        if self.fusion_type == 'moe':
            self.router = GatingRouter(input_dim=output_clip_dim, num_experts=2)
            self.classifier = nn.Linear(512, 1)
        elif self.fusion_type == 'concat':
            self.classifier = nn.Linear(1024, 1)
        elif self.fusion_type == 'add':
            self.classifier = nn.Linear(512, 1)
        elif self.fusion_type == 'spatial_only' or self.fusion_type == 'frequency_only':
            self.classifier = nn.Linear(512, 1)
        else:
            raise ValueError(f"不支持的 fusion_type: {fusion_type}")

    def forward(self, patch_tokens, class_token):
        # 结果字典初始化
        result = {}
        
        # 1. 提取单模态特征
        if hasattr(self, 'spatial_expert'):
            F_s = self.spatial_expert(patch_tokens)
            result['F_s'] = F_s
        else:
            F_s = torch.zeros(patch_tokens.size(0), 512, device=patch_tokens.device)
            result['F_s'] = F_s
            
        if hasattr(self, 'frequency_expert'):
            F_f = self.frequency_expert(patch_tokens)
            result['F_f'] = F_f
        else:
            F_f = torch.zeros(patch_tokens.size(0), 512, device=patch_tokens.device)
            result['F_f'] = F_f
        
        # 2. 特征融合与分类
        if self.fusion_type == 'moe':
            weights = self.router(class_token)
            w_s, w_f = weights[:, 0].unsqueeze(1), weights[:, 1].unsqueeze(1)
            fused_feature = w_s * F_s + w_f * F_f
            logits = self.classifier(fused_feature)
            result['w_s'] = weights[:, 0]
            result['w_f'] = weights[:, 1]
            
        elif self.fusion_type == 'concat':
            fused_feature = torch.cat([F_s, F_f], dim=1)
            logits = self.classifier(fused_feature)
            
        elif self.fusion_type == 'add':
            fused_feature = F_s + F_f
            logits = self.classifier(fused_feature)
            
        elif self.fusion_type == 'spatial_only':
            logits = self.classifier(F_s)
            
        elif self.fusion_type == 'frequency_only':
            logits = self.classifier(F_f)
            
        result['logits'] = logits
        return result
