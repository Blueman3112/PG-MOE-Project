# code/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import open_clip

class SpatialAdapter(nn.Module):
    """空间感知专家"""
    def __init__(self, input_dim=1024, output_dim=512):
        super().__init__()
        # 设计一个轻量级的 CNN 结构
        self.conv_block = nn.Sequential(
            # 输入通道为 CLIP 特征维度，输出通道自定义
            nn.Conv2d(input_dim, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, output_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            # 全局平均池化，将 HxW 的特征图变为 1x1
            nn.AdaptiveAvgPool2d((1, 1))
        )

    def forward(self, patch_tokens):
        # CLIP ViT-L/14 输出的 patch_tokens 形状: [batch_size, 257, 1024]
        # 第一个 token 是 [class] token，我们需要忽略它
        patches = patch_tokens[:, 1:, :]
        batch_size, num_patches, dim = patches.shape
        
        # 假设 patch 数量是 16x16=256
        h = w = int(num_patches**0.5)
        
        # 将序列化的 patch token 重塑为图像的 HxW 格式
        # [B, N, D] -> [B, D, N] -> [B, D, H, W]
        patches = patches.permute(0, 2, 1).reshape(batch_size, dim, h, w)
        
        feature = self.conv_block(patches)
        return feature.view(batch_size, -1) # 展平为 [batch_size, output_dim]

class FrequencyAdapter(nn.Module):
    """频域感知专家"""
    def __init__(self, input_dim=1024, num_heads=8, output_dim=512):
        super().__init__()
        # 使用一个标准的 Transformer Encoder 层
        self.transformer_block = nn.TransformerEncoderLayer(
            d_model=input_dim, nhead=num_heads, batch_first=True
        )
        self.fc = nn.Linear(input_dim, output_dim)

    def forward(self, all_tokens):
        # all_tokens 包含 [class] token 和 patch tokens
        # 形状: [batch_size, 257, 1024]
        
        # 沿序列维度进行快速傅里叶变换 (FFT)，并取实部
        freq_tokens = torch.fft.fft(all_tokens, dim=1).real
        
        # 通过 Transformer Block
        transformed_tokens = self.transformer_block(freq_tokens)
        
        # 使用变换后的 [class] token 作为全局频域特征的代表
        feature = self.fc(transformed_tokens[:, 0, :]) # 取第一个 token
        return feature # [batch_size, output_dim]

class GatingRouter(nn.Module):
    """门控路由，用于决定专家权重"""
    def __init__(self, input_dim=1024, num_experts=2):
        super().__init__()
        self.layer = nn.Linear(input_dim, num_experts)

    def forward(self, class_token):
        # 使用 CLIP 输出的 [class] token 来动态计算专家权重
        return F.softmax(self.layer(class_token), dim=-1)

class PGMoE(nn.Module):
    """PG-MoE 完整模型架构"""
    def __init__(self, model_name='ViT-L-14', pretrained='laion2b_s32b_b82k'):
        super().__init__()
        
        # 1. 加载并冻结 CLIP 骨干网络
        print("正在加载 CLIP 模型...")
        self.clip, _, _ = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
        for param in self.clip.parameters():
            param.requires_grad = False # 冻结参数
        print("CLIP 模型加载并冻结完毕。")

        # 获取 CLIP 输出的特征维度
        clip_dim = self.clip.visual.ln_post.normalized_shape[0]

        # 2. 初始化双流物理专家
        self.spatial_expert = SpatialAdapter(input_dim=clip_dim, output_dim=512)
        self.frequency_expert = FrequencyAdapter(input_dim=clip_dim, output_dim=512)
        
        # 3. 初始化门控路由
        self.router = GatingRouter(input_dim=clip_dim, num_experts=2)
        
        # 4. 初始化最终的分类器
        self.classifier = nn.Linear(512, 1) # 输出一个 Logit 用于二分类

    def forward(self, image):
        # 1. 从 CLIP 提取特征
        # 使用 internal_features=True 来获取中间层的 patch tokens
        image_features, internal_features = self.clip.visual(image, internal_features=True)
        # 形状: [B, 257, 1024] (对于 ViT-L/14)
        patch_tokens = internal_features['trunk_pre_norm'] 
        # [class] token 就是 CLIP 的最终图像输出
        # 形状: [B, 1024]
        class_token = image_features 

        # 2. 将特征送入各个专家
        F_s = self.spatial_expert(patch_tokens)
        F_f = self.frequency_expert(patch_tokens)
        
        # 3. 计算专家权重
        weights = self.router(class_token) # 形状: [B, 2]
        w_s, w_f = weights[:, 0].unsqueeze(1), weights[:, 1].unsqueeze(1)
        
        # 4. 加权融合专家特征
        fused_feature = w_s * F_s + w_f * F_f
        
        # 5. 通过分类器得到最终输出
        logits = self.classifier(fused_feature)
        
        # 返回一个字典，方便在计算损失时清晰地获取各个部分
        return {
            "logits": logits,
            "F_s": F_s,
            "F_f": F_f
        }