
import torch
from torchvision import transforms
from PIL import Image
import os
import argparse
import glob
import sys

# 导入项目模块
from model import PGMoE

def get_default_device():
    return "cuda" if torch.cuda.is_available() else "cpu"

def load_image(image_path, image_size=224):
    """
    加载并预处理图片，与验证集预处理保持一致。
    """
    transform = transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), 
            (0.26862954, 0.26130258, 0.27577711)
        ),
    ])
    
    try:
        image = Image.open(image_path)
        image_tensor = transform(image).unsqueeze(0) # [1, 3, 224, 224]
        return image_tensor
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return None

def find_best_model(results_dir="../results"):
    """
    自动寻找 results 目录下 AUC 最高的模型文件。
    """
    if not os.path.exists(results_dir):
        # 尝试兼容 v2 目录
        results_dir_v2 = "../results_v2"
        if os.path.exists(results_dir_v2):
            results_dir = results_dir_v2
        else:
            return None
        
    pth_files = glob.glob(os.path.join(results_dir, "*.pth"))
    if not pth_files:
        return None
        
    best_file = None
    max_auc = -1.0
    
    for f in pth_files:
        basename = os.path.basename(f)
        try:
            # 假设文件名格式如 best_model_epoch17_auc0.9949-0129-B.pth
            # 寻找 "auc" 后面的数字
            import re
            match = re.search(r"auc(\d+\.\d+)", basename)
            if match:
                auc = float(match.group(1))
                if auc > max_auc:
                    max_auc = auc
                    best_file = f
        except:
            pass
            
    # 如果没解析到，就用最新的
    if best_file is None:
        best_file = max(pth_files, key=os.path.getmtime)
        
    return best_file

def predict(image_path, model_path=None, device=None):
    if device is None:
        device = get_default_device()
        
    # 1. 确定模型路径
    if model_path is None:
        model_path = find_best_model()
        if model_path is None:
            print("错误：未指定模型路径，且无法自动找到默认模型。")
            return None
            
    print(f"正在加载模型: {model_path}")
    
    # 2. 加载模型
    try:
        model = PGMoE()
        # 加载权重
        checkpoint = torch.load(model_path, map_location=device)
        # 兼容处理：如果 checkpoint 包含 'state_dict' 键
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"加载模型失败: {e}")
        return None

    # 3. 处理图片
    image_tensor = load_image(image_path)
    if image_tensor is None:
        return None
    image_tensor = image_tensor.to(device)
    
    # 4. 推理
    with torch.no_grad():
        outputs = model(image_tensor)
        logits = outputs['logits']
        weights = outputs.get('weights', None) # 获取专家权重
        
        # Sigmoid 输出的是属于 Class 1 (Real) 的概率
        prob = torch.sigmoid(logits).item()
        prob_real = prob
        prob_fake = 1.0 - prob
        
        # 判定逻辑：dataset-B 中 0=fake, 1=real
        if prob_real > 0.5:
            prediction = "Real (Authentic)"
            confidence = prob_real
            is_fake = False
        else:
            prediction = "Fake (Generated)"
            confidence = prob_fake
            is_fake = True
        
        print("\n" + "="*40)
        print(f"图片: {image_path}")
        print(f"预测结果: {prediction}")
        print(f"伪造概率: {prob_fake:.4f}")
        print(f"置信度: {confidence:.2%}")
        
        expert_info = {}
        if weights is not None:
            # weights shape [1, 2] -> [Spatial, Frequency]
            w_spatial = weights[0, 0].item()
            w_freq = weights[0, 1].item()
            expert_info = {"spatial": w_spatial, "frequency": w_freq}
            
            print("-" * 20)
            print("专家关注度分析:")
            print(f"  空间专家 (Spatial): {w_spatial:.2%}")
            print(f"  频域专家 (Frequency): {w_freq:.2%}")
            if w_spatial > w_freq:
                print("  -> 模型主要依据【空间纹理/伪影】进行判断")
            else:
                print("  -> 模型主要依据【频域/周期性特征】进行判断")
        print("="*40 + "\n")
        
        return {
            "prediction": prediction,
            "is_fake": is_fake,
            "prob_fake": prob_fake,
            "confidence": confidence,
            "expert_info": expert_info
        }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PG-MoE 图像检测推理工具")
    parser.add_argument("--image", type=str, required=True, help="待检测图片的路径")
    parser.add_argument("--model", type=str, default=None, help="模型文件路径 (.pth)，默认自动寻找最佳模型")
    
    args = parser.parse_args()
    
    predict(args.image, args.model)
