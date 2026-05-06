import torch
import torch.nn.functional as F
import os
import csv
from tqdm import tqdm
from torchvision import datasets
from model_fastnew import PGMoEFast
import argparse

def main():
    parser = argparse.ArgumentParser(description="Export fast model test features and predictions with sub-category parsing")
    parser.add_argument("--dataset", type=str, default="dataset-B", help="Dataset name")
    parser.add_argument("--model_path", type=str, default="../results/PG-MoE_bestModel.pth", help="Path to the trained pth")
    parser.add_argument("--features_root", type=str, default="../datasets_features", help="Path to extracted features")
    parser.add_argument("--images_root", type=str, default="../datasets", help="Path to raw images")
    parser.add_argument("--output_csv", type=str, default="../results/test_report_v2.csv", help="Output file")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # 1. 提取路径 & 分类及子分类解析
    test_image_dir = os.path.join(args.images_root, args.dataset, "test")
    if not os.path.exists(test_image_dir):
        print(f"找不到原图路径 {test_image_dir}，将生成虚拟序号作为 sample_id")
        image_paths = None
        sub_categories = None
    else:
        # 只依靠 ImageFolder 做路径映射
        test_dataset = datasets.ImageFolder(root=test_image_dir)
        image_paths = [item[0] for item in test_dataset.samples]
        
        # 解析子分类
        sub_categories = []
        for path in image_paths:
            # 获取相对路径或文件名用于解析
            # test_dataset.samples 中 path 是完整绝对路径
            # 判断逻辑: test/fake 目录下包含 'sd' -> SDV5，否则 -> proGAN; 对于 real 为 Real
            normalized_path = path.replace("\\", "/") # 兼容Windows路径符
            if "/test/fake/" in normalized_path:
                filename = os.path.basename(normalized_path).lower()
                if 'sd' in filename:
                    sub_categories.append("SDV5")
                else:
                    sub_categories.append("proGAN")
            elif "/test/real/" in normalized_path:
                sub_categories.append("Real")
            else:
                sub_categories.append("Unknown")

    # 2. 读取极速提取的特征
    test_features_path = os.path.join(args.features_root, args.dataset, "test_features.pt")
    print(f"Loading offline features from {test_features_path} ...")
    data = torch.load(test_features_path, map_location="cpu", mmap=True)
    patch_tokens = data['patch_tokens']
    class_tokens = data['class_token']
    labels = data['labels']
    
    total_samples = len(labels)
    if image_paths is not None and len(image_paths) != total_samples:
        print(f"警告：原始图片数({len(image_paths)})与离线特征样本数({total_samples})不对齐，退化为虚拟ID。")
        image_paths = None
        sub_categories = ["Unknown"] * total_samples
    elif sub_categories is None:
        sub_categories = ["Unknown"] * total_samples

    # 初始化模型。PGMoEFast 默认 fusion_type="moe" (有 Router 的标准版)
    model = PGMoEFast(fusion_type="moe").to(DEVICE)
    print(f"Loading model weights from {args.model_path} ...")
    
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE), strict=False)
    model.eval()

    # 3. 准备输出
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    # 加入了 sub_category 字段
    header = ["image_path", "gt_label", "sub_category", "pred_label", "pred_prob", "cosine_similarity", "w_s", "w_f"]
    
    with open(args.output_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        batch_size = 512
        for start_idx in tqdm(range(0, total_samples, batch_size), desc="Running Inference"):
            end_idx = min(start_idx + batch_size, total_samples)
            
            pt_batch = patch_tokens[start_idx:end_idx].to(torch.float32).to(DEVICE)
            ct_batch = class_tokens[start_idx:end_idx].to(torch.float32).to(DEVICE)
            lbl_batch = labels[start_idx:end_idx]
            
            with torch.no_grad():
                outputs = model(pt_batch, ct_batch)
                logits = outputs["logits"].squeeze(1) # shape: (B,)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int().cpu().numpy()
                
                F_s = outputs.get("F_s", torch.zeros(end_idx-start_idx, 512, device=DEVICE))
                F_f = outputs.get("F_f", torch.zeros(end_idx-start_idx, 512, device=DEVICE))
                w_s = outputs.get("w_s", torch.zeros(end_idx-start_idx, device=DEVICE))
                w_f = outputs.get("w_f", torch.zeros(end_idx-start_idx, device=DEVICE))
                
                cos_sim = F.cosine_similarity(F_s, F_f, dim=1).cpu().numpy()
            
            probs_np = probs.cpu().numpy()
            lbl_np = lbl_batch.numpy()
            w_s_np = w_s.cpu().numpy()
            w_f_np = w_f.cpu().numpy()
            
            for i in range(end_idx - start_idx):
                global_idx = start_idx + i
                img_path = image_paths[global_idx] if image_paths else f"sample_{global_idx}"
                sub_cat = sub_categories[global_idx]
                
                writer.writerow([
                    img_path,
                    lbl_np[i],
                    sub_cat,
                    preds[i],
                    f"{probs_np[i]:.6f}",
                    f"{cos_sim[i]:.6f}",
                    f"{w_s_np[i]:.6f}",
                    f"{w_f_np[i]:.6f}"
                ])

    print(f"===== 评估报告(V2)提取完毕！=====")
    print(f"数据已保存在: {args.output_csv}")

if __name__ == "__main__":
    main()
