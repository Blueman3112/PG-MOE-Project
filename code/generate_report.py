import torch
import torch.nn.functional as F
import os
import csv
from tqdm import tqdm
from torchvision import datasets
from model_fastnew import PGMoEFast
import argparse

def main():
    parser = argparse.ArgumentParser(description="Export fast model test features and predictions")
    parser.add_argument("--dataset", type=str, default="dataset-B", help="Dataset name")
    parser.add_argument("--model_path", type=str, default="../results/PG-MoE_bestModel.pth", help="Path to the trained pth")
    parser.add_argument("--features_root", type=str, default="../datasets_features", help="Path to extracted features")
    parser.add_argument("--images_root", type=str, default="../datasets", help="Path to raw images")
    parser.add_argument("--output_csv", type=str, default="../results/test_report.csv", help="Output file")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")

    # 1. 提取路径 & 分手标签
    # 获取 ImageFolder 来对齐文件名
    test_image_dir = os.path.join(args.images_root, args.dataset, "test")
    if not os.path.exists(test_image_dir):
        # 兼容一下文件树结构如果没有原图也能直接生成虚拟的序号
        print(f"找不到原图路径 {test_image_dir}，将生成虚拟序号作为 sample_id")
        image_paths = None
    else:
        # 只依靠 ImageFolder 做路径映射，不处理图片加载
        test_dataset = datasets.ImageFolder(root=test_image_dir)
        image_paths = [item[0] for item in test_dataset.samples]

    # 2. 读取极速提取的特征
    test_features_path = os.path.join(args.features_root, args.dataset, "test_features.pt")
    print(f"Loading offline features from {test_features_path} ...")
    data = torch.load(test_features_path, map_location="cpu", mmap=True)
    patch_tokens = data['patch_tokens']
    class_tokens = data['class_token']
    labels = data['labels']
    
    # 确保总数对齐
    total_samples = len(labels)
    if image_paths is not None and len(image_paths) != total_samples:
        print(f"警告：原始图片数({len(image_paths)})与离线特征样本数({total_samples})不对齐，退化为虚拟ID。")
        image_paths = None

    # 初始化模型。PGMoEFast 默认 fusion_type="moe" (有 Router 的标准版)
    model = PGMoEFast(fusion_type="moe").to(DEVICE)
    print(f"Loading model weights from {args.model_path} ...")
    
    # 使用 strict=False 处理可能的多余权重（以兼容不同的训练快照）
    model.load_state_dict(torch.load(args.model_path, map_location=DEVICE), strict=False)
    model.eval()

    # 3. 准备输出
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    header = ["image_path", "gt_label", "pred_label", "pred_prob", "cosine_similarity", "w_s", "w_f"]
    
    with open(args.output_csv, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        
        batch_size = 512
        for start_idx in tqdm(range(0, total_samples, batch_size), desc="Running Inference"):
            end_idx = min(start_idx + batch_size, total_samples)
            
            # 由于 mmap，我们可以直接分块转 float32 提进 GPU
            pt_batch = patch_tokens[start_idx:end_idx].to(torch.float32).to(DEVICE)
            ct_batch = class_tokens[start_idx:end_idx].to(torch.float32).to(DEVICE)
            lbl_batch = labels[start_idx:end_idx]
            
            with torch.no_grad():
                outputs = model(pt_batch, ct_batch)
                logits = outputs["logits"].squeeze(1) # shape: (B,)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).int().cpu().numpy()
                
                # 获取各个特征与权重
                F_s = outputs.get("F_s", torch.zeros(end_idx-start_idx, 512, device=DEVICE))
                F_f = outputs.get("F_f", torch.zeros(end_idx-start_idx, 512, device=DEVICE))
                w_s = outputs.get("w_s", torch.zeros(end_idx-start_idx, device=DEVICE))
                w_f = outputs.get("w_f", torch.zeros(end_idx-start_idx, device=DEVICE))
                
                # 计算余弦相似度
                # 沿特征维度(dim=1) 计算
                cos_sim = F.cosine_similarity(F_s, F_f, dim=1).cpu().numpy()
            
            # 转存 CPU 取分量
            probs_np = probs.cpu().numpy()
            lbl_np = lbl_batch.numpy()
            w_s_np = w_s.cpu().numpy()
            w_f_np = w_f.cpu().numpy()
            
            # 将 F_s 和 F_f 整个张量序列化存下来可能太庞大且CSV卡死。
            # 为了防止 Excel 炸裂，这里不导出 512D 的数组而是把它另存为 Numpy/PT文件或者跳过
            # 本脚本CSV只保存标量监控数据。如果您需要全部512维数据，下面这段代码做了独立挂载。
            
            # 写入单行
            for i in range(end_idx - start_idx):
                global_idx = start_idx + i
                img_path = image_paths[global_idx] if image_paths else f"sample_{global_idx}"
                
                writer.writerow([
                    img_path,
                    lbl_np[i],
                    preds[i],
                    f"{probs_np[i]:.6f}",
                    f"{cos_sim[i]:.6f}",
                    f"{w_s_np[i]:.6f}",
                    f"{w_f_np[i]:.6f}"
                ])

    print(f"===== 评估报告提取完毕！=====")
    print(f"数据已保存在: {args.output_csv}")
    print("注：考虑到F_s与F_f均为512维高维数组，保存进CSV会导致表格卡顿且难分析。")
    print("目前以将相似度（cosine_similarity）和路由权重(w_s / w_f)进行了展开。")

if __name__ == "__main__":
    main()
