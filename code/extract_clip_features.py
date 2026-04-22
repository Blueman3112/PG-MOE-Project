import torch
import open_clip
import os
import argparse
from tqdm import tqdm
from dataset import create_dataloaders
from model import PGMoE

def extract_features(dataset_name, data_root="./datasets", output_root="./datasets_features", batch_size=256, num_workers=8):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_path = os.path.join(data_root, dataset_name)
    output_dir = os.path.join(output_root, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"初始化特征提取，来源: {dataset_path}")
    print(f"目标保存路径: {output_dir}")

    # 1. 载入模型 (复用 PGMoE 以保证预处理完全一致)
    print("正在加载 CLIP 并配置 Hook...")
    model = PGMoE().to(device)
    model.eval()

    # 2. 载入数据
    train_loader, val_loader, test_loader = create_dataloaders(dataset_path, batch_size=batch_size, num_workers=num_workers)

    splits = {
        "train": train_loader,
        "val": val_loader,
        "test": test_loader
    }

    # 3. 提取特征
    with torch.no_grad():
        for split_name, loader in splits.items():
            print(f"\n开始提取 [{split_name}] 集合的特征...")
            num_samples = len(loader.dataset)
            
            # 预分配连续内存，以应对极大的 tensor (使用 Float16 压缩一半空间)
            # patch_tokens 维度: [N, 257, 1024], class_token 维度: [N, 768]
            all_patch_tokens = torch.empty((num_samples, 257, 1024), dtype=torch.float16)
            all_class_tokens = torch.empty((num_samples, 768), dtype=torch.float16)
            all_labels = torch.empty((num_samples,), dtype=torch.long)

            start_idx = 0
            for images, labels in tqdm(loader, desc=f"Extracting {split_name}"):
                images = images.to(device)
                
                # 执行一次前向，截获特征
                class_token = model.clip.visual(images)
                patch_tokens = model.captured_tokens
                
                batch_size_cur = images.size(0)
                end_idx = start_idx + batch_size_cur

                # 存入 CPU 内存并转为 float16
                all_patch_tokens[start_idx:end_idx] = patch_tokens.cpu().to(torch.float16)
                all_class_tokens[start_idx:end_idx] = class_token.cpu().to(torch.float16)
                all_labels[start_idx:end_idx] = labels.cpu()

                start_idx = end_idx

            # 保存为单一大文件
            save_path = os.path.join(output_dir, f"{split_name}_features.pt")
            print(f"正在写入硬盘 (这可能需要几秒到几十秒): {save_path}...")
            torch.save({
                'patch_tokens': all_patch_tokens,
                'class_token': all_class_tokens,
                'labels': all_labels
            }, save_path)
            print(f"写入完成: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="dataset-A")
    parser.add_argument("--batch_size", type=int, default=256)
    args = parser.parse_args()
    
    extract_features(dataset_name=args.dataset, batch_size=args.batch_size)
