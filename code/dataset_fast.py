import torch
from torch.utils.data import Dataset, DataLoader
import os

class PreextractedFeatureDataset(Dataset):
    def __init__(self, features_path):
        super().__init__()
        print(f"正在以零拷贝内存映射(mmap)模式加载离线特征: {features_path} ...")
        # 核心改动：加入 mmap=True，让 36G 的特征文件瞬间加载完毕且几乎不占物理内存！
        data = torch.load(features_path, map_location="cpu", mmap=True)
        self.patch_tokens = data['patch_tokens']
        self.class_tokens = data['class_token']
        self.labels = data['labels']
        
        # 释放原始字典引用
        del data
        print("特征加载完成！")
        
    def __len__(self):
        return len(self.labels)
        
    def __getitem__(self, idx):
        # 注意：由于我们在特征提取时存的是 float16，
        # 在这里取出来喂给网络时需要转回 float32 保证训练精度
        pt = self.patch_tokens[idx].to(torch.float32)
        ct = self.class_tokens[idx].to(torch.float32)
        label = self.labels[idx].to(torch.float32) # 对于 BCE/Focal loss 需要 float32 标签
        
        return pt, ct, label

def create_fast_dataloaders(features_dir, batch_size=256, num_workers=4):
    """
    直接读取 .pt 特征返回极速 DataLoader
    """
    train_ds = PreextractedFeatureDataset(os.path.join(features_dir, "train_features.pt"))
    val_ds = PreextractedFeatureDataset(os.path.join(features_dir, "val_features.pt"))
    test_ds = PreextractedFeatureDataset(os.path.join(features_dir, "test_features.pt"))
    
    # 极速特征加载时，CPU瓶颈几乎没有，可以用较少的 num_workers 甚至 0
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader
