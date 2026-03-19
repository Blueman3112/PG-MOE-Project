import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class DCTNumpyDataset(Dataset):
    """
    自定义 Dataset 用于加载保存为 .npy 的 DCT 频域矩阵文件。
    目录结构应类似于标准的 ImageFolder:
    root/
        real/
            img1.npy
            img2.npy
        fake/
            img1.npy
            img2.npy
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.classes = sorted([d.name for d in os.scandir(root_dir) if d.is_dir()])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        self.samples = []
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            # 获取所有的 .npy 文件
            for npy_path in glob.glob(os.path.join(class_dir, "*.npy")):
                self.samples.append((npy_path, class_idx))
                
        if len(self.samples) == 0:
            raise RuntimeError(f"Found 0 .npy files in subfolders of: {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npy_path, label = self.samples[idx]
        
        # 加载 .npy 矩阵 (已经是归一化好的 float32)
        try:
            dct_matrix = np.load(npy_path)
        except Exception as e:
            print(f"Error loading {npy_path}: {e}")
            # 如果某个文件损坏，返回一个全零张量以防崩溃
            dct_matrix = np.zeros((256, 256), dtype=np.float32)
            
        # 转换为 PyTorch Tensor，并增加通道维度 [C, H, W] -> [1, 256, 256]
        # 因为 DCT 是单通道灰度频域图
        tensor = torch.from_numpy(dct_matrix).unsqueeze(0)
        
        return tensor, label

def create_dataloaders_dct(dataset_path, batch_size=32, num_workers=4):
    """
    根据给定的数据集路径创建训练、验证和测试的 DataLoader。
    专门针对 DCT 的 .npy 格式。
    """
    train_dir = os.path.join(dataset_path, "train")
    val_dir = os.path.join(dataset_path, "val")
    test_dir = os.path.join(dataset_path, "test")
    
    train_dataset = DCTNumpyDataset(train_dir)
    val_dataset = DCTNumpyDataset(val_dir)
    test_dataset = DCTNumpyDataset(test_dir)
    
    print(f"训练集信息: 共 {len(train_dataset)} 个样本。 类别: {train_dataset.classes} (0: real, 1: fake)")
    print(f"验证集信息: 共 {len(val_dataset)} 个样本。 类别: {val_dataset.classes}")
    print(f"测试集信息: 共 {len(test_dataset)} 个样本。 类别: {test_dataset.classes}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    return train_loader, val_loader, test_loader

if __name__ == '__main__':
    # 简单的本地测试
    pass
