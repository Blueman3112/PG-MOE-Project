
from torchvision import datasets
import os

def check_classes():
    dataset_path = "../datasets/dataset-B/train"
    if not os.path.exists(dataset_path):
        print("数据集路径不存在")
        return

    dataset = datasets.ImageFolder(root=dataset_path)
    print("类别映射 (Class to Index):")
    print(dataset.class_to_idx)
    
    # 反向映射
    idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    print("索引 0 代表:", idx_to_class[0])
    print("索引 1 代表:", idx_to_class[1])

if __name__ == "__main__":
    check_classes()
