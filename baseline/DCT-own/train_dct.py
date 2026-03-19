import torch
import time
import datetime
import logging
import csv
import shutil
import sys
import os
import argparse
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, precision_score, recall_score
import torchvision.models as models
import torch.nn as nn

# 导入我们为 DCT 专门写的 DataLoader
from dataset_dct import create_dataloaders_dct

def setup_logging(log_file):
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def calculate_metrics(labels, preds_prob):
    preds_binary = np.round(preds_prob)
    
    auc = roc_auc_score(labels, preds_prob)
    acc = accuracy_score(labels, preds_binary)
    f1 = f1_score(labels, preds_binary, zero_division=0)
    precision = precision_score(labels, preds_binary, zero_division=0)
    recall = recall_score(labels, preds_binary, zero_division=0)
    
    return {
        "auc": auc,
        "acc": acc,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }

def get_args():
    parser = argparse.ArgumentParser(description="DCT (Frequency Analysis) Training Script")
    
    parser.add_argument("--dataset", type=str, default="dataset-A-DCT", help="Name of the dataset")
    parser.add_argument("--data_root", type=str, default="./datasets", help="Root directory for datasets")
    
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save results")
    
    return parser.parse_args()

def run():
    args = get_args()
    
    start_time = time.time()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_NAME = args.dataset
    
    # 路径解析逻辑
    if os.path.exists(os.path.join(args.data_root, DATASET_NAME)):
        DATASET_PATH = os.path.join(args.data_root, DATASET_NAME)
    elif os.path.exists(os.path.join(os.path.dirname(args.data_root), DATASET_NAME)):
        DATASET_PATH = os.path.join(os.path.dirname(args.data_root), DATASET_NAME)
    else:
        DATASET_PATH = os.path.join(args.data_root, DATASET_NAME)
        
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EPOCHS = args.epochs
    NUM_WORKERS = args.num_workers
    RESULTS_ROOT = args.results_dir
    
    now = datetime.datetime.now()
    date_str = now.strftime("%m-%d")
    time_str = now.strftime("%H%M%S")
    run_id = f"{date_str}-{time_str}"
    
    if not os.path.exists(RESULTS_ROOT):
        os.makedirs(RESULTS_ROOT)
        
    temp_folder_name = f"DCT_Temp_{DATASET_NAME}_{run_id}"
    OUTPUT_DIR = os.path.join(RESULTS_ROOT, temp_folder_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    log_file = os.path.join(OUTPUT_DIR, "train.log")
    setup_logging(log_file)
    logging.info(f"--- 开始训练任务: {run_id} ---")
    logging.info(f"使用设备: {DEVICE}")
    logging.info(f"结果输出目录: {OUTPUT_DIR}")
    logging.info(f"配置参数: {args}")

    # --- 1. 数据准备 ---
    if not os.path.exists(DATASET_PATH):
        logging.error(f"数据集路径不存在: {DATASET_PATH}")
        return

    logging.info(f"加载数据集: {DATASET_PATH}")
    # 使用针对 .npy 的 DataLoader
    train_loader, val_loader, test_loader = create_dataloaders_dct(DATASET_PATH, BATCH_SIZE, NUM_WORKERS)
    
    # --- 2. 模型与优化器 ---
    # DCT 方法推荐使用相对较小的网络，如 ResNet18
    # 我们不使用 pretrained=True，因为我们要从头学习频域特征（自然图像预训练的 RGB 权重在这里用处不大，甚至是干扰）
    model = models.resnet18(pretrained=False)
    
    # 修改第一层卷积以接收 1 通道 (而不是 3 通道) 的输入
    original_conv1 = model.conv1
    model.conv1 = nn.Conv2d(1, original_conv1.out_channels, 
                            kernel_size=original_conv1.kernel_size, 
                            stride=original_conv1.stride, 
                            padding=original_conv1.padding, 
                            bias=False)
                            
    # 修改最后一层全连接为 2 分类
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) 
    
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()

    params_to_train = model.parameters()
    optimizer = torch.optim.AdamW(params_to_train, lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # --- 3. 训练循环 ---
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    patience = 20
    
    csv_file = os.path.join(OUTPUT_DIR, "training_metrics.csv")
    csv_headers = ["epoch", "lr", "train_loss", "val_loss", "val_acc", "val_auc", "val_f1", "val_precision", "val_recall", "inference_fps"]
    
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    logging.info("开始训练...")
    
    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"\n--- Epoch {epoch+1}/{EPOCHS} (LR: {current_lr:.6f}) ---")
        
        model.train()
        train_loss_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
            optimizer.step()
            
            train_loss_total += loss.item()
        
        scheduler.step()
        avg_train_loss = train_loss_total / len(train_loader)
        logging.info(f"Train Loss: {avg_train_loss:.4f}")

        model.eval()
        val_loss_total = 0
        all_preds, all_labels = [], []
        
        val_start_time = time.time()
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc=f"Epoch {epoch+1} Val"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss_total += loss.item()
                
                preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        val_end_time = time.time()
        
        avg_val_loss = val_loss_total / len(val_loader)
        val_metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
        
        num_val_images = len(val_loader.dataset)
        inference_time = val_end_time - val_start_time
        fps = num_val_images / inference_time if inference_time > 0 else 0
        
        logging.info(f"Val Loss: {avg_val_loss:.4f} | FPS: {fps:.2f}")
        logging.info(f"Val Metrics -> AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        if DEVICE == "cuda":
            max_memory = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)
            logging.info(f"当前 Epoch 真实显存峰值占用: {max_memory:.2f} GB")

        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, f"{current_lr:.6f}", f"{avg_train_loss:.4f}", f"{avg_val_loss:.4f}", 
                f"{val_metrics['acc']:.4f}", f"{val_metrics['auc']:.4f}", f"{val_metrics['f1']:.4f}", 
                f"{val_metrics['precision']:.4f}", f"{val_metrics['recall']:.4f}", f"{fps:.2f}"
            ])

        if val_metrics['auc'] > best_val_auc:
            if val_metrics['auc'] - best_val_auc > 1e-4:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            best_val_auc = val_metrics['auc']
            best_val_acc = val_metrics['acc']
            best_epoch = epoch + 1
                
            for f in os.listdir(OUTPUT_DIR):
                if f.endswith(".pth") and f.startswith("best_model_"):
                    try: os.remove(os.path.join(OUTPUT_DIR, f))
                    except OSError as e: pass
            
            model_save_name = f"best_model_epoch{epoch+1}_auc{best_val_auc:.4f}.pth"
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, model_save_name))
            logging.info(f"*** 发现新最佳模型 (AUC: {best_val_auc:.4f})，已保存 ***")
        else:
            epochs_no_improve += 1
            logging.info(f"验证集 AUC 未提升. 已连续 {epochs_no_improve} 轮未提升。")

        if epochs_no_improve >= patience:
            logging.info(f"早停触发 (Early Stopping)! 停止训练。")
            break

    # --- 4. 测试阶段 ---
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f"\n--- 训练结束。总耗时: {total_time_str} ---")
    
    logging.info("开始在测试集上评估最佳模型...")
    best_model_files = [f for f in os.listdir(OUTPUT_DIR) if f.startswith("best_model_") and f.endswith(".pth")]
    if best_model_files:
        best_model_path = os.path.join(OUTPUT_DIR, best_model_files[0])
        model.load_state_dict(torch.load(best_model_path))
        
        model.eval()
        test_preds, test_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="Testing"):
                images = images.to(DEVICE)
                outputs = model(images)
                preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                test_preds.extend(preds)
                test_labels.extend(labels.numpy())
        
        test_metrics = calculate_metrics(np.array(test_labels), np.array(test_preds))
        logging.info(f"测试集最终结果 -> AUC: {test_metrics['auc']:.4f}, Acc: {test_metrics['acc']:.4f}, F1: {test_metrics['f1']:.4f}")
        
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Test_Set", "-", "-", "-", 
                f"{test_metrics['acc']:.4f}", f"{test_metrics['auc']:.4f}", f"{test_metrics['f1']:.4f}", 
                f"{test_metrics['precision']:.4f}", f"{test_metrics['recall']:.4f}", "-"
            ])
    else:
        logging.warning("未找到最佳模型文件，跳过测试集评估。")
        test_metrics = {"auc": 0, "acc": 0}

    # --- 5. 生成报告 ---
    info_file = os.path.join(OUTPUT_DIR, "train_info.txt")
    with open(info_file, "w") as f:
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Method: DCT Frequency Analysis (ResNet18 backbone, 1-channel input)\n")
        f.write("-" * 30 + "\n")
        f.write(f"Best Validation Results (Epoch {best_epoch}):\n")
        f.write(f"  AUC: {best_val_auc:.4f}\n")
        f.write(f"  Accuracy: {best_val_acc:.4f}\n")
        if best_model_files:
             f.write(f"Test Set Results:\n")
             f.write(f"  AUC: {test_metrics['auc']:.4f}\n")
             f.write(f"  Accuracy: {test_metrics['acc']:.4f}\n")

    final_folder_name = f"DCT_{DATASET_NAME}_AUC{best_val_auc:.4f}_ACC{best_val_acc:.4f}_{run_id}"
    FINAL_OUTPUT_DIR = os.path.join(RESULTS_ROOT, final_folder_name)
    
    try:
        os.rename(OUTPUT_DIR, FINAL_OUTPUT_DIR)
        logging.info(f"结果文件夹已重命名为: {FINAL_OUTPUT_DIR}")
    except Exception as e:
        logging.error(f"重命名文件夹失败: {e}")

if __name__ == '__main__':
    run()