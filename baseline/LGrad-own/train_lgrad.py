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

# Add code directory to path to import dataset
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../code')))
from dataset import create_dataloaders

def setup_logging(log_file):
    """配置日志，同时输出到控制台和文件"""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )

def calculate_metrics(labels, preds_prob):
    """计算二分类的各种指标"""
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
    parser = argparse.ArgumentParser(description="LGrad Training Script")
    
    # 数据集配置
    parser.add_argument("--dataset", type=str, default="dataset-A-LGrad", help="Name of the dataset (folder name in datasets/)")
    parser.add_argument("--data_root", type=str, default="./datasets", help="Root directory for datasets")
    
    # 训练超参数
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    
    # 结果保存路径
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save results")
    
    return parser.parse_args()

def run():
    args = get_args()
    
    # --- 0. 基础配置与目录初始化 ---
    start_time = time.time()
    
    # 超参数
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_NAME = args.dataset
    
    # 处理 LGrad 数据集路径逻辑：如果传入的是 -LGrad 结尾的数据集，直接在根目录或指定路径找
    # 因为之前转换脚本可能把它放在了项目根目录 /hy-tmp/PG-MOE-Project/dataset-X-LGrad
    if os.path.exists(os.path.join(args.data_root, DATASET_NAME)):
        DATASET_PATH = os.path.join(args.data_root, DATASET_NAME)
    elif os.path.exists(os.path.join(os.path.dirname(args.data_root), DATASET_NAME)):
        DATASET_PATH = os.path.join(os.path.dirname(args.data_root), DATASET_NAME)
    else:
        DATASET_PATH = os.path.join(args.data_root, DATASET_NAME) # 默认
        
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    EPOCHS = args.epochs
    NUM_WORKERS = args.num_workers
    RESULTS_ROOT = args.results_dir
    
    # 获取当前日期和时间，生成唯一ID
    now = datetime.datetime.now()
    date_str = now.strftime("%m-%d")
    time_str = now.strftime("%H%M%S")
    run_id = f"{date_str}-{time_str}"
    
    # 初始结果目录 (临时名称，最后会重命名)
    if not os.path.exists(RESULTS_ROOT):
        os.makedirs(RESULTS_ROOT)
        
    temp_folder_name = f"LGrad_Temp_{DATASET_NAME}_{run_id}"
    OUTPUT_DIR = os.path.join(RESULTS_ROOT, temp_folder_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(OUTPUT_DIR, "train.log")
    setup_logging(log_file)
    logging.info(f"--- 开始训练任务: {run_id} ---")
    logging.info(f"使用设备: {DEVICE}")
    logging.info(f"结果输出目录: {OUTPUT_DIR}")
    logging.info(f"配置参数: {args}")

    # --- 1. 数据准备 ---
    if not os.path.exists(DATASET_PATH):
        logging.error(f"数据集路径不存在: {DATASET_PATH}")
        logging.error("请确保数据集已准备好。")
        return

    logging.info(f"加载数据集: {DATASET_PATH}")
    train_loader, val_loader, test_loader = create_dataloaders(DATASET_PATH, BATCH_SIZE, NUM_WORKERS)
    
    # --- 2. 模型与优化器 ---
    # LGrad 直接使用 ResNet50 作为检测器骨干
    model = models.resnet50(pretrained=True)
    # 修改最后的全连接层为二分类 (Real vs Fake)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2) 
    model = model.to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()

    params_to_train = model.parameters()
    optimizer = torch.optim.AdamW(params_to_train, lr=LEARNING_RATE)
    
    # 学习率调度器 (Cosine Annealing)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # --- 3. 训练循环 ---
    best_val_auc = 0.0
    best_val_acc = 0.0
    best_epoch = 0
    epochs_no_improve = 0
    patience = 20
    
    # CSV 记录文件
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
        
        # --- 训练阶段 ---
        model.train()
        train_loss_total = 0
        
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1} Train"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            # 梯度裁剪 (Gradient Clipping)
            torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
            
            optimizer.step()
            
            train_loss_total += loss.item()
        
        # 更新学习率
        scheduler.step()
        
        avg_train_loss = train_loss_total / len(train_loader)
        logging.info(f"Train Loss: {avg_train_loss:.4f}")

        # --- 验证阶段 ---
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
                
                # CrossEntropy 输出的是 logits，取 softmax 后的第 1 类概率作为 Fake 的概率
                preds = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        val_end_time = time.time()
        
        # 计算验证指标
        avg_val_loss = val_loss_total / len(val_loader)
        val_metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
        
        # 计算推理速度 (FPS) - 基于验证集
        num_val_images = len(val_loader.dataset)
        inference_time = val_end_time - val_start_time
        fps = num_val_images / inference_time if inference_time > 0 else 0
        
        logging.info(f"Val Loss: {avg_val_loss:.4f} | FPS: {fps:.2f}")
        logging.info(f"Val Metrics -> AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        # 打印真实显存峰值
        if DEVICE == "cuda":
            max_memory = torch.cuda.max_memory_allocated(DEVICE) / (1024 ** 3)
            logging.info(f"当前 Epoch 真实显存峰值占用: {max_memory:.2f} GB")

        # 写入 CSV
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, 
                f"{current_lr:.6f}",
                f"{avg_train_loss:.4f}",
                f"{avg_val_loss:.4f}", 
                f"{val_metrics['acc']:.4f}", f"{val_metrics['auc']:.4f}", f"{val_metrics['f1']:.4f}", 
                f"{val_metrics['precision']:.4f}", f"{val_metrics['recall']:.4f}",
                f"{fps:.2f}"
            ])

        # --- 保存最佳模型 ---
        if val_metrics['auc'] > best_val_auc:
            if val_metrics['auc'] - best_val_auc > 1e-4:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            best_val_auc = val_metrics['auc']
            best_val_acc = val_metrics['acc']
            best_epoch = epoch + 1
                
            # 清理旧的最佳模型
            for f in os.listdir(OUTPUT_DIR):
                if f.endswith(".pth"):
                    if f.startswith("best_model_"):
                        try:
                            os.remove(os.path.join(OUTPUT_DIR, f))
                        except OSError as e:
                            logging.warning(f"删除旧模型失败: {f}, 错误: {e}")
            
            # 保存新的最佳模型
            model_save_name = f"best_model_epoch{epoch+1}_auc{best_val_auc:.4f}.pth"
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, model_save_name))
            logging.info(f"*** 发现新最佳模型 (AUC: {best_val_auc:.4f})，已保存 ***")
        else:
            epochs_no_improve += 1
            logging.info(f"验证集 AUC 未提升 (当前: {val_metrics['auc']:.4f}, 最佳: {best_val_auc:.4f}). 已连续 {epochs_no_improve} 轮未提升。")

        # --- Early Stopping 检查 ---
        if epochs_no_improve >= patience:
            logging.info(f"早停触发 (Early Stopping)! 连续 {patience} 轮验证集 AUC 未显著提升。停止训练。")
            break

    # --- 4. 训练结束与最终评估 ---
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logging.info(f"\n--- 训练结束。总耗时: {total_time_str} ---")
    
    # 在测试集上评估最佳模型
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
        
        # 将测试集结果追加到 CSV 文件
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "Test_Set", 
                "-", "-", "-", 
                f"{test_metrics['acc']:.4f}", 
                f"{test_metrics['auc']:.4f}", 
                f"{test_metrics['f1']:.4f}", 
                f"{test_metrics['precision']:.4f}", 
                f"{test_metrics['recall']:.4f}",
                "-"
            ])
            
    else:
        logging.warning("未找到最佳模型文件，跳过测试集评估。")
        test_metrics = {"auc": 0, "acc": 0}

    # --- 5. 生成报告与重命名文件夹 ---
    info_file = os.path.join(OUTPUT_DIR, "train_info.txt")
    with open(info_file, "w") as f:
        f.write(f"Dataset: {DATASET_NAME}\n")
        f.write(f"Date: {date_str}\n")
        f.write(f"Run ID: {run_id}\n")
        f.write(f"Total Time: {total_time_str}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Method: LGrad (ResNet50 backbone)\n")
        f.write("-" * 30 + "\n")
        f.write(f"Hyperparameters:\n")
        f.write(f"  Epochs: {EPOCHS}\n")
        f.write(f"  Batch Size: {BATCH_SIZE}\n")
        f.write(f"  Learning Rate: {LEARNING_RATE}\n")
        f.write(f"  LR Scheduler: CosineAnnealingLR (eta_min=1e-6)\n")
        f.write(f"  Gradient Clipping: Max Norm = 1.0\n")
        f.write("-" * 30 + "\n")
        f.write(f"Best Validation Results (Epoch {best_epoch}):\n")
        f.write(f"  AUC: {best_val_auc:.4f}\n")
        f.write(f"  Accuracy: {best_val_acc:.4f}\n")
        if best_model_files:
             f.write(f"Test Set Results:\n")
             f.write(f"  AUC: {test_metrics['auc']:.4f}\n")
             f.write(f"  Accuracy: {test_metrics['acc']:.4f}\n")
             f.write(f"  F1: {test_metrics['f1']:.4f}\n")
             f.write(f"  Precision: {test_metrics['precision']:.4f}\n")
             f.write(f"  Recall: {test_metrics['recall']:.4f}\n")

    # 重命名结果文件夹，加上 LGrad_ 前缀
    final_folder_name = f"LGrad_{DATASET_NAME}_AUC{best_val_auc:.4f}_ACC{best_val_acc:.4f}_{run_id}"
    FINAL_OUTPUT_DIR = os.path.join(RESULTS_ROOT, final_folder_name)
    
    try:
        os.rename(OUTPUT_DIR, FINAL_OUTPUT_DIR)
        logging.info(f"结果文件夹已重命名为: {FINAL_OUTPUT_DIR}")
    except Exception as e:
        logging.error(f"重命名文件夹失败: {e}")

    logging.info("--- 任务全部完成 ---")

if __name__ == '__main__':
    run()