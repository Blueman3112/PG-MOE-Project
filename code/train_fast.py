import torch
import time
import datetime
import logging
import csv
import sys
import os
import argparse
from tqdm import tqdm
import numpy as np

# 从我们新的极速模块中导入
from dataset_fast import create_fast_dataloaders
from model_fast import PGMoEFast
from train import setup_logging, calculate_metrics
from loss import OrthogonalLoss

def get_args():
    parser = argparse.ArgumentParser(description="PG-MoE FAST Training Script")
    
    parser.add_argument("--dataset", type=str, default="dataset-A", help="Name of the dataset")
    parser.add_argument("--features_root", type=str, default="./datasets_features", help="Root directory for extracted features")
    parser.add_argument("--fusion_type", type=str, default="moe", choices=['moe', 'concat'], help="Fusion mechanism to use")
    
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for FAST mode (can be very large!)")
    parser.add_argument("--lambda_orth", type=float, default=0.05, help="Weight for orthogonal loss")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of data loading workers")
    parser.add_argument("--results_dir", type=str, default="./results", help="Directory to save results")
    return parser.parse_args()

def run():
    args = get_args()
    start_time = time.time()
    
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_NAME = args.dataset
    FEATURES_PATH = os.path.join(args.features_root, DATASET_NAME)
    
    # 获取当前日期和时间
    now = datetime.datetime.now()
    run_id = f"{now.strftime('%m-%d-%H%M%S')}"
    
    # 构建包含融合类型标识的输出目录名
    save_prefix = f"Fast_{args.fusion_type}"
    if args.fusion_type == 'concat':
        args.lambda_orth = 0.0 # 强制如果是拼接网络的话，不需要正交损失
        
    temp_folder_name = f"Temp_{save_prefix}_{DATASET_NAME}_{run_id}"
    OUTPUT_DIR = os.path.join(args.results_dir, temp_folder_name)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    setup_logging(os.path.join(OUTPUT_DIR, "train.log"))
    logging.info(f"--- >> 极速离线训练模式 << ---")
    logging.info(f"任务: {run_id}")
    logging.info(f"融合类型 Fusion Type: {args.fusion_type.upper()}")
    logging.info(f"结果输出目录: {OUTPUT_DIR}")
    logging.info(f"配置参数: {args}")

    if not os.path.exists(FEATURES_PATH):
        logging.error(f"特征缓存路径不存在: {FEATURES_PATH}。请先运行 extract_clip_features.py！")
        return

    logging.info(f"把极速内存特征管线拉起来...")
    train_loader, val_loader, test_loader = create_fast_dataloaders(FEATURES_PATH, args.batch_size, args.num_workers)
    
    # 极速模型装载
    model = PGMoEFast(fusion_type=args.fusion_type).to(DEVICE)
    criterion = OrthogonalLoss(lambda_orth=args.lambda_orth)

    # 取出真正需要训练的参数
    params_to_train = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params_to_train, lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_auc = 0.0
    best_val_acc = 0.0
    epochs_no_improve = 0
    patience = 5 # 极速模式下允许更快早停

    csv_file = os.path.join(OUTPUT_DIR, "training_metrics.csv")
    csv_headers = ["epoch", "lr", "train_loss", "train_focal", "train_orth", 
                   "val_loss", "val_acc", "val_auc", "val_f1", "val_precision", "val_recall", "inference_fps"]
    
    with open(csv_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)

    logging.info("--> 坐好扶稳，极速训练开始 (几秒一个 Epoch) <--")
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        current_lr = optimizer.param_groups[0]['lr']
        logging.info(f"\n--- Epoch {epoch+1}/{args.epochs} (LR: {current_lr:.6f}) ---")
        
        model.train()
        train_loss_total, train_loss_focal, train_loss_orth = 0, 0, 0
        
        # tqdm 就不显示描述了，因为快到闪眼
        loader_bar = tqdm(train_loader, desc=f"Ep {epoch+1} Train", leave=False)
        for patch_tokens, class_tokens, labels in loader_bar:
            patch_tokens = patch_tokens.to(DEVICE)
            class_tokens = class_tokens.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(patch_tokens, class_tokens)
            loss, l_focal, l_orth = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_train, max_norm=1.0)
            optimizer.step()
            
            train_loss_total += loss.item()
            train_loss_focal += l_focal.item()
            train_loss_orth += l_orth.item()
        
        scheduler.step()
        
        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_focal = train_loss_focal / len(train_loader)
        avg_train_orth = train_loss_orth / len(train_loader)
        logging.info(f"Train Loss -> Total: {avg_train_loss:.4f}, Focal: {avg_train_focal:.4f}, Orth: {avg_train_orth:.4f}")

        # 验证
        model.eval()
        val_loss_total = 0
        all_preds, all_labels = [], []
        
        val_start_time = time.time()
        with torch.no_grad():
            for patch_tokens, class_tokens, labels in val_loader:
                patch_tokens = patch_tokens.to(DEVICE)
                class_tokens = class_tokens.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(patch_tokens, class_tokens)
                loss, _, _ = criterion(outputs, labels)
                val_loss_total += loss.item()
                
                preds = torch.sigmoid(outputs['logits']).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        val_end_time = time.time()
        
        avg_val_loss = val_loss_total / len(val_loader)
        val_metrics = calculate_metrics(np.array(all_labels), np.array(all_preds))
        
        num_val_images = len(val_loader.dataset)
        inference_time = val_end_time - val_start_time
        fps = num_val_images / inference_time if inference_time > 0 else 0
        
        logging.info(f"Val Loss: {avg_val_loss:.4f} | Inference FPS: {fps:.2f}")
        logging.info(f"Val Metrics -> AUC: {val_metrics['auc']:.4f}, Acc: {val_metrics['acc']:.4f}, F1: {val_metrics['f1']:.4f}")
        
        with open(csv_file, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                epoch + 1, f"{current_lr:.6f}",
                f"{avg_train_loss:.4f}", f"{avg_train_focal:.4f}", f"{avg_train_orth:.4f}",
                f"{avg_val_loss:.4f}", 
                f"{val_metrics['acc']:.4f}", f"{val_metrics['auc']:.4f}", f"{val_metrics['f1']:.4f}", 
                f"{val_metrics['precision']:.4f}", f"{val_metrics['recall']:.4f}",
                f"{fps:.2f}"
            ])

        if val_metrics['auc'] > best_val_auc:
            if val_metrics['auc'] - best_val_auc > 1e-4:
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
            
            best_val_auc = val_metrics['auc']
            best_val_acc = val_metrics['acc']
            best_epoch = epoch + 1
            
            torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "best_model.pth"))
            logging.info(f"*** 发现新最佳极速模型 (AUC: {best_val_auc:.4f})，已保存 ***")
        else:
            epochs_no_improve += 1
            logging.info(f"验证集表现未提升 (Best AUC: {best_val_auc:.4f} at Epoch {best_epoch}). Patience: {epochs_no_improve}/{patience}")
            
        if epochs_no_improve >= patience:
            logging.info(f"连续 {patience} 个 Epoch 验证集没有实质提升，触发 Early Stopping!")
            break

    total_time = time.time() - start_time
    logging.info(f"--- 极速训练彻底完成！ ---")
    logging.info(f"最强 Epoch: {best_epoch}, 最强 AUC: {best_val_auc:.4f}, 对应的 ACC: {best_val_acc:.4f}")
    logging.info(f"本轮实验总耗时: {total_time/60:.2f} 分钟")

if __name__ == "__main__":
    run()
