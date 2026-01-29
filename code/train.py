# code/train.py

import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import os
import time  # 添加时间模块
from datetime import datetime  # 添加日期时间模块

# 从我们自己的模块中导入
from dataset import create_dataloaders
from model import PGMoE
from loss import OrthogonalLoss

def run():
    # --- 1. 超参数与配置 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_PATH = "../datasets/dataset-B"
    BATCH_SIZE = 32
    LEARNING_RATE = 1e-4
    EPOCHS = 20
    LAMBDA_ORTH = 0.05 # 正交损失的权重超参数
    NUM_WORKERS = 4 # 如果你的机器支持多线程，可以设为 4 或 8
    
    BEST_MODEL_SAVE_PATH = "../results" # 模型保存的文件夹
    if not os.path.exists(BEST_MODEL_SAVE_PATH):
        os.makedirs(BEST_MODEL_SAVE_PATH)
    
    print(f"使用设备: {DEVICE}")

    # --- 2. 准备数据、模型、损失函数、优化器 ---
    train_loader, val_loader, test_loader = create_dataloaders(DATASET_PATH, BATCH_SIZE, NUM_WORKERS)
    model = PGMoE().to(DEVICE)
    criterion = OrthogonalLoss(lambda_orth=LAMBDA_ORTH)

    # 关键：只将需要训练的参数传给优化器
    params_to_train = list(model.spatial_expert.parameters()) + \
                      list(model.frequency_expert.parameters()) + \
                      list(model.router.parameters()) + \
                      list(model.classifier.parameters())
    optimizer = torch.optim.AdamW(params_to_train, lr=LEARNING_RATE)

    # --- 3. 训练与验证循环 ---
    best_val_auc = 0.0

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()  # 记录epoch开始时间
        start_timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")  # 获取当前时间戳
        print(f"{start_timestamp} Epoch {epoch+1}/{EPOCHS}")  # 打印epoch开始时间戳
        
        # --- 训练阶段 ---
        model.train()
        train_loss_total, train_loss_bce, train_loss_orth = 0, 0, 0
        
        for images, labels in tqdm(train_loader, desc="训练中"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss, l_bce, l_orth = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss_total += loss.item()
            train_loss_bce += l_bce.item()
            train_loss_orth += l_orth.item()
        
        avg_train_loss = train_loss_total / len(train_loader)
        avg_train_bce = train_loss_bce / len(train_loader)
        avg_train_orth = train_loss_orth / len(train_loader)
        print(f"训练损失 -> 总计: {avg_train_loss:.4f}, BCE: {avg_train_bce:.4f}, Orth: {avg_train_orth:.4f}")

        # --- 验证阶段 ---
        model.eval()
        val_loss_total = 0
        all_preds, all_labels = [], []
        
        with torch.no_grad():
            for images, labels in tqdm(val_loader, desc="验证中"):
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                loss, _, _ = criterion(outputs, labels)
                val_loss_total += loss.item()
                
                preds = torch.sigmoid(outputs['logits']).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
        
        avg_val_loss = val_loss_total / len(val_loader)
        val_auc = roc_auc_score(all_labels, all_preds)
        val_accuracy = accuracy_score(all_labels, np.round(all_preds))
        print(f"验证损失 -> 总计: {avg_val_loss:.4f} | 验证集 Acc: {val_accuracy:.4f}, AUC: {val_auc:.4f}")

        # --- 保存最佳模型 ---
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            model_save_name = f"best_model_epoch{epoch+1}_auc{best_val_auc:.4f}.pth"
            model_save_path = os.path.join(BEST_MODEL_SAVE_PATH, model_save_name)

            # 删除之前的最佳模型文件
            previous_best_models = [f for f in os.listdir(BEST_MODEL_SAVE_PATH) if f.startswith("best_model") and f.endswith(".pth")]
            for old_model in previous_best_models:
                os.remove(os.path.join(BEST_MODEL_SAVE_PATH, old_model))

            # 保存当前最佳模型
            torch.save(model.state_dict(), model_save_path)
            print(f"发现更优模型 (AUC: {best_val_auc:.4f})，已保存至 {model_save_name}")

        epoch_end_time = time.time()  # 记录epoch结束时间
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch+1} 用时: {epoch_duration:.2f} 秒")  # 打印epoch用时

    print("\n--- 所有训练轮次结束 ---")

    # --- 4. 最终测试阶段 ---
    print("\n开始在测试集上进行最终评估...")
    # 找到之前保存的最佳模型文件
    best_model_filename = sorted([f for f in os.listdir(BEST_MODEL_SAVE_PATH) if f.endswith(".pth")])[-1]
    print(f"加载最佳模型: {best_model_filename}")
    
    model.load_state_dict(torch.load(os.path.join(BEST_MODEL_SAVE_PATH, best_model_filename)))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="测试中"):
            images = images.to(DEVICE)
            outputs = model(images)
            preds = torch.sigmoid(outputs['logits']).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
            
    test_auc = roc_auc_score(all_labels, all_preds)
    test_accuracy = accuracy_score(all_labels, np.round(all_preds))
    print(f"\n--- 最终测试结果 ---")
    print(f"测试集 Accuracy: {test_accuracy:.4f}")
    print(f"测试集 AUC: {test_auc:.4f}")
    print("--- 项目评估完成 ---")


if __name__ == '__main__':
    run()