
import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import os
import time
from datetime import datetime
import torch.cuda.amp as amp  # [优化] 引入混合精度训练模块

from dataset import create_dataloaders
from model import PGMoE
from loss import OrthogonalLoss

def run():
    # --- 1. 超参数与配置 ---
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DATASET_PATH = "../datasets/dataset-B"
    BATCH_SIZE = 32 # [优化] 如果显存允许，使用 AMP 后可以尝试增大 Batch Size (例如 64)
    LEARNING_RATE = 1e-4
    EPOCHS = 20
    LAMBDA_ORTH = 0.05
    NUM_WORKERS = 4 
    
    BEST_MODEL_SAVE_PATH = "../results_v2" # [修改] 保存到新目录，以免覆盖
    if not os.path.exists(BEST_MODEL_SAVE_PATH):
        os.makedirs(BEST_MODEL_SAVE_PATH)
    
    print(f"使用设备: {DEVICE}")
    print("已启用: 自动混合精度训练 (AMP) & 余弦退火学习率调度器")

    # --- 2. 准备数据、模型、损失函数、优化器 ---
    train_loader, val_loader, test_loader = create_dataloaders(DATASET_PATH, BATCH_SIZE, NUM_WORKERS)
    model = PGMoE().to(DEVICE)
    criterion = OrthogonalLoss(lambda_orth=LAMBDA_ORTH)

    params_to_train = list(model.spatial_expert.parameters()) + \
                      list(model.frequency_expert.parameters()) + \
                      list(model.router.parameters()) + \
                      list(model.classifier.parameters())
    
    optimizer = torch.optim.AdamW(params_to_train, lr=LEARNING_RATE)
    
    # [优化] 学习率调度器：余弦退火，让 LR 随 Epoch 逐渐降低
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)

    # [优化] 初始化 GradScaler 用于混合精度训练
    scaler = amp.GradScaler()

    # --- 3. 训练与验证循环 ---
    best_val_auc = 0.0

    for epoch in range(EPOCHS):
        epoch_start_time = time.time()
        start_timestamp = datetime.now().strftime("[%Y-%m-%d %H:%M:%S]")
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"{start_timestamp} Epoch {epoch+1}/{EPOCHS} | LR: {current_lr:.2e}")
        
        # --- 训练阶段 ---
        model.train()
        train_loss_total, train_loss_bce, train_loss_orth = 0, 0, 0
        
        for images, labels in tqdm(train_loader, desc="训练中"):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            
            # [优化] 使用 autocast 上下文管理器
            with amp.autocast():
                outputs = model(images)
                loss, l_bce, l_orth = criterion(outputs, labels)
            
            # [优化] 使用 scaler 进行反向传播和步进
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss_total += loss.item()
            train_loss_bce += l_bce.item()
            train_loss_orth += l_orth.item()
        
        # [优化] 更新学习率
        scheduler.step()
        
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
                # 验证阶段通常不需要 autocast，但为了保持一致性也可以加
                with amp.autocast():
                    outputs = model(images)
                    loss, _, _ = criterion(outputs, labels)
                
                val_loss_total += loss.item()
                
                preds = torch.sigmoid(outputs['logits']).float().cpu().numpy() # 确保转回 float32
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

            previous_best_models = [f for f in os.listdir(BEST_MODEL_SAVE_PATH) if f.startswith("best_model") and f.endswith(".pth")]
            for old_model in previous_best_models:
                os.remove(os.path.join(BEST_MODEL_SAVE_PATH, old_model))

            torch.save(model.state_dict(), model_save_path)
            print(f"发现更优模型 (AUC: {best_val_auc:.4f})，已保存至 {model_save_name}")

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        print(f"Epoch {epoch+1} 用时: {epoch_duration:.2f} 秒")

    print("\n--- 所有训练轮次结束 ---")

    # --- 4. 最终测试阶段 ---
    print("\n开始在测试集上进行最终评估...")
    best_model_filename = sorted([f for f in os.listdir(BEST_MODEL_SAVE_PATH) if f.endswith(".pth")])[-1]
    print(f"加载最佳模型: {best_model_filename}")
    
    model.load_state_dict(torch.load(os.path.join(BEST_MODEL_SAVE_PATH, best_model_filename)))
    model.eval()
    
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="测试中"):
            images = images.to(DEVICE)
            with amp.autocast():
                outputs = model(images)
            preds = torch.sigmoid(outputs['logits']).float().cpu().numpy()
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
