
import re
import sys

def analyze_log(log_path):
    print(f"--- 分析日志文件: {log_path} ---")
    
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("错误: 找不到日志文件")
        return

    # 正则表达式匹配模式
    epoch_pattern = re.compile(r"Epoch (\d+)/(\d+)")
    train_loss_pattern = re.compile(r"训练损失 -> 总计: ([\d\.]+), BCE: ([\d\.]+), Orth: ([\d\.]+)")
    val_metrics_pattern = re.compile(r"验证损失 -> 总计: ([\d\.]+) \| 验证集 Acc: ([\d\.]+), AUC: ([\d\.]+)")
    time_pattern = re.compile(r"Epoch \d+ 用时: ([\d\.]+) 秒")
    test_acc_pattern = re.compile(r"测试集 Accuracy: ([\d\.]+)")
    test_auc_pattern = re.compile(r"测试集 AUC: ([\d\.]+)")

    # 存储提取的数据
    epochs_data = {}
    current_epoch = None
    
    # 测试集结果
    test_result = {}

    for line in lines:
        line = line.strip()
        
        # 匹配 Epoch 开始
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            epochs_data[current_epoch] = {}
            continue

        if current_epoch is not None:
            # 匹配训练损失
            train_match = train_loss_pattern.search(line)
            if train_match:
                epochs_data[current_epoch]['train_loss'] = float(train_match.group(1))
                epochs_data[current_epoch]['train_bce'] = float(train_match.group(2))
                epochs_data[current_epoch]['train_orth'] = float(train_match.group(3))

            # 匹配验证指标
            val_match = val_metrics_pattern.search(line)
            if val_match:
                epochs_data[current_epoch]['val_loss'] = float(val_match.group(1))
                epochs_data[current_epoch]['val_acc'] = float(val_match.group(2))
                epochs_data[current_epoch]['val_auc'] = float(val_match.group(3))

            # 匹配用时
            time_match = time_pattern.search(line)
            if time_match:
                epochs_data[current_epoch]['time'] = float(time_match.group(1))
        
        # 匹配测试集结果 (通常在最后)
        test_acc_match = test_acc_pattern.search(line)
        if test_acc_match:
            test_result['acc'] = float(test_acc_match.group(1))
            
        test_auc_match = test_auc_pattern.search(line)
        if test_auc_match:
            test_result['auc'] = float(test_auc_match.group(1))

    # --- 输出简报 ---
    print(f"{'Epoch':<6} | {'Train Loss':<10} | {'Val Loss':<10} | {'Val Acc':<8} | {'Val AUC':<8} | {'Time (s)':<8}")
    print("-" * 70)
    
    for epoch in sorted(epochs_data.keys()):
        data = epochs_data[epoch]
        t_loss = f"{data.get('train_loss', 'N/A'):.4f}" if isinstance(data.get('train_loss'), float) else "N/A"
        v_loss = f"{data.get('val_loss', 'N/A'):.4f}" if isinstance(data.get('val_loss'), float) else "N/A"
        v_acc = f"{data.get('val_acc', 'N/A'):.4f}" if isinstance(data.get('val_acc'), float) else "N/A"
        v_auc = f"{data.get('val_auc', 'N/A'):.4f}" if isinstance(data.get('val_auc'), float) else "N/A"
        time_taken = f"{data.get('time', 'N/A'):.1f}" if isinstance(data.get('time'), float) else "N/A"
        
        print(f"{epoch:<6} | {t_loss:<10} | {v_loss:<10} | {v_acc:<8} | {v_auc:<8} | {time_taken:<8}")

    print("-" * 70)
    if test_result:
        print(f"最终测试集结果 -> Accuracy: {test_result.get('acc', 'N/A')}, AUC: {test_result.get('auc', 'N/A')}")
    else:
        print("未找到最终测试集结果。")

    # --- 趋势分析 ---
    if epochs_data:
        first_epoch = min(epochs_data.keys())
        last_epoch = max(epochs_data.keys())
        
        # 检查是否过拟合：Val Loss 在后期是否上升
        val_losses = [epochs_data[e].get('val_loss') for e in sorted(epochs_data.keys()) if epochs_data[e].get('val_loss') is not None]
        if len(val_losses) > 3:
            min_val_loss_idx = val_losses.index(min(val_losses))
            if min_val_loss_idx < len(val_losses) - 3:
                print("\n[警告] 可能存在过拟合：验证集 Loss 在训练后期有所上升。")
            else:
                print("\n[正常] 验证集 Loss 趋势正常。")
                
        # 检查 AUC 提升
        aucs = [epochs_data[e].get('val_auc') for e in sorted(epochs_data.keys()) if epochs_data[e].get('val_auc') is not None]
        if aucs:
            print(f"AUC 变化: {aucs[0]:.4f} -> {max(aucs):.4f}")

if __name__ == "__main__":
    analyze_log("train0128B-log.txt")
