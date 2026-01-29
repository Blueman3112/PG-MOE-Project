
import torch
import open_clip
from model import PGMoE

def test_model_forward():
    print("--- 开始模型前向传播测试 ---")
    
    # 1. 实例化模型
    # 注意：这里我们假设预训练模型路径是正确的，或者让它自动下载
    # 为了测试方便，如果本地没有权重，可能会报错。
    # 我们可以尝试用一个小一点的模型或者 try-catch
    try:
        # 使用默认参数，指向本地预训练模型
        print("尝试加载本地 CLIP 模型...")
        model = PGMoE() 
    except Exception as e:
        print(f"加载模型失败: {e}")
        # 如果加载失败，尝试加载一个伪造的/随机初始化的 CLIP (如果 open_clip 支持)
        # 或者提示用户
        return

    model.eval()
    
    # 2. 创建伪造输入 [Batch, 3, 224, 224]
    dummy_image = torch.randn(2, 3, 224, 224)
    print(f"输入图像形状: {dummy_image.shape}")
    
    # 3. 运行前向传播
    try:
        # --- 前向传播测试 ---
        outputs = model(dummy_image)
        
        # 4. 检查输出
        logits = outputs['logits']
        F_s = outputs['F_s']
        F_f = outputs['F_f']
        
        print("\n--- 输出形状检查 ---")
        print(f"Logits shape: {logits.shape} (预期: [2, 1])")
        
        # 5. 检查 Hook
        if model.captured_tokens is not None:
            if model.captured_tokens.shape[1] == 257:
                print("Token 形状: [Batch, Sequence, Dim] -> 符合预期")
            else:
                print(f"Token 形状异常: {model.captured_tokens.shape}")
        
        # --- 反向传播测试 ---
        print("\n--- 开始反向传播测试 ---")
        from loss import OrthogonalLoss
        criterion = OrthogonalLoss()
        # 伪造标签
        labels = torch.tensor([0.0, 1.0]) # Float type
        
        loss, _, _ = criterion(outputs, labels)
        print(f"计算得到 Loss: {loss.item():.4f}")
        
        loss.backward()
        print("反向传播成功！梯度已计算。")
        
        # 检查参数是否有梯度
        if model.classifier.weight.grad is not None:
            print("Classifier 层已获得梯度。")
        else:
            print("警告：Classifier 层没有梯度！")
            
        if model.clip.visual.conv1.weight.grad is None:
            print("CLIP Backbone (冻结层) 无梯度 -> 符合预期。")
        else:
            print("警告：CLIP Backbone 有梯度，冻结失败！")

        print("\n--- 所有测试通过 ---")
        
    except Exception as e:
        print(f"\n前向传播发生错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    test_model_forward()
