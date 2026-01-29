
import gradio as gr
import torch
import os
import glob
from torchvision import transforms
from model import PGMoE

# 全局配置
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CURRENT_MODEL = None
CURRENT_MODEL_PATH = None

def list_models():
    """扫描所有可能的模型目录"""
    possible_dirs = ["../results", "../results_v2"]
    all_files = []
    
    for d in possible_dirs:
        if os.path.exists(d):
            files = glob.glob(os.path.join(d, "*.pth"))
            all_files.extend(files)
            
    # 按修改时间排序，最新的在前
    if all_files:
        all_files.sort(key=os.path.getmtime, reverse=True)
        
    return all_files

def load_model_cached(model_path):
    """加载模型并缓存，避免重复加载"""
    global CURRENT_MODEL, CURRENT_MODEL_PATH
    
    if model_path == CURRENT_MODEL_PATH and CURRENT_MODEL is not None:
        return CURRENT_MODEL
        
    print(f"Loading model from {model_path}...")
    try:
        model = PGMoE()
        checkpoint = torch.load(model_path, map_location=DEVICE)
        
        # 兼容处理
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()
        
        CURRENT_MODEL = model
        CURRENT_MODEL_PATH = model_path
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None

def inference(image, model_path):
    if image is None:
        return None, None, "请先上传图片"
        
    if not model_path:
        return None, None, "请先选择模型文件"
    
    model = load_model_cached(model_path)
    if model is None:
        return None, None, f"模型加载失败: {model_path}"
        
    # 预处理 (与训练保持一致)
    transform = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        lambda image: image.convert("RGB"),
        transforms.ToTensor(),
        transforms.Normalize(
            (0.48145466, 0.4578275, 0.40821073), 
            (0.26862954, 0.26130258, 0.27577711)
        ),
    ])
    
    try:
        image_tensor = transform(image).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            logits = outputs['logits']
            weights = outputs.get('weights', None)
            
            prob = torch.sigmoid(logits).item()
            prob_real = prob
            prob_fake = 1.0 - prob
            
            # 结果逻辑 (0=Fake, 1=Real)
            # 输出给 Gradio 的 Label 字典
            confidences = {"Real (真实)": prob_real, "Fake (伪造)": prob_fake}
            
            # 专家分析文本
            expert_analysis = ""
            if weights is not None:
                w_spatial = weights[0, 0].item()
                w_freq = weights[0, 1].item()
                expert_analysis = (
                    f"### 专家权重分析\n"
                    f"- **空间专家 (Spatial)**: {w_spatial:.2%}\n"
                    f"- **频域专家 (Frequency)**: {w_freq:.2%}\n\n"
                    f"*{'模型更关注局部纹理异常' if w_spatial > w_freq else '模型更关注频域周期性伪影'}*"
                )
            
            return confidences, expert_analysis, "检测成功"
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, None, f"推理出错: {str(e)}"

def create_demo():
    model_list = list_models()
    default_model = model_list[0] if model_list else None
    
    with gr.Blocks(title="PG-MoE 合成图像检测平台", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🕵️ PG-MoE: Synthetic Image Detection
            ### 基于物理先验混合专家的合成图像检测系统
            """
        )
        
        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="pil", label="上传待检测图片")
                model_selector = gr.Dropdown(
                    choices=model_list, 
                    value=default_model, 
                    label="选择模型权重 (.pth)",
                    info="默认已自动选择最新/最佳模型"
                )
                submit_btn = gr.Button("开始检测", variant="primary", size="lg")
                
            with gr.Column(scale=1):
                output_label = gr.Label(label="检测结果", num_top_classes=2)
                output_expert = gr.Markdown(label="专家分析")
                status_text = gr.Textbox(label="系统状态", value="就绪", interactive=False)
                
        submit_btn.click(
            fn=inference,
            inputs=[input_image, model_selector],
            outputs=[output_label, output_expert, status_text]
        )
        
    return demo

if __name__ == "__main__":
    print("正在启动 Web 服务...")
    # 启用 share=True 以允许外部访问 (如果运行在云端或 Colab)
    demo = create_demo()
    demo.launch(server_name="0.0.0.0", share=True, inbrowser=True)
