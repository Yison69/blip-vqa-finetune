import gradio as gr
from PIL import Image
import torch
from transformers import BlipProcessor, BlipForQuestionAnswering
import os 

# --- 1. 定义模型加载和预测的函数 ---
# (这部分逻辑与 prediction.py 类似，但封装在一个函数里)

MODEL_SAVE_DIR = "/root/autodl-tmp/VQA_BLIP/blip-vqa-ft/blip-vqa-finetune/Model/blip-saved-model" # 确保路径正确
processor = None
model = None
device = None

def load_model_once():
    """只在应用启动时加载一次模型"""
    global processor, model, device
    if model is None:
        print(f"正在从 '{MODEL_SAVE_DIR}' 加载模型和处理器...")
        if not os.path.exists(MODEL_SAVE_DIR):
             raise FileNotFoundError(f"错误：找不到模型目录 {MODEL_SAVE_DIR}")
        processor = BlipProcessor.from_pretrained(MODEL_SAVE_DIR)
        model = BlipForQuestionAnswering.from_pretrained(MODEL_SAVE_DIR)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        print(f"模型加载完毕并移至 {device}")
    return processor, model, device

def predict_vqa(image, question):
    """接收 PIL 图片和问题字符串，返回答案字符串"""
    processor, model, device = load_model_once() # 获取加载好的模型

    if image is None or not question:
        return "请上传图片并输入问题"

    try:
        # 1. 预处理输入
        inputs = processor(image, question, return_tensors="pt").to(device)

        # 2. 模型推理
        print("模型开始生成答案...")
        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_length=20)
        print("答案生成完毕。")

        # 3. 解码答案
        answer = processor.decode(generated_ids[0], skip_special_tokens=True)
        print(f"解码答案: {answer}")
        return answer
    except Exception as e:
        print(f"预测时出错: {e}")
        return f"预测出错: {e}"

# --- 2. 创建 Gradio 界面 ---
iface = gr.Interface(
    fn=predict_vqa, # 指定处理函数
    inputs=[
        gr.Image(type="pil", label="上传图片"), # 图片上传组件，输入类型为 PIL Image
        gr.Textbox(label="输入问题")             # 文本输入框
    ],
    outputs=gr.Textbox(label="模型回答"),        # 文本输出框
    title="BLIP 视觉问答 (VQA)",
    description="上传一张图片，输入一个关于图片的问题，模型会尝试回答。",
    examples=[
        ["./Data/coco2014/val2014/COCO_val2014_000000000042.jpg", "What is on the plate?"],
        # 您可以添加更多示例图片路径和问题
    ]
)

# --- 3. 启动应用 ---
if __name__ == "__main__":
    load_model_once() # 应用启动时先预加载模型
    print("正在尝试启动 Gradio 界面...") # <--- 添加这一行
    iface.launch() # 启动 Gradio 服务