import os
import argparse # 用于接收命令行参数
from transformers import BlipProcessor, BlipForQuestionAnswering
import torch
from PIL import Image

# --- 1. 配置模型路径 ---
# 这个路径必须和 finetuning_self.py 中 model.save_pretrained() 使用的路径完全一致
MODEL_SAVE_DIR = "/root/autodl-tmp/VQA_BLIP/blip-vqa-ft/blip-vqa-finetune/Model/blip-saved-model"

def run_prediction(image_path, question):
    """
    加载微调后的模型，对给定的图片和问题进行预测。

    Args:
        image_path (str): 输入图片的路径。
        question (str): 需要模型回答的问题。

    Returns:
        str: 模型生成的答案。
    """
    # --- 2. 检查模型文件是否存在 ---
    if not os.path.exists(MODEL_SAVE_DIR):
        print(f"错误：找不到已保存的模型目录: {MODEL_SAVE_DIR}")
        print("请先确保您已经成功运行 finetuning_self.py 并保存了模型。")
        return None

    # --- 3. 加载微调后的模型和对应的处理器 ---
    print(f"从 '{MODEL_SAVE_DIR}' 加载微调后的模型和处理器...")
    try:
        # 必须从保存的目录加载，这样能保证 processor 和 model 是一致的
        processor = BlipProcessor.from_pretrained(MODEL_SAVE_DIR)
        model = BlipForQuestionAnswering.from_pretrained(MODEL_SAVE_DIR)
        print("模型和处理器加载成功！")
    except Exception as e:
        print(f"加载模型或处理器时出错: {e}")
        return None

    # --- 4. 设置运行设备 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval() # 设置为评估模式，这会关闭 dropout 等训练特有的层
    print(f"模型已移至设备: {device}，并设置为评估模式。")

    # --- 5. 准备输入数据 ---
    try:
        print(f"加载图片: {image_path}")
        raw_image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"错误：找不到图片文件: {image_path}")
        return None
    except Exception as e:
        print(f"加载图片时出错: {e}")
        return None

    print(f"处理输入 - 问题: '{question}'")
    # 使用加载的 processor 处理图片和文本
    # 注意：推理时不需要 labels
    inputs = processor(raw_image, question, return_tensors="pt").to(device)

    # --- 6. 模型推理 ---
    print("模型开始生成答案...")
    with torch.no_grad(): # 推理时不需要计算梯度
        # 使用 generate 方法生成答案 ID
        # max_length 可以根据需要调整，控制答案的最大长度
        generated_ids = model.generate(**inputs, max_length=20) 
        
    print("答案生成完毕。")

    # --- 7. 解码答案 ---
    # 使用 processor 将生成的 ID 解码回文本
    # skip_special_tokens=True 会移除像 [CLS], [SEP], [PAD] 这样的特殊标记
    answer = processor.decode(generated_ids[0], skip_special_tokens=True)
    print("答案解码完成。")

    return answer

# --- 8. 主程序入口 ---
if __name__ == "__main__":
    # 设置命令行参数解析
    parser = argparse.ArgumentParser(description="使用微调后的 BLIP 模型进行 VQA 推理")
    parser.add_argument("--image_path", type=str, required=True, help="输入图片的路径")
    parser.add_argument("--question", type=str, required=True, help="需要模型回答的问题")
    
    args = parser.parse_args()

    # 执行预测
    predicted_answer = run_prediction(args.image_path, args.question)

    # 打印结果
    if predicted_answer is not None:
        print("\n" + "="*20 + " 预测结果 " + "="*20)
        print(f"图片: {args.image_path}")
        print(f"问题: {args.question}")
        print(f"模型回答: {predicted_answer}")
        print("="*50)