import os
import json # 需要导入 json 库来读取标注文件
import requests
from transformers import BlipProcessor, BlipForQuestionAnswering
# 不再需要 datasets 库了
# from datasets import load_dataset
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset # 需要导入 Dataset
from tqdm import tqdm
import pickle

# --- 1. 定义数据加载函数 ---
def load_vqa_data(question_file, annotation_file):
    """加载 VQAv2 问题和标注文件，并合并它们"""
    print(f"Loading questions from: {question_file}")
    with open(question_file, 'r') as f:
        questions_data = json.load(f)['questions']
    
    print(f"Loading annotations from: {annotation_file}")
    with open(annotation_file, 'r') as f:
        annotations_data = json.load(f)['annotations']

    # 创建 question_id 到 question 文本的映射，方便查找
    questions_map = {q['question_id']: q for q in questions_data}
    
    # 合并数据
    merged_data = []
    missing_questions = 0
    for ann in annotations_data:
        q_id = ann['question_id']
        if q_id in questions_map:
            # 合并问题和标注信息
            merged_entry = {
                'question': questions_map[q_id]['question'],
                'answer': ann['multiple_choice_answer'], # 使用最常见的答案
                'image_id': ann['image_id'],
                # 可以根据需要添加其他字段，比如 ann['answers']
            }
            merged_data.append(merged_entry)
        else:
            missing_questions += 1
            
    if missing_questions > 0:
         print(f"Warning: Skipped {missing_questions} annotations due to missing question_id.")
         
    return merged_data

# --- 2. 定义 VQADataset 类 (修改版) ---
class VQADataset(Dataset): # 继承自 torch.utils.data.Dataset
    """适用于标准 VQAv2 和 COCO 的数据集类"""

    def __init__(self, data_list, processor, coco_root, split):
        self.data_list = data_list
        self.processor = processor
        self.coco_root = coco_root
        self.split = split # 'train' or 'val'

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        question = item['question']
        answer = item['answer']
        image_id = item['image_id']

        # 构建正确的 COCO 图片路径 (注意补零和 .jpg 后缀)
        image_filename = f"COCO_{self.split}2014_{image_id:012d}.jpg"
        image_path = os.path.join(self.coco_root, f"{self.split}2014", image_filename)

        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            # 如果找不到图片，打印警告并返回一个空的占位符或跳过
            # （更好的做法是在加载数据时就过滤掉找不到图片的条目）
            print(f"Warning: Image not found at {image_path}. Returning None.")
            # 为了让 DataLoader 继续工作，我们可能需要返回一些东西，但这会影响训练
            # 这里简单返回 None，DataLoader 的 collate_fn 需要处理 None 值
            # 或者直接引发错误 raise FileNotFoundError(f"Image not found: {image_path}")
            # 为了简单起见，我们先假设图片都存在
            # 如果频繁报错，需要回来处理这里的逻辑
            image = Image.new('RGB', (224, 224)) # 返回一个空白图片以避免崩溃

        text = question
        
        # 使用 processor 处理图片和文本
        encoding = self.processor(image, text, padding="max_length", truncation=True, return_tensors="pt")
        # 对答案进行编码作为标签
        labels = self.processor.tokenizer.encode(
            answer, max_length=8, padding='max_length', truncation=True, return_tensors='pt'
        )
        # 将标签添加到编码结果中
        encoding["labels"] = labels
        # 移除批次维度，因为 DataLoader 会自动添加
        for k,v in encoding.items():
            encoding[k] = v.squeeze()
        return encoding

# --- 3. 设置路径并加载数据 ---
data_root = "/root/autodl-tmp/VQA_BLIP/blip-vqa-ft/blip-vqa-finetune/Data"
coco_root = os.path.join(data_root, "coco2014")
vqav2_root = os.path.join(data_root, "vqav2")

# 确保必要的标注文件存在
train_q_file = os.path.join(vqav2_root, "v2_OpenEnded_mscoco_train2014_questions.json")
train_a_file = os.path.join(vqav2_root, "v2_mscoco_train2014_annotations.json")
val_q_file = os.path.join(vqav2_root, "v2_OpenEnded_mscoco_val2014_questions.json")
val_a_file = os.path.join(vqav2_root, "v2_mscoco_val2014_annotations.json")

# 检查文件是否存在，如果不存在则报错
required_files = [train_q_file, train_a_file, val_q_file, val_a_file]
for f_path in required_files:
    if not os.path.exists(f_path):
        raise FileNotFoundError(f"错误：必需的标注文件未找到: {f_path}. 请确保已正确下载并放置 VQAv2 数据。")

# 加载并合并训练集和验证集数据
training_data = load_vqa_data(train_q_file, train_a_file)
validation_data = load_vqa_data(val_q_file, val_a_file)

print(f"Training items loaded: {len(training_data)}")
print(f"Validation items loaded: {len(validation_data)}")

# --- 4. 初始化模型和处理器 ---
print("Loading model and processor...")
model = BlipForQuestionAnswering.from_pretrained("Salesforce/blip-vqa-base")
processor = BlipProcessor.from_pretrained("Salesforce/blip-vqa-base")
print("Model and processor loaded.")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print(f"Model moved to device: {device}")

torch.cuda.empty_cache()
torch.manual_seed(42)

# --- 5. 创建 Dataset 和 DataLoader ---
print("Creating datasets...")
train_dataset = VQADataset(data_list=training_data,
                           processor=processor,
                           coco_root=coco_root,
                           split='train')
valid_dataset = VQADataset(data_list=validation_data,
                           processor=processor,
                           coco_root=coco_root,
                           split='val')
print("Datasets created.")

# 定义 collate_fn 来处理可能的 None 值 (如果图片加载失败)
# 注意：更健壮的做法是在 load_vqa_data 时就过滤掉图片不存在的数据
# def collate_fn(batch):
#     batch = [item for item in batch if item is not None] # 过滤掉 None
#     if not batch:
#         return None # 如果整个批次都是 None
#     # 使用默认的 collate 函数处理剩余的有效数据
#     return torch.utils.data.dataloader.default_collate(batch)
    
print("Creating dataloaders...")
batch_size = 16 # 根据显存调整，3090/4090 可以尝试 32, 64 或更高
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4) # 打开 shuffle，设置 num_workers
valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
print("Dataloaders created.")

# --- 6. 设置优化器、调度器和训练参数 ---
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5) # 学习率可以调整
# scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9, last_epoch=-1, verbose=False) # 可以选择不同的 scheduler

num_epochs = 5 # 先训练几轮看看效果
patience = 3 # 早停轮数
min_eval_loss = float("inf")
early_stopping_hook = 0
tracking_information = []
scaler = torch.cuda.amp.GradScaler() # 用于混合精度训练

model_save_dir = "/root/autodl-tmp/VQA_BLIP/blip-vqa-ft/blip-vqa-finetune/Model/blip-saved-model"
os.makedirs(model_save_dir, exist_ok=True) # 确保目录存在
tracking_file = "tracking_information.pkl"

# --- 7. 训练循环 ---
print("Starting training loop...")
for epoch in range(num_epochs):
    epoch_loss = 0
    model.train()
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
    
    # 训练步骤
    train_progress_bar = tqdm(train_dataloader, desc=f'Training Epoch {epoch+1}')
    for batch in train_progress_bar:
        # 将批次数据移动到 GPU
        input_ids = batch.pop('input_ids').to(device)
        pixel_values = batch.pop('pixel_values').to(device)
        attention_mask = batch.pop('attention_mask').to(device) # 注意这里用了 attention_mask
        labels = batch.pop('labels').to(device)
        
        optimizer.zero_grad() # 梯度清零
        
        # 使用混合精度进行前向传播
        with torch.amp.autocast(device_type=str(device), dtype=torch.float16):
            outputs = model(input_ids=input_ids,
                            pixel_values=pixel_values,
                            attention_mask=attention_mask, # 传入 attention_mask
                            labels=labels)
            loss = outputs.loss
        
        # 检查 loss 是否有效 (防止 NaN)
        if torch.isnan(loss):
            print("Warning: Loss is NaN, skipping batch.")
            continue
            
        epoch_loss += loss.item()
        
        # 使用 GradScaler 进行反向传播和优化器步骤
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # 更新进度条显示
        train_progress_bar.set_postfix({'loss': loss.item()})
    
    # 验证步骤
    model.eval()
    eval_loss = 0
    val_progress_bar = tqdm(valid_dataloader, desc=f'Validating Epoch {epoch+1}')
    with torch.no_grad(): # 验证时不需要计算梯度
        for batch in val_progress_bar:
            input_ids = batch.pop('input_ids').to(device)
            pixel_values = batch.pop('pixel_values').to(device)
            attention_mask = batch.pop('attention_mask').to(device)
            labels = batch.pop('labels').to(device)

            # 验证时也使用混合精度
            with torch.amp.autocast(device_type=str(device), dtype=torch.float16):
                outputs = model(input_ids=input_ids,
                                pixel_values=pixel_values,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss
            
            if not torch.isnan(loss):
                 eval_loss += loss.item()
            
            val_progress_bar.set_postfix({'loss': loss.item()})

    # 计算平均损失
    avg_epoch_loss = epoch_loss / len(train_dataloader) if len(train_dataloader) > 0 else 0
    avg_eval_loss = eval_loss / len(valid_dataloader) if len(valid_dataloader) > 0 else float('inf') # 如果验证集为空，损失设为无穷大
    
    current_lr = optimizer.param_groups[0]["lr"]
    tracking_information.append((avg_epoch_loss, avg_eval_loss, current_lr))
    
    print(f"\nEpoch {epoch+1} Summary:")
    print(f"  Training Loss: {avg_epoch_loss:.4f}")
    print(f"  Validation Loss: {avg_eval_loss:.4f}")
    print(f"  Current Learning Rate: {current_lr}")
    
    # scheduler.step() # 如果使用 ExponentialLR 或其他基于 epoch 的 scheduler，在这里 step

    # 早停逻辑和模型保存
    if avg_eval_loss < min_eval_loss:
        print(f"Validation loss decreased ({min_eval_loss:.4f} --> {avg_eval_loss:.4f}). Saving model...")
        model.save_pretrained(model_save_dir, safe_serialization=True) # 使用 safe serialization
        processor.save_pretrained(model_save_dir) # 保存 processor 配置也很重要
        print(f"Model saved to {model_save_dir}")
        min_eval_loss = avg_eval_loss
        early_stopping_hook = 0
    else:
        early_stopping_hook += 1
        print(f"Validation loss did not improve for {early_stopping_hook} epoch(s).")
        if early_stopping_hook >= patience:
            print(f"Early stopping triggered after {epoch+1} epochs.")
            break # 触发早停，跳出训练循环

# --- 8. 保存训练追踪信息 ---
with open(tracking_file, "wb") as f:
    pickle.dump(tracking_information, f)
print(f"Tracking information saved to {tracking_file}")
print("The finetuning process has done!")