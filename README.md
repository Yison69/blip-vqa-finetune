# Visual Question Answering using BLIP pre-trained model!

This implementation applies the BLIP pre-trained model to solve the icon domain task. 
![The BLIP model for VQA task](https://i.postimg.cc/ncnxSnJw/image.png)
|  ![enter image description here](https://i.postimg.cc/1zSYsrmm/image.png)|  |
|--|--|
| How many dots are there? | 36 |

# Description
**Note: The test dataset does not have labels. I evaluated the model via Kaggle competition and got 96% in accuracy manner. Obviously, you can use a partition of the training set as a testing set.
## Create data folder
##Copy all data following the example form
//You can download data [here](https://drive.google.com/file/d/1tt6qJbOgevyPpfkylXpKYy-KaT4_aCYZ/view?usp=sharing)

##download dataset

# 进入您想存放数据的父目录 (例如项目根目录)
cd ~/autodl-tmp/blip-vqa-ft/blip-vqa-finetune  

# 创建并进入存放 COCO 图片的目录
mkdir -p Data/coco2014
cd Data/coco2014

# 下载训练集和验证集的压缩包
echo "开始下载 train2014.zip (约13.5GB)..."
aria2c -x 16 http://images.cocodataset.org/zips/train2014.zip //若不存在该包 先 pip install

echo "开始下载 val2014.zip (约6.6GB)..."
aria2c -x 16 http://images.cocodataset.org/zips/val2014.zip

# 解压 (需要较长时间)
echo "正在解压 train2014.zip..."
unzip train2014.zip
echo "正在解压 val2014.zip..."
unzip val2014.zip

# (可选) 删除压缩包
rm train2014.zip val2014.zip

# 返回上一级目录 (Data 目录)
cd ..

# 确保在 Data/ 目录下，然后创建并进入 vqav2 目录
mkdir -p vqav2
cd vqav2

# 下载问题和答案的压缩包
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Val_mscoco.zip
wget https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Val_mscoco.zip

# 下载额外的必需 json 文件
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/vqav2/answer_list.json
wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/datasets/vqav2/vqa_val_eval.json

# 解压 zip 文件
unzip \*.zip

# (可选) 删除 zip 压缩包
rm *.zip

# 返回项目根目录
cd ../..

## Install requirements.txt

    pip install -r requirements.txt

## Run finetuning code

    python finetuning_self.py / 训练好的模型保存在Model目录下

## Run prediction

    python predicting_self.py

## create web front to visualize

    python app_gradio.py

### References:

##other :
    autodl学术加速命令 ： source /etc/network_turbo

> Nguyen Van Tuan (2023). JAIST_Advanced Machine Learning_Visual_Question_Answering

