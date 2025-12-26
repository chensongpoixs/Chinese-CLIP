import torch 
from PIL import Image


# root@b489b4b71208:~/Chinese-CLIP# python --version
# Python 3.10.12
# root@b489b4b71208:~/Chinese-CLIP# python demo.py
# 2.3.0+cu121

import torch
print(torch.__version__)


device = "cuda" if torch.cuda.is_available() else "cpu";


print('device:', device);



import torch # 如果pytorch安装成功即可导入
print(torch.cuda.is_available()) # 查看CUDA是否可用，如果True表示可以使用
print(torch.cuda.device_count()) # 查看可用的CUDA数量，0表示有一个
print(torch.version.cuda) # 查看CUDA的版本号




import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

# Available models: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']


#device = "cpu";
# 如本地模型不存在，自动从ModelScope下载模型，需要提前安装`modelscope`包
#model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./', use_modelscope=True)

# 加载自己训练的模型
# 方法1: 使用 load_from_name 加载检查点文件（推荐）
# 需要指定检查点文件路径、视觉模型名称、文本模型名称和输入分辨率
# checkpoint_path = "datapath/experiments/muge_finetune_vit-b-16_roberta-base_bs48_1gpu/checkpoints/epoch_latest.pt"  # 或 epoch1.pt
# print(f"Loading checkpoint from: {checkpoint_path}")
# model, preprocess = load_from_name(
#     checkpoint_path,  # 检查点文件路径
#     device=device,
#     vision_model_name="ViT-B-16",  # 视觉模型名称（必须与训练时使用的模型一致）
#     text_model_name="RoBERTa-wwm-ext-base-chinese",  # 文本模型名称（必须与训练时使用的模型一致）
#     input_resolution=224  # 输入图像分辨率（ViT-B-16 使用 224）
# )
# print("Model loaded successfully!")


# model.eval()
# image = preprocess(Image.open("examples/pokemon.jpeg")).unsqueeze(0).to(device)
# text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)

# with torch.no_grad():
#     image_features = model.encode_image(image)
#     text_features = model.encode_text(text)
#     # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
#     image_features /= image_features.norm(dim=-1, keepdim=True) 
#     text_features /= text_features.norm(dim=-1, keepdim=True)    

#     logits_per_image, logits_per_text = model.get_similarity(image, text)
#     probs = logits_per_image.softmax(dim=-1).cpu().numpy()

# print("Label probs:", probs)  # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]






# 官方模型测试
def  test_clip():
    print("test clip ...");
    device = "cuda" if torch.cuda.is_available() else "cpu";
    print('device:', device);
    print("Available models:", available_models());
    # 如本地模型不存在，自动从ModelScope下载模型，需要提前安装`modelscope`包
    model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./', use_modelscope=True);
    print("Model loaded successfully!");
    
    model.eval()
    image = preprocess(Image.open("examples/pokemon.jpeg")).unsqueeze(0).to(device)
    text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device)

    with torch.no_grad():
        image_features = model.encode_image(image)
        text_features = model.encode_text(text)
        # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
        image_features /= image_features.norm(dim=-1, keepdim=True) 
        text_features /= text_features.norm(dim=-1, keepdim=True)    

        logits_per_image, logits_per_text = model.get_similarity(image, text)
        probs = logits_per_image.softmax(dim=-1).cpu().numpy()

    print("Label probs:", probs)  # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]




# 自己训练模型测试
def test_rtx5080_clip():
    print("test_rtx5080_clip ...");
    device = "cuda" if torch.cuda.is_available() else "cpu";
    print('device:', device);
    # 加载自己训练的模型
    # 方法1: 使用 load_from_name 加载检查点文件（推荐）
    # 需要指定检查点文件路径、视觉模型名称、文本模型名称和输入分辨率
    checkpoint_path = "datapath/experiments/muge_finetune_vit-b-16_roberta-base_bs48_1gpu/checkpoints/epoch_latest.pt"  # 或 epoch1.pt
    print(f"Loading checkpoint from: {checkpoint_path}");
    model, preprocess = load_from_name(
        checkpoint_path,  # 检查点文件路径
        device=device,
        vision_model_name="ViT-B-16",  # 视觉模型名称（必须与训练时使用的模型一致）
        text_model_name="RoBERTa-wwm-ext-base-chinese",  # 文本模型名称（必须与训练时使用的模型一致）
        input_resolution=224  # 输入图像分辨率（ViT-B-16 使用 224）
    );
    print("Model loaded successfully!");


    model.eval();
    image = preprocess(Image.open("examples/pokemon.jpeg")).unsqueeze(0).to(device);
    text = clip.tokenize(["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]).to(device);

    with torch.no_grad():
        image_features = model.encode_image(image);
        text_features = model.encode_text(text);
        # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
        image_features /= image_features.norm(dim=-1, keepdim=True);
        text_features /= text_features.norm(dim=-1, keepdim=True); 

        logits_per_image, logits_per_text = model.get_similarity(image, text);
        probs = logits_per_image.softmax(dim=-1).cpu().numpy();

    print("Label probs:", probs)  # [[1.268734e-03 5.436878e-02 6.795761e-04 9.436829e-01]]


# ============================================================================
# 程序入口
# ============================================================================
if __name__ == "__main__":
    test_clip();
    print("========================================================================");
    test_rtx5080_clip();






