"""
文本到图像检索 - 简化版 Demo
============================
基于您提供的示例代码，实现文本到图像的检索功能
"""

import torch
from PIL import Image
import os
import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

# ============================================================================
# 1. 加载模型
# ============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Available models:", available_models())  # ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

# 如本地模型不存在，自动从ModelScope下载模型，需要提前安装`modelscope`包
model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./', use_modelscope=True)
model.eval()

# ============================================================================
# 2. 准备图像库（从文件夹中加载多张图像）
# ============================================================================
# 图像文件夹路径（请修改为您自己的图像文件夹）
image_folder = "examples"
image_paths = []

# 收集图像文件
for filename in os.listdir(image_folder):
    if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
        image_paths.append(os.path.join(image_folder, filename))

if not image_paths:
    # 如果没有找到图像，使用示例图像
    image_paths = ["examples/pokemon.jpeg"]
    print(f"使用示例图像: {image_paths[0]}")

print(f"\n图像库包含 {len(image_paths)} 张图像")

# ============================================================================
# 3. 计算所有图像的特征
# ============================================================================
print("\n正在计算图像特征...")
image_features_list = []

with torch.no_grad():
    for img_path in image_paths:
        image = preprocess(Image.open(img_path).convert('RGB')).unsqueeze(0).to(device)
        image_features = model.encode_image(image)
        # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
        image_features /= image_features.norm(dim=-1, keepdim=True)
        image_features_list.append(image_features)

# 将所有图像特征堆叠
image_features_all = torch.cat(image_features_list, dim=0)
print(f"✓ 图像特征计算完成，特征维度: {image_features_all.shape}")

# ============================================================================
# 4. 文本查询和检索
# ============================================================================
# 文本查询（可以根据需要修改）
text_queries = ["杰尼龟", "妙蛙种子", "小火龙", "皮卡丘"]

print(f"\n文本查询: {text_queries}")

text = clip.tokenize(text_queries).to(device)

with torch.no_grad():
    text_features = model.encode_text(text)
    # 对特征进行归一化，请使用归一化后的图文特征用于下游任务
    text_features /= text_features.norm(dim=-1, keepdim=True)
    
    # 计算文本特征和所有图像特征的相似度
    # similarity: [num_queries, num_images]
    similarity = text_features @ image_features_all.T
    
    # 对每个文本查询，计算与所有图像的相似度概率
    logits_per_text = similarity  # 文本到图像的相似度
    probs = logits_per_text.softmax(dim=-1).cpu().numpy()

# ============================================================================
# 5. 显示检索结果
# ============================================================================
print("\n" + "=" * 80)
print("文本到图像检索结果:")
print("=" * 80)

for i, query in enumerate(text_queries):
    print(f"\n查询文本: \"{query}\"")
    print("-" * 80)
    
    # 获取当前查询对所有图像的相似度
    query_probs = probs[i]
    
    # 获取最相似的前3张图像
    top_k = min(3, len(image_paths))
    top_indices = query_probs.argsort()[-top_k:][::-1]
    
    for rank, idx in enumerate(top_indices, 1):
        image_name = os.path.basename(image_paths[idx])
        similarity_score = query_probs[idx]
        print(f"  Top {rank}: {image_name} (相似度: {similarity_score:.4f})")

print("\n" + "=" * 80)
print("所有查询的相似度概率矩阵:")
print("=" * 80)
print("Prob shape:", probs.shape)  # [num_queries, num_images]
print("Probs:\n", probs)

