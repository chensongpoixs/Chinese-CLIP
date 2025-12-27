"""
文本到图像检索 Demo
===================
根据文本描述从图像库中检索出最匹配的图像

使用方法：
1. 准备一个图像文件夹（包含多张图片）
2. 运行此脚本
3. 输入文本查询，查看检索结果
"""

import torch
from PIL import Image
import os
from pathlib import Path
import numpy as np

import cn_clip.clip as clip
from cn_clip.clip import load_from_name, available_models

# ============================================================================
# 步骤1: 加载模型
# ============================================================================
print("=" * 80)
print("文本到图像检索 Demo")
print("=" * 80)

# 检查可用模型
print("\n可用模型:", available_models())
# 输出: ['ViT-B-16', 'ViT-L-14', 'ViT-L-14-336', 'ViT-H-14', 'RN50']

# 设置设备
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\n使用设备: {device}")

# 加载模型和预处理函数
# 如果本地模型不存在，会自动从ModelScope下载（需要提前安装`modelscope`包）
model_name = "ViT-B-16"  # 可以选择其他模型，如 'ViT-L-14', 'ViT-H-14' 等
print(f"\n正在加载模型: {model_name}...")
model, preprocess = load_from_name(
    model_name, 
    device=device, 
    download_root='./', 
    use_modelscope=True
)
model.eval()  # 设置为评估模式
print("✓ 模型加载完成")


# ============================================================================
# 步骤2: 准备图像库
# ============================================================================
# 图像文件夹路径（请修改为您自己的图像文件夹路径）
image_folder = "examples"  # 示例：使用examples文件夹中的图片

# 支持的图像格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.gif', '.webp'}

# 获取图像文件夹中的所有图像文件
image_paths = []
if os.path.exists(image_folder):
    for ext in IMAGE_EXTENSIONS:
        image_paths.extend(Path(image_folder).glob(f"*{ext}"))
        image_paths.extend(Path(image_folder).glob(f"*{ext.upper()}"))
    image_paths = [str(p) for p in image_paths]
else:
    print(f"\n警告: 图像文件夹 '{image_folder}' 不存在，将使用单个示例图像")
    # 如果文件夹不存在，使用示例图像
    if os.path.exists("examples/pokemon.jpeg"):
        image_paths = ["examples/pokemon.jpeg"]

if not image_paths:
    print("\n错误: 未找到任何图像文件！请检查图像文件夹路径。")
    exit(1)

print(f"\n找到 {len(image_paths)} 张图像:")
for i, path in enumerate(image_paths[:5], 1):  # 只显示前5个
    print(f"  {i}. {os.path.basename(path)}")
if len(image_paths) > 5:
    print(f"  ... 还有 {len(image_paths) - 5} 张图像")


# ============================================================================
# 步骤3: 计算图像库中所有图像的特征
# ============================================================================
print("\n" + "=" * 80)
print("步骤3: 计算图像库特征")
print("=" * 80)

image_features_list = []
valid_image_paths = []

with torch.no_grad():
    for i, img_path in enumerate(image_paths):
        try:
            # 加载并预处理图像
            image = Image.open(img_path).convert('RGB')
            image_tensor = preprocess(image).unsqueeze(0).to(device)
            
            # 计算图像特征
            image_features = model.encode_image(image_tensor)
            
            # 归一化特征（重要：用于计算余弦相似度）
            image_features /= image_features.norm(dim=-1, keepdim=True)
            
            image_features_list.append(image_features.cpu())
            valid_image_paths.append(img_path)
            
            if (i + 1) % 10 == 0:
                print(f"  已处理 {i + 1}/{len(image_paths)} 张图像...")
        except Exception as e:
            print(f"  警告: 无法处理图像 {img_path}: {e}")
            continue

if not image_features_list:
    print("\n错误: 没有成功处理任何图像！")
    exit(1)

# 将所有图像特征堆叠成一个张量
image_features_all = torch.cat(image_features_list, dim=0)
print(f"\n✓ 成功处理 {len(valid_image_paths)} 张图像，特征维度: {image_features_all.shape}")


# ============================================================================
# 步骤4: 文本查询和检索
# ============================================================================
print("\n" + "=" * 80)
print("步骤4: 文本查询和检索")
print("=" * 80)

# 示例文本查询列表
example_queries = [
    "杰尼龟",
    "妙蛙种子", 
    "小火龙",
    "皮卡丘",
    "一只可爱的小动物",
    "卡通角色",
]

def text_to_image_retrieval(text_queries, top_k=5):
    """
    文本到图像检索函数
    
    参数:
        text_queries: 文本查询列表（可以是单个字符串或字符串列表）
        top_k: 返回最相似的前k张图像（默认5张）
    
    返回:
        results: 检索结果列表，每个结果包含 (图像路径, 相似度分数, 排名)
    """
    # 如果是单个字符串，转换为列表
    if isinstance(text_queries, str):
        text_queries = [text_queries]
    
    # 将文本转换为token
    text_tokens = clip.tokenize(text_queries).to(device)
    
    with torch.no_grad():
        # 计算文本特征
        text_features = model.encode_text(text_tokens)
        
        # 归一化特征（重要：用于计算余弦相似度）
        text_features /= text_features.norm(dim=-1, keepdim=True)
        
        # 计算文本特征和所有图像特征的相似度
        # image_features_all: [num_images, feature_dim]
        # text_features: [num_queries, feature_dim]
        # similarity: [num_queries, num_images]
        similarity = text_features @ image_features_all.T
        
        # 对于每个文本查询，获取最相似的前top_k张图像
        results = []
        for i, query in enumerate(text_queries):
            # 获取相似度分数和索引
            scores, indices = similarity[i].topk(top_k)
            
            query_results = []
            for rank, (score, idx) in enumerate(zip(scores, indices), 1):
                query_results.append({
                    'rank': rank,
                    'image_path': valid_image_paths[idx],
                    'image_name': os.path.basename(valid_image_paths[idx]),
                    'similarity_score': float(score.item())
                })
            
            results.append({
                'query': query,
                'results': query_results
            })
    
    return results


# ============================================================================
# 步骤5: 运行示例查询
# ============================================================================
print("\n正在执行文本到图像检索...")

# 对所有示例查询进行检索
all_results = text_to_image_retrieval(example_queries, top_k=3)

# 打印结果
print("\n" + "=" * 80)
print("检索结果:")
print("=" * 80)

for query_result in all_results:
    query = query_result['query']
    results = query_result['results']
    
    print(f"\n查询文本: \"{query}\"")
    print("-" * 80)
    for result in results:
        print(f"  Top {result['rank']}: {result['image_name']}")
        print(f"    相似度分数: {result['similarity_score']:.4f}")
        print(f"    图像路径: {result['image_path']}")


# ============================================================================
# 步骤6: 交互式查询（可选）
# ============================================================================
print("\n" + "=" * 80)
print("交互式查询模式")
print("=" * 80)
print("提示: 输入 'quit' 或 'exit' 退出")

while True:
    try:
        user_query = input("\n请输入查询文本: ").strip()
        
        if user_query.lower() in ['quit', 'exit', '退出', 'q']:
            print("退出交互式查询模式")
            break
        
        if not user_query:
            print("请输入有效的查询文本")
            continue
        
        # 执行检索
        results = text_to_image_retrieval(user_query, top_k=5)
        
        # 显示结果
        if results and results[0]['results']:
            query_result = results[0]
            print(f"\n查询: \"{query_result['query']}\"")
            print("-" * 80)
            for result in query_result['results']:
                print(f"  Top {result['rank']}: {result['image_name']} "
                      f"(相似度: {result['similarity_score']:.4f})")
        else:
            print("未找到匹配结果")
            
    except KeyboardInterrupt:
        print("\n\n退出交互式查询模式")
        break
    except Exception as e:
        print(f"错误: {e}")
        continue

print("\n" + "=" * 80)
print("Demo 结束")
print("=" * 80)

