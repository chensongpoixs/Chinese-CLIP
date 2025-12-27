"""
文字生成图片 Demo - 使用 Stable Diffusion
==========================================
注意：这个示例需要安装 diffusers 库和 Stable Diffusion 模型

安装命令：
pip install diffusers transformers accelerate torch torchvision
"""

import torch
from PIL import Image
import os

# ============================================================================
# 方案1: 使用 diffusers 库（推荐）
# ============================================================================
def generate_image_with_diffusers(prompt, negative_prompt="", num_inference_steps=50, guidance_scale=7.5):
    """
    使用 diffusers 库生成图片
    
    参数:
        prompt: 文本提示词（中文或英文）
        negative_prompt: 负面提示词（不希望出现的内容）
        num_inference_steps: 推理步数（越多质量越好，但速度越慢）
        guidance_scale: 引导强度（越大越遵循提示词）
    
    返回:
        PIL.Image: 生成的图片
    """
    try:
        from diffusers import StableDiffusionPipeline
        
        # 检查是否有GPU
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"使用设备: {device}")
        
        # 加载模型（首次运行会自动下载模型，约4-7GB）
        print("正在加载 Stable Diffusion 模型...")
        print("提示: 首次运行需要下载模型，请耐心等待...")
        
        # 使用中文 Stable Diffusion 模型（如果有）
        # 如果没有中文模型，可以使用英文模型，但中文提示词效果可能不佳
        model_id = "runwayml/stable-diffusion-v1-5"  # 英文模型
        # model_id = "IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1"  # 中文模型（如果可用）
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,  # 禁用安全检查器（可选）
            requires_safety_checker=False
        )
        pipe = pipe.to(device)
        
        # 如果显存不足，可以启用内存优化
        if device == "cuda":
            pipe.enable_attention_slicing()  # 降低显存占用
            # pipe.enable_xformers_memory_efficient_attention()  # 需要安装 xformers
        
        print("✓ 模型加载完成")
        
        # 生成图片
        print(f"\n正在生成图片...")
        print(f"提示词: {prompt}")
        
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt else None,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=512,
                width=512
            ).images[0]
        
        print("✓ 图片生成完成")
        return image
        
    except ImportError:
        print("错误: 未安装 diffusers 库")
        print("请运行: pip install diffusers transformers accelerate")
        return None
    except Exception as e:
        print(f"生成图片时出错: {e}")
        return None


# ============================================================================
# 方案2: 使用 Stable Diffusion WebUI API（如果已安装 WebUI）
# ============================================================================
def generate_image_with_webui_api(prompt, api_url="http://127.0.0.1:7860"):
    """
    通过 Stable Diffusion WebUI 的 API 生成图片
    
    需要先启动 WebUI:
    cd stable-diffusion-webui
    python launch.py --api
    
    参数:
        prompt: 文本提示词
        api_url: WebUI API 地址
    
    返回:
        PIL.Image: 生成的图片
    """
    try:
        import requests
        import base64
        from io import BytesIO
        
        # API 请求
        payload = {
            "prompt": prompt,
            "negative_prompt": "",
            "steps": 20,
            "width": 512,
            "height": 512
        }
        
        response = requests.post(f"{api_url}/sdapi/v1/txt2img", json=payload)
        response.raise_for_status()
        
        result = response.json()
        image_base64 = result['images'][0]
        
        # 解码图片
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        
        return image
        
    except ImportError:
        print("错误: 未安装 requests 库")
        print("请运行: pip install requests")
        return None
    except Exception as e:
        print(f"通过 API 生成图片时出错: {e}")
        print("请确保 Stable Diffusion WebUI 已启动并启用了 API")
        return None


# ============================================================================
# 方案3: 结合 CLIP 评估生成结果
# ============================================================================
def evaluate_with_clip(image, text_prompt, clip_model=None, preprocess=None):
    """
    使用 CLIP 评估生成的图片与文本提示词的匹配度
    
    参数:
        image: PIL.Image 图片
        text_prompt: 文本提示词
        clip_model: CLIP 模型（如果已加载）
        preprocess: CLIP 预处理函数（如果已加载）
    
    返回:
        float: 相似度分数（0-1之间，越高越匹配）
    """
    try:
        import cn_clip.clip as clip
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 如果没有传入模型，则加载模型
        if clip_model is None or preprocess is None:
            from cn_clip.clip import load_from_name
            clip_model, preprocess = load_from_name("ViT-B-16", device=device, download_root='./', use_modelscope=True)
            clip_model.eval()
        
        # 处理图像和文本
        image_tensor = preprocess(image).unsqueeze(0).to(device)
        text_tokens = clip.tokenize([text_prompt]).to(device)
        
        with torch.no_grad():
            # 提取特征
            image_features = clip_model.encode_image(image_tensor)
            text_features = clip_model.encode_text(text_tokens)
            
            # 归一化
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            
            # 计算相似度
            similarity = (image_features @ text_features.T).item()
        
        return similarity
        
    except Exception as e:
        print(f"CLIP 评估时出错: {e}")
        return None


# ============================================================================
# 主程序示例
# ============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("文字生成图片 Demo")
    print("=" * 80)
    print("\n注意: 这个示例需要安装 Stable Diffusion 相关库")
    print("如果未安装，请先安装: pip install diffusers transformers accelerate")
    print("=" * 80)
    
    # 文本提示词
    prompt = "一只可爱的小猫，坐在窗台上，阳光洒在它身上，高清，细节丰富"
    
    print(f"\n提示词: {prompt}")
    print("\n请选择生成方式:")
    print("1. 使用 diffusers 库（需要下载模型，约4-7GB）")
    print("2. 使用 Stable Diffusion WebUI API（需要先启动 WebUI）")
    print("3. 仅演示 CLIP 评估功能（需要先生成图片）")
    
    choice = input("\n请选择 (1/2/3): ").strip()
    
    if choice == "1":
        # 使用 diffusers 生成
        image = generate_image_with_diffusers(prompt)
        if image:
            # 保存图片
            output_path = "generated_image.png"
            image.save(output_path)
            print(f"\n✓ 图片已保存到: {output_path}")
            
            # 使用 CLIP 评估
            print("\n使用 CLIP 评估生成结果...")
            similarity = evaluate_with_clip(image, prompt)
            if similarity:
                print(f"CLIP 相似度分数: {similarity:.4f} (1.0为完全匹配)")
    
    elif choice == "2":
        # 使用 WebUI API
        image = generate_image_with_webui_api(prompt)
        if image:
            output_path = "generated_image.png"
            image.save(output_path)
            print(f"\n✓ 图片已保存到: {output_path}")
    
    elif choice == "3":
        # 仅演示 CLIP 评估
        image_path = input("请输入图片路径: ").strip()
        if os.path.exists(image_path):
            image = Image.open(image_path)
            similarity = evaluate_with_clip(image, prompt)
            if similarity:
                print(f"\nCLIP 相似度分数: {similarity:.4f} (1.0为完全匹配)")
        else:
            print("图片文件不存在")
    
    else:
        print("无效的选择")

