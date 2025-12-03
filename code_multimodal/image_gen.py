
import torch
import datetime
from pathlib import Path
from typing import List
from diffusers import StableDiffusionPipeline
from .config import TEXT2IMG_MODEL

def build_image_prompts(query: str, contexts: List[dict]) -> List[str]:
    """构建图片生成提示"""
    prompts = []
    base_instruction = "digital art, highly detailed, cinematic lighting"
    
    for ctx in contexts[:2]:  # 最多生成2张图片
        meta = ctx["meta"]
        prompt = (
            f"{base_instruction}. Scene from {meta['source']}: "
            f"{ctx['text'][:100]}... Focus on key elements from the text."
        )
        prompts.append(prompt)
    
    return prompts[:2]


def generate_images(prompts: List[str], output_dir: Path) -> List[Path]:
    """生成图片"""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    image_paths = []
    
    if not prompts:
        return image_paths
        
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            TEXT2IMG_MODEL,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            safety_checker=None,
        )
        pipe = pipe.to(device)
        if device == "cuda":
            pipe.enable_attention_slicing()
            
    except Exception as e:
        print(f"加载文生图模型失败: {e}")
        return image_paths

    for idx, prompt in enumerate(prompts, start=1):
        try:
            image = pipe(prompt, num_inference_steps=20).images[0]
            img_path = output_dir / f"picture_{idx}.png"
            image.save(img_path)
            image_paths.append(img_path)
        except Exception as e:
            print(f"生成图片失败: {e}")
            
    return image_paths

