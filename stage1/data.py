import os
import torch
import random
import json
import gc
from diffusers import StableDiffusion3Pipeline
from tqdm import tqdm

# ================= 配置区域 =================
# 1. 本地模型绝对路径 (请确认路径无误)
MODEL_PATH = "/home/HDD/cxy/.cache/huggingface/hub/models--stabilityai--stable-diffusion-3-medium-diffusers"

# 2. 数据集保存路径
OUTPUT_DIR = "/home/HDD/cxy/SD3/dataset1024"
METADATA_FILE = "metadata.jsonl"

# 3. 生成参数
NUM_IMAGES = 10000
BATCH_SIZE = 1
RESOLUTION = 1024


# ===========================================

# --- 智能路径修复逻辑 (保持不变) ---
def get_actual_model_path(base_path):
    if os.path.exists(os.path.join(base_path, "model_index.json")):
        return base_path
    snapshots_dir = os.path.join(base_path, "snapshots")
    if os.path.exists(snapshots_dir):
        subfolders = [f for f in os.listdir(snapshots_dir) if os.path.isdir(os.path.join(snapshots_dir, f))]
        if subfolders:
            return os.path.join(snapshots_dir, subfolders[0])
    return base_path


# ================= PRO 版 Prompt 生成引擎 =================

def gen_photography_prompt():
    """模式1: 真实摄影 (覆盖人像、风景、微距)"""
    subjects = ["a portrait of an old man", "a wide angle shot of a cyber city", "macro photography of a dew drop",
                "a street photographer capturing a busy market", "an aerial view of a coastline",
                "a fashion model posing",
                "a lonely tree in a snow field", "a busy kitchen with chefs cooking", "a race car speeding on a track"]
    lighting = ["natural sunlight", "studio softbox lighting", "neon signs reflection", "harsh flash photography",
                "golden hour"]
    camera = ["shot on Sony A7RIV", "captured with 35mm film", "shot on iPhone 15 Pro", "Canon 5D Mark IV",
              "Fujifilm simulation"]
    return f"{random.choice(subjects)}, {random.choice(lighting)}, {random.choice(camera)}, highly detailed, photorealistic, 4k"


def gen_art_prompt():
    """模式2: 艺术风格 (覆盖油画、水彩、素描)"""
    styles = ["Oil painting by Claude Monet", "Pencil sketch", "Watercolor painting", "Ukiyo-e style",
              "Abstract expressionism",
              "Pop art style", "Gothic fantasy illustration", "Cyberpunk digital art", "Pixel art 16-bit"]
    subjects = ["a chaotic battle scene", "a peaceful village", "a bouquet of flowers", "a futuristic robot",
                "a dragon sleeping on gold"]
    return f"{random.choice(subjects)}, {random.choice(styles)}, vibrant colors, artistic composition, masterpiece"


def gen_texture_prompt():
    """模式3: 纯纹理/材质 (这对水印训练至关重要！提供高频信息)"""
    textures = ["rough concrete wall texture", "fluffy white fur texture", "green grass field top view",
                "rusty metal surface", "woven fabric pattern", "liquid marble texture",
                "carbon fiber pattern", "wooden floor boards", "crumpled paper texture"]
    return f"Full screen texture of {random.choice(textures)}, high fidelity, uniform lighting, seamless pattern, 8k texture"


def gen_flat_prompt():
    """模式4: 极简/平滑/纯色 (这是水印的'地狱难度'，测试隐蔽性)"""
    subjects = ["clear blue sky with one small cloud", "a white minimalist room with no furniture",
                "a gradient background from blue to pink", "a blank canvas", "smooth ceramic surface",
                "foggy morning with low visibility"]
    return f"{random.choice(subjects)}, minimalist, clean, smooth, negative space, soft lighting"


def gen_text_prompt():
    """模式5: 包含文字/排版 (SD3 的强项，测试边缘水印)"""
    texts = ["'HELLO WORLD'", "'STABLE DIFFUSION'", "'WATERMARK'", "'FUTURE'", "'CYBERPUNK'"]
    styles = ["neon sign on a brick wall", "written in chalk on a blackboard", "printed on a t-shirt",
              "gold lettering on a book cover", "graffiti on a subway train"]
    return f"Text spelling {random.choice(texts)}, {random.choice(styles)}, clear typography, legible text"


def gen_weird_prompt():
    """模式6: 抽象/超现实 (增加数据分布的离散性)"""
    concepts = ["a melting clock in a desert", "an astronaut riding a horse in space",
                "a geometric shape made of water", "a transparent computer showing circuits",
                "a brain made of fiber optic cables"]
    return f"{random.choice(concepts)}, surrealism, dreamlike, dali style, impossible geometry"


def get_super_diverse_prompt():
    """路由器：随机选择一种模式"""
    generators = [
        (gen_photography_prompt, 0.3),  # 30% 摄影
        (gen_art_prompt, 0.2),  # 20% 艺术
        (gen_texture_prompt, 0.2),  # 20% 纹理 (重点！)
        (gen_flat_prompt, 0.1),  # 10% 极简 (难点！)
        (gen_text_prompt, 0.1),  # 10% 文字
        (gen_weird_prompt, 0.1)  # 10% 抽象
    ]

    # 根据权重随机选择函数
    func = random.choices([g[0] for g in generators], weights=[g[1] for g in generators])[0]
    return func()


# =========================================================

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    metadata_path = os.path.join(OUTPUT_DIR, METADATA_FILE)

    real_model_path = get_actual_model_path(MODEL_PATH)

    existing_files = [f for f in os.listdir(OUTPUT_DIR) if f.endswith(".jpg")]
    current_count = len(existing_files)

    if current_count >= NUM_IMAGES:
        print(f"✅ 已完成 {current_count} 张，退出。")
        return

    print(f"🚀 [Pro版] 目标: {NUM_IMAGES} | 当前: {current_count}")
    print(f"🎨 将生成: 摄影, 艺术, 纯纹理, 极简, 文字, 超现实等多种风格")

    # --- 加载模型 ---
    print("⏳ 正在加载模型...")
    try:
        pipe = StableDiffusion3Pipeline.from_pretrained(
            real_model_path,
            torch_dtype=torch.float16,
            use_safetensors=True,
            local_files_only=True
        )
        pipe = pipe.to("cuda")
        pipe.set_progress_bar_config(disable=True)
    except Exception as e:
        print(f"\n❌ 模型加载失败: {e}")
        return

    # --- 生成循环 ---
    print("⚡ 开始生成...")
    with open(metadata_path, 'a', encoding='utf-8') as f_meta:
        pbar = tqdm(total=NUM_IMAGES, initial=current_count, unit="img")

        while current_count < NUM_IMAGES:
            try:
                # 使用 Pro 版 Prompt 生成器
                prompt = get_super_diverse_prompt()

                with torch.no_grad():
                    image = pipe(prompt, num_inference_steps=28, guidance_scale=7.0, height=RESOLUTION,
                                 width=RESOLUTION).images[0]

                file_name = f"image_{current_count:05d}.jpg"
                save_path = os.path.join(OUTPUT_DIR, file_name)
                image.save(save_path, quality=95)

                meta_entry = {"file_name": file_name, "text": prompt}
                f_meta.write(json.dumps(meta_entry) + "\n")
                f_meta.flush()

                current_count += 1
                pbar.update(1)

                if current_count % 50 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

            except KeyboardInterrupt:
                print("\n🛑 手动停止。")
                break
            except Exception as e:
                print(f"\n❌ 生成出错: {e}")
                continue


if __name__ == "__main__":
    main()