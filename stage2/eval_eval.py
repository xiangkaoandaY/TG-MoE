import torch
import torch.nn.functional as F
from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel, StableDiffusion3Pipeline
import watermarkModel
import lora_moe
import os
import tqdm
import numpy as np
import random
from torchvision import transforms
from safetensors.torch import load_file
# 确保你的 utils.py 文件里有 distorsion_unit 函数
from utils import distorsion_unit
import argparse

# 引入质量评价库
from skimage.metrics import peak_signal_noise_ratio as compute_psnr
from skimage.metrics import structural_similarity as compute_ssim

# ==========================================
# 🚀 引入高级评测指标库
# ==========================================
try:
    from transformers import CLIPProcessor, CLIPModel

    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    print("⚠️ 未检测到 transformers 库，将跳过 CLIP 测试。")

try:
    from pytorch_fid.fid_score import calculate_fid_given_paths

    HAS_FID = True
except ImportError:
    HAS_FID = False
    print("⚠️ 未检测到 pytorch-fid 库，将跳过 FID 分布差异测试。")

# 👉 新增：引入 ROC 曲线计算库
try:
    from sklearn.metrics import roc_curve

    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("⚠️ 未检测到 scikit-learn，将跳过 TPR/FPR 测试。")

parser = argparse.ArgumentParser()
parser.add_argument('--base_model', type=str, required=True, help='SD3 官方基础模型路径 (包含 model_index.json)')
parser.add_argument('--transformer_dir', type=str, required=True, help='你训练出的 checkpoint 里的 transformer 文件夹')
parser.add_argument('--pretrainedWM_dir', type=str, required=True, help='预训练水印提取器路径 (包含 decoder.pth)')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--clip_dir', type=str, default="/home/HDD/Checkpoint/clip-vit-base-patch32",
                    help='本地 CLIP 模型的绝对路径')
parser.add_argument('--num_experts', type=int, default=4, help='消融实验对应的专家数量')
parser.add_argument('--rank', type=int, default=16, help='消融实验对应的 LoRA Rank 大小')
args = parser.parse_args()


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(args.seed)


def calculate_bit_acc(decoded_result, GT):
    predictions = decoded_result.cpu()
    ground_truth = GT.cpu()
    rounded_predictions = torch.round(predictions)
    correct_predictions = (rounded_predictions == ground_truth).sum().item()
    accuracy = correct_predictions / ground_truth.numel()
    return accuracy


prompts = [
    # ---- 🏞️ 自然与风景 ----
    "A high quality photo of a landscape with mountains and a lake at sunset",
    "A dense fog rolling through a dark pine forest, cinematic lighting",
    "A pristine white sand beach with crystal clear turquoise water",
    "A spectacular aurora borealis over a snow-covered cabin in Norway",
    "A drone view of a winding river through a vibrant autumn forest",
    "A vast desert with golden sand dunes under a clear blue sky",
    "A close-up of a cascading waterfall in a lush green tropical jungle",
    "A field of blooming sunflowers under a bright summer sun",
    "A dramatic thunderstorm over a grand canyon with lightning strikes",
    "A peaceful zen garden with a koi pond and cherry blossom trees",

    # ---- 🏙️ 城市与建筑 ----
    "A futuristic cyberpunk city street at night with neon lights and rain",
    "A cobblestone street in a historic European town with warm streetlamps",
    "A bustling Tokyo intersection during rush hour with glowing billboards",
    "An interior shot of a cozy modern living room with a fireplace",
    "A highly detailed gothic cathedral with stained glass windows",
    "A massive abandoned factory reclaimed by nature with vines growing",
    "A sleek modern glass skyscraper reflecting the sunset",
    "A cozy coffee shop interior with warm lighting and wooden furniture",
    "An ancient ruined temple in the middle of a dense jungle",
    "A panoramic view of the New York City skyline at twilight",

    # ---- 🦊 动物与生物 ----
    "A cute golden retriever playing in a sunny green park",
    "A majestic Bengal tiger resting on a rock in the jungle",
    "A macro shot of a colorful jumping spider with water droplets",
    "A highly detailed portrait of a bald eagle looking into the distance",
    "A school of vibrant clownfish swimming in a coral reef",
    "A fluffy orange cat sleeping on a windowsill in the sunlight",
    "A wild horse running through a shallow river with water splashing",
    "A close-up of a chameleon blending into a green leaf",
    "A family of elephants walking across the African savanna",
    "A mysterious glowing jellyfish in the deep dark ocean",

    # ---- 🍔 静物与美食 ----
    "A cup of hot coffee on a wooden table next to an open book",
    "A delicious pepperoni pizza with melting cheese and fresh basil",
    "A macro shot of a vibrant red rose with dew drops on its petals",
    "A vintage typewriter sitting on an old mahogany desk",
    "A freshly baked chocolate chip cookie breaking apart with melted chocolate",
    "A highly detailed mechanical pocket watch with exposed gears",
    "A bowl of fresh ramen with steam rising, soft lighting",
    "A glowing glass potion bottle filled with magical purple liquid",
    "A classic acoustic guitar resting against a brick wall",
    "An elegant glass of red wine catching the light on a dark table",

    # ---- 🧑‍🚀 科幻、奇幻与人物 ----
    "An astronaut riding a horse on the moon, highly detailed",
    "A brave medieval knight in shining armor standing on a battlefield",
    "A futuristic glowing robot repairing a spaceship in orbit",
    "A beautiful elven wizard casting a glowing blue magic spell",
    "A steampunk inventor working in a cluttered workshop with brass pipes",
    "A portrait of a cyberpunk hacker with glowing neon tattoos",
    "A massive alien spaceship hovering over an ancient pyramid",
    "A glowing magical sword stuck in a stone in a dark forest",
    "A steampunk airship flying through fluffy white clouds",
    "A cybernetic samurai with a glowing katana in a dark alleyway"
]

device = "cuda"
weight_dtype = torch.bfloat16

print("Loading base SD3 Transformer & Pipeline...")
transformer = SD3Transformer2DModel.from_pretrained(args.base_model, subfolder="transformer")
transformer = lora_moe.inject_moe_lora_to_sd3(transformer, num_experts=args.num_experts, rank=args.rank)
if os.path.exists(os.path.join(args.transformer_dir, "diffusion_pytorch_model.safetensors")):
    state_dict = load_file(os.path.join(args.transformer_dir, "diffusion_pytorch_model.safetensors"))
else:
    state_dict = torch.load(os.path.join(args.transformer_dir, "diffusion_pytorch_model.bin"), map_location="cpu")
transformer.load_state_dict(state_dict, strict=True)
transformer = transformer.to(device, dtype=weight_dtype)

pipe = StableDiffusion3Pipeline.from_pretrained(
    args.base_model,
    transformer=transformer,
    torch_dtype=weight_dtype
)
pipe = pipe.to(device)
pipe.set_progress_bar_config(disable=True)
vae = pipe.vae

secret_pt_path = f"{args.pretrainedWM_dir}/secret.pt"
GT_secret = torch.load(secret_pt_path).to(device, dtype=weight_dtype)
GT_secret_batch = GT_secret.view(1, 48)

watermark_extractor = watermarkModel.Extractor_forLatent(secret_size=48)
watermark_extractor.load_state_dict(torch.load(os.path.join(args.pretrainedWM_dir, "decoder.pth")))
watermark_extractor = watermark_extractor.to(device, dtype=weight_dtype)
watermark_extractor.eval()

if HAS_CLIP:
    print(f"Loading CLIP Model from local path: {args.clip_dir}")
    try:
        clip_processor = CLIPProcessor.from_pretrained(args.clip_dir, local_files_only=True)
        clip_model = CLIPModel.from_pretrained(args.clip_dir, local_files_only=True).to(device)
        clip_model.eval()
    except Exception as e:
        print(f"\n⚠️ 无法加载本地 CLIP 模型权重，自动跳过 CLIP 测试！(报错信息: {e})")
        HAS_CLIP = False

total_acc_WM = []
resize_wm_acc, blur_wm_acc, noise_wm_acc, jpeg_compress_wm_acc = [], [], [], []
sharpness_wm_acc, brightness_wm_acc, contrast_wm_acc, saturation_wm_acc = [], [], [], []

total_psnr = []
total_ssim = []
total_clip_nowm = []
total_clip_wm = []

# 👉 新增：用于存放 TPR/FPR 的真实标签和预测分数
y_true_list = []
y_score_list = []

distortion_list = ['blur', 'noise', 'jpeg_compress', 'resize', 'sharpness', "brightness", "contrast", "saturation"]

noWM_dir = "./Evaluation/noWM"
WM_dir = "./Evaluation/WM"
os.makedirs(noWM_dir, exist_ok=True)
os.makedirs(WM_dir, exist_ok=True)

transform_to_latent = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

print("\nStarting Evaluation Pipeline (Generating 50 image pairs)...")
with torch.no_grad():
    for i, prompt in enumerate(tqdm.tqdm(prompts)):

        lora_moe.set_moe_context(pipe.transformer, secret_bits=None, image_context=None)
        gen_noWM = torch.Generator(device=device).manual_seed(args.seed + i)
        img_noWM = pipe(prompt=prompt, num_inference_steps=28, guidance_scale=7.0, generator=gen_noWM).images[0]
        img_noWM.save(os.path.join(noWM_dir, f"{i}.png"))

        lora_moe.set_moe_context(pipe.transformer, secret_bits=None, image_context=None)
        gen_WM = torch.Generator(device=device).manual_seed(args.seed + i)


        def wm_step_callback(pipe, step_index, timestep, callback_kwargs):
            t_float = timestep.item() / 1000.0
            if t_float < 0.4:
                lora_moe.set_moe_context(pipe.transformer, secret_bits=GT_secret_batch, image_context=None)
            else:
                lora_moe.set_moe_context(pipe.transformer, secret_bits=None, image_context=None)
            return callback_kwargs


        img_WM = pipe(
            prompt=prompt,
            num_inference_steps=28,
            guidance_scale=7.0,
            generator=gen_WM,
            callback_on_step_end=wm_step_callback
        ).images[0]
        img_WM.save(os.path.join(WM_dir, f"{i}.png"))

        img_noWM_np = np.array(img_noWM)
        img_WM_np = np.array(img_WM)

        current_psnr = compute_psnr(img_noWM_np, img_WM_np, data_range=255)
        current_ssim = compute_ssim(img_noWM_np, img_WM_np, data_range=255, channel_axis=-1)
        total_psnr.append(current_psnr)
        total_ssim.append(current_ssim)

        if HAS_CLIP:
            inputs_nowm = clip_processor(text=[prompt], images=img_noWM, return_tensors="pt", padding=True).to(device)
            clip_score_nowm = clip_model(**inputs_nowm).logits_per_image[0][0].item()
            total_clip_nowm.append(clip_score_nowm)

            inputs_wm = clip_processor(text=[prompt], images=img_WM, return_tensors="pt", padding=True).to(device)
            clip_score_wm = clip_model(**inputs_wm).logits_per_image[0][0].item()
            total_clip_wm.append(clip_score_wm)

        # ----------------------------------------
        # 计算基础准确率与 TPR/FPR
        # ----------------------------------------
        validation_image_tensors = torch.stack([transform_to_latent(img_noWM), transform_to_latent(img_WM)]).to(device,
                                                                                                                dtype=weight_dtype)
        validation_latent_tensors = vae.encode(
            validation_image_tensors).latent_dist.sample() * vae.config.scaling_factor

        decoded_results = torch.sigmoid(watermark_extractor(validation_latent_tensors))
        GT_secret_repeated = GT_secret.view(1, 48)

        # 1. 计算水图的提取率 (Positive Sample)
        decoded_result_WM = decoded_results[1].unsqueeze(0)
        acc_wm = calculate_bit_acc(decoded_result_WM, GT_secret_repeated)
        total_acc_WM.append(acc_wm)

        # 2. 计算原图的提取率 (Negative Sample)
        decoded_result_noWM = decoded_results[0].unsqueeze(0)
        acc_nowm = calculate_bit_acc(decoded_result_noWM, GT_secret_repeated)

        # 👉 收集用于算 ROC 的数据
        if HAS_SKLEARN:
            y_true_list.append(1)  # 水图标签为 1
            y_score_list.append(acc_wm)  # 提取得分
            y_true_list.append(0)  # 原图标签为 0
            y_score_list.append(acc_nowm)  # 误触得分

        # ----------------------------------------
        # 鲁棒性攻击测试 (针对水印图)
        # ----------------------------------------
        for distortion in distortion_list:
            distorted_image = distorsion_unit(transforms.ToTensor()(img_WM).unsqueeze(0).to(device), distortion)
            distorted_image = F.interpolate(distorted_image, size=(1024, 1024), mode='bilinear')
            distorted_image = distorted_image * 2.0 - 1.0

            distorted_latent = vae.encode(
                distorted_image.to(weight_dtype)).latent_dist.sample() * vae.config.scaling_factor
            reveal_output = watermark_extractor(distorted_latent)
            results = torch.round(torch.sigmoid(reveal_output))
            distort_acc = (results == GT_secret_repeated).sum().item() / GT_secret_repeated.numel()

            if distortion == 'resize':
                resize_wm_acc.append(distort_acc)
            elif distortion == 'brightness':
                brightness_wm_acc.append(distort_acc)
            elif distortion == 'contrast':
                contrast_wm_acc.append(distort_acc)
            elif distortion == 'saturation':
                saturation_wm_acc.append(distort_acc)
            elif distortion == 'blur':
                blur_wm_acc.append(distort_acc)
            elif distortion == 'noise':
                noise_wm_acc.append(distort_acc)
            elif distortion == 'jpeg_compress':
                jpeg_compress_wm_acc.append(distort_acc)
            elif distortion == 'sharpness':
                sharpness_wm_acc.append(distort_acc)

fid_score_str = "N/A (Skipped)"
if HAS_FID:
    print("\nCalculating FID Score between Clean and Watermarked image distributions...")
    fid_score = calculate_fid_given_paths([noWM_dir, WM_dir], batch_size=8, device=device, dims=2048)
    fid_score_str = f"{fid_score:.4f}"

# 👉 新增：计算 FPR @ TPR
fpr_at_tpr99_str = "N/A"
fpr_at_tpr95_str = "N/A"
if HAS_SKLEARN and len(y_true_list) > 0:
    fpr, tpr, thresholds = roc_curve(y_true_list, y_score_list)
    fpr_at_tpr99 = fpr[np.argmax(tpr >= 0.99)] if any(tpr >= 0.99) else float('nan')
    fpr_at_tpr95 = fpr[np.argmax(tpr >= 0.95)] if any(tpr >= 0.95) else float('nan')
    fpr_at_tpr99_str = f"{fpr_at_tpr99:.6f}"
    fpr_at_tpr95_str = f"{fpr_at_tpr95:.6f}"

print('\n================ IMAGE QUALITY & DISTRIBUTION ================')
print(f'Average PSNR         : {sum(total_psnr) / len(total_psnr):.4f} dB')
print(f'Average SSIM         : {sum(total_ssim) / len(total_ssim):.4f}')
print(f'FID (Clean vs WM)    : {fid_score_str}')
print('================== TEXT-IMAGE ALIGNMENT (CLIP) ===============')
if HAS_CLIP:
    print(f'CLIP Score (Clean)   : {sum(total_clip_nowm) / len(total_clip_nowm):.4f}')
    print(f'CLIP Score (WM)      : {sum(total_clip_wm) / len(total_clip_wm):.4f}')
else:
    print("CLIP Score           : N/A (Skipped)")
print('======================== ROBUSTNESS ==========================')
print(f'Original WM ACC : {sum(total_acc_WM) / len(total_acc_WM):.4f}')
print(f'Resize ACC      : {sum(resize_wm_acc) / len(resize_wm_acc):.4f}')
print(f'Blur ACC        : {sum(blur_wm_acc) / len(blur_wm_acc):.4f}')
print(f'Noise ACC       : {sum(noise_wm_acc) / len(noise_wm_acc):.4f}')
print(f'JPEG ACC        : {sum(jpeg_compress_wm_acc) / len(jpeg_compress_wm_acc):.4f}')
print(f'Sharpness ACC   : {sum(sharpness_wm_acc) / len(sharpness_wm_acc):.4f}')
print(f'Brightness ACC  : {sum(brightness_wm_acc) / len(brightness_wm_acc):.4f}')
print(f'Contrast ACC    : {sum(contrast_wm_acc) / len(contrast_wm_acc):.4f}')
print(f'Saturation ACC  : {sum(saturation_wm_acc) / len(saturation_wm_acc):.4f}')
print('================== WATERMARK DETECTION (ROC) =================')
print(f'FPR @ TPR=0.99  : {fpr_at_tpr99_str}')
print(f'FPR @ TPR=0.95  : {fpr_at_tpr95_str}')
print('==============================================================')