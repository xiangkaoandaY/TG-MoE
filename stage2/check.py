import torch
from diffusers import StableDiffusion3Pipeline
from safetensors.torch import load_file
from torchvision import transforms
import watermarkModel
import lora_moe

# ================= 配置区 =================
MODEL_ID = "stabilityai/stable-diffusion-3-medium-diffusers"
# 注意检查这里的路径是不是你的 20000 步权重路径
CHECKPOINT_DIR = "/home/HDD/cxy/SD3/stage2/output/checkpoint-20000/transformer"
SECRET_PATH = "/home/HDD/cxy/SD3/stage2/pretrainedWM/secret.pt"
EXTRACTOR_PATH = "/home/HDD/cxy/SD3/stage2/pretrainedWM/decoder.pth"
DEVICE = "cuda"
WEIGHT_DTYPE = torch.bfloat16


# ==========================================

def test_e2e_accuracy():
    print("1. 加载基础 Pipeline 和 VAE...")
    pipe = StableDiffusion3Pipeline.from_pretrained(MODEL_ID, torch_dtype=WEIGHT_DTYPE)

    print("2. 注入 MoE 结构并加载 20000 步权重...")
    pipe.transformer = lora_moe.inject_moe_lora_to_sd3(pipe.transformer, num_experts=4, rank=32)
    pipe.transformer = pipe.transformer.to(device=DEVICE, dtype=WEIGHT_DTYPE)
    pipe = pipe.to(DEVICE)

    weight_path = f"{CHECKPOINT_DIR}/diffusion_pytorch_model.safetensors"
    state_dict = load_file(weight_path, device=DEVICE)
    pipe.transformer.load_state_dict(state_dict, strict=False)
    pipe.transformer.eval()

    print("3. 加载提取器和 Secret...")
    secret = torch.load(SECRET_PATH).to(DEVICE, dtype=WEIGHT_DTYPE)
    extractor = watermarkModel.Extractor_forLatent(secret_size=48).to(DEVICE, dtype=WEIGHT_DTYPE)
    extractor.load_state_dict(torch.load(EXTRACTOR_PATH, map_location=DEVICE))
    extractor.eval()

    print("4. 开始生成图片并进行双重 ACC 测试...")
    generator = torch.Generator(device=DEVICE).manual_seed(42)

    # 关闭图像上下文，仅依赖 secret_bits
    secret_bits_batch = secret.view(1, 48)
    lora_moe.set_moe_context(pipe.transformer, secret_bits=secret_bits_batch, image_context=None)

    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=WEIGHT_DTYPE):
            # 步骤 A：生成潜变量 (Latent)
            latents = pipe(
                prompt="A high quality photo of a landscape",
                num_inference_steps=28,
                generator=generator,
                output_type="latent"
            ).images

            # ---------------------------------------------------------
            # 🎯 测试 1: 原生潜变量准确率 (Latent ACC)
            # ---------------------------------------------------------
            pred_latent = torch.round(torch.sigmoid(extractor(latents)))
            correct_latent = (pred_latent == secret).sum().item()
            acc_latent = correct_latent / 48.0

            # ---------------------------------------------------------
            # 🔬 测试 2: 端到端图片准确率 (End-to-End ACC)
            # ---------------------------------------------------------
            # 1. 潜变量 -> 像素图像 (模拟真实保存的图片)
            img_tensor = pipe.vae.decode(latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
            img = pipe.image_processor.postprocess(img_tensor, output_type="pil")[0]

            # 2. 图像 -> 重新转为 Tensor -> VAE 重新编码回潜变量
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
            ])
            val_img_tensors = transform(img).unsqueeze(0).to(dtype=WEIGHT_DTYPE, device=DEVICE)
            val_reencoded_latents = pipe.vae.encode(
                val_img_tensors).latent_dist.sample() * pipe.vae.config.scaling_factor

            # 3. 在经过 VAE 折腾后的潜变量上提取水印
            pred_e2e = torch.round(torch.sigmoid(extractor(val_reencoded_latents)))
            correct_e2e = (pred_e2e == secret).sum().item()
            acc_e2e = correct_e2e / 48.0

    print("\n" + "=" * 50)
    print(f"🎯 [内部指标] 潜变量准确率 (Latent ACC): {acc_latent:.4f}  <-- 这个应该在 0.99 左右")
    print(f"🔬 [应用指标] 端到端准确率 (E2E ACC)   : {acc_e2e:.4f}  <-- 这是我们现在最关心的指标！")
    print("=" * 50 + "\n")

    if acc_e2e > 0.8:
        print("🎉 恭喜！端到端测试也拿下了！即使经过 VAE 解码编码，3 倍强度的水印依然存活！下一步只需调低权重恢复画质即可。")
    else:
        print("⚠️ Latent ACC 很高，但 E2E ACC 较低。说明 VAE 强大的有损压缩还是把相框里的水印洗掉了一部分。")


if __name__ == "__main__":
    test_e2e_accuracy()