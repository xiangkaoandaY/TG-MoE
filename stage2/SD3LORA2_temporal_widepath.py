import os
import math
import copy
import shutil
import logging
import argparse
import json
from pathlib import Path
from tqdm.auto import tqdm
import lora_moe_temporal_widepath as lora_moe
import watermarkModel
import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# PyTorch 相关
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageDraw, ImageOps

# Accelerate
from accelerate import Accelerator
from accelerate.utils import ProjectConfiguration, set_seed
from accelerate.logging import get_logger

# Transformers & Diffusers 工具
import transformers
import diffusers
from transformers import BitsAndBytesConfig
from diffusers import DiffusionPipeline
from diffusers.optimization import get_scheduler
from diffusers.training_utils import cast_training_params
from diffusers.utils.torch_utils import is_compiled_module
from diffusers import FlowMatchEulerDiscreteScheduler, SD3Transformer2DModel, AutoencoderKL
from transformers import (
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast,
    SiglipVisionModel,
    SiglipImageProcessor,
)
import wandb
from utils import (
    DreamBoothDataset_modified,
    collate,
    encode_prompt,
    coefficient_wm,
    coefficient_preserve
)

logger = get_logger(__name__)



def parse_float_csv(value):
    if isinstance(value, (list, tuple)):
        return [float(x) for x in value]
    return [float(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_int_csv(value):
    if isinstance(value, (list, tuple)):
        return [int(x) for x in value]
    return [int(x.strip()) for x in str(value).split(",") if x.strip()]


def parse_str_csv(value):
    if isinstance(value, (list, tuple)):
        return [str(x) for x in value]
    return [x.strip() for x in str(value).split(",") if x.strip()]


def append_jsonl(path, record):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def make_fixed_wrong_secret(owner_secret, seed=20260724):
    owner = owner_secret.detach().float().reshape(1, -1)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    wrong = torch.randint(0, 2, owner.shape, generator=generator, dtype=torch.float32)
    # Avoid an accidentally identical key without using the trivial complement key.
    if torch.equal(wrong, owner.cpu()):
        wrong[0, 0] = 1.0 - wrong[0, 0]
    return wrong.to(owner_secret.device, dtype=owner_secret.dtype)


def load_initial_transformer_weights(model, transformer_dir):
    if not transformer_dir:
        return
    directory = Path(transformer_dir)
    candidates = [
        directory / "diffusion_pytorch_model.safetensors",
        directory / "diffusion_pytorch_model.bin",
        directory / "pytorch_model.bin",
    ]
    state_path = next((path for path in candidates if path.is_file()), None)
    if state_path is None:
        raise FileNotFoundError(
            f"No transformer state found in {directory}; tried: "
            + ", ".join(path.name for path in candidates)
        )
    if state_path.suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(state_path))
    else:
        state = torch.load(state_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    # New temporal-wide-path code adds no persistent parameters, so unexpected
    # or missing trainable adapter keys indicate a genuinely incompatible file.
    serious_missing = [
        key for key in missing
        if any(token in key for token in ("lora_A", "lora_B", "lora_bias", "router", "bit_router"))
    ]
    serious_unexpected = [
        key for key in unexpected
        if any(token in key for token in ("lora_A", "lora_B", "lora_bias", "router", "bit_router"))
    ]
    if serious_missing or serious_unexpected:
        raise RuntimeError(
            "Incompatible TG-MoE checkpoint. "
            f"missing={serious_missing[:8]}, unexpected={serious_unexpected[:8]}"
        )
    print(
        f"[Init] loaded TG-MoE transformer from {state_path}; "
        f"missing={len(missing)}, unexpected={len(unexpected)}"
    )


def watermark_probe_metrics(logits, bits):
    bits = bits.float()
    pred = (torch.sigmoid(logits.float()) >= 0.5).float()
    signs = bits * 2.0 - 1.0
    return {
        "bit_acc": float((pred == bits).float().mean().item()),
        "owner_margin": float((signs * logits.float()).mean().item()),
        "matched_probability": float(
            (bits * torch.sigmoid(logits.float()) + (1.0 - bits) * (1.0 - torch.sigmoid(logits.float()))).mean().item()
        ),
    }


@torch.no_grad()
def evaluate_shadow_ft_probe(
    transformer,
    transformer_frozen,
    watermark_extractor,
    model_input,
    prompt_embeds,
    pooled_prompt_embeds,
    owner_secret,
    wrong_secret,
    eval_times,
    noise_scheduler,
    accelerator,
    noise_seed,
):
    device = model_input.device
    dtype = model_input.dtype
    bsz = model_input.shape[0]
    generator = torch.Generator(device=device).manual_seed(int(noise_seed))
    fixed_noise = torch.randn(model_input.shape, generator=generator, device=device, dtype=dtype)
    per_time = []
    for t_value in eval_times:
        t_batch = torch.full((bsz,), float(t_value), device=device, dtype=torch.float32)
        timesteps = t_batch * noise_scheduler.config.num_train_timesteps
        t_view = t_batch.view(-1, 1, 1, 1).to(dtype=dtype)
        noisy = (1.0 - t_view) * model_input + t_view * fixed_noise
        with accelerator.autocast():
            target_clean = transformer_frozen(
                hidden_states=noisy,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
        lora_moe.set_moe_context(
            transformer,
            secret_bits=owner_secret,
            timestep=t_batch,
            wrong_secret_bits=wrong_secret,
        )
        with accelerator.autocast():
            pred = transformer(
                hidden_states=noisy,
                timestep=timesteps,
                encoder_hidden_states=prompt_embeds,
                pooled_projections=pooled_prompt_embeds,
                return_dict=False,
            )[0]
        pred_x0 = noisy.float() - t_batch.view(-1, 1, 1, 1) * pred.float()
        extractor_param = next(watermark_extractor.parameters())
        logits = watermark_extractor(
            pred_x0.to(
                device=extractor_param.device,
                dtype=extractor_param.dtype,
            )
        )
        wm = watermark_probe_metrics(logits, owner_secret.float())
        path = lora_moe.get_temporal_path_losses(transformer)
        per_time.append({
            "t": float(t_value),
            **wm,
            "target_mass": float(path["path_target_mass"].detach().item()),
            "wrong_target_mass": float(path["path_wrong_target_mass"].detach().item()),
            "path_entropy": float(path["path_entropy"].detach().item()),
            "clean_velocity_mse": float(F.mse_loss(pred.float(), target_clean.float()).item()),
        })
    lora_moe.set_moe_context(transformer, None, timestep=None, wrong_secret_bits=None)
    keys = [
        "bit_acc", "owner_margin", "matched_probability", "target_mass",
        "wrong_target_mass", "path_entropy", "clean_velocity_mse",
    ]
    output = {key: float(sum(row[key] for row in per_time) / len(per_time)) for key in keys}
    output["per_time"] = per_time
    return output



def _fit_pil_to_square(image, side):
    image = image.convert("RGB")
    fitted = ImageOps.contain(image, (side, side))

    canvas = Image.new(
        "RGB",
        (side, side),
        "white",
    )

    left = (side - fitted.width) // 2
    top = (side - fitted.height) // 2
    canvas.paste(fitted, (left, top))

    return canvas


def save_ft_probe_pipeline_visuals(
    args,
    images,
    global_step,
    ft_steps,
):
    """
    保存当前LoRA微调步数下的：

    1. 水印生成图
    2. 无密钥干净生成图
    3. 左右对照图
    """

    if len(images) % 2 != 0:
        raise ValueError(
            "log_validation返回的图片数量必须为偶数，"
            f"当前数量为：{len(images)}"
        )

    root = (
        Path(args.output_dir)
        / "ft_probe_pipeline"
        / f"train_step_{int(global_step):06d}"
    )

    step_dir = root / f"ft_{int(ft_steps):05d}"
    step_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    side = int(args.ft_probe_pipeline_thumbnail)
    header_height = 40

    saved_pairs = []

    for prompt_index in range(len(images) // 2):
        # log_validation返回顺序：
        # 水印图0、干净图0、水印图1、干净图1……
        image_wm = images[2 * prompt_index].convert("RGB")
        image_clean = images[2 * prompt_index + 1].convert("RGB")

        wm_path = (
            step_dir
            / f"prompt_{prompt_index:02d}_watermarked.png"
        )

        clean_path = (
            step_dir
            / f"prompt_{prompt_index:02d}_clean.png"
        )

        pair_path = (
            step_dir
            / f"prompt_{prompt_index:02d}_pair.png"
        )

        # 保存完整分辨率图
        image_wm.save(wm_path)
        image_clean.save(clean_path)

        # 生成左右对比缩略图
        wm_preview = _fit_pil_to_square(
            image_wm,
            side,
        )

        clean_preview = _fit_pil_to_square(
            image_clean,
            side,
        )

        pair = Image.new(
            "RGB",
            (2 * side, side + header_height),
            "white",
        )

        pair.paste(
            wm_preview,
            (0, header_height),
        )

        pair.paste(
            clean_preview,
            (side, header_height),
        )

        draw = ImageDraw.Draw(pair)

        draw.text(
            (10, 12),
            f"Watermarked | LoRA FT step {ft_steps}",
            fill="black",
        )

        draw.text(
            (side + 10, 12),
            f"Clean | LoRA FT step {ft_steps}",
            fill="black",
        )

        pair.save(pair_path)
        saved_pairs.append(pair_path)

    logger.info(
        "[FT-Pipeline] train_step=%d "
        "attack_steps=%d saved=%s",
        global_step,
        ft_steps,
        step_dir,
    )

    return saved_pairs


def build_ft_probe_pipeline_timeline(
    args,
    global_step,
    ft_steps_list,
):
    """
    把0、256、500……3000步的Pipeline生成图
    拼成一张纵向时间线。
    """

    root = (
        Path(args.output_dir)
        / "ft_probe_pipeline"
        / f"train_step_{int(global_step):06d}"
    )

    if not root.exists():
        return []

    unique_steps = sorted(
        set(int(value) for value in ft_steps_list)
    )

    prompt_indices = set()

    for step in unique_steps:
        step_dir = root / f"ft_{step:05d}"

        for pair_path in step_dir.glob(
            "prompt_*_pair.png"
        ):
            try:
                prompt_index = int(
                    pair_path.stem.split("_")[1]
                )
                prompt_indices.add(prompt_index)

            except (IndexError, ValueError):
                continue

    timeline_paths = []

    for prompt_index in sorted(prompt_indices):
        rows = []

        for step in unique_steps:
            pair_path = (
                root
                / f"ft_{step:05d}"
                / f"prompt_{prompt_index:02d}_pair.png"
            )

            if pair_path.is_file():
                image = Image.open(
                    pair_path
                ).convert("RGB")

                rows.append(
                    (step, image)
                )

        if not rows:
            continue

        width = max(
            image.width
            for _, image in rows
        )

        title_height = 48
        row_gap = 8

        total_height = (
            title_height
            + sum(image.height for _, image in rows)
            + row_gap * max(0, len(rows) - 1)
        )

        timeline = Image.new(
            "RGB",
            (width, total_height),
            "white",
        )

        draw = ImageDraw.Draw(timeline)

        draw.text(
            (10, 15),
            (
                "Pipeline visualization across "
                "LoRA fine-tuning milestones "
                f"| prompt {prompt_index}"
            ),
            fill="black",
        )

        y = title_height

        for _, image in rows:
            timeline.paste(
                image,
                (0, y),
            )

            y += image.height + row_gap

        timeline_path = (
            root
            / f"timeline_prompt_{prompt_index:02d}.png"
        )

        timeline.save(timeline_path)
        timeline_paths.append(timeline_path)

        logger.info(
            "[FT-Pipeline] saved timeline: %s",
            timeline_path,
        )

    return timeline_paths

def run_shadow_lora_finetune_probe(
    args,
    transformer,
    transformer_frozen,
    watermark_extractor,
    model_input,
    prompt_embeds,
    pooled_prompt_embeds,
    owner_secret,
    wrong_secret,
    noise_scheduler,
    vae,
    text_encoder_1,
    tokenizer_1,
    text_encoder_2,
    tokenizer_2,
    text_encoder_3,
    tokenizer_3,
    accelerator,
    global_step,
):
    """Train a temporary downstream LoRA on clean flow matching, evaluate, then delete it.

    This does not modify the main TG-MoE parameters or its optimizer. It is an
    in-training robustness proxy, not a replacement for the final checkpoint-level
    multi-step/RGB fine-tuning experiment.
    """
    if accelerator.num_processes != 1:
        logger.warning("In-training FT probe currently supports one process; skipping.")
        return []

    unwrapped = accelerator.unwrap_model(transformer)
    was_training = unwrapped.training
    attack_params = lora_moe.enable_probe_attack_lora(
        unwrapped,
        rank=args.ft_probe_rank,
        alpha=args.ft_probe_alpha,
        blocks=parse_int_csv(args.ft_probe_blocks),
        projections=parse_str_csv(args.ft_probe_projections),
    )
    optimizer_probe = torch.optim.AdamW(
        attack_params,
        lr=args.ft_probe_lr,
        weight_decay=args.ft_probe_weight_decay,
    )
    milestones = sorted(set(parse_int_csv(args.ft_probe_milestones) + [0, args.ft_probe_steps]))
    milestones = [value for value in milestones if 0 <= value <= args.ft_probe_steps]
    eval_times = parse_float_csv(args.ft_probe_eval_times)
    output_records = []
    pipeline_visual_steps = []

    def maybe_generate_pipeline_visuals(ft_steps):
        if not args.ft_probe_save_pipeline_images:
            return

        if args.validation_prompt is None:
            logger.warning(
                "[FT-Pipeline] validation_prompt为空，跳过。"
            )
            return

        try:
            unwrapped.eval()

            with torch.no_grad():
                _, generated_images = log_validation(
                    args,
                    unwrapped,
                    transformer_frozen,
                    noise_scheduler,
                    vae,
                    text_encoder_1,
                    tokenizer_1,
                    text_encoder_2,
                    tokenizer_2,
                    text_encoder_3,
                    tokenizer_3,
                    accelerator,
                )

            save_ft_probe_pipeline_visuals(
                args,
                generated_images,
                global_step,
                ft_steps,
            )

            pipeline_visual_steps.append(
                int(ft_steps)
            )

        except Exception:
            logger.exception(
                "[FT-Pipeline] generation failed "
                "at train_step=%d, attack_steps=%d",
                global_step,
                ft_steps,
            )

    model_input = model_input.detach()
    prompt_embeds = prompt_embeds.detach()
    pooled_prompt_embeds = pooled_prompt_embeds.detach()
    owner_secret = owner_secret.detach()
    wrong_secret = wrong_secret.detach()
    bsz = model_input.shape[0]
    generator = torch.Generator(device=model_input.device).manual_seed(
        int(args.seed + 1000003 + global_step)
    )

    try:
        unwrapped.eval()
        if 0 in milestones:
            metrics = evaluate_shadow_ft_probe(
                transformer, transformer_frozen, watermark_extractor,
                model_input, prompt_embeds, pooled_prompt_embeds,
                owner_secret, wrong_secret, eval_times, noise_scheduler,
                accelerator, noise_seed=args.seed + global_step + 77,
            )
            output_records.append({"ft_steps": 0, **metrics})
            maybe_generate_pipeline_visuals(0)

        unwrapped.train()
        for probe_step in range(1, args.ft_probe_steps + 1):
            t_batch = torch.rand((bsz,), generator=generator, device=model_input.device)
            timesteps = t_batch * noise_scheduler.config.num_train_timesteps
            t_view = t_batch.view(-1, 1, 1, 1).to(model_input.dtype)
            noise = torch.randn(
                model_input.shape,
                generator=generator,
                device=model_input.device,
                dtype=model_input.dtype,
            )
            noisy = (1.0 - t_view) * model_input + t_view * noise
            # Standard clean Flow-Matching target:
            # x_t = (1-t)x_0 + t*noise, therefore dx_t/dt = noise - x_0.
            flow_target = noise.float() - model_input.float()

            # Normal downstream fine-tuning does not know the owner secret.
            # Only the temporary attack LoRA is optimized.
            lora_moe.set_moe_context(
                transformer, secret_bits=None, timestep=t_batch, wrong_secret_bits=None
            )
            with accelerator.autocast():
                pred = transformer(
                    hidden_states=noisy,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds,
                    pooled_projections=pooled_prompt_embeds,
                    return_dict=False,
                )[0]

            clean_ft_loss = F.mse_loss(
                pred.float(),
                flow_target,
            )
            optimizer_probe.zero_grad(set_to_none=True)
            clean_ft_loss.backward()
            torch.nn.utils.clip_grad_norm_(attack_params, args.ft_probe_max_grad_norm)
            optimizer_probe.step()

            if probe_step in milestones:
                unwrapped.eval()
                metrics = evaluate_shadow_ft_probe(
                    transformer, transformer_frozen, watermark_extractor,
                    model_input, prompt_embeds, pooled_prompt_embeds,
                    owner_secret, wrong_secret, eval_times, noise_scheduler,
                    accelerator, noise_seed=args.seed + global_step + 77,
                )
                output_records.append({
                    "ft_steps": probe_step,
                    "last_clean_ft_loss": float(clean_ft_loss.detach().item()),
                    **metrics,
                })
                maybe_generate_pipeline_visuals(probe_step)
                unwrapped.train()
    finally:
        lora_moe.set_moe_context(transformer, None, timestep=None, wrong_secret_bits=None)
        lora_moe.disable_probe_attack_lora(unwrapped)
        for parameter in unwrapped.parameters():
            parameter.grad = None
        if not was_training:
            unwrapped.eval()
        torch.cuda.empty_cache()

    if pipeline_visual_steps:
        build_ft_probe_pipeline_timeline(
            args,
            global_step,
            pipeline_visual_steps,
        )

    log_path = Path(args.output_dir) / "finetune_probe.jsonl"
    for record in output_records:
        complete = {
            "train_step": int(global_step),
            "probe_type": "temporary_clean_downstream_lora",
            "attack_rank": int(args.ft_probe_rank),
            "attack_lr": float(args.ft_probe_lr),
            **record,
        }
        append_jsonl(log_path, complete)
        logger.info(
            "[FT-Probe] train_step=%d attack_steps=%d bit_acc=%.4f margin=%.4f target_mass=%.4f clean_mse=%.6f",
            global_step,
            record["ft_steps"],
            record["bit_acc"],
            record["owner_margin"],
            record["target_mass"],
            record["clean_velocity_mse"],
        )
        accelerator.log({
            f"FTProbe/bit_acc_{record['ft_steps']}": record["bit_acc"],
            f"FTProbe/owner_margin_{record['ft_steps']}": record["owner_margin"],
            f"FTProbe/target_mass_{record['ft_steps']}": record["target_mass"],
            f"FTProbe/clean_mse_{record['ft_steps']}": record["clean_velocity_mse"],
            "FTProbe/train_step": global_step,
        })
    return output_records

def log_avg_gradient_norm(transformer_params_to_optimize):
    """计算可训练参数的平均梯度范数"""
    total_grad_norm = 0.0
    count = 0
    for param in transformer_params_to_optimize:
        if param.grad is not None:
            grad_norm = torch.norm(param.grad).item()
            total_grad_norm += grad_norm ** 2
            count += param.numel()
    if count == 0:
        return torch.tensor(0.0)
    avg_grad_norm = torch.sqrt(torch.tensor(total_grad_norm) / count)
    return avg_grad_norm


def log_validation(
        args,
        transformer: SD3Transformer2DModel,
        transformer_frozen: SD3Transformer2DModel,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModelWithProjection,
        tokenizer: CLIPTokenizer,
        text_encoder_2: CLIPTextModelWithProjection,
        tokenizer_2: CLIPTokenizer,
        text_encoder_3: T5EncoderModel,
        tokenizer_3: T5TokenizerFast,
        accelerator: Accelerator,
        image_encoder: SiglipVisionModel = None,
        feature_extractor: SiglipImageProcessor = None,
):
    pipeline = DiffusionPipeline.from_pretrained(
        args.pretrained_model_name_or_path,
        vae=vae, text_encoder=text_encoder, text_encoder_2=text_encoder_2, text_encoder_3=text_encoder_3,
        tokenizer=tokenizer, tokenizer_2=tokenizer_2, tokenizer_3=tokenizer_3,
        transformer=transformer, scheduler=scheduler,
        image_encoder=image_encoder, feature_extractor=feature_extractor,
    )
    pipeline.scheduler = FlowMatchEulerDiscreteScheduler.from_config(pipeline.scheduler.config)
    pipeline = pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    weight_dtype = next(transformer.parameters()).dtype

    watermarked_latents = []
    watermarked_images = []

    import lora_moe_temporal_widepath as lora_moe
    GT_secret_val = torch.load(args.secret_pt_path).to(accelerator.device, dtype=weight_dtype)
    secret_bits_batch_val = GT_secret_val.view(1, 48)

    with accelerator.autocast():
        # 根据 args.num_validation_images 生成对应数量的“对比图组”
        for i in range(args.num_validation_images):
            # 给每组对比图分配一个固定的随机种子（保证水图和原图的构图一模一样）
            current_seed = (args.seed + i) if args.seed is not None else torch.seed()

            # ==========================================
            # 1. 生成【水印图】
            # ==========================================
            # ==========================================
            # 1. 生成【水印图】 (带推理期时间门控)
            # ==========================================
            gen_wm = torch.Generator(device=accelerator.device).manual_seed(current_seed)

            # 初始时先关闭开关（因为去噪从 t=1.0 开始，前期绝不允许加水印破坏构图！）
            lora_moe.set_moe_context(transformer, secret_bits=None, image_context=None)

            # 👉 核心回调函数：在生成的每一步实时控制水印开关
            def wm_step_callback(pipe, step_index, timestep, callback_kwargs):
                # SD3 的 timestep 是从 1000 降到 0
                t_float = timestep.item() / 1000.0

                # 当进入最后的 40% 细节刻画阶段时，才物理开启水印开关！
                if t_float < 0.4:
                    lora_moe.set_moe_context(
                        transformer,
                        secret_bits=secret_bits_batch_val,
                        image_context=None,
                        timestep=torch.tensor([t_float], device=accelerator.device),
                    )
                else:
                    lora_moe.set_moe_context(
                        transformer,
                        secret_bits=None,
                        image_context=None,
                        timestep=torch.tensor([t_float], device=accelerator.device),
                    )

                return callback_kwargs

            # 生成图片，并把我们写好的 callback 传进去
            latents_wm = pipeline(
                prompt=args.validation_prompt,
                num_inference_steps=28,
                generator=gen_wm,
                guidance_scale=7.0,
                output_type="latent",
                callback_on_step_end=wm_step_callback  # 👈 挂载门控！
            ).images

            # 记录水印图的潜变量 (只用水印图来测 ACC)
            watermarked_latents.append(latents_wm[0])

            # 解码成像素图
            img_wm = vae.decode(latents_wm / vae.config.scaling_factor, return_dict=False)[0]
            img_wm = pipeline.image_processor.postprocess(img_wm, output_type="pil")[0]
            watermarked_images.append(img_wm)  # 加入列表

            # ==========================================
            # 2. 生成【干净原图】
            # ==========================================
            # 彻底关闭水印开关
            lora_moe.set_moe_context(transformer, secret_bits=None, image_context=None)

            # 重新实例化 generator，使用与上面【完全相同】的 seed
            gen_clean = torch.Generator(device=accelerator.device).manual_seed(current_seed)

            latents_clean = pipeline(
                prompt=args.validation_prompt,
                num_inference_steps=28,
                generator=gen_clean,
                guidance_scale=7.0,
                output_type="latent"
            ).images

            # 解码成像素图
            img_clean = vae.decode(latents_clean / vae.config.scaling_factor, return_dict=False)[0]
            img_clean = pipeline.image_processor.postprocess(img_clean, output_type="pil")[0]
            watermarked_images.append(img_clean)  # 紧接着加入列表

    # 验证结束后清空上下文，以免影响后续训练
    lora_moe.set_moe_context(transformer, secret_bits=None, image_context=None)

    del pipeline
    torch.cuda.empty_cache()

    # 返回：[水图的潜变量...] 和 [水图1, 原图1, 水图2, 原图2...]
    return watermarked_latents, watermarked_images

def main(args):
    logging_dir = Path(args.output_dir, "logs")
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    # 修复1：使用 args.gradient_accumulation_steps
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="wandb",
        project_config=accelerator_project_config,
    )
    # LPIPS is not used in this training script.
    # Disabled to avoid unnecessary AlexNet weight download at startup.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO)
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    if args.seed is not None:
        set_seed(args.seed)

    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)

    # 加载 tokenizers
    tokenizer_1 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer")
    tokenizer_2 = CLIPTokenizer.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_2")
    tokenizer_3 = T5TokenizerFast.from_pretrained(args.pretrained_model_name_or_path, subfolder="tokenizer_3")

    # 加载文本编码器
    text_encoder_1 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_2")
    text_encoder_3 = T5EncoderModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder_3")

    # 加载 scheduler
    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(args.pretrained_model_name_or_path, subfolder="scheduler")

    # 加载 VAE
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_name_or_path, subfolder="vae")
    # 强制bf16
    weight_dtype = torch.bfloat16
    # 加载两个 transformer（学生和教师）
    transformer = SD3Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer")
    transformer_frozen = SD3Transformer2DModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="transformer")
    transformer.to(accelerator.device, dtype=weight_dtype)
    transformer_frozen.to(accelerator.device, dtype=weight_dtype)
    # 加载水印组件
    GT_secret = torch.load(args.secret_pt_path)
    watermark_extractor = watermarkModel.Extractor_forLatent(secret_size=48)
    watermark_extractor.load_state_dict(torch.load(os.path.join(args.pretrainedWM_dir, "decoder.pth")))
    WM_residual = torch.load(args.wm_residual_path)

    # 冻结所有不需要训练的部分
    vae.requires_grad_(False)
    transformer_frozen.requires_grad_(False)
    text_encoder_1.requires_grad_(False)
    text_encoder_2.requires_grad_(False)
    text_encoder_3.requires_grad_(False)
    watermark_extractor.requires_grad_(False)


    # 移动冻结组件到 GPU（半精度）
    vae = vae.to(accelerator.device, dtype=weight_dtype)
    text_encoder_1 = text_encoder_1.to(accelerator.device, dtype=weight_dtype)
    text_encoder_2 = text_encoder_2.to(accelerator.device, dtype=weight_dtype)
    text_encoder_3 = text_encoder_3.to(accelerator.device, dtype=weight_dtype)
    transformer_frozen = transformer_frozen.to(accelerator.device, dtype=weight_dtype)
    watermark_extractor = watermark_extractor.to(accelerator.device, dtype=weight_dtype)
    GT_secret = GT_secret.to(accelerator.device, dtype=weight_dtype)
    WM_residual = WM_residual.to(accelerator.device, dtype=weight_dtype)

    # 注入 MoE-LoRA
    transformer.requires_grad_(False)
    transformer.to(accelerator.device, dtype=weight_dtype)
    transformer = lora_moe.inject_moe_lora_to_sd3(
        transformer,
        num_experts=args.num_experts,
        rank=args.rank
    )
    load_initial_transformer_weights(transformer, args.init_transformer_dir)
    lora_moe.configure_temporal_widepath(
        transformer,
        band_edges=parse_float_csv(args.path_band_edges),
        target_k=args.path_target_k,
        path_blocks=parse_int_csv(args.path_blocks),
        path_projections=parse_str_csv(args.path_projections),
        target_eps=args.path_target_eps,
        kl_weight=args.path_kl_weight,
        route_margin=args.path_route_margin,
        separation_margin=args.path_separation_margin,
        routing_temperature=args.path_routing_temperature,
    )
    transformer.to(accelerator.device, dtype=weight_dtype)
    target_dtype = next(transformer.parameters()).dtype

    for name, param in transformer.named_parameters():
        if "lora_" in name or "router" in name:
            param.requires_grad = True
            param.data = param.data.float()  # ⭐ fp32 master weight
        else:
            param.requires_grad = False
    # 在设置 requires_grad 后，定义优化器前添加
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    print(f"Number of trainable parameters: {len(trainable_params)}")
    if len(trainable_params) == 0:
        raise ValueError("No trainable parameters found! Check MoE-LoRA injection.")
    else:
        print("First 5 trainable parameter names:")
        for n, p in transformer.named_parameters():
            if p.requires_grad:
                print(n)
                break  # 只打印第一个

    # 设置可训练参数（仅 MoE 相关），并强制转为 fp32
    for name, param in transformer.named_parameters():
        if "lora_" in name or "router" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

    if args.gradient_checkpointing:
        transformer.enable_gradient_checkpointing()

    transformer.train()



    # 将可训练参数强制转为 fp32（仅一次）
    #cast_training_params([transformer], dtype=torch.float32)
    trainable_params = [p for p in transformer.parameters() if p.requires_grad]
    # 保存/加载钩子（修复：不依赖 PeftModel）
    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    def save_model_hook(models, weights, output_dir):
        if accelerator.is_main_process:
            for model in models:
                model = unwrap_model(model)
                # 直接保存整个 transformer 的 state_dict（或仅保存 adapter）
                # 这里选择保存完整模型（后续可用 from_pretrained 加载）
                model.save_pretrained(os.path.join(output_dir, "transformer"))
                if len(weights) > 0:
                    weights.pop()

    def load_model_hook(models, input_dir):
        while len(models) > 0:
            model = models.pop()
            model = unwrap_model(model)
            load_dir = os.path.join(input_dir, "transformer")
            if os.path.exists(load_dir):
                # 加载完整模型（需确保结构一致）
                model.load_state_dict(torch.load(os.path.join(load_dir, "diffusion_pytorch_model.bin")))
            else:
                logger.warning(f"Transformer weights not found in {load_dir}, skipping load.")

    accelerator.register_save_state_pre_hook(save_model_hook)
    accelerator.register_load_state_pre_hook(load_model_hook)

    # 数据集
    train_dataset = DreamBoothDataset_modified(
        instance_data_root=args.instance_data_dir,
        tokenizers=[tokenizer_1, tokenizer_2, tokenizer_3],
        size=args.resolution,
        center_crop=args.center_crop,
        tokenizer_max_length=args.tokenizer_max_length,
        prompt_trigger=args.trigger,
        use_null_prompt=args.use_null_prompt
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        collate_fn=lambda examples: collate(examples),
        num_workers=args.dataloader_num_workers,
    )
    # 🔥 第一次 prepare（模型 + dataloader）
    transformer, train_dataloader = accelerator.prepare(
        transformer, train_dataloader
    )

    # 重新拿可训练参数
    params_to_optimize = [p for p in transformer.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=args.max_train_steps,
    )

    # 🔥 第二次 prepare（optimizer + scheduler）
    optimizer, lr_scheduler = accelerator.prepare(
        optimizer, lr_scheduler
    )

    # 计算训练步数
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / accelerator.num_processes)
    num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # WandB 初始化
    if args.wandb_run_name is None:
        args.wandb_run_name = args.output_dir
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        accelerator.init_trackers(project_name=args.wandb_project_name, config=tracker_config,
                                  init_kwargs={"wandb": {"name": args.wandb_run_name}})

    # 训练信息
    total_batch_size = args.train_batch_size * accelerator.num_processes
    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    global_step = 0
    first_epoch = 0
    if args.resume_from_checkpoint:
        path = os.path.basename(args.resume_from_checkpoint)
        accelerator.print(f"Resuming from checkpoint {path}")
        accelerator.load_state(args.resume_from_checkpoint)
        global_step = int(path.split("-")[1])
        first_epoch = int(global_step // num_update_steps_per_epoch)

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process
    )
    initial_ft_probe_done = bool(args.skip_ft_probe_at_start)

    watermark_extractor.eval()

    for epoch in range(first_epoch, num_train_epochs):
        for step, batch in enumerate(train_dataloader):
            # 前向传播前确保 MoE 上下文清空
            for m in transformer.modules():
                if hasattr(m, "current_logits"):
                    m.current_logits = None
                if hasattr(m, "set_global_context"):
                    m.set_global_context(None)

            transformer.train()
            pixel_values = batch["pixel_values"]

            # VAE 编码
            with accelerator.autocast():
                model_input = vae.encode(pixel_values).latent_dist.sample()
            model_input = model_input * vae.config.scaling_factor

            noise = torch.randn_like(model_input)
            bsz = model_input.shape[0]

            # 采样时间步（修复：使用 diff_t_prob 实现非均匀采样）
            # ==========================================================
            # 👉 终极修正：SD3 Flow Matching 专属连续时间步随机采样
            # ==========================================================
            # 1. 在 [0.0, 1.0) 之间生成连续的 float 随机数
            t_float_sampler = torch.rand((bsz,), device=model_input.device)

            # 2. 转换成 Transformer 需要的时间步输入格式 (通常是 t * 1000)
            timesteps = t_float_sampler * noise_scheduler.config.num_train_timesteps

            # 3. 转换形状供 Flow Matching 加噪使用
            t_sigmas = t_float_sampler.view(-1, 1, 1, 1).to(dtype=weight_dtype)
            noisy_model_input = (1.0 - t_sigmas) * model_input + t_sigmas * noise

            # 获取文本 embeddings
            input_ids_list = batch['input_ids_list']
            input_ids_trigger_list = batch['input_ids_trigger_list']
            text_encoders = [text_encoder_1, text_encoder_2, text_encoder_3]

            prompt_embeds, pooled_prompt_embeds = encode_prompt(text_encoders, input_ids_list)
            prompt_embeds = prompt_embeds.to(weight_dtype)
            pooled_prompt_embeds = pooled_prompt_embeds.to(weight_dtype)

            # Non-triggered ownership watermark: the student receives the same normal prompt.
            # Secret+timestep controls the internal MoE route; no trigger token is required.
            prompt_embeds_WM = prompt_embeds
            pooled_prompt_embeds_WM = pooled_prompt_embeds

            secret_bits_batch = GT_secret.to(weight_dtype).repeat(bsz, 1)
            wrong_secret_bits_batch = make_fixed_wrong_secret(
                GT_secret, seed=args.wrong_key_seed
            ).to(weight_dtype).repeat(bsz, 1)

            # Test the loaded checkpoint before its first continuation update.
            # This produces train_step=0 records under the same temporary
            # clean downstream-LoRA attack used at later training checkpoints.
            if (
                not initial_ft_probe_done
                and args.ft_probe_steps > 0
                and accelerator.is_main_process
            ):
                run_shadow_lora_finetune_probe(
                    args,
                    transformer,
                    transformer_frozen,
                    watermark_extractor,
                    model_input,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    secret_bits_batch,
                    wrong_secret_bits_batch,
                    noise_scheduler,
                    vae,
                    text_encoder_1,
                    tokenizer_1,
                    text_encoder_2,
                    tokenizer_2,
                    text_encoder_3,
                    tokenizer_3,
                    accelerator,
                    global_step,
                )
                initial_ft_probe_done = True
                transformer.train()

            # 教师模型预测（干净目标）
            with torch.no_grad():
                with accelerator.autocast():
                    target_v_clean = transformer_frozen(
                        hidden_states=noisy_model_input,
                        timestep=timesteps,
                        encoder_hidden_states=prompt_embeds,
                        pooled_projections=pooled_prompt_embeds,
                        return_dict=False,
                    )[0]

                # 反推干净潜变量（你的 MoE 路由需要用到这个特征，所以必须保留它！）
                t_norm = timesteps.float().view(-1, 1, 1, 1) / noise_scheduler.config.num_train_timesteps
                x0_clean_pred = noisy_model_input - t_norm * target_v_clean

                # watermark 强度
                wm_strength = 0.02

                wm_res_batch = wm_strength * WM_residual.to(x0_clean_pred.dtype).repeat(bsz, 1, 1, 1)

                # 在 x0 上加 watermark
                x0_watermarked = x0_clean_pred + wm_res_batch

                # 重新计算 velocity (Flow Matching)
                target_v_watermarked = (noisy_model_input - x0_watermarked) / (t_norm + 1e-6)

            unwrapped = accelerator.unwrap_model(transformer)
            # 注入到所有 MoE 层
            lora_moe.set_moe_context(
                transformer,
                secret_bits=secret_bits_batch,
                image_context=None,
                timestep=t_float_sampler,
                wrong_secret_bits=wrong_secret_bits_batch,
            )
            # =============================================
            # 学生模型预测
            with accelerator.autocast():
                model_pred_v = transformer(
                    hidden_states=noisy_model_input,
                    timestep=timesteps,
                    encoder_hidden_states=prompt_embeds_WM,
                    pooled_projections=pooled_prompt_embeds_WM,
                    return_dict=False
                )[0]
                model_pred_v = model_pred_v.to(torch.float32)
                target_v_watermarked = target_v_watermarked.to(torch.float32)
                target_v_clean = target_v_clean.to(torch.float32)
            # 计算损失
                # =====================================================
                # ⬇️ 替换开始：终极修复版 Loss 计算 ⬇️
                # =====================================================

                # -----------------------------------------------------
                # 1. 铁腕画质保真 (改用 Smooth L1，全时间段死守结构)
                # -----------------------------------------------------
                # 注意：变量名统一改成 loss_clean_final，避免和上面的旧代码冲突
            loss_clean_raw = F.smooth_l1_loss(model_pred_v, target_v_clean, reduction='none').mean(dim=[1, 2, 3])
            loss_clean_final = loss_clean_raw.mean()

                # -----------------------------------------------------
                # 2. 水印残差 MSE
                # -----------------------------------------------------
            loss_wm_raw = F.mse_loss(model_pred_v, target_v_watermarked, reduction='none').mean(dim=[1, 2, 3])

                # -----------------------------------------------------
                # 3. 提取器 Hinge 损失
                # -----------------------------------------------------
            t_float = timesteps.float() / noise_scheduler.config.num_train_timesteps
            t_norm = t_float.view(-1, 1, 1, 1)
            pred_x0 = noisy_model_input.to(torch.float32) - t_norm.to(torch.float32) * model_pred_v

            extracted_logits = watermark_extractor(pred_x0.to(weight_dtype))  # [batch, 48]
            targets_sign = secret_bits_batch.float() * 2.0 - 1.0
            margin = 2.0
            hinge_loss_raw = F.relu(margin - targets_sign * extracted_logits).mean(dim=1)  # [batch]

                # -----------------------------------------------------
                # 4. 【核心修复】共享的时间门控
                # -----------------------------------------------------
                # 只允许在 t < 0.4 (生成中后期细节刻画) 时生效
            time_weight = torch.sigmoid((0.4 - t_float) / 0.05)

                # 👉 提取器和残差 都必须戴上时间枷锁！
            loss_extractor = (hinge_loss_raw * time_weight).mean()
            loss_wm_gated = (loss_wm_raw * time_weight).mean()  # 必须乘 time_weight！

                # -----------------------------------------------------
                # 5. MoE 负载均衡
                # -----------------------------------------------------
            loss_moe_bal = lora_moe.get_moe_bal_loss(transformer)
            path_stats = lora_moe.get_temporal_path_losses(transformer)
            loss_path_align = path_stats["path_align_loss"]
            loss_path_margin = path_stats["path_margin_loss"]
            loss_path_sep = path_stats["path_separation_loss"]
            path_target_mass = path_stats["path_target_mass"]
            path_wrong_target_mass = path_stats["path_wrong_target_mass"]
            path_entropy = path_stats["path_entropy"]

                # -----------------------------------------------------
                # 6. 纯净 Total Loss 组装
                # -----------------------------------------------------
            lambda_clean = 5.0
            lambda_wm = 2.0
            lambda_ext = 0.5
            lambda_bal = args.lambda_bal

            total_loss = (
                        lambda_clean * loss_clean_final  # 保真随时生效
                        + lambda_wm * loss_wm_gated  # 水印只在后期生效
                        + lambda_ext * loss_extractor  # 雷达只在后期生效
                        + lambda_bal * loss_moe_bal
                        + args.lambda_path_align * loss_path_align
                        + args.lambda_path_margin * loss_path_margin
                        + args.lambda_path_separation * loss_path_sep
            )

            wm_norm = wm_res_batch.abs().mean()
            loss_strength = F.relu(0.015 - wm_norm)
            total_loss = total_loss + 0.5 * loss_strength

                # =====================================================
                # ⬆️ 替换结束 ⬆️
                # =====================================================
            # 反向传播
            optimizer.zero_grad()
            accelerator.backward(total_loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(transformer.parameters(), args.max_grad_norm)
            optimizer.step()
            lr_scheduler.step()

            # 清理 MoE 状态
            for m in transformer.modules():
                if hasattr(m, "current_logits"):
                    m.current_logits = None
                if hasattr(m, "set_global_context"):
                    m.set_global_context(None)
            # 清理 MoE 上下文
            lora_moe.set_moe_context(transformer, None, None)
            # 同时清除可能残留的 logits
            for m in transformer.modules():
                if hasattr(m, "current_logits"):
                    m.current_logits = None
            progress_bar.update(1)
            global_step += 1
            if global_step % 100 == 0:
                print(f"V_clean scale: {target_v_clean.abs().mean().item():.6f}")
                print(f"WM_res scale: {wm_res_batch.abs().mean().item():.6f}")
            if global_step % 500 == 0:
                print(f"MoE balance loss: {loss_moe_bal.item():.6f}")
                # 日志（使用 accelerator.log 保证安全）
                # 日志（使用 accelerator.log 保证安全）
            if global_step % 100 == 0:
                print(
                    f"wm_norm={wm_norm.item():.4f} | "
                    f"clean={loss_clean_final.item():.4f} | "  # 👈 改成了 loss_clean_final
                    f"wm={loss_wm_gated.item():.4f} | "  # 👈 改成了 loss_wm_gated
                    f"ext={loss_extractor.item():.4f}"
                )
                logs = {
                    "Train/loss_clean": loss_clean_final.item(),  # 👈 这里
                    "Train/loss_wm": loss_wm_gated.item(),  # 👈 这里
                    "Train/loss_extractor": loss_extractor.item(),
                    "Train/loss_strength": loss_strength.item(),
                    "Train/loss_moe_bal": loss_moe_bal.item(),
                    "Train/loss_path_align": loss_path_align.item(),
                    "Train/loss_path_margin": loss_path_margin.item(),
                    "Train/loss_path_separation": loss_path_sep.item(),
                    "Train/path_target_mass": path_target_mass.item(),
                    "Train/path_wrong_target_mass": path_wrong_target_mass.item(),
                    "Train/path_entropy": path_entropy.item(),
                    "Train/wm_norm": wm_norm.item(),
                    "Train/total_loss": total_loss.item(),
                    "Train/lr": optimizer.param_groups[0]["lr"],
                    "step": global_step
                }
                accelerator.log(logs)

            progress_bar.set_postfix(**{
                "L_clean": f"{loss_clean_final.item():.4f}",  # 👈 这里
                "L_wm": f"{loss_wm_gated.item():.4f}",  # 👈 这里
                "L_ext": f"{loss_extractor.item():.4f}",
                "L_str": f"{loss_strength.item():.4f}",
                "wm": f"{wm_norm.item():.4f}",
                "L_bal": f"{loss_moe_bal.item():.4f}",
                "L_path": f"{loss_path_align.item():.4f}",
                "Pmass": f"{path_target_mass.item():.3f}",
                "Loss": f"{total_loss.item():.4f}",
                "step": global_step
            })
            # Periodic temporary downstream-LoRA fine-tuning robustness probe.
            # It adds attack LoRA branches to the base path, trains them on clean
            # flow matching, evaluates watermark retention, then deletes them.
            if (
                args.ft_probe_every > 0
                and global_step > 0
                and global_step % args.ft_probe_every == 0
                and accelerator.is_main_process
            ):
                run_shadow_lora_finetune_probe(
                    args,
                    transformer,
                    transformer_frozen,
                    watermark_extractor,
                    model_input,
                    prompt_embeds,
                    pooled_prompt_embeds,
                    secret_bits_batch,
                    wrong_secret_bits_batch,
                    noise_scheduler,
                    vae,
                    text_encoder_1,
                    tokenizer_1,
                    text_encoder_2,
                    tokenizer_2,
                    text_encoder_3,
                    tokenizer_3,
                    accelerator,
                    global_step,
                )
                transformer.train()

            # 保存 checkpoint
            if accelerator.is_main_process and global_step % args.checkpointing_steps == 0:
                torch.cuda.empty_cache()
                if args.checkpoints_total_limit is not None:
                    checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith("checkpoint")]
                    checkpoints = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))
                    if len(checkpoints) >= args.checkpoints_total_limit:
                        num_to_remove = len(checkpoints) - args.checkpoints_total_limit + 1
                        for rem in checkpoints[:num_to_remove]:
                            shutil.rmtree(os.path.join(args.output_dir, rem))
                            logger.info(f"Removed old checkpoint {rem}")
                save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                accelerator.save_state(save_path)
                logger.info(f"Saved state to {save_path}")

            # 验证
            if accelerator.is_main_process and args.validation_prompt is not None and global_step % args.validation_steps == 0:
                torch.cuda.empty_cache()
                transformer.eval()
                watermark_extractor.eval()
                try:
                    with torch.no_grad():
                        # 获取生成的潜变量和图片
                        val_latents_list, watermarked_imgs = log_validation(
                            args, transformer, transformer_frozen, noise_scheduler, vae,
                            text_encoder_1, tokenizer_1, text_encoder_2, tokenizer_2, text_encoder_3, tokenizer_3,
                            accelerator
                        )

                        # 👉 核心修改：直接在纯净的潜变量上提水印，绝对不经过 VAE Encode！
                        val_latent_tensors = torch.stack(val_latents_list).to(dtype=weight_dtype,
                                                                              device=accelerator.device)

                        decoded_result = torch.round(torch.sigmoid(watermark_extractor(val_latent_tensors))).cpu()
                        GT_secret_repeated = GT_secret.view(1, 48).repeat(val_latent_tensors.shape[0], 1).cpu()
                        correct = (decoded_result == GT_secret_repeated).sum().item()
                        acc = correct / GT_secret_repeated.numel()

                        logger.info(f"🎯 Validation Latent ACC: {acc:.4f}")

                        # 👉 核心修复：只挑出列表里偶数索引的图片（即“水印图”），不要去测“原图”
                        imgs_for_e2e = watermarked_imgs[::2]

                        transform = transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize([0.5], [0.5])
                        ])

                        # 把挑出来的水图转为 Tensor
                        val_img_tensors = torch.stack([transform(img) for img in imgs_for_e2e]).to(dtype=weight_dtype,
                                                                                                   device=accelerator.device)
                        val_reencoded_latents = vae.encode(
                            val_img_tensors).latent_dist.sample() * vae.config.scaling_factor

                        # 再次通过提取器
                        e2e_preds = torch.round(torch.sigmoid(watermark_extractor(val_reencoded_latents))).cpu()

                        # 此时 e2e_preds 是 2 行，GT_secret_repeated 也是 2 行，完美对齐！
                        e2e_correct = (e2e_preds == GT_secret_repeated).sum().item()
                        e2e_acc = e2e_correct / GT_secret_repeated.numel()

                        logger.info(f"🔬 Validation End-to-End (Pixel) ACC: {e2e_acc:.4f}")
                        # 记录图像和准确率
                        accelerator.log({
                            "Validation/Generated_Images": [wandb.Image(img, caption=f"Val_{i}") for i, img in
                                                            enumerate(watermarked_imgs)],
                            "Validation/accuracy": acc,
                            "Validation/global_step": global_step
                        })
                finally:
                    transformer.train()

            if global_step >= args.max_train_steps:
                break

    accelerator.wait_for_everyone()
    accelerator.end_training()

if __name__ == "__main__":
    def parse_args():
        parser = argparse.ArgumentParser(description="SD3 MoE-LoRA Watermark Training")
        parser.add_argument("--lambda_wm", type=float, default=1.0,help="水印速度跟踪损失的权重")
        parser.add_argument("--lambda_clean", type=float, default=1.0,help="画质保持损失的权重")
        parser.add_argument("--lambda_ext", type=float, default=0.01,help="提取器损失的权重")
        parser.add_argument("--extractor_time_threshold", type=float, default=0.1,help="时间门控阈值：仅当 t_norm < 该值时启用提取器损失")
        parser.add_argument("--lambda_lpips", type=float, default=0.1, help="")
        parser.add_argument("--lpips_freq", type=float, default=20,help="")
        # 基础路径
        parser.add_argument("--pretrained_model_name_or_path", type=str, required=True)
        parser.add_argument("--output_dir", type=str, default="sd3_moe_lora_wm")
        parser.add_argument("--instance_data_dir", type=str, required=True)
        parser.add_argument("--secret_pt_path", type=str, required=True)
        parser.add_argument("--wm_residual_path", type=str, required=True)
        parser.add_argument("--pretrainedWM_dir", type=str, required=True)
        parser.add_argument(
            "--init_transformer_dir",
            type=str,
            default=None,
            help="Optional existing TG-MoE checkpoint-*/transformer directory used as initialization.",
        )

        # MoE-LoRA 参数
        parser.add_argument("--num_experts", type=int, default=4)
        parser.add_argument("--rank", type=int, default=32)
        parser.add_argument("--lambda_bal", type=float, default=0.01)

        # Non-triggered secret x timestep temporal wide-path routing.
        parser.add_argument("--path_band_edges", type=str, default="0.0,0.2,0.4")
        parser.add_argument("--path_target_k", type=int, default=2)
        parser.add_argument("--path_blocks", type=str, default="20,21,22,23")
        parser.add_argument("--path_projections", type=str, default="to_q,to_k,to_v,to_out")
        parser.add_argument("--path_target_eps", type=float, default=1e-4)
        parser.add_argument("--path_kl_weight", type=float, default=0.2)
        parser.add_argument("--path_route_margin", type=float, default=0.10)
        parser.add_argument("--path_separation_margin", type=float, default=0.10)
        parser.add_argument("--path_routing_temperature", type=float, default=1.0)
        parser.add_argument("--lambda_path_align", type=float, default=0.10)
        parser.add_argument("--lambda_path_margin", type=float, default=0.10)
        parser.add_argument("--lambda_path_separation", type=float, default=0.05)
        parser.add_argument("--wrong_key_seed", type=int, default=20260724)

        # Automatic temporary downstream-LoRA fine-tuning test during training.
        parser.add_argument("--ft_probe_every", type=int, default=500)
        parser.add_argument("--ft_probe_steps", type=int, default=16)
        parser.add_argument("--ft_probe_milestones", type=str, default="0,4,8,16")
        parser.add_argument("--ft_probe_eval_times", type=str, default="0.1,0.3")
        parser.add_argument("--ft_probe_rank", type=int, default=8)
        parser.add_argument("--ft_probe_alpha", type=float, default=8.0)
        parser.add_argument("--ft_probe_lr", type=float, default=1e-4)
        parser.add_argument("--ft_probe_weight_decay", type=float, default=1e-4)
        parser.add_argument("--ft_probe_max_grad_norm", type=float, default=1.0)
        parser.add_argument("--ft_probe_blocks", type=str, default="20,21,22,23")
        parser.add_argument("--ft_probe_projections", type=str, default="to_q,to_k,to_v,to_out")
        parser.add_argument(
            "--ft_probe_save_pipeline_images",
            action="store_true",
            help=(
                "Run full SD3 pipeline at every "
                "FT-probe milestone and save images."
            ),
        )
        parser.add_argument(
            "--ft_probe_pipeline_thumbnail",
            type=int,
            default=384,
            help=(
                "Thumbnail size used for "
                "pair and timeline previews."
            ),
        )
        parser.add_argument(
            "--skip_ft_probe_at_start",
            action="store_true",
            help="Skip the train_step=0 probe of the loaded TG-MoE checkpoint.",
        )

        # 显存与精度
        parser.add_argument("--mixed_precision", type=str, default="bf16", choices=["no", "fp16", "bf16"])
        parser.add_argument("--train_batch_size", type=int, default=1)
        parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
        parser.add_argument("--gradient_checkpointing", action="store_true")

        # 时间步门控
        parser.add_argument("--loss_t_threshold", type=float, nargs=2, default=[0.1, 0.6],
                            help="水印注入的时间步范围 [min, max] (归一化0~1)")
        parser.add_argument("--wmLoss_weight", type=float, default=5)
        parser.add_argument("--coeff_steepness", type=float, default=12.0)
        parser.add_argument("--max_grad_norm", type=float, default=5)

        # 优化器与调度
        parser.add_argument("--learning_rate", type=float, default=2e-4)
        parser.add_argument("--max_train_steps", type=int, default=20000
                            )
        parser.add_argument("--lr_scheduler", type=str, default="constant_with_warmup")
        parser.add_argument("--lr_warmup_steps", type=int, default=100)
        parser.add_argument("--lr_num_cycles", type=int, default=1)
        parser.add_argument("--lr_power", type=float, default=1.0)
        parser.add_argument("--adam_beta1", type=float, default=0.9)
        parser.add_argument("--adam_beta2", type=float, default=0.999)
        parser.add_argument("--adam_weight_decay", type=float, default=1e-4)
        parser.add_argument("--adam_epsilon", type=float, default=1e-08)

        # 验证与保存
        parser.add_argument("--checkpointing_steps", type=int, default=500)
        parser.add_argument("--checkpoints_total_limit", type=int, default=3)
        parser.add_argument("--validation_steps", type=int, default=500)
        parser.add_argument("--validation_prompt", type=str, default="A high quality photo of a landscape")
        parser.add_argument("--num_validation_images", type=int, default=2)

        # 其他
        parser.add_argument("--seed", type=int, default=42)
        parser.add_argument("--resolution", type=int, default=1024)
        parser.add_argument("--dataloader_num_workers", type=int, default=4)
        parser.add_argument("--wandb_project_name", type=str, default="sd3-moe-watermark")
        parser.add_argument("--wandb_run_name", type=str, default=None)
        parser.add_argument("--trigger", type=str, default="")
        parser.add_argument("--use_null_prompt", action="store_true")
        parser.add_argument("--center_crop", action="store_true")
        parser.add_argument("--random_flip", action="store_true")
        parser.add_argument("--tokenizer_max_length", type=int, default=None)
        parser.add_argument("--diff_t_prob", action="store_true", help="启用非均匀时间步采样（使用倒数权重）")
        parser.add_argument("--resume_from_checkpoint", type=str, default=None)

        return parser.parse_args()

    args = parse_args()
    main(args)












