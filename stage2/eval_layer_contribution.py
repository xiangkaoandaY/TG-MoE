import os
import csv
import argparse
from pathlib import Path

import torch
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from safetensors.torch import load_file

from diffusers import (
    DiffusionPipeline,
    SD3Transformer2DModel,
)

import lora_moe_temporal_widepath as lora_moe
import watermarkModel


# ============================================================
# Utils
# ============================================================

def get_block_index(layer_name):
    """
    transformer_blocks.20.attn.to_q -> 20
    """
    import re

    m = re.search(
        r"transformer_blocks\.(\d+)",
        layer_name
    )

    if m is None:
        return None

    return int(m.group(1))


def load_tgmoe_weights(
    transformer,
    transformer_dir,
):
    """
    Load trained TG-MoE transformer weights.
    """

    transformer_dir = Path(transformer_dir)

    candidates = [
        transformer_dir
        / "diffusion_pytorch_model.safetensors",

        transformer_dir
        / "diffusion_pytorch_model.bin",

        transformer_dir
        / "pytorch_model.bin",
    ]

    state_path = None

    for path in candidates:
        if path.exists():
            state_path = path
            break

    if state_path is None:
        raise FileNotFoundError(
            f"No transformer checkpoint found in "
            f"{transformer_dir}"
        )

    print(
        f"[Load] TG-MoE checkpoint: {state_path}"
    )

    if state_path.suffix == ".safetensors":

        state = load_file(
            str(state_path)
        )

    else:

        state = torch.load(
            state_path,
            map_location="cpu"
        )

        if (
            isinstance(state, dict)
            and "state_dict" in state
        ):
            state = state["state_dict"]

    missing, unexpected = (
        transformer.load_state_dict(
            state,
            strict=False,
        )
    )

    serious_missing = [
        x for x in missing
        if any(
            key in x
            for key in [
                "lora_A",
                "lora_B",
                "lora_bias",
                "router",
                "bit_router",
            ]
        )
    ]

    if serious_missing:
        print(
            "[Warning] Missing TG-MoE keys:"
        )

        for x in serious_missing[:20]:
            print("   ", x)

    print(
        f"[Load] missing={len(missing)}, "
        f"unexpected={len(unexpected)}"
    )


# ============================================================
# Block ablation
# ============================================================

def save_original_scaling(transformer):

    scaling = {}

    for name, module in transformer.named_modules():

        if isinstance(
            module,
            lora_moe.MoELoRALayer,
        ):
            scaling[name] = module.scaling

    return scaling


def restore_scaling(
    transformer,
    original_scaling,
):

    for name, module in transformer.named_modules():

        if not isinstance(
            module,
            lora_moe.MoELoRALayer,
        ):
            continue

        if name in original_scaling:
            module.scaling = (
                original_scaling[name]
            )


def disable_block(
    transformer,
    block_id,
):
    """
    Disable only the SGML residual of one block.
    The frozen SD3 base path remains unchanged.
    """

    count = 0

    for name, module in transformer.named_modules():

        if not isinstance(
            module,
            lora_moe.MoELoRALayer,
        ):
            continue

        block = get_block_index(
            module.layer_name
        )

        if block == block_id:

            module.scaling = 0.0
            count += 1

    print(
        f"[Ablation] Block {block_id}: "
        f"disabled {count} SGML projections"
    )

    return count


# ============================================================
# RGB -> latent -> extractor
# ============================================================

image_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5],
    ),
])


@torch.no_grad()
def extract_rgb_logits(
    images,
    vae,
    extractor,
    device,
    dtype,
):

    tensors = torch.stack([
        image_transform(
            image.convert("RGB")
        )
        for image in images
    ])

    tensors = tensors.to(
        device=device,
        dtype=dtype,
    )

    posterior = vae.encode(
        tensors
    ).latent_dist

    # 与论文 Appendix 中的 deterministic
    # RGB re-encoding 保持一致
    latents = posterior.mode()

    latents = (
        latents
        * vae.config.scaling_factor
    )

    extractor_dtype = (
        next(
            extractor.parameters()
        ).dtype
    )

    logits = extractor(
        latents.to(
            dtype=extractor_dtype
        )
    )

    return logits.float()


def calculate_metrics(
    logits,
    owner_secret,
):

    secret = (
        owner_secret
        .float()
        .reshape(1, -1)
    )

    secret = secret.expand(
        logits.shape[0],
        -1,
    )

    pred = (
        torch.sigmoid(logits)
        >= 0.5
    ).float()

    # image-wise RGB BitAcc
    image_bitacc = (
        pred == secret
    ).float().mean(dim=1)

    bitacc = (
        image_bitacc.mean().item()
    )

    # signed owner score
    signs = (
        secret * 2.0 - 1.0
    )

    owner_score = (
        signs * logits
    ).mean().item()

    # aggregate logits first,
    # corresponding to model-level extraction
    aggregate_logits = (
        logits.mean(dim=0)
    )

    aggregate_pred = (
        aggregate_logits >= 0
    ).float()

    aggregate_bitacc = (
        aggregate_pred
        == secret[0]
    ).float().mean().item()

    return {
        "bitacc":
            bitacc,

        "aggregate_bitacc":
            aggregate_bitacc,

        "owner_score":
            owner_score,

        "std":
            image_bitacc.std(
                unbiased=False
            ).item(),
    }


# ============================================================
# Image generation
# ============================================================

@torch.no_grad()
def generate_images(
    pipe,
    transformer,
    prompts,
    seeds,
    owner_secret,
    device,
):

    images = []

    owner_secret = (
        owner_secret
        .reshape(1, 48)
        .to(
            device=device,
            dtype=next(
                transformer.parameters()
            ).dtype,
        )
    )

    num_train_timesteps = getattr(
        pipe.scheduler.config,
        "num_train_timesteps",
        1000,
    )

    for index, prompt in enumerate(prompts):

        seed = seeds[index]

        print(
            f"  image "
            f"{index + 1}/{len(prompts)}"
        )

        generator = (
            torch.Generator(
                device=device
            )
            .manual_seed(seed)
        )

        # generation begins with watermark disabled
        lora_moe.set_moe_context(
            transformer,
            secret_bits=None,
            timestep=None,
        )

        def callback(
            pipe,
            step_index,
            timestep,
            callback_kwargs,
        ):

            if torch.is_tensor(timestep):
                t_value = (
                    float(
                        timestep
                        .detach()
                        .float()
                        .item()
                    )
                    / float(
                        num_train_timesteps
                    )
                )
            else:
                t_value = (
                    float(timestep)
                    / float(
                        num_train_timesteps
                    )
                )

            # same inference gate as your training code
            if t_value < 0.4:

                lora_moe.set_moe_context(
                    transformer,
                    secret_bits=owner_secret,
                    timestep=torch.tensor(
                        [t_value],
                        device=device,
                    ),
                )

            else:

                lora_moe.set_moe_context(
                    transformer,
                    secret_bits=None,
                    timestep=torch.tensor(
                        [t_value],
                        device=device,
                    ),
                )

            return callback_kwargs

        output = pipe(
            prompt=prompt,
            num_inference_steps=28,
            guidance_scale=7.0,
            generator=generator,
            callback_on_step_end=callback,
        )

        images.append(
            output.images[0]
        )

    lora_moe.set_moe_context(
        transformer,
        secret_bits=None,
        timestep=None,
    )

    return images


# ============================================================
# One ablation condition
# ============================================================

@torch.no_grad()
def evaluate_condition(
    condition,
    disabled_block,
    pipe,
    transformer,
    vae,
    extractor,
    owner_secret,
    prompts,
    seeds,
    original_scaling,
    device,
    dtype,
):

    print()
    print("=" * 70)
    print(
        f"Condition: {condition}"
    )
    print("=" * 70)

    # restore complete TG-MoE first
    restore_scaling(
        transformer,
        original_scaling,
    )

    if disabled_block is not None:

        count = disable_block(
            transformer,
            disabled_block,
        )

        if count == 0:
            raise RuntimeError(
                f"No SGML modules found "
                f"in block {disabled_block}"
            )

    images = generate_images(
        pipe=pipe,
        transformer=transformer,
        prompts=prompts,
        seeds=seeds,
        owner_secret=owner_secret,
        device=device,
    )

    logits = extract_rgb_logits(
        images=images,
        vae=vae,
        extractor=extractor,
        device=device,
        dtype=dtype,
    )

    metrics = calculate_metrics(
        logits,
        owner_secret,
    )

    print(
        f"RGB BitAcc       = "
        f"{metrics['bitacc']:.4f}"
    )

    print(
        f"Aggregate BitAcc = "
        f"{metrics['aggregate_bitacc']:.4f}"
    )

    print(
        f"Owner Score      = "
        f"{metrics['owner_score']:.4f}"
    )

    return metrics


# ============================================================
# Main
# ============================================================

def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base_model",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--tgmoe_transformer_dir",
        type=str,
        required=True,
        help=(
            "TG-MoE transformer directory, e.g. "
            "checkpoint-20000/transformer"
        ),
    )

    parser.add_argument(
        "--secret_pt",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--pretrainedWM_dir",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="./layer_contribution",
    )

    parser.add_argument(
        "--num_images",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--blocks",
        type=str,
        default="20,21,22,23",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0",
    )

    args = parser.parse_args()

    device = args.device
    dtype = torch.bfloat16

    output_dir = Path(
        args.output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    blocks = [
        int(x)
        for x in args.blocks.split(",")
    ]

    # ========================================================
    # Fixed prompts
    # ========================================================

    prompt_pool = [
        "A photo of a dog sitting on grass.",
        "A red sports car parked on a city street.",
        "A mountain landscape during sunset.",
        "A portrait photo of a young woman.",
        "A wooden house beside a quiet lake.",
        "A cat sitting on a sofa.",
        "A futuristic city at night.",
        "A bowl of fruit on a wooden table.",
        "A lighthouse beside the ocean.",
        "A bicycle parked near a brick wall.",
        "A snow-covered mountain under blue sky.",
        "A small boat sailing on calm water.",
        "A yellow flower in a green field.",
        "A city street during a rainy evening.",
        "A castle on top of a hill.",
        "A bird sitting on a tree branch.",
        "A forest path during autumn.",
        "A cup of coffee on a wooden desk.",
        "A beach during golden hour.",
        "A modern building with glass windows.",
    ]

    if args.num_images > len(
        prompt_pool
    ):
        prompts = (
            prompt_pool
            * (
                args.num_images
                // len(prompt_pool)
                + 1
            )
        )[:args.num_images]

    else:
        prompts = (
            prompt_pool[
                :args.num_images
            ]
        )

    seeds = [
        1000 + i
        for i in range(
            len(prompts)
        )
    ]

    # ========================================================
    # Load base transformer
    # ========================================================

    print(
        "[1/5] Loading base SD3 transformer..."
    )

    transformer = (
        SD3Transformer2DModel
        .from_pretrained(
            args.base_model,
            subfolder="transformer",
            torch_dtype=dtype,
        )
    )

    transformer.requires_grad_(
        False
    )

    # ========================================================
    # Inject exact TG-MoE structure
    # ========================================================

    print(
        "[2/5] Injecting TG-MoE..."
    )

    transformer = (
        lora_moe
        .inject_moe_lora_to_sd3(
            transformer,
            num_experts=4,
            rank=32,
        )
    )

    lora_moe.configure_temporal_widepath(
        transformer,
        band_edges=(
            0.0,
            0.2,
            0.4,
        ),
        target_k=2,
        path_blocks=(
            20,
            21,
            22,
            23,
        ),
        path_projections=(
            "to_q",
            "to_k",
            "to_v",
            "to_out",
        ),
        target_eps=1e-4,
        kl_weight=0.2,
        route_margin=0.10,
        separation_margin=0.10,
        routing_temperature=1.0,
    )

    # ========================================================
    # Load trained TG-MoE
    # ========================================================

    print(
        "[3/5] Loading TG-MoE weights..."
    )

    load_tgmoe_weights(
        transformer,
        args.tgmoe_transformer_dir,
    )

    transformer = transformer.to(
        device=device,
        dtype=dtype,
    )

    transformer.eval()

    # ========================================================
    # Load pipeline
    # ========================================================

    print(
        "[4/5] Loading pipeline..."
    )

    pipe = (
        DiffusionPipeline
        .from_pretrained(
            args.base_model,
            transformer=transformer,
            torch_dtype=dtype,
        )
    )

    pipe = pipe.to(device)

    pipe.set_progress_bar_config(
        disable=True
    )

    vae = pipe.vae

    # ========================================================
    # Load extractor / owner secret
    # ========================================================

    print(
        "[5/5] Loading watermark extractor..."
    )

    extractor = (
        watermarkModel
        .Extractor_forLatent(
            secret_size=48
        )
    )

    decoder_path = os.path.join(
        args.pretrainedWM_dir,
        "decoder.pth",
    )

    extractor.load_state_dict(
        torch.load(
            decoder_path,
            map_location="cpu",
        )
    )

    extractor = extractor.to(
        device=device,
        dtype=dtype,
    )

    extractor.eval()

    extractor.requires_grad_(False)

    owner_secret = torch.load(
        args.secret_pt,
        map_location="cpu",
    )

    owner_secret = (
        owner_secret
        .float()
        .reshape(1, 48)
        .to(device)
    )

    # ========================================================
    # Save original SGML scale
    # ========================================================

    original_scaling = (
        save_original_scaling(
            transformer
        )
    )

    print(
        f"\nFound "
        f"{len(original_scaling)} "
        f"MoE-LoRA modules."
    )

    # ========================================================
    # Conditions
    # ========================================================

    conditions = [
        ("Full TG-MoE", None)
    ]

    for block in blocks:
        conditions.append(
            (
                f"w/o Block {block}",
                block,
            )
        )

    results = []

    for (
        condition,
        disabled_block,
    ) in conditions:

        metrics = evaluate_condition(
            condition=condition,
            disabled_block=disabled_block,
            pipe=pipe,
            transformer=transformer,
            vae=vae,
            extractor=extractor,
            owner_secret=owner_secret,
            prompts=prompts,
            seeds=seeds,
            original_scaling=original_scaling,
            device=device,
            dtype=dtype,
        )

        results.append({
            "condition":
                condition,

            "disabled_block":
                (
                    -1
                    if disabled_block
                    is None
                    else disabled_block
                ),

            **metrics,
        })

    # restore model
    restore_scaling(
        transformer,
        original_scaling,
    )

    # ========================================================
    # CSV
    # ========================================================

    csv_path = (
        output_dir
        / "layer_contribution.csv"
    )

    with open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=[
                "condition",
                "disabled_block",
                "bitacc",
                "aggregate_bitacc",
                "owner_score",
                "std",
            ],
        )

        writer.writeheader()
        writer.writerows(
            results
        )

    print(
        f"\nSaved CSV: {csv_path}"
    )

    # ========================================================
    # Plot BitAcc
    # ========================================================

    labels = [
        x["condition"]
        for x in results
    ]

    values = [
        x["bitacc"]
        for x in results
    ]

    plt.figure(
        figsize=(6.0, 3.8)
    )

    x = np.arange(
        len(labels)
    )

    plt.bar(
        x,
        values,
    )

    plt.xticks(
        x,
        labels,
        rotation=20,
        ha="right",
    )

    plt.ylabel(
        "RGB-level BitAcc"
    )

    plt.ylim(
        0.0,
        1.0,
    )

    plt.axhline(
        0.5,
        linestyle="--",
        linewidth=1,
    )

    plt.tight_layout()

    figure_path = (
        output_dir
        / "layer_contribution.png"
    )

    plt.savefig(
        figure_path,
        dpi=400,
        bbox_inches="tight",
    )

    plt.close()

    print(
        f"Saved figure: "
        f"{figure_path}"
    )

    print()

    print(
        "=" * 65
    )

    print(
        "Final results"
    )

    print(
        "=" * 65
    )

    for result in results:

        print(
            f"{result['condition']:15s} "
            f"| BitAcc="
            f"{result['bitacc']:.4f} "
            f"| Agg="
            f"{result['aggregate_bitacc']:.4f} "
            f"| Score="
            f"{result['owner_score']:.4f}"
        )


if __name__ == "__main__":
    main()