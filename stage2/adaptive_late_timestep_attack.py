#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Adaptive Late-Timestep Attack for TG-MoE on Stable Diffusion 3.

Attack definition used here
---------------------------
The current TG-MoE implementation returns the frozen SD3 base-layer output
whenever secret_bits=None. Therefore, during the selected final denoising
region, the attack switches every MoE-LoRA layer to the clean base path:

    watermark path:  secret_bits = owner secret
    clean path:      secret_bits = None

For a watermark gate t < wm_gate_end and an attack size rho:

    t >= wm_gate_end          -> clean path (normal TG-MoE behavior)
    rho <= t < wm_gate_end    -> watermark path
    t < rho                   -> clean path (adaptive attack)

Thus rho=0.10 attacks the last 10% of normalized model time, and rho=0.40
removes the entire t<0.40 watermark window.

The script:
1. loads the trained TG-MoE SD3 transformer;
2. generates Clean, normal TG-MoE, and attacked outputs with identical seeds;
3. extracts the 48-bit watermark from RGB outputs using the local SD3 VAE;
4. reports BitAcc, signed owner margin, matched probability, PSNR and SSIM;
5. optionally evaluates black-box ownership TPR using previously frozen
   thresholds from blackbox_ownership_verify.py.

Expected checkpoint style matches the user's eval_eval_adv.py:
- inject MoE-LoRA into SD3 to_q/to_k/to_v/to_out[0];
- load transformer_dir/diffusion_pytorch_model.safetensors or .bin;
- Stage-1 extractor uses posterior.mode() * vae.config.scaling_factor.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from diffusers import SD3Transformer2DModel, StableDiffusion3Pipeline
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms
from tqdm.auto import tqdm

try:
    from safetensors.torch import load_file as load_safetensors
except ImportError:
    load_safetensors = None


# ---------------------------------------------------------------------------
# Default 50 prompts: identical to the current evaluation script
# ---------------------------------------------------------------------------

DEFAULT_PROMPTS = [
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
    "An astronaut riding a horse on the moon, highly detailed",
    "A brave medieval knight in shining armor standing on a battlefield",
    "A futuristic glowing robot repairing a spaceship in orbit",
    "A beautiful elven wizard casting a glowing blue magic spell",
    "A steampunk inventor working in a cluttered workshop with brass pipes",
    "A portrait of a cyberpunk hacker with glowing neon tattoos",
    "A massive alien spaceship hovering over an ancient pyramid",
    "A glowing magical sword stuck in a stone in a dark forest",
    "A steampunk airship flying through fluffy white clouds",
    "A cybernetic samurai with a glowing katana in a dark alleyway",
]


# ---------------------------------------------------------------------------
# General utilities
# ---------------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "t"}


def parse_float_list(value: str) -> List[float]:
    values = sorted({float(item.strip()) for item in value.split(",") if item.strip()})
    if not values:
        raise argparse.ArgumentTypeError("The list cannot be empty.")
    if any(not (0.0 <= item <= 1.0) for item in values):
        raise argparse.ArgumentTypeError("All values must lie in [0,1].")
    return values


def parse_int_list(value: str) -> List[int]:
    values = sorted({int(item.strip()) for item in value.split(",") if item.strip()})
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("All query budgets must be positive.")
    return values


def dtype_from_name(name: str) -> torch.dtype:
    if name == "fp16":
        return torch.float16
    if name == "bf16":
        return torch.bfloat16
    return torch.float32


def safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def safe_std(values: Sequence[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0


def load_prompts(path: Optional[str], limit: int) -> List[str]:
    if not path:
        prompts = list(DEFAULT_PROMPTS)
    else:
        prompt_path = Path(path)
        if not prompt_path.exists():
            raise FileNotFoundError(prompt_path)

        suffix = prompt_path.suffix.lower()
        prompts: List[str] = []

        if suffix in {".txt", ".list"}:
            prompts = [
                line.strip()
                for line in prompt_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        elif suffix == ".json":
            obj = json.loads(prompt_path.read_text(encoding="utf-8"))
            if isinstance(obj, list):
                for item in obj:
                    if isinstance(item, str):
                        prompts.append(item)
                    elif isinstance(item, dict):
                        value = item.get("prompt", item.get("text", item.get("caption")))
                        if value:
                            prompts.append(str(value))
            elif isinstance(obj, dict):
                data = obj.get("prompts", obj.get("data", []))
                for item in data:
                    if isinstance(item, str):
                        prompts.append(item)
                    elif isinstance(item, dict):
                        value = item.get("prompt", item.get("text", item.get("caption")))
                        if value:
                            prompts.append(str(value))
        elif suffix == ".jsonl":
            for line in prompt_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                value = item.get("prompt", item.get("text", item.get("caption")))
                if value:
                    prompts.append(str(value))
        elif suffix == ".csv":
            with prompt_path.open("r", encoding="utf-8-sig", newline="") as f:
                for row in csv.DictReader(f):
                    value = row.get("prompt", row.get("text", row.get("caption")))
                    if value:
                        prompts.append(str(value))
        else:
            raise ValueError(f"Unsupported prompt file: {prompt_path}")

    if limit > 0:
        prompts = prompts[:limit]
    if not prompts:
        raise RuntimeError("No prompts were loaded.")
    return prompts


# ---------------------------------------------------------------------------
# Exact MoE-LoRA structure used by the current evaluation implementation
# ---------------------------------------------------------------------------

class MoELoRALayer(nn.Module):
    def __init__(
        self,
        base_layer: nn.Module,
        num_experts: int = 4,
        rank: int = 32,
        network_alpha: Optional[float] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.num_experts = num_experts
        self.rank = rank

        in_features = base_layer.in_features
        out_features = base_layer.out_features

        self.lora_A = nn.Parameter(torch.zeros(num_experts, in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(num_experts, rank, out_features))
        self.lora_bias = nn.Parameter(torch.zeros(num_experts, out_features))
        self.router = nn.Linear(in_features, num_experts)
        self.bit_router = nn.Linear(48, num_experts)

        self.current_secret_bits: Optional[torch.Tensor] = None
        self.current_logits: Optional[torch.Tensor] = None
        self.scaling = network_alpha / rank if network_alpha else 1.0
        self.dropout = nn.Dropout(dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.zeros_(self.lora_bias)

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        base_output = self.base_layer(x, *args, **kwargs)
        if self.current_secret_bits is None:
            return base_output

        secret_bits = self.current_secret_bits.to(device=x.device, dtype=x.dtype)
        logits = self.router(x)
        bit_logits = self.bit_router(secret_bits).unsqueeze(1)
        logits = logits + bit_logits
        self.current_logits = logits

        routing_probs = F.softmax(logits, dim=-1)
        batch_size, seq_len, _ = x.shape
        lora_delta = torch.zeros(
            batch_size,
            seq_len,
            self.base_layer.out_features,
            device=x.device,
            dtype=x.dtype,
        )

        for expert_index in range(self.num_experts):
            expert_A = self.lora_A[expert_index].to(x.dtype)
            expert_B = self.lora_B[expert_index].to(x.dtype)
            expert_bias = self.lora_bias[expert_index].to(x.dtype)
            out = torch.matmul(torch.matmul(x, expert_A), expert_B) + expert_bias
            weight = routing_probs[..., expert_index].unsqueeze(-1)
            lora_delta += weight * out

        return base_output + self.dropout(lora_delta).to(x.dtype) * self.scaling


def inject_moe_lora_to_sd3(
    transformer: nn.Module,
    num_experts: int,
    rank: int,
) -> nn.Module:
    transformer.requires_grad_(False)

    # Freeze the module list before replacement, matching the existing code.
    for name, module in list(transformer.named_modules()):
        if any(target in name for target in ["to_q", "to_k", "to_v"]):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = transformer.get_submodule(parent_name)
            old_linear = getattr(parent, child_name)
            setattr(
                parent,
                child_name,
                MoELoRALayer(old_linear, num_experts=num_experts, rank=rank),
            )
        elif "to_out" in name and isinstance(module, nn.ModuleList):
            module[0] = MoELoRALayer(
                module[0],
                num_experts=num_experts,
                rank=rank,
            )

    return transformer


def set_moe_context(
    model: nn.Module,
    secret_bits: Optional[torch.Tensor],
) -> None:
    for module in model.modules():
        if isinstance(module, MoELoRALayer):
            module.current_secret_bits = secret_bits


# ---------------------------------------------------------------------------
# Stage-1 48-bit extractor
# ---------------------------------------------------------------------------

class Conv2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: Optional[str] = "relu",
        strides: int = 1,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            strides,
            int((kernel_size - 1) / 2),
        )
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "selu":
            self.act = nn.SELU(inplace=True)
        else:
            self.act = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.conv(inputs)
        return outputs if self.act is None else self.act(outputs)


class Linear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation: Optional[str] = "relu",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if activation == "relu":
            self.act = nn.ReLU(inplace=True)
        elif activation == "selu":
            self.act = nn.SELU(inplace=True)
        else:
            self.act = None

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.linear(inputs)
        return outputs if self.act is None else self.act(outputs)


class Flatten(nn.Module):
    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs.contiguous().view(inputs.size(0), -1)


class ExtractorForLatent(nn.Module):
    def __init__(self, secret_size: int = 48) -> None:
        super().__init__()
        self.decoder = nn.Sequential(
            Conv2D(16, 64, 3, strides=2, activation="selu"),
            Conv2D(64, 64, 3, activation="selu"),
            Conv2D(64, 128, 3, strides=2, activation="selu"),
            Conv2D(128, 128, 3, activation="selu"),
            Conv2D(128, 256, 3, strides=2, activation="selu"),
            Conv2D(256, 256, 3, activation="selu"),
            Conv2D(256, 512, 3, strides=2, activation="selu"),
            Conv2D(512, 512, 3, activation="selu"),
            Flatten(),
        )
        self.mlps = nn.Sequential(
            Linear(32768, 2048, activation="selu"),
            Linear(2048, 2048, activation="selu"),
            Linear(2048, 2048, activation="selu"),
            nn.Dropout(p=0.1),
            Linear(2048, secret_size, activation=None),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        return self.mlps(self.decoder(latent))


def extract_tensor(obj, preferred_keys: Sequence[str]) -> torch.Tensor:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for key in preferred_keys:
            value = obj.get(key)
            if torch.is_tensor(value):
                return value
        tensor_values = [value for value in obj.values() if torch.is_tensor(value)]
        if len(tensor_values) == 1:
            return tensor_values[0]
    raise ValueError(f"Could not find tensor; tried keys={preferred_keys}")


def clean_state_dict(obj) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("decoder", "extractor", "state_dict"):
            if isinstance(obj.get(key), dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint does not contain a state_dict.")

    result: Dict[str, torch.Tensor] = {}
    for key, value in obj.items():
        if not torch.is_tensor(value):
            continue
        while key.startswith("module."):
            key = key[len("module."):]
        result[key] = value
    if not result:
        raise ValueError("No tensor parameters found in checkpoint.")
    return result


def load_secret(path: Path, bit_dim: int, device: torch.device) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    bits = extract_tensor(
        obj,
        ("secret", "bits", "GT_secret", "watermark_secret"),
    ).detach().float().flatten()

    if bits.numel() != bit_dim:
        raise ValueError(f"Secret contains {bits.numel()} bits; expected {bit_dim}.")
    if float(bits.min()) < 0.0 or float(bits.max()) > 1.0:
        bits = (bits > 0).float()
    else:
        bits = (bits >= 0.5).float()
    return bits.view(1, bit_dim).to(device)


def load_extractor(
    path: Path,
    bit_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    model = ExtractorForLatent(secret_size=bit_dim)
    model.load_state_dict(clean_state_dict(torch.load(path, map_location="cpu")), strict=True)
    model.requires_grad_(False)
    model.eval()
    return model.to(device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Transformer loading
# ---------------------------------------------------------------------------

def locate_transformer_state(transformer_dir: Path) -> Path:
    candidates = [
        transformer_dir / "diffusion_pytorch_model.safetensors",
        transformer_dir / "diffusion_pytorch_model.bin",
        transformer_dir / "pytorch_model.bin",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"No transformer state found in {transformer_dir}. "
        "Expected diffusion_pytorch_model.safetensors or .bin."
    )


def load_transformer_state(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        if load_safetensors is None:
            raise ImportError("Install safetensors to load this checkpoint.")
        return load_safetensors(str(path))
    obj = torch.load(path, map_location="cpu")
    return clean_state_dict(obj)


def infer_moe_shape(state: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    for key, value in state.items():
        if key.endswith("lora_A") and value.ndim == 3:
            return int(value.shape[0]), int(value.shape[2])
    raise ValueError(
        "Could not infer num_experts/rank because no 3-D lora_A tensor was found."
    )


# ---------------------------------------------------------------------------
# Generation and attack
# ---------------------------------------------------------------------------

def normalized_timestep(timestep: torch.Tensor, num_train_timesteps: float) -> float:
    return float(torch.as_tensor(timestep).detach().float().item()) / num_train_timesteps


def secret_for_condition(
    condition: str,
    timestep_norm: float,
    owner_secret: torch.Tensor,
    wm_gate_end: float,
    attack_fraction: float,
) -> Optional[torch.Tensor]:
    if condition == "clean":
        return None

    # Normal TG-MoE only activates below wm_gate_end.
    watermark_active = timestep_norm < wm_gate_end
    if not watermark_active:
        return None

    if condition == "wm":
        return owner_secret

    if condition == "attack":
        # Adaptive clean-path switch in the final rho region.
        if timestep_norm < attack_fraction:
            return None
        return owner_secret

    raise ValueError(f"Unknown condition: {condition}")


@torch.inference_mode()
def generate_image(
    pipe: StableDiffusion3Pipeline,
    prompt: str,
    seed: int,
    condition: str,
    owner_secret: torch.Tensor,
    wm_gate_end: float,
    attack_fraction: float,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    device: torch.device,
) -> Image.Image:
    scheduler = pipe.scheduler
    scheduler.set_timesteps(num_inference_steps, device=device)
    schedule = scheduler.timesteps.detach().clone()

    num_train_timesteps = float(
        getattr(scheduler.config, "num_train_timesteps", 1000.0)
    )
    if num_train_timesteps <= 1:
        num_train_timesteps = 1000.0

    # Set the context for the first denoising step explicitly.
    first_t = normalized_timestep(schedule[0], num_train_timesteps)
    set_moe_context(
        pipe.transformer,
        secret_for_condition(
            condition,
            first_t,
            owner_secret,
            wm_gate_end,
            attack_fraction,
        ),
    )

    def callback_on_step_end(
        pipeline,
        step_index: int,
        timestep: torch.Tensor,
        callback_kwargs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        del timestep
        next_index = step_index + 1
        if next_index < len(schedule):
            next_t = normalized_timestep(
                schedule[next_index],
                num_train_timesteps,
            )
            context = secret_for_condition(
                condition,
                next_t,
                owner_secret,
                wm_gate_end,
                attack_fraction,
            )
            set_moe_context(pipeline.transformer, context)
        else:
            set_moe_context(pipeline.transformer, None)
        return callback_kwargs

    generator = torch.Generator(device=device).manual_seed(seed)
    result = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        height=height,
        width=width,
        callback_on_step_end=callback_on_step_end,
    ).images[0]

    set_moe_context(pipe.transformer, None)
    return result


# ---------------------------------------------------------------------------
# RGB output verification
# ---------------------------------------------------------------------------

PIL_TO_M11 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


@torch.inference_mode()
def extract_logits_from_pil(
    image: Image.Image,
    vae: nn.Module,
    extractor: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    latent_resolution: int,
) -> torch.Tensor:
    image_tensor = PIL_TO_M11(image).unsqueeze(0).to(device=device, dtype=dtype)
    posterior = vae.encode(image_tensor).latent_dist

    # Match the existing Stage-1 pixel re-encoding convention.
    latent = posterior.mode().float() * float(vae.config.scaling_factor)
    if latent.shape[-2:] != (latent_resolution, latent_resolution):
        latent = F.interpolate(
            latent,
            size=(latent_resolution, latent_resolution),
            mode="bilinear",
            align_corners=False,
        )
    return extractor(latent.to(dtype=dtype)).float()


def watermark_metrics(
    logits: torch.Tensor,
    owner_secret: torch.Tensor,
) -> Dict[str, float]:
    bits = owner_secret.float()
    probabilities = torch.sigmoid(logits)
    predictions = (probabilities >= 0.5).float()
    signs = bits * 2.0 - 1.0

    bit_acc = (predictions == bits).float().mean()
    owner_margin = (signs * logits).mean()
    matched_probability = (
        bits * probabilities + (1.0 - bits) * (1.0 - probabilities)
    ).mean()

    return {
        "bit_acc": float(bit_acc.item()),
        "owner_margin": float(owner_margin.item()),
        "matched_probability": float(matched_probability.item()),
    }


def image_quality(reference: Image.Image, target: Image.Image) -> Tuple[float, float]:
    reference_np = np.asarray(reference.convert("RGB"))
    target_np = np.asarray(target.convert("RGB"))
    psnr = peak_signal_noise_ratio(reference_np, target_np, data_range=255)
    ssim = structural_similarity(
        reference_np,
        target_np,
        data_range=255,
        channel_axis=-1,
    )
    return float(psnr), float(ssim)


# ---------------------------------------------------------------------------
# Black-box evaluation with frozen thresholds
# ---------------------------------------------------------------------------

def owner_scores_from_trials(
    logits: np.ndarray,
    owner_bits: np.ndarray,
    query_budget: int,
    num_trials: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if len(logits) < query_budget:
        raise ValueError(
            f"Only {len(logits)} images are available for Q={query_budget}."
        )

    signs = owner_bits.astype(np.float32) * 2.0 - 1.0
    scores = np.empty(num_trials, dtype=np.float32)
    for trial in range(num_trials):
        indices = rng.choice(len(logits), size=query_budget, replace=False)
        aggregated = logits[indices].mean(axis=0)
        scores[trial] = float((aggregated * signs).mean())
    return scores


def evaluate_frozen_blackbox(
    logits_by_condition: Dict[str, List[np.ndarray]],
    thresholds_path: Path,
    query_budgets: Sequence[int],
    num_trials: int,
    seed: int,
    output_dir: Path,
) -> List[dict]:
    with thresholds_path.open("r", encoding="utf-8") as f:
        frozen = json.load(f)

    owner_bits = np.asarray(frozen["owner_bits"], dtype=np.int8)
    thresholds = frozen["thresholds"]
    rows: List[dict] = []

    for condition, values in logits_by_condition.items():
        logits = np.stack(values, axis=0).astype(np.float32)
        for q in query_budgets:
            if str(q) not in thresholds:
                print(
                    f"[Black-box][Skip] no frozen threshold for Q={q}; "
                    f"available={sorted(thresholds)}"
                )
                continue
            if len(logits) < q:
                print(
                    f"[Black-box][Skip] condition={condition}, "
                    f"images={len(logits)} < Q={q}"
                )
                continue

            rng = np.random.default_rng(seed + 10000 + q)
            scores = owner_scores_from_trials(
                logits,
                owner_bits,
                q,
                num_trials,
                rng,
            )
            threshold = float(thresholds[str(q)])
            verification_rate = float(np.mean(scores >= threshold))
            rows.append({
                "condition": condition,
                "query_budget": q,
                "threshold": threshold,
                "verification_rate": verification_rate,
                "mean_owner_score": float(scores.mean()),
                "std_owner_score": float(scores.std(ddof=1)),
                "num_trials": num_trials,
            })

    if rows:
        with (output_dir / "blackbox_attack_results.csv").open(
            "w", encoding="utf-8", newline=""
        ) as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

        plt.figure(figsize=(6.2, 4.2))
        grouped: Dict[str, List[dict]] = defaultdict(list)
        for row in rows:
            grouped[row["condition"]].append(row)
        for condition, condition_rows in grouped.items():
            condition_rows = sorted(
                condition_rows,
                key=lambda item: item["query_budget"],
            )
            plt.plot(
                [item["query_budget"] for item in condition_rows],
                [item["verification_rate"] for item in condition_rows],
                marker="o",
                label=condition,
            )
        plt.xlabel("Number of black-box queries")
        plt.ylabel("Ownership verification rate")
        plt.ylim(-0.02, 1.02)
        plt.grid(alpha=0.25)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(
            output_dir / "blackbox_attack_query_curve.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    dtype = dtype_from_name(args.dtype)
    if device.type == "cpu" and dtype != torch.float32:
        print("[Warning] CPU selected; forcing fp32.")
        dtype = torch.float32

    prompts = load_prompts(args.prompts_file, args.num_prompts)
    print(f"[Prompts] {len(prompts)}")

    transformer_dir = Path(args.transformer_dir)
    state_path = locate_transformer_state(transformer_dir)
    state = load_transformer_state(state_path)

    inferred_experts, inferred_rank = infer_moe_shape(state)
    num_experts = args.num_experts or inferred_experts
    rank = args.rank or inferred_rank
    print(
        f"[MoE] inferred experts={inferred_experts}, rank={inferred_rank}; "
        f"using experts={num_experts}, rank={rank}"
    )

    if num_experts != inferred_experts or rank != inferred_rank:
        print(
            "[Warning] CLI MoE shape differs from checkpoint inference. "
            "Strict loading may fail."
        )

    print("[Load] base SD3 transformer")
    transformer = SD3Transformer2DModel.from_pretrained(
        args.base_model,
        subfolder="transformer",
        local_files_only=args.local_files_only,
    )
    transformer = inject_moe_lora_to_sd3(
        transformer,
        num_experts=num_experts,
        rank=rank,
    )
    transformer.load_state_dict(state, strict=True)
    transformer = transformer.to(device=device, dtype=dtype)
    transformer.eval()
    transformer.requires_grad_(False)

    print("[Load] SD3 pipeline")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.base_model,
        transformer=transformer,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    if args.enable_vae_tiling:
        pipe.vae.enable_tiling()
    pipe.vae.eval()
    pipe.vae.requires_grad_(False)

    pretrained_wm_dir = Path(args.pretrainedWM_dir)
    owner_secret = load_secret(
        pretrained_wm_dir / "secret.pt",
        args.bit_dim,
        device,
    ).to(dtype=dtype)
    extractor = load_extractor(
        pretrained_wm_dir / "decoder.pth",
        args.bit_dim,
        device,
        dtype,
    )

    attack_fractions = [
        value
        for value in args.attack_fractions
        if value <= args.wm_gate_end + 1e-12
    ]
    if len(attack_fractions) != len(args.attack_fractions):
        print(
            "[Warning] Attack fractions above wm_gate_end were removed because "
            "they would also target timesteps where TG-MoE is already inactive."
        )

    conditions: List[Tuple[str, str, float]] = [
        ("clean", "clean", 0.0),
        ("wm", "wm", 0.0),
    ]
    for fraction in attack_fractions:
        label = f"attack_last_{int(round(fraction * 100)):02d}"
        conditions.append((label, "attack", fraction))

    for label, _, _ in conditions:
        (output_dir / label).mkdir(parents=True, exist_ok=True)

    per_image_rows: List[dict] = []
    logits_by_condition: Dict[str, List[np.ndarray]] = defaultdict(list)

    for index, prompt in enumerate(tqdm(prompts, desc="Prompts")):
        image_seed = args.seed + index

        clean_image = generate_image(
            pipe=pipe,
            prompt=prompt,
            seed=image_seed,
            condition="clean",
            owner_secret=owner_secret,
            wm_gate_end=args.wm_gate_end,
            attack_fraction=0.0,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            device=device,
        )
        clean_image.save(output_dir / "clean" / f"{index:04d}.png")

        wm_image = generate_image(
            pipe=pipe,
            prompt=prompt,
            seed=image_seed,
            condition="wm",
            owner_secret=owner_secret,
            wm_gate_end=args.wm_gate_end,
            attack_fraction=0.0,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            height=args.height,
            width=args.width,
            device=device,
        )
        wm_image.save(output_dir / "wm" / f"{index:04d}.png")

        generated: Dict[str, Tuple[Image.Image, float]] = {
            "clean": (clean_image, 0.0),
            "wm": (wm_image, 0.0),
        }

        for fraction in attack_fractions:
            label = f"attack_last_{int(round(fraction * 100)):02d}"
            attacked_image = generate_image(
                pipe=pipe,
                prompt=prompt,
                seed=image_seed,
                condition="attack",
                owner_secret=owner_secret,
                wm_gate_end=args.wm_gate_end,
                attack_fraction=fraction,
                num_inference_steps=args.num_inference_steps,
                guidance_scale=args.guidance_scale,
                height=args.height,
                width=args.width,
                device=device,
            )
            attacked_image.save(output_dir / label / f"{index:04d}.png")
            generated[label] = (attacked_image, fraction)

        for label, (image, fraction) in generated.items():
            logits = extract_logits_from_pil(
                image=image,
                vae=pipe.vae,
                extractor=extractor,
                device=device,
                dtype=dtype,
                latent_resolution=args.latent_resolution,
            )
            metrics = watermark_metrics(logits, owner_secret)
            logits_by_condition[label].append(
                logits.squeeze(0).detach().cpu().numpy().astype(np.float32)
            )

            psnr_to_wm, ssim_to_wm = image_quality(wm_image, image)
            psnr_to_clean, ssim_to_clean = image_quality(clean_image, image)

            per_image_rows.append({
                "prompt_index": index,
                "seed": image_seed,
                "prompt": prompt,
                "condition": label,
                "attack_fraction": fraction,
                **metrics,
                "psnr_to_wm": psnr_to_wm,
                "ssim_to_wm": ssim_to_wm,
                "psnr_to_clean": psnr_to_clean,
                "ssim_to_clean": ssim_to_clean,
            })

    # Save per-image measurements.
    per_image_path = output_dir / "adaptive_attack_per_image.csv"
    with per_image_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(per_image_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_image_rows)

    # Aggregate summary.
    grouped_rows: Dict[str, List[dict]] = defaultdict(list)
    for row in per_image_rows:
        grouped_rows[row["condition"]].append(row)

    summary_rows: List[dict] = []
    metric_names = [
        "bit_acc",
        "owner_margin",
        "matched_probability",
        "psnr_to_wm",
        "ssim_to_wm",
        "psnr_to_clean",
        "ssim_to_clean",
    ]
    for condition, rows in grouped_rows.items():
        summary = {
            "condition": condition,
            "attack_fraction": rows[0]["attack_fraction"],
            "num_images": len(rows),
        }
        for metric in metric_names:
            values = [float(row[metric]) for row in rows]
            summary[f"{metric}_mean"] = safe_mean(values)
            summary[f"{metric}_std"] = safe_std(values)
        summary_rows.append(summary)

    summary_rows.sort(key=lambda row: (
        0 if row["condition"] == "clean"
        else 1 if row["condition"] == "wm"
        else 2,
        row["attack_fraction"],
    ))

    with (output_dir / "adaptive_attack_summary.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    with (output_dir / "adaptive_attack_summary.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(
            {
                "configuration": vars(args),
                "results": summary_rows,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    # Plot attack strength versus image-level watermark recovery.
    attack_summary = [
        row for row in summary_rows if row["condition"].startswith("attack_last_")
    ]
    if attack_summary:
        attack_summary = sorted(
            attack_summary,
            key=lambda row: row["attack_fraction"],
        )
        x = [row["attack_fraction"] * 100.0 for row in attack_summary]

        plt.figure(figsize=(5.8, 4.0))
        plt.plot(
            x,
            [row["bit_acc_mean"] for row in attack_summary],
            marker="o",
            label="BitAcc",
        )
        plt.plot(
            x,
            [row["matched_probability_mean"] for row in attack_summary],
            marker="s",
            label="Matched probability",
        )
        plt.xlabel("Attacked final denoising region (%)")
        plt.ylabel("Watermark recovery")
        plt.ylim(0.0, 1.02)
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            output_dir / "adaptive_attack_watermark_curve.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        plt.figure(figsize=(5.8, 4.0))
        plt.plot(
            x,
            [row["psnr_to_wm_mean"] for row in attack_summary],
            marker="o",
            label="PSNR to unattacked TG-MoE",
        )
        plt.plot(
            x,
            [row["psnr_to_clean_mean"] for row in attack_summary],
            marker="s",
            label="PSNR to Clean SD3",
        )
        plt.xlabel("Attacked final denoising region (%)")
        plt.ylabel("PSNR (dB)")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(
            output_dir / "adaptive_attack_quality_curve.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    # Optional model-level black-box evaluation with the original frozen threshold.
    if args.frozen_thresholds:
        evaluate_frozen_blackbox(
            logits_by_condition=logits_by_condition,
            thresholds_path=Path(args.frozen_thresholds),
            query_budgets=args.query_budgets,
            num_trials=args.blackbox_trials,
            seed=args.seed,
            output_dir=output_dir,
        )

    print("\n==================== DONE ====================")
    print(f"Images:          {output_dir}")
    print(f"Per-image CSV:   {per_image_path}")
    print(f"Summary CSV:     {output_dir / 'adaptive_attack_summary.csv'}")
    if args.frozen_thresholds:
        print(
            f"Black-box CSV:   "
            f"{output_dir / 'blackbox_attack_results.csv'}"
        )
    print("==============================================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Adaptive Late-Timestep Attack for TG-MoE SD3"
    )
    parser.add_argument("--base_model", required=True)
    parser.add_argument(
        "--transformer_dir",
        required=True,
        help="Trained checkpoint transformer folder.",
    )
    parser.add_argument(
        "--pretrainedWM_dir",
        required=True,
        help="Stage-1 folder containing decoder.pth and secret.pt.",
    )
    parser.add_argument("--output_dir", default="./Evaluation/adaptive_late_attack")

    parser.add_argument("--prompts_file", default=None)
    parser.add_argument("--num_prompts", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)

    parser.add_argument(
        "--wm_gate_end",
        type=float,
        default=0.40,
        help="Current TG-MoE activation rule is t < wm_gate_end.",
    )
    parser.add_argument(
        "--attack_fractions",
        type=parse_float_list,
        default=[0.10, 0.20, 0.30, 0.40],
        help="Comma-separated final normalized-time regions to switch to clean path.",
    )

    parser.add_argument(
        "--num_experts",
        type=int,
        default=0,
        help="0 means infer from checkpoint.",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=0,
        help="0 means infer from checkpoint.",
    )
    parser.add_argument("--bit_dim", type=int, default=48)
    parser.add_argument("--latent_resolution", type=int, default=128)

    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--enable_vae_tiling", type=str2bool, default=False)
    parser.add_argument("--local_files_only", type=str2bool, default=True)

    parser.add_argument(
        "--frozen_thresholds",
        default=None,
        help="Optional frozen_thresholds.json from the previous black-box benchmark.",
    )
    parser.add_argument(
        "--query_budgets",
        type=parse_int_list,
        default=[1, 5, 10, 20],
    )
    parser.add_argument("--blackbox_trials", type=int, default=2000)

    args = parser.parse_args()
    if not (0.0 < args.wm_gate_end <= 1.0):
        parser.error("--wm_gate_end must lie in (0,1].")
    if args.num_prompts <= 0:
        parser.error("--num_prompts must be positive.")
    if args.num_inference_steps <= 0:
        parser.error("--num_inference_steps must be positive.")
    if args.height <= 0 or args.width <= 0:
        parser.error("--height and --width must be positive.")
    return args


if __name__ == "__main__":
    main(parse_args())
