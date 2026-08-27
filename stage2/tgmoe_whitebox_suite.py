#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TG-MoE white-box experiment suite for the user's current SD3 implementation.

Supported experiments
---------------------
1. expert_dropout
   Disable selected MoE experts globally and test distributed expert redundancy.

2. routing_ablation
   Compare full secret+feature routing, feature-only, secret-only, uniform,
   and fixed single-expert routing.

3. adapter_dropout
   Disable random or structured subsets of MoE-LoRA layers.

4. parameter_robustness
   Apply magnitude pruning, Gaussian parameter noise, or fake INT8/INT4
   quantization only to watermark adapter/router parameters.

5. routing_profile
   Record expert usage, normalized routing entropy, top-1 load, and
   pairwise expert update cosine similarity.

6. layer_contribution
   Disable one block, one projection family, or one individual adapter at
   a time and measure its contribution to owner margin and BitAcc.

The script matches the current project:
- MoE-LoRA is inserted into SD3 to_q, to_k, to_v and to_out[0].
- The router uses feature logits + 48-bit secret logits.
- TG-MoE is active for normalized model time t < --wm_gate_end.
- RGB verification uses posterior.mode() * VAE scaling_factor without shift.
- A local frozen Stage-1 48-bit extractor is used.
- Optional model-level verification uses the already frozen thresholds from
  blackbox_ownership_verify.py; thresholds are never re-selected here.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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


# =============================================================================
# Utilities
# =============================================================================

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def str2bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).lower() in {"1", "true", "yes", "y", "t"}


def parse_float_list(value: str) -> List[float]:
    result = [float(x.strip()) for x in value.split(",") if x.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one float.")
    return result


def parse_int_list(value: str) -> List[int]:
    result = [int(x.strip()) for x in value.split(",") if x.strip()]
    if not result:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    return result


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


def prompt_digest(prompts: Sequence[str]) -> str:
    payload = "\n".join(prompts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_prompts(path: Optional[str], limit: int) -> List[str]:
    if path is None:
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
            data = obj if isinstance(obj, list) else obj.get("prompts", obj.get("data", []))
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
            raise ValueError(f"Unsupported prompts file: {prompt_path}")

    if limit > 0:
        prompts = prompts[:limit]
    if not prompts:
        raise RuntimeError("No prompts were loaded.")
    return prompts


# =============================================================================
# Controlled MoE-LoRA
# =============================================================================

class MoELoRALayer(nn.Module):
    """
    State-dict-compatible extension of the user's current MoELoRALayer.

    Extra control attributes are ordinary Python attributes and do not alter
    checkpoint keys, so strict checkpoint loading remains possible.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        layer_name: str,
        num_experts: int = 4,
        rank: int = 32,
        network_alpha: Optional[float] = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.layer_name = layer_name
        self.num_experts = int(num_experts)
        self.rank = int(rank)

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

        # White-box controls.
        self.adapter_enabled = True
        self.routing_mode = "full"
        self.single_expert_index = 0
        self.expert_mask: Optional[torch.Tensor] = None
        self.renormalize_expert_mask = False
        self.record_routing = False
        self.reset_routing_stats()

    def reset_controls(self) -> None:
        self.adapter_enabled = True
        self.routing_mode = "full"
        self.single_expert_index = 0
        self.expert_mask = None
        self.renormalize_expert_mask = False
        self.record_routing = False
        self.current_logits = None

    def reset_routing_stats(self) -> None:
        self.routing_prob_sum = torch.zeros(self.num_experts, dtype=torch.float64)
        self.routing_top1_count = torch.zeros(self.num_experts, dtype=torch.float64)
        self.routing_entropy_sum = 0.0
        self.routing_token_count = 0

    def _compute_routing_probs(
        self,
        x: torch.Tensor,
        secret_bits: torch.Tensor,
    ) -> torch.Tensor:
        feature_logits = self.router(x)
        bit_logits = self.bit_router(secret_bits).unsqueeze(1)

        if self.routing_mode == "full":
            logits = feature_logits + bit_logits
            probs = F.softmax(logits, dim=-1)
        elif self.routing_mode == "feature_only":
            logits = feature_logits
            probs = F.softmax(logits, dim=-1)
        elif self.routing_mode == "secret_only":
            logits = bit_logits.expand(
                feature_logits.shape[0],
                feature_logits.shape[1],
                -1,
            )
            probs = F.softmax(logits, dim=-1)
        elif self.routing_mode == "uniform":
            logits = torch.zeros_like(feature_logits)
            probs = torch.full_like(feature_logits, 1.0 / self.num_experts)
        elif self.routing_mode == "single":
            index = int(self.single_expert_index)
            if not 0 <= index < self.num_experts:
                raise ValueError(
                    f"single_expert_index={index} outside [0,{self.num_experts - 1}]"
                )
            logits = torch.full_like(feature_logits, -1e4)
            logits[..., index] = 0.0
            probs = torch.zeros_like(feature_logits)
            probs[..., index] = 1.0
        else:
            raise ValueError(f"Unknown routing_mode={self.routing_mode}")

        self.current_logits = logits

        if self.expert_mask is not None:
            mask = self.expert_mask.to(device=probs.device, dtype=probs.dtype)
            if mask.numel() != self.num_experts:
                raise ValueError(
                    f"Mask has {mask.numel()} experts; expected {self.num_experts}"
                )
            probs = probs * mask.view(1, 1, -1)
            if self.renormalize_expert_mask:
                denominator = probs.sum(dim=-1, keepdim=True)
                probs = torch.where(
                    denominator > 1e-12,
                    probs / denominator.clamp(min=1e-12),
                    probs,
                )

        if self.record_routing:
            with torch.no_grad():
                flat = probs.detach().float().reshape(-1, self.num_experts).cpu()
                self.routing_prob_sum += flat.double().sum(dim=0)
                self.routing_top1_count += torch.bincount(
                    flat.argmax(dim=-1),
                    minlength=self.num_experts,
                ).double()
                entropy = -(flat.clamp(min=1e-12) * flat.clamp(min=1e-12).log()).sum(dim=-1)
                self.routing_entropy_sum += float(entropy.double().sum().item())
                self.routing_token_count += int(flat.shape[0])

        return probs

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        base_output = self.base_layer(x, *args, **kwargs)

        if self.current_secret_bits is None or not self.adapter_enabled:
            return base_output

        secret_bits = self.current_secret_bits.to(device=x.device, dtype=x.dtype)
        routing_probs = self._compute_routing_probs(x, secret_bits)

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

    for name, module in list(transformer.named_modules()):
        if any(target in name for target in ["to_q", "to_k", "to_v"]):
            parent_name = ".".join(name.split(".")[:-1])
            child_name = name.split(".")[-1]
            parent = transformer.get_submodule(parent_name)
            old_linear = getattr(parent, child_name)
            setattr(
                parent,
                child_name,
                MoELoRALayer(
                    old_linear,
                    layer_name=name,
                    num_experts=num_experts,
                    rank=rank,
                ),
            )
        elif "to_out" in name and isinstance(module, nn.ModuleList):
            module[0] = MoELoRALayer(
                module[0],
                layer_name=f"{name}.0",
                num_experts=num_experts,
                rank=rank,
            )
    return transformer


def moe_layers(model: nn.Module) -> List[MoELoRALayer]:
    return [module for module in model.modules() if isinstance(module, MoELoRALayer)]


def layer_map(model: nn.Module) -> Dict[str, MoELoRALayer]:
    return {module.layer_name: module for module in moe_layers(model)}


def set_moe_context(model: nn.Module, secret_bits: Optional[torch.Tensor]) -> None:
    for module in moe_layers(model):
        module.current_secret_bits = secret_bits


def reset_all_controls(model: nn.Module) -> None:
    for module in moe_layers(model):
        module.reset_controls()


def reset_all_routing_stats(model: nn.Module) -> None:
    for module in moe_layers(model):
        module.reset_routing_stats()


# =============================================================================
# Stage-1 extractor
# =============================================================================

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

    state: Dict[str, torch.Tensor] = {}
    for key, value in obj.items():
        if not torch.is_tensor(value):
            continue
        while key.startswith("module."):
            key = key[len("module."):]
        state[key] = value
    if not state:
        raise ValueError("No tensor values found in checkpoint.")
    return state


def load_secret(path: Path, bit_dim: int, device: torch.device) -> torch.Tensor:
    obj = torch.load(path, map_location="cpu")
    secret = extract_tensor(
        obj,
        ("secret", "bits", "GT_secret", "watermark_secret"),
    ).detach().float().flatten()
    if secret.numel() != bit_dim:
        raise ValueError(f"Secret has {secret.numel()} bits; expected {bit_dim}.")
    if float(secret.min()) < 0.0 or float(secret.max()) > 1.0:
        secret = (secret > 0).float()
    else:
        secret = (secret >= 0.5).float()
    return secret.view(1, bit_dim).to(device)


def load_extractor(
    path: Path,
    bit_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> nn.Module:
    extractor = ExtractorForLatent(secret_size=bit_dim)
    extractor.load_state_dict(
        clean_state_dict(torch.load(path, map_location="cpu")),
        strict=True,
    )
    extractor.requires_grad_(False)
    extractor.eval()
    return extractor.to(device=device, dtype=dtype)


# =============================================================================
# Model loading
# =============================================================================

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
        f"No checkpoint state found in {transformer_dir}. "
        "Expected diffusion_pytorch_model.safetensors or .bin."
    )


def load_transformer_state(path: Path) -> Dict[str, torch.Tensor]:
    if path.suffix == ".safetensors":
        if load_safetensors is None:
            raise ImportError("Install safetensors to load this checkpoint.")
        return load_safetensors(str(path))
    return clean_state_dict(torch.load(path, map_location="cpu"))


def infer_moe_shape(state: Mapping[str, torch.Tensor]) -> Tuple[int, int]:
    for key, value in state.items():
        if key.endswith("lora_A") and value.ndim == 3:
            return int(value.shape[0]), int(value.shape[2])
    raise ValueError("No 3-D lora_A tensor found; cannot infer experts/rank.")


# =============================================================================
# Generation and RGB verification
# =============================================================================

PIL_TO_M11 = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


def normalized_timestep(timestep: torch.Tensor, num_train_timesteps: float) -> float:
    return float(torch.as_tensor(timestep).detach().float().item()) / num_train_timesteps


@torch.inference_mode()
def generate_image(
    pipe: StableDiffusion3Pipeline,
    prompt: str,
    seed: int,
    watermarked: bool,
    owner_secret: torch.Tensor,
    wm_gate_end: float,
    num_inference_steps: int,
    guidance_scale: float,
    height: int,
    width: int,
    device: torch.device,
) -> Image.Image:
    pipe.scheduler.set_timesteps(num_inference_steps, device=device)
    schedule = pipe.scheduler.timesteps.detach().clone()
    num_train_timesteps = float(
        getattr(pipe.scheduler.config, "num_train_timesteps", 1000.0)
    )
    if num_train_timesteps <= 1:
        num_train_timesteps = 1000.0

    def context_for_t(t_norm: float) -> Optional[torch.Tensor]:
        if watermarked and t_norm < wm_gate_end:
            return owner_secret
        return None

    first_t = normalized_timestep(schedule[0], num_train_timesteps)
    set_moe_context(pipe.transformer, context_for_t(first_t))

    def callback_on_step_end(
        pipeline,
        step_index: int,
        timestep: torch.Tensor,
        callback_kwargs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        del timestep
        next_index = step_index + 1
        if next_index < len(schedule):
            next_t = normalized_timestep(schedule[next_index], num_train_timesteps)
            set_moe_context(pipeline.transformer, context_for_t(next_t))
        else:
            set_moe_context(pipeline.transformer, None)
        return callback_kwargs

    generator = torch.Generator(device=device).manual_seed(seed)
    image = pipe(
        prompt=prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        height=height,
        width=width,
        callback_on_step_end=callback_on_step_end,
    ).images[0]

    set_moe_context(pipe.transformer, None)
    return image


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
    return {
        "bit_acc": float((predictions == bits).float().mean().item()),
        "owner_margin": float((signs * logits).mean().item()),
        "matched_probability": float(
            (
                bits * probabilities
                + (1.0 - bits) * (1.0 - probabilities)
            ).mean().item()
        ),
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


# =============================================================================
# Baseline cache
# =============================================================================

@dataclass
class BaselineData:
    clean_paths: List[Path]
    wm_paths: List[Path]
    clean_logits: np.ndarray
    wm_logits: np.ndarray
    rows: List[dict]


def baseline_metadata(args: argparse.Namespace, prompts: Sequence[str]) -> dict:
    return {
        "base_model": str(Path(args.base_model).resolve()),
        "transformer_dir": str(Path(args.transformer_dir).resolve()),
        "pretrainedWM_dir": str(Path(args.pretrainedWM_dir).resolve()),
        "prompt_digest": prompt_digest(prompts),
        "num_prompts": len(prompts),
        "seed": args.seed,
        "num_inference_steps": args.num_inference_steps,
        "guidance_scale": args.guidance_scale,
        "height": args.height,
        "width": args.width,
        "wm_gate_end": args.wm_gate_end,
        "latent_resolution": args.latent_resolution,
    }


def cache_is_compatible(cache_dir: Path, metadata: dict) -> bool:
    metadata_path = cache_dir / "baseline_metadata.json"
    if not metadata_path.exists():
        return False
    try:
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    return existing == metadata


def load_cached_baseline(cache_dir: Path, num_prompts: int) -> BaselineData:
    archive = np.load(cache_dir / "baseline_logits.npz")
    clean_paths = [cache_dir / "clean" / f"{i:04d}.png" for i in range(num_prompts)]
    wm_paths = [cache_dir / "wm" / f"{i:04d}.png" for i in range(num_prompts)]
    if not all(path.exists() for path in clean_paths + wm_paths):
        raise RuntimeError("Baseline image cache is incomplete.")

    with (cache_dir / "baseline_per_image.csv").open(
        "r", encoding="utf-8-sig", newline=""
    ) as f:
        rows = list(csv.DictReader(f))

    return BaselineData(
        clean_paths=clean_paths,
        wm_paths=wm_paths,
        clean_logits=archive["clean_logits"].astype(np.float32),
        wm_logits=archive["wm_logits"].astype(np.float32),
        rows=rows,
    )


def generate_baseline(
    args: argparse.Namespace,
    pipe: StableDiffusion3Pipeline,
    extractor: nn.Module,
    owner_secret: torch.Tensor,
    prompts: Sequence[str],
    cache_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
    record_routing: bool = False,
) -> BaselineData:
    cache_dir.mkdir(parents=True, exist_ok=True)
    clean_dir = cache_dir / "clean"
    wm_dir = cache_dir / "wm"
    clean_dir.mkdir(parents=True, exist_ok=True)
    wm_dir.mkdir(parents=True, exist_ok=True)

    if record_routing:
        reset_all_routing_stats(pipe.transformer)
        for layer in moe_layers(pipe.transformer):
            layer.record_routing = True

    clean_logits: List[np.ndarray] = []
    wm_logits: List[np.ndarray] = []
    rows: List[dict] = []

    for index, prompt in enumerate(tqdm(prompts, desc="Baseline")):
        image_seed = args.seed + index
        clean_image = generate_image(
            pipe,
            prompt,
            image_seed,
            False,
            owner_secret,
            args.wm_gate_end,
            args.num_inference_steps,
            args.guidance_scale,
            args.height,
            args.width,
            device,
        )
        wm_image = generate_image(
            pipe,
            prompt,
            image_seed,
            True,
            owner_secret,
            args.wm_gate_end,
            args.num_inference_steps,
            args.guidance_scale,
            args.height,
            args.width,
            device,
        )

        clean_path = clean_dir / f"{index:04d}.png"
        wm_path = wm_dir / f"{index:04d}.png"
        clean_image.save(clean_path)
        wm_image.save(wm_path)

        clean_logit = extract_logits_from_pil(
            clean_image,
            pipe.vae,
            extractor,
            device,
            dtype,
            args.latent_resolution,
        )
        wm_logit = extract_logits_from_pil(
            wm_image,
            pipe.vae,
            extractor,
            device,
            dtype,
            args.latent_resolution,
        )
        clean_logits.append(clean_logit.squeeze(0).cpu().numpy().astype(np.float32))
        wm_logits.append(wm_logit.squeeze(0).cpu().numpy().astype(np.float32))

        psnr, ssim = image_quality(clean_image, wm_image)
        rows.append({
            "prompt_index": index,
            "seed": image_seed,
            "prompt": prompt,
            "condition": "clean",
            **watermark_metrics(clean_logit, owner_secret),
            "psnr_to_clean": float("inf"),
            "ssim_to_clean": 1.0,
            "psnr_to_wm": psnr,
            "ssim_to_wm": ssim,
        })
        rows.append({
            "prompt_index": index,
            "seed": image_seed,
            "prompt": prompt,
            "condition": "wm",
            **watermark_metrics(wm_logit, owner_secret),
            "psnr_to_clean": psnr,
            "ssim_to_clean": ssim,
            "psnr_to_wm": float("inf"),
            "ssim_to_wm": 1.0,
        })

    for layer in moe_layers(pipe.transformer):
        layer.record_routing = False

    np.savez_compressed(
        cache_dir / "baseline_logits.npz",
        clean_logits=np.stack(clean_logits),
        wm_logits=np.stack(wm_logits),
    )
    with (cache_dir / "baseline_per_image.csv").open(
        "w", encoding="utf-8", newline=""
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (cache_dir / "baseline_metadata.json").write_text(
        json.dumps(
            baseline_metadata(args, prompts),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    return BaselineData(
        clean_paths=[clean_dir / f"{i:04d}.png" for i in range(len(prompts))],
        wm_paths=[wm_dir / f"{i:04d}.png" for i in range(len(prompts))],
        clean_logits=np.stack(clean_logits),
        wm_logits=np.stack(wm_logits),
        rows=rows,
    )


def get_baseline(
    args: argparse.Namespace,
    pipe: StableDiffusion3Pipeline,
    extractor: nn.Module,
    owner_secret: torch.Tensor,
    prompts: Sequence[str],
    device: torch.device,
    dtype: torch.dtype,
    force_generate: bool = False,
    record_routing: bool = False,
) -> BaselineData:
    cache_dir = Path(args.baseline_cache_dir)
    metadata = baseline_metadata(args, prompts)

    if (
        args.reuse_baseline
        and not force_generate
        and not record_routing
        and cache_is_compatible(cache_dir, metadata)
        and (cache_dir / "baseline_logits.npz").exists()
        and (cache_dir / "baseline_per_image.csv").exists()
    ):
        print(f"[Baseline] Reusing cache: {cache_dir}")
        return load_cached_baseline(cache_dir, len(prompts))

    print(f"[Baseline] Generating cache: {cache_dir}")
    return generate_baseline(
        args,
        pipe,
        extractor,
        owner_secret,
        prompts,
        cache_dir,
        device,
        dtype,
        record_routing=record_routing,
    )


# =============================================================================
# Adapter parameter snapshot and modifications
# =============================================================================

ADAPTER_PARAMETER_SUFFIXES = (
    "lora_A",
    "lora_B",
    "lora_bias",
    "router.weight",
    "router.bias",
    "bit_router.weight",
    "bit_router.bias",
)


def named_watermark_parameters(
    model: nn.Module,
    target: str = "all",
) -> List[Tuple[str, nn.Parameter]]:
    result: List[Tuple[str, nn.Parameter]] = []
    for layer in moe_layers(model):
        entries = [
            (f"{layer.layer_name}.lora_A", layer.lora_A),
            (f"{layer.layer_name}.lora_B", layer.lora_B),
            (f"{layer.layer_name}.lora_bias", layer.lora_bias),
            (f"{layer.layer_name}.router.weight", layer.router.weight),
            (f"{layer.layer_name}.router.bias", layer.router.bias),
            (f"{layer.layer_name}.bit_router.weight", layer.bit_router.weight),
            (f"{layer.layer_name}.bit_router.bias", layer.bit_router.bias),
        ]
        for name, parameter in entries:
            is_lora = ".lora_" in name
            is_router = ".router." in name or ".bit_router." in name
            if target == "all" or (target == "lora" and is_lora) or (target == "router" and is_router):
                result.append((name, parameter))
    return result


def snapshot_watermark_parameters(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in named_watermark_parameters(model, target="all")
    }


@torch.no_grad()
def restore_watermark_parameters(
    model: nn.Module,
    snapshot: Mapping[str, torch.Tensor],
) -> None:
    current = dict(named_watermark_parameters(model, target="all"))
    missing = set(snapshot) - set(current)
    if missing:
        raise KeyError(f"Snapshot keys not found in model: {sorted(missing)[:5]}")
    for name, tensor in snapshot.items():
        parameter = current[name]
        parameter.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))


@torch.no_grad()
def apply_global_magnitude_pruning(
    model: nn.Module,
    ratio: float,
    target: str,
) -> dict:
    if not 0.0 <= ratio < 1.0:
        raise ValueError("Pruning ratio must lie in [0,1).")
    parameters = named_watermark_parameters(model, target=target)
    flat_abs = torch.cat([
        parameter.detach().float().abs().reshape(-1).cpu()
        for _, parameter in parameters
    ])
    if ratio <= 0.0:
        threshold = -1.0
    else:
        total_values = int(flat_abs.numel())
        kth_index = min(
            total_values - 1,
            max(0, int(math.ceil(ratio * total_values)) - 1),
        )
        flat_abs_np = flat_abs.contiguous().numpy()
        flat_abs_np.partition(kth_index)
        threshold = float(flat_abs_np[kth_index])

    total = 0
    pruned = 0
    for _, parameter in parameters:
        mask = parameter.detach().float().abs() <= threshold
        total += int(mask.numel())
        pruned += int(mask.sum().item())
        parameter.masked_fill_(mask.to(parameter.device), 0)

    return {
        "requested_ratio": ratio,
        "actual_ratio": pruned / max(total, 1),
        "threshold": threshold,
        "target": target,
    }


@torch.no_grad()
def apply_gaussian_parameter_noise(
    model: nn.Module,
    sigma: float,
    target: str,
    seed: int,
) -> dict:
    if sigma < 0:
        raise ValueError("Noise sigma must be non-negative.")
    torch.manual_seed(seed)
    relative_rms: List[float] = []

    for _, parameter in named_watermark_parameters(model, target=target):
        original = parameter.detach().float()
        std = float(original.std(unbiased=False).item())
        if std <= 0.0:
            continue
        noise = torch.randn_like(original) * (sigma * std)
        parameter.add_(noise.to(dtype=parameter.dtype))
        relative_rms.append(
            float(noise.pow(2).mean().sqrt().item())
            / max(float(original.pow(2).mean().sqrt().item()), 1e-12)
        )

    return {
        "sigma": sigma,
        "target": target,
        "mean_relative_rms": safe_mean(relative_rms),
    }


@torch.no_grad()
def fake_symmetric_quantize(
    model: nn.Module,
    bits: int,
    target: str,
) -> dict:
    if bits < 2:
        raise ValueError("Quantization bits must be >=2.")
    qmax = float(2 ** (bits - 1) - 1)
    relative_errors: List[float] = []

    for _, parameter in named_watermark_parameters(model, target=target):
        original = parameter.detach().float()
        max_abs = float(original.abs().max().item())
        if max_abs <= 0.0:
            continue
        scale = max_abs / qmax
        quantized = torch.round(original / scale).clamp(-qmax, qmax) * scale
        error = quantized - original
        relative_errors.append(
            float(error.pow(2).mean().sqrt().item())
            / max(float(original.pow(2).mean().sqrt().item()), 1e-12)
        )
        parameter.copy_(quantized.to(dtype=parameter.dtype))

    return {
        "bits": bits,
        "target": target,
        "mean_relative_quant_error": safe_mean(relative_errors),
    }


# =============================================================================
# Conditions
# =============================================================================

@dataclass
class Condition:
    label: str
    apply: Callable[[nn.Module], dict]
    metadata: dict


def identity_condition(model: nn.Module) -> dict:
    del model
    return {}


def build_expert_dropout_conditions(
    model: nn.Module,
    active_counts: Sequence[int],
    max_masks_per_count: int,
    renormalize: bool,
    seed: int,
) -> List[Condition]:
    layers = moe_layers(model)
    num_experts = layers[0].num_experts
    conditions: List[Condition] = []

    for active_count in active_counts:
        if not 1 <= active_count < num_experts:
            continue
        combinations = list(itertools.combinations(range(num_experts), active_count))
        if max_masks_per_count > 0 and len(combinations) > max_masks_per_count:
            rng = random.Random(seed + active_count)
            combinations = rng.sample(combinations, max_masks_per_count)

        for combo in combinations:
            active = tuple(sorted(combo))
            mask_values = [1.0 if i in active else 0.0 for i in range(num_experts)]
            label = f"experts_{active_count}of{num_experts}_" + "".join(map(str, active))

            def apply_fn(
                current_model: nn.Module,
                mask_values=mask_values,
                renormalize=renormalize,
            ) -> dict:
                for layer in moe_layers(current_model):
                    layer.expert_mask = torch.tensor(mask_values, dtype=torch.float32)
                    layer.renormalize_expert_mask = renormalize
                return {
                    "active_experts": int(sum(mask_values)),
                    "expert_mask": mask_values,
                    "renormalized": renormalize,
                }

            conditions.append(
                Condition(
                    label=label,
                    apply=apply_fn,
                    metadata={
                        "active_experts": active_count,
                        "expert_mask": mask_values,
                        "renormalized": renormalize,
                    },
                )
            )
    return conditions


def build_routing_ablation_conditions(
    model: nn.Module,
    include_single_experts: bool,
) -> List[Condition]:
    num_experts = moe_layers(model)[0].num_experts
    conditions: List[Condition] = []

    for mode in ["feature_only", "secret_only", "uniform"]:
        def apply_fn(current_model: nn.Module, mode=mode) -> dict:
            for layer in moe_layers(current_model):
                layer.routing_mode = mode
            return {"routing_mode": mode}

        conditions.append(
            Condition(
                label=mode,
                apply=apply_fn,
                metadata={"routing_mode": mode},
            )
        )

    if include_single_experts:
        for expert_index in range(num_experts):
            def apply_single(
                current_model: nn.Module,
                expert_index=expert_index,
            ) -> dict:
                for layer in moe_layers(current_model):
                    layer.routing_mode = "single"
                    layer.single_expert_index = expert_index
                return {
                    "routing_mode": "single",
                    "single_expert": expert_index,
                }

            conditions.append(
                Condition(
                    label=f"single_expert_{expert_index}",
                    apply=apply_single,
                    metadata={
                        "routing_mode": "single",
                        "single_expert": expert_index,
                    },
                )
            )
    return conditions


def parse_block_index(layer_name: str) -> Optional[int]:
    match = re.search(r"transformer_blocks\.(\d+)", layer_name)
    return int(match.group(1)) if match else None


def projection_family(layer_name: str) -> str:
    if layer_name.endswith("to_q"):
        return "to_q"
    if layer_name.endswith("to_k"):
        return "to_k"
    if layer_name.endswith("to_v"):
        return "to_v"
    if "to_out" in layer_name:
        return "to_out"
    return "other"


def build_adapter_dropout_conditions(
    model: nn.Module,
    ratios: Sequence[float],
    repeats: int,
    include_structured: bool,
    seed: int,
) -> List[Condition]:
    names = sorted(layer_map(model))
    conditions: List[Condition] = []

    for ratio in ratios:
        if not 0.0 < ratio < 1.0:
            continue
        count = max(1, int(round(len(names) * ratio)))
        for repeat in range(repeats):
            rng = random.Random(seed + int(ratio * 10000) + repeat)
            disabled = sorted(rng.sample(names, count))
            label = f"random_drop_{int(round(ratio * 100)):02d}_r{repeat}"

            def apply_fn(
                current_model: nn.Module,
                disabled=disabled,
                ratio=ratio,
                repeat=repeat,
            ) -> dict:
                mapping = layer_map(current_model)
                for name in disabled:
                    mapping[name].adapter_enabled = False
                return {
                    "drop_type": "random",
                    "drop_ratio": ratio,
                    "repeat": repeat,
                    "disabled_count": len(disabled),
                }

            conditions.append(
                Condition(
                    label=label,
                    apply=apply_fn,
                    metadata={
                        "drop_type": "random",
                        "drop_ratio": ratio,
                        "repeat": repeat,
                        "disabled_count": len(disabled),
                    },
                )
            )

    if include_structured:
        block_to_names: Dict[int, List[str]] = defaultdict(list)
        for name in names:
            block = parse_block_index(name)
            if block is not None:
                block_to_names[block].append(name)
        blocks = sorted(block_to_names)
        if blocks:
            chunks = np.array_split(np.array(blocks), 3)
            for label_part, chunk in zip(["early", "middle", "late"], chunks):
                selected_blocks = [int(x) for x in chunk.tolist()]
                disabled = sorted(
                    name
                    for block in selected_blocks
                    for name in block_to_names[block]
                )
                label = f"structured_drop_{label_part}"

                def apply_structured(
                    current_model: nn.Module,
                    disabled=disabled,
                    label_part=label_part,
                    selected_blocks=selected_blocks,
                ) -> dict:
                    mapping = layer_map(current_model)
                    for name in disabled:
                        mapping[name].adapter_enabled = False
                    return {
                        "drop_type": "structured",
                        "region": label_part,
                        "disabled_blocks": selected_blocks,
                        "disabled_count": len(disabled),
                    }

                conditions.append(
                    Condition(
                        label=label,
                        apply=apply_structured,
                        metadata={
                            "drop_type": "structured",
                            "region": label_part,
                            "disabled_blocks": selected_blocks,
                            "disabled_count": len(disabled),
                        },
                    )
                )
    return conditions


def build_parameter_robustness_conditions(
    prune_ratios: Sequence[float],
    noise_sigmas: Sequence[float],
    quant_bits: Sequence[int],
    target: str,
    seed: int,
) -> List[Condition]:
    conditions: List[Condition] = []

    for ratio in prune_ratios:
        if ratio <= 0:
            continue

        def apply_prune(
            current_model: nn.Module,
            ratio=ratio,
            target=target,
        ) -> dict:
            return apply_global_magnitude_pruning(current_model, ratio, target)

        conditions.append(
            Condition(
                label=f"prune_{target}_{int(round(ratio * 100)):02d}",
                apply=apply_prune,
                metadata={
                    "attack": "magnitude_pruning",
                    "ratio": ratio,
                    "target": target,
                },
            )
        )

    for sigma in noise_sigmas:
        if sigma <= 0:
            continue

        def apply_noise(
            current_model: nn.Module,
            sigma=sigma,
            target=target,
            seed=seed,
        ) -> dict:
            return apply_gaussian_parameter_noise(
                current_model,
                sigma,
                target,
                seed + int(sigma * 100000),
            )

        conditions.append(
            Condition(
                label=f"noise_{target}_{sigma:g}",
                apply=apply_noise,
                metadata={
                    "attack": "gaussian_noise",
                    "sigma": sigma,
                    "target": target,
                },
            )
        )

    for bits in quant_bits:
        def apply_quant(
            current_model: nn.Module,
            bits=bits,
            target=target,
        ) -> dict:
            return fake_symmetric_quantize(current_model, bits, target)

        conditions.append(
            Condition(
                label=f"fake_int{bits}_{target}",
                apply=apply_quant,
                metadata={
                    "attack": "fake_quantization",
                    "bits": bits,
                    "target": target,
                },
            )
        )
    return conditions


def build_layer_contribution_conditions(
    model: nn.Module,
    scope: str,
    max_units: int,
    seed: int,
) -> List[Condition]:
    names = sorted(layer_map(model))
    groups: Dict[str, List[str]] = defaultdict(list)

    if scope == "block":
        for name in names:
            block = parse_block_index(name)
            if block is not None:
                groups[f"block_{block:02d}"].append(name)
    elif scope == "projection":
        for name in names:
            groups[projection_family(name)].append(name)
    elif scope == "module":
        for name in names:
            groups[name].append(name)
    else:
        raise ValueError(f"Unknown contribution scope: {scope}")

    items = sorted(groups.items())
    if max_units > 0 and len(items) > max_units:
        rng = random.Random(seed)
        items = sorted(rng.sample(items, max_units))

    conditions: List[Condition] = []
    for unit, disabled in items:
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "_", unit)

        def apply_fn(
            current_model: nn.Module,
            disabled=disabled,
            unit=unit,
        ) -> dict:
            mapping = layer_map(current_model)
            for name in disabled:
                mapping[name].adapter_enabled = False
            return {
                "contribution_unit": unit,
                "disabled_count": len(disabled),
            }

        conditions.append(
            Condition(
                label=f"disable_{safe_label}",
                apply=apply_fn,
                metadata={
                    "contribution_unit": unit,
                    "disabled_count": len(disabled),
                },
            )
        )
    return conditions


# =============================================================================
# Evaluation and summary
# =============================================================================

def evaluate_condition(
    args: argparse.Namespace,
    condition: Condition,
    pipe: StableDiffusion3Pipeline,
    extractor: nn.Module,
    owner_secret: torch.Tensor,
    prompts: Sequence[str],
    baseline: BaselineData,
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[List[dict], np.ndarray]:
    condition_dir = output_dir / "images" / condition.label
    condition_dir.mkdir(parents=True, exist_ok=True)

    applied_metadata = condition.apply(pipe.transformer)
    rows: List[dict] = []
    logits_list: List[np.ndarray] = []

    for index, prompt in enumerate(tqdm(prompts, desc=condition.label, leave=False)):
        image_seed = args.seed + index
        image = generate_image(
            pipe,
            prompt,
            image_seed,
            True,
            owner_secret,
            args.wm_gate_end,
            args.num_inference_steps,
            args.guidance_scale,
            args.height,
            args.width,
            device,
        )
        image.save(condition_dir / f"{index:04d}.png")

        logits = extract_logits_from_pil(
            image,
            pipe.vae,
            extractor,
            device,
            dtype,
            args.latent_resolution,
        )
        logits_list.append(logits.squeeze(0).cpu().numpy().astype(np.float32))

        with Image.open(baseline.clean_paths[index]) as clean_image:
            clean_image = clean_image.convert("RGB")
            psnr_clean, ssim_clean = image_quality(clean_image, image)
        with Image.open(baseline.wm_paths[index]) as wm_image:
            wm_image = wm_image.convert("RGB")
            psnr_wm, ssim_wm = image_quality(wm_image, image)

        row = {
            "experiment": args.experiment,
            "condition": condition.label,
            "prompt_index": index,
            "seed": image_seed,
            "prompt": prompt,
            **watermark_metrics(logits, owner_secret),
            "psnr_to_clean": psnr_clean,
            "ssim_to_clean": ssim_clean,
            "psnr_to_wm": psnr_wm,
            "ssim_to_wm": ssim_wm,
            "condition_metadata": json.dumps(
                {**condition.metadata, **applied_metadata},
                ensure_ascii=False,
                sort_keys=True,
            ),
        }
        rows.append(row)

    return rows, np.stack(logits_list)


def summarize_rows(rows: Sequence[dict]) -> List[dict]:
    grouped: Dict[str, List[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row["condition"])].append(row)

    metrics = [
        "bit_acc",
        "owner_margin",
        "matched_probability",
        "psnr_to_clean",
        "ssim_to_clean",
        "psnr_to_wm",
        "ssim_to_wm",
    ]
    summary: List[dict] = []
    for condition, values in grouped.items():
        result = {
            "condition": condition,
            "num_images": len(values),
            "condition_metadata": values[0].get("condition_metadata", "{}"),
        }
        for metric in metrics:
            metric_values = [float(row[metric]) for row in values]
            result[f"{metric}_mean"] = safe_mean(metric_values)
            result[f"{metric}_std"] = safe_std(metric_values)
        summary.append(result)
    return summary


def baseline_rows_for_output(
    baseline: BaselineData,
    experiment: str,
) -> List[dict]:
    rows: List[dict] = []
    for raw in baseline.rows:
        row = dict(raw)
        for key in [
            "prompt_index",
            "seed",
            "bit_acc",
            "owner_margin",
            "matched_probability",
            "psnr_to_clean",
            "ssim_to_clean",
            "psnr_to_wm",
            "ssim_to_wm",
        ]:
            row[key] = float(row[key]) if key not in {"prompt_index", "seed"} else int(row[key])
        row["experiment"] = experiment
        row["condition_metadata"] = "{}"
        rows.append(row)
    return rows


def save_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    all_fields: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                all_fields.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=all_fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def plot_summary(summary: Sequence[dict], output_dir: Path) -> None:
    if not summary:
        return
    labels = [row["condition"] for row in summary]

    for metric, ylabel, filename in [
        ("bit_acc_mean", "BitAcc", "summary_bitacc.png"),
        ("owner_margin_mean", "Owner margin", "summary_owner_margin.png"),
        ("psnr_to_wm_mean", "PSNR to original TG-MoE (dB)", "summary_psnr_to_wm.png"),
    ]:
        values = [float(row[metric]) for row in summary]
        plt.figure(figsize=(max(6.0, 0.42 * len(labels)), 4.2))
        plt.bar(range(len(labels)), values)
        plt.xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
        plt.ylabel(ylabel)
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=300, bbox_inches="tight")
        plt.close()


def evaluate_frozen_blackbox(
    logits_by_condition: Mapping[str, np.ndarray],
    thresholds_path: Path,
    query_budgets: Sequence[int],
    num_trials: int,
    seed: int,
) -> List[dict]:
    frozen = json.loads(thresholds_path.read_text(encoding="utf-8"))
    owner_bits = np.asarray(frozen["owner_bits"], dtype=np.int8)
    signs = owner_bits.astype(np.float32) * 2.0 - 1.0
    thresholds = frozen["thresholds"]
    rows: List[dict] = []

    for condition, logits in logits_by_condition.items():
        logits = np.asarray(logits, dtype=np.float32)
        for q in query_budgets:
            if str(q) not in thresholds or len(logits) < q:
                continue
            rng = np.random.default_rng(seed + q)
            scores = np.empty(num_trials, dtype=np.float32)
            bitaccs = np.empty(num_trials, dtype=np.float32)

            for trial in range(num_trials):
                indices = rng.choice(len(logits), size=q, replace=False)
                aggregated = logits[indices].mean(axis=0)
                scores[trial] = float((aggregated * signs).mean())
                bitaccs[trial] = float(
                    ((aggregated >= 0).astype(np.int8) == owner_bits).mean()
                )

            threshold = float(thresholds[str(q)])
            rows.append({
                "condition": condition,
                "query_budget": q,
                "threshold": threshold,
                "verification_rate": float(np.mean(scores >= threshold)),
                "mean_owner_score": float(scores.mean()),
                "std_owner_score": float(scores.std(ddof=1)),
                "aggregated_bitacc_mean": float(bitaccs.mean()),
                "num_trials": num_trials,
            })
    return rows


# =============================================================================
# Routing analysis
# =============================================================================

def save_routing_profile(
    model: nn.Module,
    output_dir: Path,
) -> None:
    routing_rows: List[dict] = []
    probability_matrix: List[List[float]] = []
    labels: List[str] = []

    for layer in moe_layers(model):
        count = max(layer.routing_token_count, 1)
        mean_probs = (layer.routing_prob_sum / count).numpy()
        top1 = (layer.routing_top1_count / count).numpy()
        normalized_entropy = (
            layer.routing_entropy_sum / count / math.log(layer.num_experts)
            if layer.num_experts > 1
            else 0.0
        )
        labels.append(layer.layer_name)
        probability_matrix.append(mean_probs.tolist())
        for expert in range(layer.num_experts):
            routing_rows.append({
                "layer": layer.layer_name,
                "expert": expert,
                "mean_probability": float(mean_probs[expert]),
                "top1_frequency": float(top1[expert]),
                "normalized_entropy": float(normalized_entropy),
                "token_count": int(layer.routing_token_count),
            })

    save_csv(output_dir / "routing_profile.csv", routing_rows)

    if probability_matrix:
        matrix = np.asarray(probability_matrix)
        plt.figure(figsize=(7.0, max(5.0, 0.16 * len(labels))))
        plt.imshow(matrix, aspect="auto")
        plt.colorbar(label="Mean routing probability")
        plt.yticks(range(len(labels)), labels, fontsize=5)
        plt.xticks(range(matrix.shape[1]), [f"E{i}" for i in range(matrix.shape[1])])
        plt.xlabel("Expert")
        plt.ylabel("MoE-LoRA layer")
        plt.tight_layout()
        plt.savefig(output_dir / "routing_probability_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

    cosine_rows: List[dict] = []
    average_matrix = None
    matrix_count = 0

    for layer in moe_layers(model):
        num_experts = layer.num_experts
        layer_matrix = np.eye(num_experts, dtype=np.float64)

        # Effective update is A @ B. Compute Frobenius cosine without
        # materializing the full in_features x out_features matrix.
        with torch.no_grad():
            A = layer.lora_A.detach().float()
            B = layer.lora_B.detach().float()
            norms = []
            for i in range(num_experts):
                gram_A = A[i].T @ A[i]
                gram_B = B[i] @ B[i].T
                norm_sq = torch.trace(gram_A @ gram_B).clamp(min=1e-20)
                norms.append(torch.sqrt(norm_sq))

            for i in range(num_experts):
                for j in range(i + 1, num_experts):
                    cross_A = A[i].T @ A[j]
                    cross_B = B[j] @ B[i].T
                    inner = torch.trace(cross_A @ cross_B)
                    cosine = float(
                        (inner / (norms[i] * norms[j]).clamp(min=1e-20)).item()
                    )
                    layer_matrix[i, j] = cosine
                    layer_matrix[j, i] = cosine
                    cosine_rows.append({
                        "layer": layer.layer_name,
                        "expert_i": i,
                        "expert_j": j,
                        "effective_update_cosine": cosine,
                    })

        average_matrix = (
            layer_matrix
            if average_matrix is None
            else average_matrix + layer_matrix
        )
        matrix_count += 1

    save_csv(output_dir / "expert_effective_update_cosine.csv", cosine_rows)

    if average_matrix is not None:
        average_matrix = average_matrix / max(matrix_count, 1)
        plt.figure(figsize=(4.8, 4.2))
        plt.imshow(average_matrix, vmin=-1.0, vmax=1.0)
        plt.colorbar(label="Mean cosine similarity")
        plt.xticks(range(average_matrix.shape[1]), [f"E{i}" for i in range(average_matrix.shape[1])])
        plt.yticks(range(average_matrix.shape[0]), [f"E{i}" for i in range(average_matrix.shape[0])])
        plt.tight_layout()
        plt.savefig(output_dir / "expert_cosine_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()


def save_layer_contribution(
    summary: List[dict],
    baseline_wm_margin: float,
    baseline_wm_bitacc: float,
    output_dir: Path,
) -> None:
    rows: List[dict] = []
    for row in summary:
        if not row["condition"].startswith("disable_"):
            continue
        result = dict(row)
        result["owner_margin_drop"] = baseline_wm_margin - float(row["owner_margin_mean"])
        result["bit_acc_drop"] = baseline_wm_bitacc - float(row["bit_acc_mean"])
        rows.append(result)

    rows.sort(key=lambda item: item["owner_margin_drop"], reverse=True)
    save_csv(output_dir / "layer_contribution_ranked.csv", rows)

    top = rows[: min(25, len(rows))]
    if top:
        plt.figure(figsize=(7.2, max(4.2, 0.28 * len(top))))
        labels = [row["condition"].replace("disable_", "") for row in top][::-1]
        values = [row["owner_margin_drop"] for row in top][::-1]
        plt.barh(range(len(labels)), values)
        plt.yticks(range(len(labels)), labels, fontsize=7)
        plt.xlabel("Owner-margin decrease after disabling unit")
        plt.tight_layout()
        plt.savefig(output_dir / "layer_contribution_top.png", dpi=300, bbox_inches="tight")
        plt.close()


# =============================================================================
# Main
# =============================================================================

def main(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    output_dir = Path(args.output_dir) / args.experiment
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

    checkpoint_path = locate_transformer_state(Path(args.transformer_dir))
    state = load_transformer_state(checkpoint_path)
    inferred_experts, inferred_rank = infer_moe_shape(state)
    num_experts = args.num_experts if args.num_experts > 0 else inferred_experts
    rank = args.rank if args.rank > 0 else inferred_rank
    print(
        f"[Checkpoint] inferred experts={inferred_experts}, rank={inferred_rank}; "
        f"using experts={num_experts}, rank={rank}"
    )

    transformer = SD3Transformer2DModel.from_pretrained(
        args.base_model,
        subfolder="transformer",
        local_files_only=args.local_files_only,
    )
    transformer = inject_moe_lora_to_sd3(transformer, num_experts, rank)
    transformer.load_state_dict(state, strict=True)
    transformer = transformer.to(device=device, dtype=dtype)
    transformer.eval()
    transformer.requires_grad_(False)

    pipe = StableDiffusion3Pipeline.from_pretrained(
        args.base_model,
        transformer=transformer,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.vae.eval()
    pipe.vae.requires_grad_(False)
    if args.enable_vae_tiling:
        pipe.vae.enable_tiling()

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

    parameter_snapshot = snapshot_watermark_parameters(pipe.transformer)
    reset_all_controls(pipe.transformer)

    # routing_profile must regenerate watermarked outputs while recording internal routing.
    if args.experiment == "routing_profile":
        baseline = get_baseline(
            args,
            pipe,
            extractor,
            owner_secret,
            prompts,
            device,
            dtype,
            force_generate=True,
            record_routing=True,
        )
        rows = baseline_rows_for_output(baseline, args.experiment)
        save_csv(output_dir / "per_image_results.csv", rows)
        summary = summarize_rows(rows)
        save_csv(output_dir / "summary.csv", summary)
        save_routing_profile(pipe.transformer, output_dir)
        print(f"[Done] {output_dir}")
        return

    baseline = get_baseline(
        args,
        pipe,
        extractor,
        owner_secret,
        prompts,
        device,
        dtype,
    )

    if args.experiment == "expert_dropout":
        conditions = build_expert_dropout_conditions(
            pipe.transformer,
            args.expert_active_counts,
            args.max_masks_per_count,
            args.renormalize_expert_mask,
            args.seed,
        )
    elif args.experiment == "routing_ablation":
        conditions = build_routing_ablation_conditions(
            pipe.transformer,
            args.include_single_experts,
        )
    elif args.experiment == "adapter_dropout":
        conditions = build_adapter_dropout_conditions(
            pipe.transformer,
            args.drop_ratios,
            args.mask_repeats,
            args.include_structured_dropout,
            args.seed,
        )
    elif args.experiment == "parameter_robustness":
        conditions = build_parameter_robustness_conditions(
            args.prune_ratios,
            args.noise_sigmas,
            args.quant_bits,
            args.parameter_target,
            args.seed,
        )
    elif args.experiment == "layer_contribution":
        conditions = build_layer_contribution_conditions(
            pipe.transformer,
            args.contribution_scope,
            args.max_contribution_units,
            args.seed,
        )
    else:
        raise ValueError(f"Unsupported experiment: {args.experiment}")

    print(f"[Experiment] {args.experiment}: {len(conditions)} conditions")

    all_rows = baseline_rows_for_output(baseline, args.experiment)
    logits_by_condition: Dict[str, np.ndarray] = {
        "clean": baseline.clean_logits,
        "wm": baseline.wm_logits,
    }

    for condition in conditions:
        restore_watermark_parameters(pipe.transformer, parameter_snapshot)
        reset_all_controls(pipe.transformer)

        rows, logits = evaluate_condition(
            args,
            condition,
            pipe,
            extractor,
            owner_secret,
            prompts,
            baseline,
            output_dir,
            device,
            dtype,
        )
        all_rows.extend(rows)
        logits_by_condition[condition.label] = logits

    restore_watermark_parameters(pipe.transformer, parameter_snapshot)
    reset_all_controls(pipe.transformer)

    save_csv(output_dir / "per_image_results.csv", all_rows)
    summary = summarize_rows(all_rows)
    save_csv(output_dir / "summary.csv", summary)
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "configuration": vars(args),
                "results": summary,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plot_summary(summary, output_dir)

    if args.frozen_thresholds:
        blackbox_rows = evaluate_frozen_blackbox(
            logits_by_condition,
            Path(args.frozen_thresholds),
            args.query_budgets,
            args.blackbox_trials,
            args.seed,
        )
        save_csv(output_dir / "blackbox_results.csv", blackbox_rows)

    if args.experiment == "layer_contribution":
        baseline_wm = next(row for row in summary if row["condition"] == "wm")
        save_layer_contribution(
            summary,
            float(baseline_wm["owner_margin_mean"]),
            float(baseline_wm["bit_acc_mean"]),
            output_dir,
        )

    print("\n==================== DONE ====================")
    print(f"Experiment: {args.experiment}")
    print(f"Output:     {output_dir}")
    print(f"Summary:    {output_dir / 'summary.csv'}")
    if args.frozen_thresholds:
        print(f"Black-box:  {output_dir / 'blackbox_results.csv'}")
    print("==============================================")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="TG-MoE SD3 white-box experiment suite"
    )
    parser.add_argument(
        "--experiment",
        required=True,
        choices=[
            "expert_dropout",
            "routing_ablation",
            "adapter_dropout",
            "parameter_robustness",
            "routing_profile",
            "layer_contribution",
        ],
    )
    parser.add_argument("--base_model", required=True)
    parser.add_argument("--transformer_dir", required=True)
    parser.add_argument("--pretrainedWM_dir", required=True)
    parser.add_argument("--output_dir", default="./Evaluation/whitebox")
    parser.add_argument(
        "--baseline_cache_dir",
        default="./Evaluation/whitebox_baseline",
    )
    parser.add_argument("--reuse_baseline", type=str2bool, default=True)

    parser.add_argument("--prompts_file", default=None)
    parser.add_argument("--num_prompts", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_inference_steps", type=int, default=28)
    parser.add_argument("--guidance_scale", type=float, default=7.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--wm_gate_end", type=float, default=0.40)

    parser.add_argument("--num_experts", type=int, default=0)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--bit_dim", type=int, default=48)
    parser.add_argument("--latent_resolution", type=int, default=128)
    parser.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="bf16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--enable_vae_tiling", type=str2bool, default=False)
    parser.add_argument("--local_files_only", type=str2bool, default=True)

    parser.add_argument("--frozen_thresholds", default=None)
    parser.add_argument("--query_budgets", type=parse_int_list, default=[1, 5, 10])
    parser.add_argument("--blackbox_trials", type=int, default=2000)

    # Expert dropout
    parser.add_argument(
        "--expert_active_counts",
        type=parse_int_list,
        default=[3, 2, 1],
    )
    parser.add_argument("--max_masks_per_count", type=int, default=2)
    parser.add_argument(
        "--renormalize_expert_mask",
        type=str2bool,
        default=False,
    )

    # Routing ablation
    parser.add_argument(
        "--include_single_experts",
        type=str2bool,
        default=True,
    )

    # Adapter dropout
    parser.add_argument(
        "--drop_ratios",
        type=parse_float_list,
        default=[0.25, 0.50],
    )
    parser.add_argument("--mask_repeats", type=int, default=3)
    parser.add_argument(
        "--include_structured_dropout",
        type=str2bool,
        default=True,
    )

    # Parameter robustness
    parser.add_argument(
        "--prune_ratios",
        type=parse_float_list,
        default=[0.20, 0.40],
    )
    parser.add_argument(
        "--noise_sigmas",
        type=parse_float_list,
        default=[0.02, 0.05],
    )
    parser.add_argument(
        "--quant_bits",
        type=parse_int_list,
        default=[8],
    )
    parser.add_argument(
        "--parameter_target",
        choices=["all", "lora", "router"],
        default="all",
    )

    # Layer contribution
    parser.add_argument(
        "--contribution_scope",
        choices=["block", "projection", "module"],
        default="projection",
    )
    parser.add_argument("--max_contribution_units", type=int, default=0)

    args = parser.parse_args()

    if not 0.0 < args.wm_gate_end <= 1.0:
        parser.error("--wm_gate_end must lie in (0,1].")
    if args.num_prompts <= 0:
        parser.error("--num_prompts must be positive.")
    if args.blackbox_trials <= 0:
        parser.error("--blackbox_trials must be positive.")
    return args


if __name__ == "__main__":
    main(parse_args())
