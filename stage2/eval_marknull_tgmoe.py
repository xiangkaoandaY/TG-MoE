#!/usr/bin/env python3
"""Evaluate MarkNull outputs with the private TG-MoE Stage-I extractor.

The attack uses SD1.5 as its public proxy, but owner-side verification must use
the original SD3 VAE together with ``decoder.pth`` and ``secret.pt`` from the
TG-MoE Stage-I owner directory.

Reported metrics:
  * BitAcc before and after MarkNull
  * TPR at the requested theoretical binomial FPR (default 1e-6)
  * optional empirical clean FPR
  * paired PSNR, SSIM, LPIPS between watermarked and attacked images
  * set-level FID between watermarked and attacked directories
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import AutoencoderKL
from PIL import Image, ImageOps
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchvision import transforms
from tqdm.auto import tqdm


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


class Conv2D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        activation: str | None = "relu",
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
            self.act: nn.Module | None = nn.ReLU(inplace=True)
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
        activation: str | None = "relu",
    ) -> None:
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        if activation == "relu":
            self.act: nn.Module | None = nn.ReLU(inplace=True)
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
    """The 16-channel SD3 Stage-I extractor used by TG-MoE."""

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate MarkNull against the TG-MoE 48-bit extractor."
    )
    parser.add_argument("--watermarked_dir", type=Path, required=True)
    parser.add_argument("--attacked_dir", type=Path, required=True)
    parser.add_argument(
        "--clean_dir",
        type=Path,
        default=None,
        help="Optional non-watermarked images for empirical clean FPR.",
    )
    parser.add_argument(
        "--base_model",
        type=Path,
        required=True,
        help="Local SD3 Diffusers model. Do not pass the SD1.5 proxy model.",
    )
    parser.add_argument(
        "--pretrainedWM_dir",
        type=Path,
        required=True,
        help="Stage-I owner directory containing decoder.pth and secret.pt.",
    )
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--bit_dim", type=int, default=48)
    parser.add_argument("--target_fpr", type=float, default=1e-6)
    parser.add_argument(
        "--threshold_bits",
        type=int,
        default=None,
        help="Use a previously frozen empirical threshold; otherwise binomial threshold.",
    )
    parser.add_argument("--latent_resolution", type=int, default=128)
    parser.add_argument("--max_images", type=int, default=None)
    parser.add_argument("--seed", type=int, default=20260822)
    parser.add_argument("--enable_vae_tiling", action="store_true")
    parser.add_argument("--allow_download", action="store_true")
    parser.add_argument("--skip_lpips", action="store_true")
    parser.add_argument("--skip_fid", action="store_true")
    parser.add_argument("--fid_batch_size", type=int, default=8)
    parser.add_argument("--fid_num_workers", type=int, default=4)
    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch_size must be positive")
    if args.bit_dim <= 0:
        parser.error("--bit_dim must be positive")
    if not (0.0 < args.target_fpr < 1.0):
        parser.error("--target_fpr must lie in (0,1)")
    if args.max_images is not None and args.max_images <= 0:
        parser.error("--max_images must be positive")
    return args


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def torch_load_compat(path: Path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def clean_state_dict(obj) -> Dict[str, torch.Tensor]:
    if isinstance(obj, dict):
        for key in ("decoder", "extractor", "state_dict", "model"):
            if isinstance(obj.get(key), dict):
                obj = obj[key]
                break
    if not isinstance(obj, dict):
        raise ValueError("Extractor checkpoint does not contain a state_dict")

    cleaned: Dict[str, torch.Tensor] = {}
    for key, value in obj.items():
        if not torch.is_tensor(value):
            continue
        while key.startswith("module."):
            key = key[len("module.") :]
        cleaned[key] = value
    if not cleaned:
        raise ValueError("No tensor parameters found in extractor checkpoint")
    return cleaned


def extract_tensor(obj, preferred_keys: Sequence[str]) -> torch.Tensor:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for key in preferred_keys:
            value = obj.get(key)
            if torch.is_tensor(value):
                return value
        tensors = [value for value in obj.values() if torch.is_tensor(value)]
        if len(tensors) == 1:
            return tensors[0]
    raise ValueError(f"Could not find tensor; tried keys={preferred_keys}")


def load_owner(
    pretrained_dir: Path,
    bit_dim: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, nn.Module]:
    secret_path = pretrained_dir / "secret.pt"
    decoder_path = pretrained_dir / "decoder.pth"
    if not secret_path.is_file():
        raise FileNotFoundError(secret_path)
    if not decoder_path.is_file():
        raise FileNotFoundError(decoder_path)

    secret_obj = torch_load_compat(secret_path)
    secret = extract_tensor(
        secret_obj,
        ("secret", "bits", "GT_secret", "watermark_secret"),
    ).detach().float().flatten()
    if secret.numel() != bit_dim:
        raise ValueError(f"Secret contains {secret.numel()} bits; expected {bit_dim}")
    secret = (secret >= 0.5).float() if float(secret.min()) >= 0 else (secret > 0).float()
    secret = secret.reshape(1, bit_dim).to(device=device)

    extractor = ExtractorForLatent(secret_size=bit_dim)
    extractor.load_state_dict(clean_state_dict(torch_load_compat(decoder_path)), strict=True)
    extractor.requires_grad_(False)
    extractor.eval()
    extractor = extractor.to(device=device, dtype=dtype)
    return secret, extractor


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}[name]


def image_map(directory: Path) -> Dict[str, Path]:
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return {
        path.name: path
        for path in directory.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def collect_pairs(
    watermarked_dir: Path,
    attacked_dir: Path,
    max_images: int | None,
) -> List[Tuple[str, Path, Path]]:
    watermarked = image_map(watermarked_dir)
    attacked = image_map(attacked_dir)
    names = sorted(set(watermarked) & set(attacked))
    if max_images is not None:
        names = names[:max_images]
    if not names:
        raise RuntimeError("No same-named image pairs were found")
    if len(names) != len(watermarked) or len(names) != len(attacked):
        print(
            f"[Warning] paired={len(names)}, watermarked={len(watermarked)}, "
            f"attacked={len(attacked)}"
        )
    return [(name, watermarked[name], attacked[name]) for name in names]


def batches(items: Sequence, batch_size: int) -> Iterable[Sequence]:
    for start in range(0, len(items), batch_size):
        yield items[start : start + batch_size]


def open_rgb(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def decode_paths(
    paths: Sequence[Path],
    vae: AutoencoderKL,
    extractor: nn.Module,
    device: torch.device,
    dtype: torch.dtype,
    latent_resolution: int,
    batch_size: int,
    description: str,
) -> torch.Tensor:
    outputs: List[torch.Tensor] = []
    to_tensor = transforms.ToTensor()
    for batch_paths in tqdm(
        list(batches(paths, batch_size)),
        desc=description,
    ):
        tensors = [to_tensor(open_rgb(path)) for path in batch_paths]
        shapes = {tuple(tensor.shape) for tensor in tensors}
        if len(shapes) != 1:
            raise ValueError(f"Images in a batch have different shapes: {shapes}")
        image_01 = torch.stack(tensors).to(device=device, dtype=dtype)
        with torch.inference_mode():
            posterior = vae.encode(image_01 * 2.0 - 1.0).latent_dist
            latent = posterior.mode() * float(
                getattr(vae.config, "scaling_factor", 1.0)
            )
            if latent.shape[-2:] != (latent_resolution, latent_resolution):
                latent = F.interpolate(
                    latent,
                    size=(latent_resolution, latent_resolution),
                    mode="bilinear",
                    align_corners=False,
                )
            reference = next(extractor.parameters())
            logits = extractor(latent.to(device=reference.device, dtype=reference.dtype))
        outputs.append(logits.float().cpu())
    return torch.cat(outputs, dim=0)


def binomial_tail(bit_dim: int, minimum_matches: int) -> float:
    numerator = sum(
        math.comb(bit_dim, value)
        for value in range(minimum_matches, bit_dim + 1)
    )
    return float(numerator / (2**bit_dim))


def threshold_for_fpr(bit_dim: int, target_fpr: float) -> Tuple[int, float]:
    for minimum_matches in range(bit_dim + 1):
        tail = binomial_tail(bit_dim, minimum_matches)
        if tail <= target_fpr:
            return minimum_matches, tail
    return bit_dim + 1, 0.0


def detection_arrays(
    logits: torch.Tensor,
    secret: torch.Tensor,
    threshold_bits: int,
) -> Dict[str, torch.Tensor]:
    predictions = (logits >= 0.0).long()
    target = secret.cpu().long().reshape(1, -1)
    matching = (predictions == target).sum(dim=1)
    bit_acc = matching.float() / target.shape[1]
    detected = matching >= threshold_bits
    probabilities = torch.sigmoid(logits)
    matched_probability = (
        target.float() * probabilities
        + (1.0 - target.float()) * (1.0 - probabilities)
    ).mean(dim=1)
    owner_margin = ((target.float() * 2.0 - 1.0) * logits).mean(dim=1)
    return {
        "matching": matching,
        "bit_acc": bit_acc,
        "detected": detected,
        "matched_probability": matched_probability,
        "owner_margin": owner_margin,
    }


def paired_quality(
    pairs: Sequence[Tuple[str, Path, Path]],
    device: torch.device,
    skip_lpips: bool,
) -> Tuple[List[Dict[str, float]], Dict[str, float]]:
    lpips_model = None
    if not skip_lpips:
        try:
            import lpips
        except ImportError as exc:
            raise RuntimeError("Install lpips or pass --skip_lpips") from exc
        lpips_model = lpips.LPIPS(net="alex").to(device).eval()

    rows: List[Dict[str, float]] = []
    for name, watermarked_path, attacked_path in tqdm(pairs, desc="Quality"):
        watermarked = open_rgb(watermarked_path)
        attacked = open_rgb(attacked_path)
        wm_np = np.asarray(watermarked, dtype=np.uint8)
        attacked_np = np.asarray(attacked, dtype=np.uint8)
        if wm_np.shape != attacked_np.shape:
            raise ValueError(
                f"Shape mismatch for {name}: {wm_np.shape} vs {attacked_np.shape}"
            )
        row: Dict[str, float] = {
            "psnr": float(peak_signal_noise_ratio(wm_np, attacked_np, data_range=255)),
            "ssim": float(
                structural_similarity(
                    wm_np,
                    attacked_np,
                    data_range=255,
                    channel_axis=-1,
                )
            ),
        }
        if lpips_model is not None:
            wm_tensor = transforms.ToTensor()(watermarked).unsqueeze(0).to(device)
            attacked_tensor = transforms.ToTensor()(attacked).unsqueeze(0).to(device)
            with torch.inference_mode():
                value = lpips_model(
                    wm_tensor * 2.0 - 1.0,
                    attacked_tensor * 2.0 - 1.0,
                ).reshape(-1)[0]
            row["lpips"] = float(value.item())
        rows.append(row)

    summary: Dict[str, float] = {}
    for key in ("psnr", "ssim", "lpips"):
        values = [row[key] for row in rows if key in row]
        if values:
            summary[f"{key}_mean"] = float(np.mean(values))
            summary[f"{key}_std"] = float(np.std(values))
    return rows, summary


def compute_fid(
    watermarked_dir: Path,
    attacked_dir: Path,
    batch_size: int,
    device: str,
    num_workers: int,
) -> float:
    try:
        from pytorch_fid.fid_score import calculate_fid_given_paths
    except ImportError as exc:
        raise RuntimeError("Install pytorch-fid or pass --skip_fid") from exc
    return float(
        calculate_fid_given_paths(
            [str(watermarked_dir), str(attacked_dir)],
            batch_size=batch_size,
            device=device,
            dims=2048,
            num_workers=num_workers,
        )
    )


def stats(arrays: Dict[str, torch.Tensor]) -> Dict[str, float]:
    bit_acc = arrays["bit_acc"].float()
    detected = arrays["detected"].float()
    return {
        "bit_acc_mean": float(bit_acc.mean().item()),
        "bit_acc_std": float(bit_acc.std(unbiased=False).item()),
        "tpr": float(detected.mean().item()),
        "owner_margin_mean": float(arrays["owner_margin"].mean().item()),
        "matched_probability_mean": float(
            arrays["matched_probability"].mean().item()
        ),
        "num_images": int(bit_acc.numel()),
    }


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    args = parse_args()
    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    device = torch.device(args.device)
    dtype = dtype_from_name(args.dtype)
    if device.type == "cpu":
        dtype = torch.float32

    pairs = collect_pairs(args.watermarked_dir, args.attacked_dir, args.max_images)
    names = [item[0] for item in pairs]
    watermarked_paths = [item[1] for item in pairs]
    attacked_paths = [item[2] for item in pairs]

    automatic_threshold, theoretical_fpr = threshold_for_fpr(
        args.bit_dim,
        args.target_fpr,
    )
    threshold_bits = (
        args.threshold_bits
        if args.threshold_bits is not None
        else automatic_threshold
    )
    if not (0 <= threshold_bits <= args.bit_dim):
        raise ValueError("threshold_bits must lie between 0 and bit_dim")
    threshold_theoretical_fpr = binomial_tail(args.bit_dim, threshold_bits)

    secret, extractor = load_owner(
        args.pretrainedWM_dir,
        args.bit_dim,
        device,
        dtype,
    )
    vae = AutoencoderKL.from_pretrained(
        str(args.base_model),
        subfolder="vae",
        torch_dtype=dtype,
        local_files_only=not args.allow_download,
    ).to(device)
    vae.eval()
    vae.requires_grad_(False)
    latent_channels = int(getattr(vae.config, "latent_channels", -1))
    if latent_channels != 16:
        raise ValueError(
            f"Expected the SD3 16-channel VAE, but latent_channels={latent_channels}. "
            "Do not use the SD1.5 MarkNull proxy model for owner verification."
        )
    if args.enable_vae_tiling:
        vae.enable_tiling()

    watermarked_logits = decode_paths(
        watermarked_paths,
        vae,
        extractor,
        device,
        dtype,
        args.latent_resolution,
        args.batch_size,
        "Decode watermarked",
    )
    attacked_logits = decode_paths(
        attacked_paths,
        vae,
        extractor,
        device,
        dtype,
        args.latent_resolution,
        args.batch_size,
        "Decode MarkNull",
    )
    watermarked_arrays = detection_arrays(watermarked_logits, secret, threshold_bits)
    attacked_arrays = detection_arrays(attacked_logits, secret, threshold_bits)

    quality_rows, quality_summary = paired_quality(
        pairs,
        device,
        args.skip_lpips,
    )

    clean_summary = None
    if args.clean_dir is not None:
        clean_map = image_map(args.clean_dir)
        clean_paths = [clean_map[name] for name in names if name in clean_map]
        if not clean_paths:
            raise RuntimeError("No clean images match the evaluated filenames")
        clean_logits = decode_paths(
            clean_paths,
            vae,
            extractor,
            device,
            dtype,
            args.latent_resolution,
            args.batch_size,
            "Decode clean",
        )
        clean_arrays = detection_arrays(clean_logits, secret, threshold_bits)
        clean_summary = {
            "owner_match_mean": float(clean_arrays["bit_acc"].mean().item()),
            "empirical_fpr": float(clean_arrays["detected"].float().mean().item()),
            "num_images": len(clean_paths),
        }

    fid_value = None
    if not args.skip_fid:
        fid_value = compute_fid(
            args.watermarked_dir,
            args.attacked_dir,
            args.fid_batch_size,
            args.device,
            args.fid_num_workers,
        )

    per_image: List[Dict[str, object]] = []
    for index, name in enumerate(names):
        row: Dict[str, object] = {
            "file_name": name,
            "wm_matching_bits": int(watermarked_arrays["matching"][index].item()),
            "wm_bit_acc": float(watermarked_arrays["bit_acc"][index].item()),
            "wm_detected": int(watermarked_arrays["detected"][index].item()),
            "marknull_matching_bits": int(attacked_arrays["matching"][index].item()),
            "marknull_bit_acc": float(attacked_arrays["bit_acc"][index].item()),
            "marknull_detected": int(attacked_arrays["detected"][index].item()),
            "marknull_owner_margin": float(
                attacked_arrays["owner_margin"][index].item()
            ),
            **quality_rows[index],
        }
        per_image.append(row)

    summary = {
        "protocol": {
            "watermarked_dir": str(args.watermarked_dir.resolve()),
            "attacked_dir": str(args.attacked_dir.resolve()),
            "base_model": str(args.base_model.resolve()),
            "pretrainedWM_dir": str(args.pretrainedWM_dir.resolve()),
            "bit_dim": args.bit_dim,
            "target_fpr": args.target_fpr,
            "threshold_correct_bits": threshold_bits,
            "threshold_bit_accuracy": threshold_bits / args.bit_dim,
            "theoretical_binomial_fpr": threshold_theoretical_fpr,
            "automatic_threshold_correct_bits": automatic_threshold,
            "automatic_threshold_binomial_fpr": theoretical_fpr,
            "latent_resolution": args.latent_resolution,
        },
        "watermarked": stats(watermarked_arrays),
        "marknull": stats(attacked_arrays),
        "quality_watermarked_vs_marknull": quality_summary,
        "fid_watermarked_vs_marknull": fid_value,
        "clean_negative": clean_summary,
    }

    write_csv(args.output_dir / "per_image_metrics.csv", per_image)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\n================ MarkNull evaluation ================")
    print(f"Images                  : {len(names)}")
    print(f"Threshold               : >= {threshold_bits}/{args.bit_dim} bits")
    print(
        f"Before BitAcc / TPR     : {summary['watermarked']['bit_acc_mean']:.6f} / "
        f"{summary['watermarked']['tpr']:.6f}"
    )
    print(
        f"MarkNull BitAcc / TPR   : {summary['marknull']['bit_acc_mean']:.6f} / "
        f"{summary['marknull']['tpr']:.6f}"
    )
    print(
        f"PSNR / SSIM             : {quality_summary['psnr_mean']:.4f} / "
        f"{quality_summary['ssim_mean']:.6f}"
    )
    if "lpips_mean" in quality_summary:
        print(f"LPIPS                   : {quality_summary['lpips_mean']:.6f}")
    if fid_value is not None:
        print(f"FID                      : {fid_value:.6f}")
    if clean_summary is not None:
        print(f"Empirical clean FPR      : {clean_summary['empirical_fpr']:.6f}")
    print(f"Saved                   : {args.output_dir.resolve()}")
    print("======================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
