#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TG-MoE / SD3 black-box model ownership verification.

The suspicious model is never loaded. The verifier only consumes RGB images,
re-encodes them with a local SD3 VAE, extracts 48-bit logits, and aggregates
multiple queries into a model-level ownership score.

Two subcommands:
  benchmark  Calibrate thresholds and evaluate watermarked vs unwatermarked groups.
  verify     Apply a frozen threshold to one suspicious model/API output folder.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from diffusers import AutoencoderKL

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}


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


def parse_named_path(value: str) -> Tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Expected NAME=/path/to/images")
    name, path = value.split("=", 1)
    name, path = name.strip(), Path(path.strip()).expanduser()
    if not name:
        raise argparse.ArgumentTypeError("Group name cannot be empty")
    return name, path


def parse_budgets(value: str) -> List[int]:
    values = sorted({int(x.strip()) for x in value.split(",") if x.strip()})
    if not values or any(x <= 0 for x in values):
        raise argparse.ArgumentTypeError("Query budgets must be positive integers")
    return values


# -----------------------------------------------------------------------------
# Exact Stage-1 extractor architecture currently used in this project
# -----------------------------------------------------------------------------
class Conv2D(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size=3, activation="relu", strides=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, strides, int((kernel_size - 1) / 2)
        )
        self.act = (
            nn.ReLU(inplace=True) if activation == "relu"
            else nn.SELU(inplace=True) if activation == "selu"
            else None
        )

    def forward(self, x):
        x = self.conv(x)
        return x if self.act is None else self.act(x)


class Linear(nn.Module):
    def __init__(self, in_features, out_features, activation="relu"):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.act = (
            nn.ReLU(inplace=True) if activation == "relu"
            else nn.SELU(inplace=True) if activation == "selu"
            else None
        )

    def forward(self, x):
        x = self.linear(x)
        return x if self.act is None else self.act(x)


class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)


class ExtractorForLatent(nn.Module):
    def __init__(self, bit_dim=48):
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
            Linear(2048, bit_dim, activation=None),
        )

    def forward(self, latent):
        return self.mlps(self.decoder(latent))


def extract_tensor(obj, keys: Sequence[str]) -> torch.Tensor:
    if torch.is_tensor(obj):
        return obj
    if isinstance(obj, dict):
        for key in keys:
            if torch.is_tensor(obj.get(key)):
                return obj[key]
        tensors = [v for v in obj.values() if torch.is_tensor(v)]
        if len(tensors) == 1:
            return tensors[0]
    raise ValueError(f"Could not find tensor; tried keys={keys}")


def extract_state_dict(obj, keys: Sequence[str]) -> Dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise ValueError("Decoder checkpoint is not a dictionary/state_dict")
    for key in keys:
        if isinstance(obj.get(key), dict):
            obj = obj[key]
            break
    state = {}
    for key, value in obj.items():
        if not torch.is_tensor(value):
            continue
        while key.startswith("module."):
            key = key[len("module."):]
        state[key] = value
    if not state:
        raise ValueError("No parameters found in decoder checkpoint")
    return state


def load_secret(path: Path, bit_dim: int) -> np.ndarray:
    obj = torch.load(path, map_location="cpu")
    bits = extract_tensor(
        obj, ("secret", "bits", "GT_secret", "watermark_secret")
    ).detach().float().flatten()
    if bits.numel() != bit_dim:
        raise ValueError(f"Secret has {bits.numel()} values; expected {bit_dim}")
    bits = (bits > 0).float() if float(bits.min()) < 0 else (bits >= 0.5).float()
    return bits.numpy().astype(np.int8)


def load_decoder(path: Path, bit_dim: int, device: torch.device) -> nn.Module:
    decoder = ExtractorForLatent(bit_dim)
    obj = torch.load(path, map_location="cpu")
    decoder.load_state_dict(
        extract_state_dict(obj, ("decoder", "extractor", "state_dict")),
        strict=True,
    )
    decoder.requires_grad_(False)
    decoder.eval()
    return decoder.to(device=device, dtype=torch.float32)


# -----------------------------------------------------------------------------
# RGB -> local SD3 VAE -> Stage-1 latent -> 48-bit logits
# -----------------------------------------------------------------------------
def list_images(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(root)
    paths = sorted(
        p for p in root.rglob("*")
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not paths:
        raise RuntimeError(f"No images found under {root}")
    return paths


class ImageDataset(Dataset):
    def __init__(self, paths: Sequence[Path], resolution: int):
        self.paths = list(paths)
        self.transform = transforms.Compose([
            transforms.Resize(
                (resolution, resolution),
                interpolation=transforms.InterpolationMode.BICUBIC,
                antialias=True,
            ),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]
        with Image.open(path) as image:
            image = ImageOps.exif_transpose(image).convert("RGB")
            tensor = self.transform(image)
        return tensor, str(path)


def dtype_from_name(name: str) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }[name]


def load_vae(args, device: torch.device, dtype: torch.dtype) -> nn.Module:
    kwargs = {
        "torch_dtype": dtype,
        "local_files_only": args.local_files_only,
    }
    if args.vae_subfolder:
        kwargs["subfolder"] = args.vae_subfolder
    vae = AutoencoderKL.from_pretrained(args.vae_path, **kwargs)
    vae.requires_grad_(False)
    vae.eval().to(device)
    if args.enable_vae_tiling:
        vae.enable_tiling()
    return vae


@torch.inference_mode()
def extract_logits(
    group_name: str,
    paths: Sequence[Path],
    vae: nn.Module,
    decoder: nn.Module,
    device: torch.device,
    vae_dtype: torch.dtype,
    resolution: int,
    latent_resolution: int,
    batch_size: int,
    num_workers: int,
) -> Tuple[np.ndarray, List[str]]:
    loader = DataLoader(
        ImageDataset(paths, resolution),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    scaling = float(getattr(vae.config, "scaling_factor", 1.0))
    all_logits, all_paths = [], []
    print(f"[Extract] {group_name}: {len(paths)} images")
    for step, (images, batch_paths) in enumerate(loader, 1):
        images = images.to(device=device, dtype=vae_dtype, non_blocking=True)
        posterior = vae.encode(images).latent_dist
        # Matches current deterministic pixel re-encoding evaluation:
        # Stage-1 latent = posterior.mode() * scaling_factor, with no shift_factor.
        latent = posterior.mode().float() * scaling
        if latent.shape[-2:] != (latent_resolution, latent_resolution):
            latent = F.interpolate(
                latent,
                (latent_resolution, latent_resolution),
                mode="bilinear",
                align_corners=False,
            )
        logits = decoder(latent.float())
        all_logits.append(logits.cpu().numpy().astype(np.float32))
        all_paths.extend(batch_paths)
        if step % 10 == 0 or step == len(loader):
            print(f"  {min(step * batch_size, len(paths))}/{len(paths)}")
    return np.concatenate(all_logits, axis=0), list(all_paths)


# -----------------------------------------------------------------------------
# Model-level aggregation and statistics
# -----------------------------------------------------------------------------
def split_group(logits, paths, fraction, rng):
    order = rng.permutation(len(logits))
    n_cal = max(1, min(len(logits) - 1, int(round(len(logits) * fraction))))
    cal, test = order[:n_cal], order[n_cal:]
    return {
        "cal": logits[cal], "test": logits[test],
        "cal_paths": [paths[i] for i in cal],
        "test_paths": [paths[i] for i in test],
    }


def aggregate_trials(logits, q, trials, rng):
    if len(logits) < q:
        raise ValueError(f"Only {len(logits)} images available, but Q={q}")
    output = np.empty((trials, logits.shape[1]), dtype=np.float32)
    for i in range(trials):
        idx = rng.choice(len(logits), size=q, replace=False)
        output[i] = logits[idx].mean(axis=0)
    return output


def score_for_key(aggregated, bits):
    signs = bits.astype(np.float32) * 2.0 - 1.0
    scores = (aggregated * signs[None]).mean(axis=1)
    predictions = (aggregated >= 0).astype(np.int8)
    bitacc = (predictions == bits[None]).mean(axis=1)
    return scores.astype(np.float32), bitacc.astype(np.float32)


def score_for_random_keys(aggregated, owner_bits, rng):
    keys = rng.integers(0, 2, size=aggregated.shape, dtype=np.int8)
    same = np.all(keys == owner_bits[None], axis=1)
    while np.any(same):
        keys[same] = rng.integers(
            0, 2, size=(int(same.sum()), aggregated.shape[1]), dtype=np.int8
        )
        same = np.all(keys == owner_bits[None], axis=1)
    signs = keys.astype(np.float32) * 2.0 - 1.0
    scores = (aggregated * signs).mean(axis=1)
    bitacc = ((aggregated >= 0).astype(np.int8) == keys).mean(axis=1)
    return scores.astype(np.float32), bitacc.astype(np.float32)


def threshold_at_fpr(scores, target_fpr):
    try:
        return float(np.quantile(scores, 1.0 - target_fpr, method="higher"))
    except TypeError:
        return float(np.quantile(scores, 1.0 - target_fpr, interpolation="higher"))


def auc_score(y_true, scores):
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y_true, scores))
    except Exception:
        pos, neg = scores[y_true == 1], scores[y_true == 0]
        total = 0.0
        for value in pos:
            total += np.sum(value > neg) + 0.5 * np.sum(value == neg)
        return float(total / (len(pos) * len(neg)))


def save_per_image_csv(path, group_logits, group_paths, bits):
    signs = bits.astype(np.float32) * 2 - 1
    fields = ["group", "path", "owner_score", "bit_acc"] + [
        f"logit_{i:02d}" for i in range(len(bits))
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for group, logits in group_logits.items():
            scores = (logits * signs[None]).mean(axis=1)
            accs = ((logits >= 0).astype(np.int8) == bits[None]).mean(axis=1)
            for image_path, score, acc, row_logits in zip(
                group_paths[group], scores, accs, logits
            ):
                row = {
                    "group": group, "path": image_path,
                    "owner_score": float(score), "bit_acc": float(acc),
                }
                row.update({f"logit_{i:02d}": float(v) for i, v in enumerate(row_logits)})
                writer.writerow(row)


def plot_query_curve(rows, output):
    budgets = [r["query_budget"] for r in rows]
    plt.figure(figsize=(5.8, 4.0))
    plt.plot(budgets, [r["positive_tpr"] for r in rows], marker="o", label="TG-MoE owner key: TPR")
    plt.plot(budgets, [r["wrong_key_fpr"] for r in rows], marker="s", label="TG-MoE wrong key: FPR")
    for name in rows[0]["negative_fpr"]:
        plt.plot(budgets, [r["negative_fpr"][name] for r in rows], marker="^", label=f"{name}: FPR")
    plt.xlabel("Number of black-box queries")
    plt.ylabel("Verification rate")
    plt.ylim(-0.02, 1.02)
    plt.xticks(budgets)
    plt.grid(alpha=0.25)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


def plot_distribution(detail, output):
    plt.figure(figsize=(6.0, 4.1))
    plt.hist(detail["positive_scores"], bins=40, alpha=0.55, density=True, label="TG-MoE + owner key")
    for name, values in detail["negative_scores"].items():
        plt.hist(values, bins=40, alpha=0.40, density=True, label=f"{name} + owner key")
    plt.hist(detail["wrong_scores"], bins=40, alpha=0.40, density=True, label="TG-MoE + wrong key")
    plt.axvline(detail["threshold"], linestyle="--", linewidth=1.5, label=f"threshold={detail['threshold']:.3f}")
    plt.xlabel("Aggregated owner score")
    plt.ylabel("Density")
    plt.title(f"Black-box ownership scores (Q={detail['query_budget']})")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(output, dpi=300, bbox_inches="tight")
    plt.close()


def get_device_and_dtype(args):
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = dtype_from_name(args.vae_dtype)
    if device.type == "cpu" and dtype != torch.float32:
        print("[Warning] CPU selected; forcing fp32")
        dtype = torch.float32
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    return device, dtype


# -----------------------------------------------------------------------------
# benchmark subcommand
# -----------------------------------------------------------------------------
def run_benchmark(args):
    seed_everything(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    device, vae_dtype = get_device_and_dtype(args)

    positive_name, positive_dir = parse_named_path(args.positive)
    negatives = dict(parse_named_path(x) for x in args.negative)
    if positive_name in negatives:
        raise ValueError("Positive and negative group names must differ")
    group_dirs = {positive_name: positive_dir, **negatives}
    image_paths = {name: list_images(path) for name, path in group_dirs.items()}
    for name, paths in image_paths.items():
        print(f"[Group] {name}: {len(paths)} images")

    bits = load_secret(Path(args.secret_path), args.bit_dim)
    decoder = load_decoder(Path(args.decoder_path), args.bit_dim, device)
    cache_path = output / "extracted_logits.npz"
    group_logits, group_path_strings = {}, {}

    if args.reuse_cache and cache_path.exists():
        print(f"[Cache] loading {cache_path}")
        cache = np.load(cache_path, allow_pickle=True)
        for name in cache["groups"].tolist():
            group_logits[str(name)] = cache[f"logits::{name}"].astype(np.float32)
            group_path_strings[str(name)] = cache[f"paths::{name}"].tolist()
        missing = set(group_dirs) - set(group_logits)
        if missing:
            raise RuntimeError(f"Cache missing groups {sorted(missing)}; use --reuse_cache false")
    else:
        vae = load_vae(args, device, vae_dtype)
        for name, paths in image_paths.items():
            logits, strings = extract_logits(
                name, paths, vae, decoder, device, vae_dtype,
                args.resolution, args.latent_resolution,
                args.batch_size, args.num_workers,
            )
            group_logits[name], group_path_strings[name] = logits, strings
        payload = {"groups": np.array(list(group_logits), dtype=object)}
        for name in group_logits:
            payload[f"logits::{name}"] = group_logits[name]
            payload[f"paths::{name}"] = np.array(group_path_strings[name], dtype=object)
        np.savez_compressed(cache_path, **payload)
        print(f"[Cache] saved {cache_path}")
        del vae
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    save_per_image_csv(
        output / "per_image_scores_and_logits.csv",
        group_logits, group_path_strings, bits,
    )

    split_rng = np.random.default_rng(args.seed + 100)
    splits = {
        name: split_group(logits, group_path_strings[name], args.calibration_fraction, split_rng)
        for name, logits in group_logits.items()
    }
    max_q = max(args.query_budgets)
    for name, split in splits.items():
        print(f"[Split] {name}: cal={len(split['cal'])}, test={len(split['test'])}")
        if len(split["cal"]) < max_q or len(split["test"]) < max_q:
            raise RuntimeError(
                f"Group {name} needs at least Q={max_q} images in both split subsets. "
                "Generate more images, lower max Q, or adjust --calibration_fraction."
            )

    rows, details = [], {}
    for q in args.query_budgets:
        print(f"\n[Evaluate] Q={q}")
        rng = np.random.default_rng(args.seed + 1000 + q)

        cal_neg = []
        for name in negatives:
            agg = aggregate_trials(splits[name]["cal"], q, args.num_trials, rng)
            cal_neg.append(score_for_key(agg, bits)[0])
        pos_cal = aggregate_trials(splits[positive_name]["cal"], q, args.num_trials, rng)
        cal_neg.append(score_for_random_keys(pos_cal, bits, rng)[0])
        threshold = threshold_at_fpr(np.concatenate(cal_neg), args.target_fpr)

        pos_agg = aggregate_trials(splits[positive_name]["test"], q, args.num_trials, rng)
        pos_scores, pos_acc = score_for_key(pos_agg, bits)

        neg_scores, neg_acc = {}, {}
        for name in negatives:
            agg = aggregate_trials(splits[name]["test"], q, args.num_trials, rng)
            neg_scores[name], neg_acc[name] = score_for_key(agg, bits)

        wrong_agg = aggregate_trials(splits[positive_name]["test"], q, args.num_trials, rng)
        wrong_scores, wrong_acc = score_for_random_keys(wrong_agg, bits, rng)

        all_neg = np.concatenate([*neg_scores.values(), wrong_scores])
        y = np.concatenate([np.ones(len(pos_scores)), np.zeros(len(all_neg))]).astype(np.int64)
        auc = auc_score(y, np.concatenate([pos_scores, all_neg]))

        row = {
            "query_budget": q,
            "threshold": threshold,
            "target_fpr": args.target_fpr,
            "positive_tpr": float(np.mean(pos_scores >= threshold)),
            "negative_fpr": {name: float(np.mean(v >= threshold)) for name, v in neg_scores.items()},
            "wrong_key_fpr": float(np.mean(wrong_scores >= threshold)),
            "auroc_all_negatives": auc,
            "positive_mean_score": float(pos_scores.mean()),
            "positive_mean_bitacc": float(pos_acc.mean()),
            "negative_mean_score": {name: float(v.mean()) for name, v in neg_scores.items()},
            "negative_mean_bitacc": {name: float(v.mean()) for name, v in neg_acc.items()},
            "wrong_key_mean_score": float(wrong_scores.mean()),
            "wrong_key_mean_bitacc": float(wrong_acc.mean()),
        }
        rows.append(row)
        details[q] = {
            **row,
            "positive_scores": pos_scores,
            "negative_scores": neg_scores,
            "wrong_scores": wrong_scores,
        }
        print(f"  threshold={threshold:.6f} TPR={row['positive_tpr']:.4f} wrong-key FPR={row['wrong_key_fpr']:.4f} AUROC={auc:.4f}")
        for name, value in row["negative_fpr"].items():
            print(f"  {name} FPR={value:.4f}")

    negative_names = list(negatives)
    fields = [
        "query_budget", "threshold", "target_fpr", "positive_tpr",
        "wrong_key_fpr", "auroc_all_negatives", "positive_mean_score",
        "positive_mean_bitacc", "wrong_key_mean_score", "wrong_key_mean_bitacc",
    ]
    for name in negative_names:
        fields += [f"{name}_fpr", f"{name}_mean_score", f"{name}_mean_bitacc"]
    with (output / "blackbox_summary.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {k: row[k] for k in fields if k in row}
            for name in negative_names:
                flat[f"{name}_fpr"] = row["negative_fpr"][name]
                flat[f"{name}_mean_score"] = row["negative_mean_score"][name]
                flat[f"{name}_mean_bitacc"] = row["negative_mean_bitacc"][name]
            writer.writerow(flat)

    thresholds = {
        "owner_bits": bits.tolist(),
        "bit_dim": args.bit_dim,
        "target_fpr": args.target_fpr,
        "score_definition": "mean((2*owner_bit-1)*mean_Q_logits)",
        "aggregation": "average logits across Q images before scoring/decoding",
        "thresholds": {str(row["query_budget"]): row["threshold"] for row in rows},
        "resolution": args.resolution,
        "latent_resolution": args.latent_resolution,
        "positive_group": positive_name,
        "negative_groups": negative_names,
        "calibration_fraction": args.calibration_fraction,
        "seed": args.seed,
    }
    (output / "frozen_thresholds.json").write_text(
        json.dumps(thresholds, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "blackbox_summary.json").write_text(
        json.dumps({"configuration": vars(args), "results": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_query_curve(rows, output / "query_budget_curve.png")
    largest_q = max(args.query_budgets)
    plot_distribution(details[largest_q], output / f"score_distribution_Q{largest_q}.png")
    print(f"\n[Done] Results saved to {output}")


# -----------------------------------------------------------------------------
# verify subcommand
# -----------------------------------------------------------------------------
def run_verify(args):
    seed_everything(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frozen = json.loads(Path(args.thresholds).read_text(encoding="utf-8"))
    q_key = str(args.query_budget)
    if q_key not in frozen["thresholds"]:
        raise KeyError(f"No threshold for Q={args.query_budget}; available={sorted(frozen['thresholds'])}")
    threshold = float(frozen["thresholds"][q_key])
    bits = np.asarray(frozen["owner_bits"], dtype=np.int8)

    paths = list_images(Path(args.suspect_images))
    if len(paths) < args.query_budget:
        raise RuntimeError(f"Need {args.query_budget} images, found {len(paths)}")
    paths = paths[:args.query_budget]

    device, vae_dtype = get_device_and_dtype(args)
    decoder = load_decoder(Path(args.decoder_path), len(bits), device)
    vae = load_vae(args, device, vae_dtype)
    logits, used_paths = extract_logits(
        "suspect", paths, vae, decoder, device, vae_dtype,
        args.resolution, args.latent_resolution,
        args.batch_size, args.num_workers,
    )
    aggregated = logits.mean(axis=0, keepdims=True)
    scores, accs = score_for_key(aggregated, bits)
    score, bitacc = float(scores[0]), float(accs[0])
    verified = bool(score >= threshold)
    result = {
        "query_budget": args.query_budget,
        "owner_score": score,
        "threshold": threshold,
        "bit_accuracy": bitacc,
        "verified": verified,
        "decision": "OWNERSHIP VERIFIED" if verified else "OWNERSHIP NOT VERIFIED",
        "recovered_bits": (aggregated[0] >= 0).astype(np.int8).tolist(),
        "owner_bits": bits.tolist(),
        "images": used_paths,
    }
    (output / "verification_result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print("\n================ BLACK-BOX OWNERSHIP VERIFICATION ================")
    print(f"Queries:      {args.query_budget}")
    print(f"Owner score:  {score:.6f}")
    print(f"Threshold:    {threshold:.6f}")
    print(f"BitAcc:       {bitacc:.4f}")
    print(f"Decision:     {result['decision']}")
    print("==================================================================")


def add_common_extract_args(parser):
    parser.add_argument("--vae_path", required=True)
    parser.add_argument("--vae_subfolder", default="vae")
    parser.add_argument("--decoder_path", required=True)
    parser.add_argument("--resolution", type=int, default=1024)
    parser.add_argument("--latent_resolution", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--vae_dtype", choices=["fp32", "fp16", "bf16"], default="fp16")
    parser.add_argument("--device", default=None)
    parser.add_argument("--enable_vae_tiling", type=str2bool, default=False)
    parser.add_argument("--local_files_only", type=str2bool, default=True)
    parser.add_argument("--seed", type=int, default=2026)


def parse_args():
    parser = argparse.ArgumentParser(description="TG-MoE black-box ownership verification")
    sub = parser.add_subparsers(dest="command", required=True)

    bench = sub.add_parser("benchmark", help="Calibrate and evaluate ownership verification")
    add_common_extract_args(bench)
    bench.add_argument("--secret_path", required=True)
    bench.add_argument("--bit_dim", type=int, default=48)
    bench.add_argument("--positive", required=True, help="NAME=/path/to/watermarked_outputs")
    bench.add_argument("--negative", action="append", required=True, help="NAME=/path/to/unwatermarked_outputs; repeatable")
    bench.add_argument("--query_budgets", type=parse_budgets, default=[1, 5, 10, 20])
    bench.add_argument("--num_trials", type=int, default=2000)
    bench.add_argument("--target_fpr", type=float, default=0.01)
    bench.add_argument("--calibration_fraction", type=float, default=0.40)
    bench.add_argument("--reuse_cache", type=str2bool, default=True)
    bench.add_argument("--output_dir", default="./blackbox_results")

    verify = sub.add_parser("verify", help="Verify one suspicious model/API output folder")
    add_common_extract_args(verify)
    verify.add_argument("--thresholds", required=True)
    verify.add_argument("--suspect_images", required=True)
    verify.add_argument("--query_budget", type=int, required=True)
    verify.add_argument("--output_dir", default="./suspect_verification")

    args = parser.parse_args()
    if args.command == "benchmark":
        if not 0 < args.calibration_fraction < 1:
            parser.error("--calibration_fraction must be in (0,1)")
        if not 0 < args.target_fpr < 1:
            parser.error("--target_fpr must be in (0,1)")
        if args.num_trials <= 0:
            parser.error("--num_trials must be positive")
    return args


if __name__ == "__main__":
    arguments = parse_args()
    if arguments.command == "benchmark":
        run_benchmark(arguments)
    else:
        run_verify(arguments)
