import hashlib
import math
import re
from typing import Dict, Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


def _parse_block_index(layer_name: str) -> Optional[int]:
    m = re.search(r"transformer_blocks\.(\d+)", layer_name)
    return int(m.group(1)) if m else None


def _projection_name(layer_name: str) -> str:
    if layer_name.endswith("to_q"):
        return "to_q"
    if layer_name.endswith("to_k"):
        return "to_k"
    if layer_name.endswith("to_v"):
        return "to_v"
    if "to_out" in layer_name:
        return "to_out"
    return "other"


def _zero_like_model(model: nn.Module) -> torch.Tensor:
    return torch.zeros((), device=next(model.parameters()).device, dtype=torch.float32)


def get_moe_bal_loss(model: nn.Module) -> torch.Tensor:
    all_logits = []
    for module in model.modules():
        if isinstance(module, MoELoRALayer) and module.current_logits is not None:
            all_logits.append(module.current_logits)
    if not all_logits:
        return _zero_like_model(model)

    logits = torch.cat(
        [x.reshape(-1, x.shape[-1]).float() for x in all_logits], dim=0
    )
    num_experts = logits.shape[-1]
    probs = F.softmax(logits, dim=-1)
    importance = probs.mean(dim=0)
    top1 = logits.argmax(dim=-1)
    load = torch.bincount(top1, minlength=num_experts).float() / max(top1.numel(), 1)
    return (num_experts * torch.sum(importance * load) - 1.0).pow(2)


def get_temporal_path_losses(model: nn.Module) -> Dict[str, torch.Tensor]:
    names = [
        "path_align_loss",
        "path_margin_loss",
        "path_separation_loss",
        "path_target_mass",
        "path_wrong_target_mass",
        "path_entropy",
    ]
    collected = {name: [] for name in names}
    active_layers = 0
    for module in model.modules():
        if not isinstance(module, MoELoRALayer):
            continue
        if module.path_active_tokens <= 0:
            continue
        active_layers += 1
        for name in names:
            value = getattr(module, "current_" + name, None)
            if value is not None:
                collected[name].append(value.float())

    output: Dict[str, torch.Tensor] = {}
    for name, values in collected.items():
        output[name] = torch.stack(values).mean() if values else _zero_like_model(model)
    output["active_layers"] = torch.tensor(
        float(active_layers), device=next(model.parameters()).device
    )
    return output


def inject_moe_lora_to_sd3(transformer, num_experts: int = 4, rank: int = 32):
    """State-dict compatible replacement for the user's original injection."""
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
            print(f"Inject Temporal-WidePath MoE-LoRA to: {name}")
        elif "to_out" in name and isinstance(module, nn.ModuleList):
            layer_name = f"{name}.0"
            module[0] = MoELoRALayer(
                module[0],
                layer_name=layer_name,
                num_experts=num_experts,
                rank=rank,
            )
            print(f"Inject Temporal-WidePath MoE-LoRA to: {layer_name}")
    return transformer


def configure_temporal_widepath(
    model: nn.Module,
    band_edges: Sequence[float] = (0.0, 0.2, 0.4),
    target_k: int = 2,
    path_blocks: Sequence[int] = (20, 21, 22, 23),
    path_projections: Sequence[str] = ("to_q", "to_k", "to_v", "to_out"),
    target_eps: float = 1e-4,
    kl_weight: float = 0.2,
    route_margin: float = 0.10,
    separation_margin: float = 0.10,
    routing_temperature: float = 1.0,
) -> None:
    edges = tuple(float(x) for x in band_edges)
    if len(edges) < 2 or any(b <= a for a, b in zip(edges, edges[1:])):
        raise ValueError(f"Invalid temporal band edges: {edges}")
    blocks = {int(x) for x in path_blocks}
    projections = {str(x) for x in path_projections}

    enabled = 0
    for module in model.modules():
        if not isinstance(module, MoELoRALayer):
            continue
        block = _parse_block_index(module.layer_name)
        projection = _projection_name(module.layer_name)
        module.path_enabled = block in blocks and projection in projections
        module.path_band_edges = edges
        module.path_target_k = min(max(1, int(target_k)), module.num_experts - 1)
        module.path_target_eps = float(target_eps)
        module.path_kl_weight = float(kl_weight)
        module.path_route_margin = float(route_margin)
        module.path_separation_margin = float(separation_margin)
        module.routing_temperature = float(routing_temperature)
        enabled += int(module.path_enabled)
    print(
        f"[TemporalWidePath] enabled_layers={enabled}, bands={edges}, "
        f"target_k={target_k}, blocks={sorted(blocks)}, projections={sorted(projections)}"
    )


def set_moe_context(
    model: nn.Module,
    secret_bits: Optional[torch.Tensor] = None,
    image_context=None,
    timestep: Optional[torch.Tensor] = None,
    wrong_secret_bits: Optional[torch.Tensor] = None,
) -> None:
    del image_context
    for module in model.modules():
        if isinstance(module, MoELoRALayer):
            module.current_secret_bits = secret_bits
            module.current_timestep = timestep
            module.current_wrong_secret_bits = wrong_secret_bits


def enable_probe_attack_lora(
    model: nn.Module,
    rank: int = 8,
    alpha: float = 8.0,
    blocks: Sequence[int] = (20, 21, 22, 23),
    projections: Sequence[str] = ("to_q", "to_k", "to_v", "to_out"),
) -> List[nn.Parameter]:
    block_set = {int(x) for x in blocks}
    projection_set = {str(x) for x in projections}
    params: List[nn.Parameter] = []
    count = 0
    for module in model.modules():
        if not isinstance(module, MoELoRALayer):
            continue
        block = _parse_block_index(module.layer_name)
        projection = _projection_name(module.layer_name)
        if block in block_set and projection in projection_set:
            module.enable_probe_attack(rank=rank, alpha=alpha)
            params.extend(module.probe_attack_parameters())
            count += 1
    if not params:
        raise RuntimeError("No modules selected for the in-training LoRA fine-tuning probe.")
    print(f"[FT-Probe] attack LoRA enabled on {count} projections; tensors={len(params)}")
    return params


def disable_probe_attack_lora(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, MoELoRALayer):
            module.disable_probe_attack()


class MoELoRALayer(nn.Module):
    """
    Original parameter names are preserved exactly:
      base_layer, lora_A, lora_B, lora_bias, router, bit_router.
    Therefore an existing TG-MoE checkpoint can be loaded into this module.
    """

    def __init__(
        self,
        base_layer: nn.Module,
        layer_name: str,
        num_experts: int = 4,
        rank: int = 32,
        network_alpha=None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.base_layer = base_layer
        self.layer_name = str(layer_name)
        self.num_experts = int(num_experts)
        self.rank = int(rank)

        in_features = base_layer.in_features
        out_features = base_layer.out_features
        self.lora_A = nn.Parameter(torch.zeros(num_experts, in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(num_experts, rank, out_features))
        self.lora_bias = nn.Parameter(torch.zeros(num_experts, out_features))
        self.router = nn.Linear(in_features, num_experts)
        self.bit_router = nn.Linear(48, num_experts)

        self.current_secret_bits = None
        self.current_wrong_secret_bits = None
        self.current_timestep = None
        self.current_logits = None
        self.current_routing_probs = None

        self.scaling = network_alpha / rank if network_alpha else 1.0
        self.dropout = nn.Dropout(dropout)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        nn.init.zeros_(self.lora_bias)

        self.path_enabled = False
        self.path_band_edges = (0.0, 0.2, 0.4)
        self.path_target_k = min(2, max(1, num_experts - 1))
        self.path_target_eps = 1e-4
        self.path_kl_weight = 0.2
        self.path_route_margin = 0.10
        self.path_separation_margin = 0.10
        self.routing_temperature = 1.0
        self._reset_path_state()

        # These are temporary and are deliberately not created in __init__, so
        # old checkpoints have no missing keys. They are added only during the
        # shadow fine-tuning probe, then deleted before checkpoint saving.
        self.probe_attack_rank = 0
        self.probe_attack_alpha = 0.0

    def _reset_path_state(self) -> None:
        self.path_active_tokens = 0
        self.current_path_align_loss = None
        self.current_path_margin_loss = None
        self.current_path_separation_loss = None
        self.current_path_target_mass = None
        self.current_path_wrong_target_mass = None
        self.current_path_entropy = None

    def enable_probe_attack(self, rank: int, alpha: float) -> None:
        self.disable_probe_attack()
        device = self.base_layer.weight.device
        self.probe_attack_rank = int(rank)
        self.probe_attack_alpha = float(alpha)
        in_features = self.base_layer.in_features
        out_features = self.base_layer.out_features
        self.probe_attack_A = nn.Parameter(
            torch.empty(in_features, rank, device=device, dtype=torch.float32)
        )
        self.probe_attack_B = nn.Parameter(
            torch.zeros(rank, out_features, device=device, dtype=torch.float32)
        )
        nn.init.normal_(self.probe_attack_A, std=0.01)

    def disable_probe_attack(self) -> None:
        for name in ("probe_attack_A", "probe_attack_B"):
            if hasattr(self, name):
                delattr(self, name)
        self.probe_attack_rank = 0
        self.probe_attack_alpha = 0.0

    def probe_attack_parameters(self) -> List[nn.Parameter]:
        if not hasattr(self, "probe_attack_A"):
            return []
        return [self.probe_attack_A, self.probe_attack_B]

    def _probe_attack_delta(self, x: torch.Tensor) -> torch.Tensor:
        if not hasattr(self, "probe_attack_A"):
            return torch.zeros(
                *x.shape[:-1], self.base_layer.out_features,
                device=x.device, dtype=x.dtype
            )
        scale = self.probe_attack_alpha / max(self.probe_attack_rank, 1)
        return (
            (x @ self.probe_attack_A.to(x.dtype))
            @ self.probe_attack_B.to(x.dtype)
        ) * scale

    def _secret_string(self, bits: torch.Tensor) -> str:
        values = (bits.detach().float().reshape(-1) >= 0.5).to(torch.int8).cpu().tolist()
        return "".join(str(int(v)) for v in values)

    def _target_indices(self, secret: torch.Tensor, band_index: int) -> List[int]:
        secret_string = self._secret_string(secret)
        scores = []
        for expert in range(self.num_experts):
            payload = (
                f"{secret_string}|{self.layer_name}|band={band_index}|expert={expert}"
            ).encode("utf-8")
            score = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
            scores.append((score, expert))
        scores.sort()
        return sorted(expert for _, expert in scores[: self.path_target_k])

    def _build_target_distribution(
        self,
        secret_bits: torch.Tensor,
        timesteps: torch.Tensor,
        seq_len: int,
        device,
        dtype,
    ):
        batch = secret_bits.shape[0]
        target = torch.full(
            (batch, self.num_experts),
            self.path_target_eps,
            device=device,
            dtype=dtype,
        )
        target_mask = torch.zeros_like(target)
        active = torch.zeros(batch, device=device, dtype=torch.bool)
        edges = self.path_band_edges
        for b in range(batch):
            t = float(timesteps[b].detach().float().item())
            band_index = None
            for index, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
                if left <= t < right:
                    band_index = index
                    break
            if band_index is None:
                continue
            active[b] = True
            indices = self._target_indices(secret_bits[b], band_index)
            target[b, indices] = 1.0 / len(indices)
            target_mask[b, indices] = 1.0
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return (
            target[:, None, :].expand(batch, seq_len, self.num_experts),
            target_mask[:, None, :].expand(batch, seq_len, self.num_experts),
            active[:, None].expand(batch, seq_len),
        )

    def _compute_path_losses(
        self,
        feature_logits: torch.Tensor,
        owner_probs: torch.Tensor,
        secret_bits: torch.Tensor,
    ) -> None:
        self._reset_path_state()
        if not self.path_enabled or self.current_timestep is None:
            return

        t = self.current_timestep
        if not torch.is_tensor(t):
            t = torch.tensor(t, device=owner_probs.device)
        t = t.to(device=owner_probs.device, dtype=torch.float32).reshape(-1)
        if t.numel() == 1 and owner_probs.shape[0] > 1:
            t = t.repeat(owner_probs.shape[0])

        target, target_mask, active = self._build_target_distribution(
            secret_bits,
            t,
            owner_probs.shape[1],
            owner_probs.device,
            owner_probs.dtype,
        )
        if not bool(active.any()):
            return

        active_f = active.to(owner_probs.dtype)
        denom = active_f.sum().clamp_min(1.0)
        mse_token = (owner_probs - target).pow(2).mean(dim=-1)
        kl_token = (
            target.clamp_min(1e-8)
            * (target.clamp_min(1e-8).log() - owner_probs.clamp_min(1e-8).log())
        ).sum(dim=-1)
        align = ((mse_token + self.path_kl_weight * kl_token) * active_f).sum() / denom

        target_values = owner_probs.masked_fill(target_mask < 0.5, 1.0)
        non_target_values = owner_probs.masked_fill(target_mask > 0.5, -1.0)
        min_target = target_values.min(dim=-1).values
        max_non_target = non_target_values.max(dim=-1).values
        margin_token = F.relu(self.path_route_margin + max_non_target - min_target)
        margin = (margin_token * active_f).sum() / denom

        owner_mass_token = (owner_probs * target_mask).sum(dim=-1)
        owner_mass = (owner_mass_token * active_f).sum() / denom
        wrong_mass = owner_mass.new_zeros(())
        separation = owner_mass.new_zeros(())

        if self.current_wrong_secret_bits is not None:
            wrong_secret = self.current_wrong_secret_bits.to(
                device=feature_logits.device, dtype=feature_logits.dtype
            )
            wrong_logits = feature_logits + self.bit_router(wrong_secret).unsqueeze(1)
            wrong_probs = F.softmax(
                wrong_logits / max(self.routing_temperature, 1e-6), dim=-1
            )
            wrong_mass_token = (wrong_probs * target_mask).sum(dim=-1)
            wrong_mass = (wrong_mass_token * active_f).sum() / denom
            separation = F.relu(
                self.path_separation_margin + wrong_mass - owner_mass
            )

        entropy_token = -(
            owner_probs.clamp_min(1e-8) * owner_probs.clamp_min(1e-8).log()
        ).sum(dim=-1) / math.log(self.num_experts)
        entropy = (entropy_token * active_f).sum() / denom

        self.path_active_tokens = int(active.sum().detach().item())
        self.current_path_align_loss = align
        self.current_path_margin_loss = margin
        self.current_path_separation_loss = separation
        self.current_path_target_mass = owner_mass
        self.current_path_wrong_target_mass = wrong_mass
        self.current_path_entropy = entropy

    def forward(self, x, *args, **kwargs):
        base_output = self.base_layer(x, *args, **kwargs)
        base_output = base_output + self._probe_attack_delta(x).to(base_output.dtype)
        self._reset_path_state()
        self.current_logits = None
        self.current_routing_probs = None

        if self.current_secret_bits is None:
            return base_output

        secret_bits = self.current_secret_bits.to(device=x.device, dtype=x.dtype)
        feature_logits = self.router(x)
        bit_logits = self.bit_router(secret_bits).unsqueeze(1)
        logits = feature_logits + bit_logits
        self.current_logits = logits
        routing_probs = F.softmax(
            logits / max(self.routing_temperature, 1e-6), dim=-1
        )
        self.current_routing_probs = routing_probs
        self._compute_path_losses(feature_logits, routing_probs, secret_bits)

        batch_size, seq_len, _ = x.shape
        lora_delta = torch.zeros(
            batch_size,
            seq_len,
            self.base_layer.out_features,
            device=x.device,
            dtype=x.dtype,
        )
        for i in range(self.num_experts):
            temp = x @ self.lora_A[i].to(x.dtype)
            out = temp @ self.lora_B[i].to(x.dtype) + self.lora_bias[i].to(x.dtype)
            lora_delta = lora_delta + routing_probs[..., i : i + 1] * out
        return base_output + self.dropout(lora_delta).to(x.dtype) * self.scaling
