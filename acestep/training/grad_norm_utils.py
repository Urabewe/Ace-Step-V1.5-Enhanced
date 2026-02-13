"""
Compute per-parameter gradient norms for one sample (e.g. during preprocessing).

Used to save gradient-norm info into preprocessed .pt files for rank/target
selection without logging during training.
"""

import re
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from acestep.training.configs import TrainingConfig
from acestep.training.trainer import (
    build_training_timesteps,
    _sample_flowmatching_t_r,
    sample_discrete_timestep,
)


def compute_decoder_grad_norms_for_sample(
    model: nn.Module,
    batch_tensors: Dict[str, torch.Tensor],
    device: torch.device,
    dtype: torch.dtype,
    training_config: Optional[TrainingConfig] = None,
) -> Dict[str, float]:
    """Run one forward+backward through the decoder and return per-parameter gradient norms.

    Args:
        model: DiT model with .decoder and .config.
        batch_tensors: Single-sample tensors (no batch dim): target_latents, attention_mask,
            encoder_hidden_states, encoder_attention_mask, context_latents. Will be unsqueezed(0) and moved to device.
        device: Device to run on.
        dtype: Dtype for computation.
        training_config: Optional; used for building timesteps. Defaults to TrainingConfig().

    Returns:
        Dict mapping parameter name (e.g. "decoder.layers.0.self_attn.q_proj.weight") to gradient L2 norm (float).
        Only includes params with finite gradients.
    """
    decoder = getattr(model, "decoder", None)
    if decoder is None:
        return {}

    config = getattr(model, "config", None)
    is_turbo = bool(getattr(config, "is_turbo", False))
    tcfg = training_config or TrainingConfig()

    # Single sample: add batch dim and move to device
    def to_batch(t: torch.Tensor) -> torch.Tensor:
        if t.dim() >= 2:
            return t.unsqueeze(0).to(device=device, dtype=dtype)
        return t.unsqueeze(0).to(device=device, dtype=dtype)

    target_latents = to_batch(batch_tensors["target_latents"])
    attention_mask = to_batch(batch_tensors["attention_mask"])
    encoder_hidden_states = to_batch(batch_tensors["encoder_hidden_states"])
    encoder_attention_mask = to_batch(batch_tensors["encoder_attention_mask"])
    context_latents = to_batch(batch_tensors["context_latents"])

    timesteps = build_training_timesteps(model, tcfg, device, dtype)

    decoder.train()
    with torch.enable_grad():
        x1 = torch.randn_like(target_latents, device=device, dtype=dtype)
        x0 = target_latents
        bsz = 1

        if is_turbo:
            t, r = sample_discrete_timestep(bsz, timesteps)
        else:
            t, r = _sample_flowmatching_t_r(
                bsz=bsz,
                device=device,
                dtype=dtype,
                data_proportion=getattr(config, "data_proportion", 0.5),
                timestep_mu=getattr(config, "timestep_mu", -0.4),
                timestep_sigma=getattr(config, "timestep_sigma", 1.0),
                use_meanflow=False,
            )
        t_ = t.unsqueeze(-1).unsqueeze(-1)  # [B, 1, 1] so xt stays 3D [B, T, C] like trainer
        xt = t_ * x1 + (1.0 - t_) * x0

        decoder_outputs = decoder(
            hidden_states=xt,
            timestep=t,
            timestep_r=r,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
        )
        flow = x1 - x0
        pred = decoder_outputs[0] if isinstance(decoder_outputs, (tuple, list)) else decoder_outputs
        # Decoder may return 4D (e.g. [B, T, C, 1] or [B, 1, T, C]); flow is 3D [B, T, C]. Match shapes.
        if pred.dim() != flow.dim() or pred.shape != flow.shape:
            if pred.numel() == flow.numel():
                pred = pred.reshape(flow.shape)
            else:
                while pred.dim() > flow.dim() and pred.shape[-1] == 1:
                    pred = pred.squeeze(-1)
                if pred.dim() == flow.dim() and pred.shape != flow.shape and pred.numel() == flow.numel():
                    pred = pred.reshape(flow.shape)
        loss = F.mse_loss(pred, flow)
        loss = loss.float()
        loss.backward()

    result: Dict[str, float] = {}
    for name, p in decoder.named_parameters():
        if p.grad is not None and torch.isfinite(p.grad).all():
            result[f"decoder.{name}"] = p.grad.norm().item()
    decoder.zero_grad()
    return result


# Common decoder module suffixes used as LoRA targets (order for tie-break).
DEFAULT_TARGET_MODULE_TYPES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _module_type_from_param_name(param_name: str) -> Optional[str]:
    """Extract module type from param name, e.g. 'decoder.layers.0.self_attn.q_proj.weight' -> 'q_proj'."""
    name = param_name.replace("decoder.", "").rstrip(".weight").rstrip(".bias")
    parts = name.split(".")
    for part in reversed(parts):
        if part in DEFAULT_TARGET_MODULE_TYPES or re.match(r"^\w+_proj$", part):
            return part
    return parts[-1] if parts else None


def get_top_target_modules_by_grad_norm(
    param_mean_norms: Dict[str, float],
    top_k: int = 4,
    candidate_types: Optional[List[str]] = None,
) -> List[str]:
    """Compute mean grad norm per module type and return the top-k for LoRA target selection.

    Args:
        param_mean_norms: Dict param_name -> mean gradient norm (e.g. from aggregate_grad_norms).
        top_k: Number of module types to keep.
        candidate_types: Allowed module type names; if None, any type seen in params is allowed.

    Returns:
        List of top-k module type names (e.g. ["q_proj", "v_proj", "k_proj", "o_proj"]) for target_modules.
    """
    if not param_mean_norms:
        return list((candidate_types or DEFAULT_TARGET_MODULE_TYPES)[:top_k])
    type_to_norms: Dict[str, List[float]] = {}
    for param_name, norm in param_mean_norms.items():
        mt = _module_type_from_param_name(param_name)
        if mt:
            type_to_norms.setdefault(mt, []).append(norm)
    type_to_mean = {t: (sum(ns) / len(ns)) if ns else 0.0 for t, ns in type_to_norms.items()}
    allowed = set(candidate_types) if candidate_types else set(type_to_mean.keys())
    sorted_types = sorted(
        (t for t in type_to_mean if t in allowed),
        key=lambda t: type_to_mean[t],
        reverse=True,
    )
    result = sorted_types[:top_k]
    if len(result) < top_k and candidate_types:
        for t in candidate_types:
            if t not in result and len(result) < top_k:
                result.append(t)
    return result if result else list(DEFAULT_TARGET_MODULE_TYPES)[:top_k]
