"""
LoKR Utilities for ACE-Step (via LyCORIS)

Provides utilities for injecting LoKR (Low-Rank Kronecker Product) adapters
into the DiT decoder model using the lycoris-lora library.
"""

import os
from typing import Optional, Dict, Any, Tuple
from loguru import logger

import torch

try:
    from lycoris import create_lycoris, LycorisNetwork
    LYCORIS_AVAILABLE = True
except ImportError:
    LYCORIS_AVAILABLE = False
    logger.warning(
        "LyCORIS library not installed. LoKR training will not be available. "
        "Install with: pip install lycoris-lora"
    )

from acestep.training.configs import LoKRConfig, LoRAConfig


def build_optimizer_state_by_name(
    optimizer: torch.optim.Optimizer,
    param_id_to_name: Dict[int, str],
) -> Dict[str, Dict[str, Any]]:
    """Build a dict of optimizer state keyed by param name for resume across config changes.
    State (e.g. exp_avg, exp_avg_sq) is copied to CPU for saving.
    """
    state_by_name: Dict[str, Dict[str, Any]] = {}
    for group in optimizer.param_groups:
        for p in group["params"]:
            name = param_id_to_name.get(id(p))
            if name is None:
                continue
            if p not in optimizer.state:
                continue
            raw = optimizer.state[p]
            state_by_name[name] = {}
            for k, v in raw.items():
                if isinstance(v, torch.Tensor):
                    state_by_name[name][k] = v.detach().cpu().clone()
                else:
                    state_by_name[name][k] = v
    return state_by_name


def load_optimizer_state_by_name(
    optimizer: torch.optim.Optimizer,
    state_by_name: Dict[str, Dict[str, Any]],
    param_id_to_name: Dict[int, str],
    device: torch.device,
) -> int:
    """Load optimizer state for params that match by name. Returns number of params restored."""
    restored = 0
    for group in optimizer.param_groups:
        for p in group["params"]:
            name = param_id_to_name.get(id(p))
            if name is None or name not in state_by_name:
                continue
            saved = state_by_name[name]
            target_device = getattr(p, "device", device)
            loaded = {}
            for k, v in saved.items():
                if isinstance(v, torch.Tensor):
                    loaded[k] = v.to(target_device)
                else:
                    loaded[k] = v
            optimizer.state[p] = loaded
            restored += 1
    return restored


def check_lycoris_available() -> bool:
    """Check if LyCORIS library is available."""
    return LYCORIS_AVAILABLE


def inject_lokr_into_dit(
    model,
    lokr_config: LoKRConfig,
    multiplier: float = 1.0,
) -> Tuple[Any, "LycorisNetwork", Dict[str, Any]]:
    """Inject LoKR adapters into the DiT decoder using LyCORIS.

    Args:
        model: The AceStepConditionGenerationModel
        lokr_config: LoKR configuration
        multiplier: LoKR output multiplier (default 1.0)

    Returns:
        Tuple of (model, lycoris_network, info_dict)
    """
    if not LYCORIS_AVAILABLE:
        raise ImportError(
            "LyCORIS library is required for LoKR training. "
            "Install with: pip install lycoris-lora"
        )

    decoder = model.decoder

    # Freeze all non-LoKR parameters BEFORE injection so newly created LoKR params
    # are not accidentally frozen.
    for _, param in model.named_parameters():
        param.requires_grad = False

    # Apply preset to filter target modules
    LycorisNetwork.apply_preset(
        {"target_name": lokr_config.target_modules}
    )

    # Create LyCORIS network with LoKR algorithm
    lycoris_net = create_lycoris(
        decoder,
        multiplier,
        linear_dim=lokr_config.linear_dim,
        linear_alpha=lokr_config.linear_alpha,
        algo="lokr",
        factor=lokr_config.factor,
        decompose_both=lokr_config.decompose_both,
        use_tucker=lokr_config.use_tucker,
        use_scalar=lokr_config.use_scalar,
        full_matrix=lokr_config.full_matrix,
        bypass_mode=lokr_config.bypass_mode,
        rs_lora=lokr_config.rs_lora,
        unbalanced_factorization=lokr_config.unbalanced_factorization,
    )

    if lokr_config.weight_decompose:
        # DoRA mode: set via kwargs if supported
        try:
            lycoris_net2 = create_lycoris(
                decoder,
                multiplier,
                linear_dim=lokr_config.linear_dim,
                linear_alpha=lokr_config.linear_alpha,
                algo="lokr",
                factor=lokr_config.factor,
                decompose_both=lokr_config.decompose_both,
                use_tucker=lokr_config.use_tucker,
                use_scalar=lokr_config.use_scalar,
                full_matrix=lokr_config.full_matrix,
                bypass_mode=lokr_config.bypass_mode,
                rs_lora=lokr_config.rs_lora,
                unbalanced_factorization=lokr_config.unbalanced_factorization,
                dora_wd=True,
            )
            lycoris_net = lycoris_net2
        except Exception as e:
            logger.warning(f"DoRA (weight_decompose) not supported in this LyCORIS version: {e}")

    # Apply the LoKR wrappers to the decoder
    lycoris_net.apply_to()

    # Register LyCORIS network on decoder
    if not hasattr(decoder, "_lycoris_net"):
        decoder._lycoris_net = lycoris_net

    # Enable gradients for LoKR parameters
    lokr_param_list = []
    for m in getattr(lycoris_net, "loras", []) or []:
        for p in m.parameters():
            p.requires_grad = True
            lokr_param_list.append(p)
    if not lokr_param_list:
        for p in lycoris_net.parameters():
            p.requires_grad = True
            lokr_param_list.append(p)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    uniq = {}
    for p in lokr_param_list:
        uniq[id(p)] = p
    lokr_params = sum(p.numel() for p in uniq.values())
    trainable_params = sum(p.numel() for p in uniq.values() if p.requires_grad)

    info = {
        "total_params": total_params,
        "lokr_params": lokr_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        "linear_dim": lokr_config.linear_dim,
        "linear_alpha": lokr_config.linear_alpha,
        "factor": lokr_config.factor,
        "algo": "lokr",
        "target_modules": lokr_config.target_modules,
    }

    logger.info("LoKR injected into DiT decoder:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  LoKR parameters: {lokr_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({info['trainable_ratio']:.2%})")
    logger.info(f"  linear_dim: {lokr_config.linear_dim}, linear_alpha: {lokr_config.linear_alpha}")
    logger.info(f"  factor: {lokr_config.factor}, decompose_both: {lokr_config.decompose_both}")

    return model, lycoris_net, info


def inject_lora_into_dit_lycoris(
    model,
    lora_config: LoRAConfig,
    multiplier: float = 1.0,
) -> Tuple[Any, "LycorisNetwork", Dict[str, Any]]:
    """Inject LoRA adapters into the DiT decoder using LyCORIS."""
    if not LYCORIS_AVAILABLE:
        raise ImportError(
            "LyCORIS library is required for LoRA training. "
            "Install with: pip install lycoris-lora"
        )

    decoder = model.decoder

    for _, param in model.named_parameters():
        param.requires_grad = False

    LycorisNetwork.apply_preset({"target_name": lora_config.target_modules})

    lycoris_net = create_lycoris(
        decoder,
        multiplier,
        linear_dim=lora_config.r,
        linear_alpha=lora_config.alpha,
        algo="lora",
    )

    lycoris_net.apply_to()

    if not hasattr(decoder, "_lycoris_net"):
        decoder._lycoris_net = lycoris_net

    lora_param_list = []
    for m in getattr(lycoris_net, "loras", []) or []:
        for p in m.parameters():
            p.requires_grad = True
            lora_param_list.append(p)
    if not lora_param_list:
        for p in lycoris_net.parameters():
            p.requires_grad = True
            lora_param_list.append(p)

    total_params = sum(p.numel() for p in model.parameters())
    uniq = {}
    for p in lora_param_list:
        uniq[id(p)] = p
    lora_params = sum(p.numel() for p in uniq.values())
    trainable_params = sum(p.numel() for p in uniq.values() if p.requires_grad)

    info = {
        "total_params": total_params,
        "lora_params": lora_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        "linear_dim": lora_config.r,
        "linear_alpha": lora_config.alpha,
        "algo": "lora",
        "target_modules": lora_config.target_modules,
    }

    logger.info("LoRA injected into DiT decoder (LyCORIS):")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  LoRA parameters: {lora_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({info['trainable_ratio']:.2%})")
    logger.info(f"  linear_dim: {lora_config.r}, linear_alpha: {lora_config.alpha}")

    return model, lycoris_net, info


def save_lokr_weights(
    lycoris_net: "LycorisNetwork",
    output_dir: str,
    dtype: Optional[torch.dtype] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """Save LoKR adapter weights."""
    os.makedirs(output_dir, exist_ok=True)

    weights_path = os.path.join(output_dir, "lokr_weights.safetensors")

    save_metadata = {"algo": "lokr", "format": "lycoris"}
    if metadata:
        save_metadata.update(metadata)

    lycoris_net.save_weights(weights_path, dtype=dtype, metadata=save_metadata)
    logger.info(f"LoKR weights saved to {weights_path}")

    return weights_path


def save_lora_weights_lycoris(
    lycoris_net: "LycorisNetwork",
    output_dir: str,
    dtype: Optional[torch.dtype] = None,
    metadata: Optional[Dict[str, str]] = None,
) -> str:
    """Save LoRA adapter weights (LyCORIS single-file)."""
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "lora_weights.safetensors")
    save_metadata = {"algo": "lora", "format": "lycoris"}
    if metadata:
        save_metadata.update(metadata)
    lycoris_net.save_weights(weights_path, dtype=dtype, metadata=save_metadata)
    logger.info(f"LoRA weights saved to {weights_path}")
    return weights_path


def load_lora_weights_lycoris(
    lycoris_net: "LycorisNetwork",
    weights_path: str,
) -> Dict[str, Any]:
    """Load LoRA adapter weights into an existing LyCORIS network."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"LoRA weights not found: {weights_path}")
    result = lycoris_net.load_weights(weights_path)
    logger.info(f"LoRA weights loaded from {weights_path}")
    return result


def save_lora_training_checkpoint_lycoris(
    lycoris_net: "LycorisNetwork",
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    output_dir: str,
    lora_config: Optional[LoRAConfig] = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a training checkpoint including LoRA weights and training state."""
    os.makedirs(output_dir, exist_ok=True)
    save_lora_weights_lycoris(lycoris_net, output_dir)

    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }
    if lora_config is not None:
        training_state["lora_config"] = {
            "r": lora_config.r,
            "alpha": lora_config.alpha,
            "dropout": lora_config.dropout,
            "target_modules": lora_config.target_modules,
            "bias": lora_config.bias,
        }
    if extra_state:
        training_state.update(extra_state)

    state_path = os.path.join(output_dir, "training_state.pt")
    torch.save(training_state, state_path)
    logger.info(f"LoRA training checkpoint saved to {output_dir} (epoch {epoch}, step {global_step})")
    return output_dir


def load_lora_training_checkpoint_lycoris(
    checkpoint_dir: str,
    lycoris_net: Optional["LycorisNetwork"] = None,
    optimizer=None,
    scheduler=None,
    device: torch.device = None,
    param_id_to_name: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Load LoRA training checkpoint. If param_id_to_name is provided and full optimizer load
    fails (e.g. different rank), loads optimizer state by param name from optimizer_state_by_name when present.
    """
    result = {
        "epoch": 0,
        "global_step": 0,
        "weights_path": None,
        "loaded_optimizer": False,
        "loaded_scheduler": False,
        "lora_config": None,
    }

    weights_path = os.path.join(checkpoint_dir, "lora_weights.safetensors")
    if os.path.exists(weights_path):
        result["weights_path"] = weights_path
        if lycoris_net is not None:
            load_lora_weights_lycoris(lycoris_net, weights_path)

    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if os.path.exists(state_path):
        map_location = device if device else "cpu"
        try:
            training_state = torch.load(state_path, map_location=map_location, weights_only=False)
        except TypeError:
            training_state = torch.load(state_path, map_location=map_location)
        result["epoch"] = training_state.get("epoch", 0)
        result["global_step"] = training_state.get("global_step", 0)
        result["lora_config"] = training_state.get("lora_config", None)
        optimizer_loaded = False
        if optimizer is not None and "optimizer_state_dict" in training_state:
            try:
                saved = training_state["optimizer_state_dict"]
                curr = optimizer.state_dict()
                if len(saved.get("param_groups", [])) != len(curr.get("param_groups", [])):
                    logger.warning(
                        "Parameter group count mismatch; trying to load optimizer state by param name from .pt file."
                    )
                else:
                    optimizer.load_state_dict(saved)
                    result["loaded_optimizer"] = True
                    optimizer_loaded = True
                    logger.info("Optimizer state loaded from LoRA checkpoint")
            except Exception as e:
                logger.warning(
                    "Full optimizer load failed: %s. Trying to load by param name from .pt file.",
                    e,
                )
        if not optimizer_loaded and optimizer is not None and param_id_to_name is not None and "optimizer_state_by_name" in training_state:
            try:
                dev = device if device is not None else torch.device("cpu")
                n = load_optimizer_state_by_name(
                    optimizer,
                    training_state["optimizer_state_by_name"],
                    param_id_to_name,
                    dev,
                )
                if n > 0:
                    result["loaded_optimizer"] = True
                    logger.info("Loaded optimizer state from .pt file for %d params (by name)", n)
            except Exception as e:
                logger.warning("Could not load optimizer state by name: %s", e)
        if scheduler is not None and "scheduler_state_dict" in training_state:
            try:
                scheduler.load_state_dict(training_state["scheduler_state_dict"])
                result["loaded_scheduler"] = True
                logger.info("Scheduler state loaded from LoRA checkpoint")
            except Exception as e:
                logger.warning(f"Failed to load scheduler state: {e}")

        logger.info(f"Loaded LoRA checkpoint from epoch {result['epoch']}, step {result['global_step']}")
    else:
        import re
        match = re.search(r'epoch_(\d+)', checkpoint_dir)
        if match:
            result["epoch"] = int(match.group(1))

    return result


def load_lokr_weights(
    lycoris_net: "LycorisNetwork",
    weights_path: str,
) -> Dict[str, Any]:
    """Load LoKR adapter weights into an existing LyCORIS network."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"LoKR weights not found: {weights_path}")

    result = lycoris_net.load_weights(weights_path)
    logger.info(f"LoKR weights loaded from {weights_path}")

    return result


def save_lokr_training_checkpoint(
    lycoris_net: "LycorisNetwork",
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    output_dir: str,
    lokr_config: Optional[LoKRConfig] = None,
    extra_state: Optional[Dict[str, Any]] = None,
) -> str:
    """Save a training checkpoint including LoKR weights and training state."""
    os.makedirs(output_dir, exist_ok=True)

    save_lokr_weights(lycoris_net, output_dir)

    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    if lokr_config is not None:
        training_state["lokr_config"] = lokr_config.to_dict()
    if extra_state:
        training_state.update(extra_state)

    state_path = os.path.join(output_dir, "training_state.pt")
    torch.save(training_state, state_path)

    logger.info(f"LoKR training checkpoint saved to {output_dir} (epoch {epoch}, step {global_step})")
    return output_dir


def load_lokr_training_checkpoint(
    checkpoint_dir: str,
    lycoris_net: Optional["LycorisNetwork"] = None,
    optimizer=None,
    scheduler=None,
    device: torch.device = None,
    param_id_to_name: Optional[Dict[int, str]] = None,
) -> Dict[str, Any]:
    """Load LoKR training checkpoint. If param_id_to_name is provided and full optimizer load
    fails, loads optimizer state by param name from optimizer_state_by_name when present.
    """
    result = {
        "epoch": 0,
        "global_step": 0,
        "weights_path": None,
        "loaded_optimizer": False,
        "loaded_scheduler": False,
        "lokr_config": None,
    }

    weights_path = os.path.join(checkpoint_dir, "lokr_weights.safetensors")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(checkpoint_dir, "lokr_weights.pt")
    if os.path.exists(weights_path):
        result["weights_path"] = weights_path
        if lycoris_net is not None:
            load_lokr_weights(lycoris_net, weights_path)

    state_path = os.path.join(checkpoint_dir, "training_state.pt")
    if os.path.exists(state_path):
        try:
            training_state = torch.load(state_path, map_location=device or "cpu", weights_only=False)
        except TypeError:
            training_state = torch.load(state_path, map_location=device or "cpu")

        result["epoch"] = training_state.get("epoch", 0)
        result["global_step"] = training_state.get("global_step", 0)
        result["lokr_config"] = training_state.get("lokr_config", None)

        optimizer_loaded = False
        if optimizer is not None and "optimizer_state_dict" in training_state:
            try:
                saved = training_state["optimizer_state_dict"]
                curr = optimizer.state_dict()
                if len(saved.get("param_groups", [])) != len(curr.get("param_groups", [])):
                    logger.warning(
                        "Parameter group count mismatch; trying to load optimizer state by param name from .pt file."
                    )
                else:
                    optimizer.load_state_dict(saved)
                    result["loaded_optimizer"] = True
                    optimizer_loaded = True
                    logger.info("Optimizer state loaded from LoKR checkpoint")
            except Exception as e:
                logger.warning(
                    "Full optimizer load failed: %s. Trying to load by param name from .pt file.",
                    e,
                )
        if not optimizer_loaded and optimizer is not None and param_id_to_name is not None and "optimizer_state_by_name" in training_state:
            try:
                dev = device if device is not None else torch.device("cpu")
                n = load_optimizer_state_by_name(
                    optimizer,
                    training_state["optimizer_state_by_name"],
                    param_id_to_name,
                    dev,
                )
                if n > 0:
                    result["loaded_optimizer"] = True
                    logger.info("Loaded optimizer state from .pt file for %d params (by name)", n)
            except Exception as e:
                logger.warning("Could not load optimizer state by name: %s", e)

        if scheduler is not None and "scheduler_state_dict" in training_state:
            try:
                scheduler.load_state_dict(training_state["scheduler_state_dict"])
                result["loaded_scheduler"] = True
                logger.info("Scheduler state loaded from LoKR checkpoint")
            except Exception as e:
                logger.warning(f"Failed to load scheduler state: {e}")

        logger.info(f"Loaded LoKR checkpoint from epoch {result['epoch']}, step {result['global_step']}")
    else:
        import re
        match = re.search(r'epoch_(\d+)', checkpoint_dir)
        if match:
            result["epoch"] = int(match.group(1))

    return result


def restore_lokr(lycoris_net: "LycorisNetwork") -> None:
    """Remove LoKR adapters and restore the original model weights."""
    if lycoris_net is not None:
        lycoris_net.restore()
        logger.info("LoKR adapters removed, original model restored")


def get_lokr_info(lycoris_net: "LycorisNetwork") -> Dict[str, Any]:
    """Get information about LoKR adapters."""
    info = {
        "has_lokr": False,
        "lokr_params": 0,
        "num_modules": 0,
    }

    if lycoris_net is None:
        return info

    lokr_params = sum(p.numel() for p in lycoris_net.parameters())
    num_modules = len(list(lycoris_net.loras))

    info["has_lokr"] = lokr_params > 0
    info["lokr_params"] = lokr_params
    info["num_modules"] = num_modules

    return info
