"""
LoRA Trainer for ACE-Step

Lightning Fabric-based trainer for LoRA fine-tuning of ACE-Step DiT decoder.
Supports training from preprocessed tensor files for optimal performance.
"""

import os
import time
from typing import Optional, List, Dict, Any, Tuple, Generator
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, LinearLR, SequentialLR

try:
    from lightning.fabric import Fabric
    from lightning.fabric.loggers import TensorBoardLogger
    LIGHTNING_AVAILABLE = True
except ImportError:
    LIGHTNING_AVAILABLE = False
    logger.warning("Lightning Fabric not installed. Training will use basic training loop.")

# Optional bitsandbytes optimizer
try:
    import bitsandbytes as bnb
    HAS_BNB = True
except ImportError:
    HAS_BNB = False
    logger.warning("bitsandbytes not installed. Using standard AdamW.")

from acestep.training.configs import LoRAConfig, LoKRConfig, TrainingConfig
from acestep.training.lokr_utils import (
    inject_lora_into_dit_lycoris,
    inject_lokr_into_dit,
    save_lora_weights_lycoris,
    save_lora_training_checkpoint_lycoris,
    load_lora_training_checkpoint_lycoris,
    save_lokr_weights,
    save_lokr_training_checkpoint,
    load_lokr_training_checkpoint,
    build_optimizer_state_by_name,
    check_lycoris_available,
)
from acestep.training.data_module import PreprocessedDataModule, PreprocessedTensorDataset


# Turbo model shift=3.0 discrete timesteps (8 steps, same as inference)
TURBO_SHIFT3_TIMESTEPS = [1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75, 0.6428571428571429, 0.5, 0.3]


def _apply_shift(timesteps: torch.Tensor, shift: float) -> torch.Tensor:
    """Apply timestep shift (t = shift * t / (1 + (shift - 1) * t))."""
    if shift == 1.0:
        return timesteps
    return shift * timesteps / (1.0 + (shift - 1.0) * timesteps)


def _extract_timesteps_from_config(config: Optional[Any]) -> Optional[List[float]]:
    if config is None:
        return None
    for key in ("timesteps", "timestep", "sampling_timesteps"):
        raw = getattr(config, key, None)
        if isinstance(raw, (list, tuple)) and raw:
            try:
                return [float(x) for x in raw]
            except Exception:
                return None
    return None


def _ensure_trainable_params_fp32(module: nn.Module) -> Tuple[int, int]:
    """Force trainable floating-point parameters to fp32."""
    casted = 0
    total = 0
    for p in module.parameters():
        if not p.requires_grad:
            continue
        total += 1
        if p.is_floating_point() and p.dtype != torch.float32:
            with torch.no_grad():
                p.data = p.data.float()
            casted += 1
    return casted, total


def _build_param_name_lookup(module: nn.Module, extra_module: Optional[nn.Module] = None) -> Dict[int, str]:
    """Build a best-effort id(param) -> name lookup for debug logging."""
    lookup: Dict[int, str] = {}
    for name, p in module.named_parameters():
        lookup[id(p)] = name
    if extra_module is not None:
        for name, p in extra_module.named_parameters():
            lookup.setdefault(id(p), f"lycoris_net.{name}")
    return lookup


def _count_nonfinite_grads_detailed(
    params: List[torch.nn.Parameter],
    param_name_lookup: Dict[int, str],
    detail_limit: int = 8,
) -> Tuple[int, int, List[str]]:
    """Count non-finite grads and return up to `detail_limit` offending tensor details."""
    nonfinite = 0
    total_with_grad = 0
    details: List[str] = []

    for p in params:
        g = p.grad
        if g is None:
            continue
        total_with_grad += 1
        if torch.isfinite(g).all():
            continue

        nonfinite += 1
        if len(details) >= detail_limit:
            continue

        pname = param_name_lookup.get(id(p), f"<unnamed:{id(p)}>")
        g32 = g.detach().float()
        nan_count = int(torch.isnan(g32).sum().item())
        inf_count = int(torch.isinf(g32).sum().item())
        finite_vals = g32[torch.isfinite(g32)]
        max_abs_finite = float(finite_vals.abs().max().item()) if finite_vals.numel() else float("nan")

        p32 = p.detach().float()
        param_nonfinite = int((~torch.isfinite(p32)).sum().item())
        details.append(
            f"{pname} | shape={tuple(p.shape)} grad_dtype={g.dtype} "
            f"nan={nan_count} inf={inf_count} max_abs_finite={max_abs_finite:.3e} "
            f"param_nonfinite={param_nonfinite}"
        )

    return nonfinite, total_with_grad, details


def _unwrap_stale_fabric_decoder(model: nn.Module) -> bool:
    """Unwrap stale Lightning Fabric wrappers from decoder left by previous runs."""
    if model is None or not hasattr(model, "decoder"):
        return False
    decoder = model.decoder
    unwrapped = False
    while hasattr(decoder, "_forward_module") and isinstance(getattr(decoder, "_forward_module"), nn.Module):
        decoder = decoder._forward_module
        unwrapped = True
    if unwrapped:
        model.decoder = decoder
    return unwrapped


def _select_optimizer(
    params: List[torch.nn.Parameter],
    lr: float,
    weight_decay: float,
    device_type: str,
) -> torch.optim.Optimizer:
    """Select AdamW or bitsandbytes AdamW8bit if available."""
    if HAS_BNB and device_type == "cuda":
        return bnb.optim.AdamW8bit(params, lr=lr, weight_decay=weight_decay)
    return AdamW(params, lr=lr, weight_decay=weight_decay)


def _sample_flowmatching_t_r(
    bsz: int,
    device: torch.device,
    dtype: torch.dtype,
    data_proportion: float,
    timestep_mu: float,
    timestep_sigma: float,
    use_meanflow: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample continuous timesteps for base flow-matching training."""
    t = torch.sigmoid(torch.randn((bsz,), device=device, dtype=dtype) * timestep_sigma + timestep_mu)
    r = torch.sigmoid(torch.randn((bsz,), device=device, dtype=dtype) * timestep_sigma + timestep_mu)
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    if not use_meanflow:
        data_proportion = 1.0
    data_size = int(bsz * data_proportion)
    zero_mask = torch.arange(bsz, device=device) < data_size
    r = torch.where(zero_mask, t, r)
    return t, r


def build_training_timesteps(
    model: nn.Module,
    training_config: TrainingConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build training timesteps based on the loaded model configuration."""
    config = getattr(model, "config", None)
    is_turbo = bool(getattr(config, "is_turbo", False))

    if is_turbo:
        return torch.tensor(TURBO_SHIFT3_TIMESTEPS, device=device, dtype=dtype)

    config_timesteps = _extract_timesteps_from_config(config)
    if config_timesteps:
        ts = torch.tensor(config_timesteps, device=device, dtype=dtype)
        return ts.clamp(0.0, 1.0)

    shift = getattr(config, "shift", training_config.shift)
    num_steps = getattr(config, "num_inference_steps", training_config.num_inference_steps)

    # If still on turbo defaults, switch to base-friendly defaults.
    if shift == 3.0:
        shift = 1.0
    if num_steps == 8:
        num_steps = 32

    timesteps = torch.linspace(1.0, 0.0, int(num_steps), device=device, dtype=dtype)
    return _apply_shift(timesteps, shift)


def sample_discrete_timestep(bsz: int, timesteps: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample timesteps from a precomputed discrete schedule."""
    indices = torch.randint(0, timesteps.numel(), (bsz,), device=timesteps.device)
    t = timesteps[indices]
    r = t
    return t, r


class PreprocessedLoRAModule(nn.Module):
    """LoRA Training Module using preprocessed tensors.
    
    This module trains only the DiT decoder with LoRA adapters.
    All inputs are pre-computed tensors - no VAE or text encoder needed!
    
    Training flow:
    1. Load pre-computed tensors (target_latents, encoder_hidden_states, context_latents)
    2. Sample noise and timestep
    3. Forward through decoder (with LoRA)
    4. Compute flow matching loss
    """
    
    def __init__(
        self,
        model: nn.Module,
        lora_config: LoRAConfig,
        training_config: TrainingConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        """Initialize the training module.
        
        Args:
            model: The AceStepConditionGenerationModel
            lora_config: LoRA configuration
            training_config: Training configuration
            device: Device to use
            dtype: Data type to use
        """
        super().__init__()
        
        self.lora_config = lora_config
        self.training_config = training_config
        self.device = device
        self.dtype = dtype
        self.lycoris_net = None
        
        # Inject LoRA into the decoder only
        if check_lycoris_available():
            self.model, self.lycoris_net, self.lora_info = inject_lora_into_dit_lycoris(model, lora_config)
            logger.info(f"LoRA injected: {self.lora_info.get('trainable_params', 0):,} trainable params")
        else:
            self.model = model
            self.lora_info = {}
            logger.warning("LyCORIS not available, cannot train LoRA adapters")
        
        # Model config for flow matching
        self.config = model.config
        self.training_timesteps = build_training_timesteps(
            model=self.model,
            training_config=self.training_config,
            device=self.device,
            dtype=self.dtype,
        )
        logger.info(
            f"Training schedule: steps={self.training_timesteps.numel()}, shift={getattr(self.config, 'shift', self.training_config.shift)}, "
            f"turbo={getattr(self.config, 'is_turbo', False)}"
        )
        
        # Store training losses
        self.training_losses = []
    
    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step using preprocessed tensors.
        
        Note: This uses the model's timestep schedule; turbo models are discrete.
        
        Args:
            batch: Dictionary containing pre-computed tensors:
                - target_latents: [B, T, 64] - VAE encoded audio
                - attention_mask: [B, T] - Valid audio mask
                - encoder_hidden_states: [B, L, D] - Condition encoder output
                - encoder_attention_mask: [B, L] - Condition mask
                - context_latents: [B, T, 128] - Source context
            
        Returns:
            Loss tensor (float32 for stable backward)
        """
        # Use autocast for mixed precision training (bf16 on CUDA, fp16 on MPS)
        _device_type = self.device if isinstance(self.device, str) else self.device.type
        _autocast_dtype = torch.float16 if _device_type == "mps" else torch.bfloat16
        with torch.autocast(device_type=_device_type, dtype=_autocast_dtype):
            # Get tensors from batch (already on device from Fabric dataloader)
            target_latents = batch["target_latents"].to(self.device)  # x0
            attention_mask = batch["attention_mask"].to(self.device)
            encoder_hidden_states = batch["encoder_hidden_states"].to(self.device)
            encoder_attention_mask = batch["encoder_attention_mask"].to(self.device)
            context_latents = batch["context_latents"].to(self.device)
            
            bsz = target_latents.shape[0]
            
            # Classifier-free guidance dropout: null condition for cfg_dropout_prob of the batch
            if self.training_config.cfg_dropout_prob > 0:
                cfg_mask = torch.rand(bsz, device=self.device, dtype=encoder_hidden_states.dtype) < self.training_config.cfg_dropout_prob
                encoder_hidden_states = encoder_hidden_states.clone()
                encoder_attention_mask = encoder_attention_mask.clone()
                encoder_hidden_states[cfg_mask] = 0
                encoder_attention_mask[cfg_mask] = 0
            
            # Flow matching: sample noise x1 and interpolate with data x0
            x1 = torch.randn_like(target_latents)  # Noise
            x0 = target_latents  # Data
            
            # Sample timesteps: continuous (logit-normal) or discrete schedule
            use_continuous = self.training_config.use_continuous_timestep
            if use_continuous or not getattr(self.config, "is_turbo", False):
                t, r = _sample_flowmatching_t_r(
                    bsz=bsz,
                    device=self.device,
                    dtype=target_latents.dtype,
                    data_proportion=getattr(self.config, "data_proportion", 0.5),
                    timestep_mu=getattr(self.config, "timestep_mu", -0.4),
                    timestep_sigma=getattr(self.config, "timestep_sigma", 1.0),
                    use_meanflow=False,
                )
            else:
                if self.training_timesteps.dtype != target_latents.dtype:
                    self.training_timesteps = self.training_timesteps.to(dtype=target_latents.dtype)
                t, r = sample_discrete_timestep(bsz, self.training_timesteps)
            t_ = t.unsqueeze(-1).unsqueeze(-1)
            
            # Interpolate: x_t = t * x1 + (1 - t) * x0
            xt = t_ * x1 + (1.0 - t_) * x0
            
            # Forward through decoder (distilled turbo model, no CFG)
            decoder_outputs = self.model.decoder(
                hidden_states=xt,
                timestep=t,
                timestep_r=t,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
            )
            
            # Flow matching loss: predict the flow field v = x1 - x0
            flow = x1 - x0
            diffusion_loss = F.mse_loss(decoder_outputs[0], flow)
        
        # Convert loss to float32 for stable backward pass
        diffusion_loss = diffusion_loss.float()

        # Optional: weight loss by precomputed sample importance (faster convergence)
        if self.training_config.use_grad_norm_sample_weighting and batch.get("grad_norms"):
            gn_list = batch["grad_norms"]
            weights = []
            for gn in gn_list:
                if gn and isinstance(gn, dict) and gn:
                    weights.append(sum(gn.values()) / len(gn))
                else:
                    weights.append(1.0)
            if weights:
                batch_w = sum(weights) / len(weights)
                batch_w = min(2.0, max(0.25, batch_w))
                diffusion_loss = diffusion_loss * batch_w
        
        self.training_losses.append(diffusion_loss.item())
        
        return diffusion_loss


class PreprocessedLoKRModule(nn.Module):
    """LoKR Training Module using preprocessed tensors."""

    def __init__(
        self,
        model: nn.Module,
        lokr_config: LoKRConfig,
        training_config: TrainingConfig,
        device: torch.device,
        dtype: torch.dtype,
    ):
        super().__init__()

        self.lokr_config = lokr_config
        self.training_config = training_config
        self.device = device
        self.dtype = dtype
        self.lycoris_net = None
        self.lokr_info = {}

        if check_lycoris_available():
            self.model, self.lycoris_net, self.lokr_info = inject_lokr_into_dit(model, lokr_config)
            logger.info(f"LoKR injected: {self.lokr_info.get('trainable_params', 0):,} trainable params")
        else:
            self.model = model
            self.lokr_info = {}
            logger.warning("LyCORIS not available, cannot train LoKR adapters")

        self.config = model.config
        self.training_losses = []
        self.training_timesteps = build_training_timesteps(
            model=self.model,
            training_config=self.training_config,
            device=self.device,
            dtype=self.dtype,
        )
        logger.info(
            f"Training schedule: steps={self.training_timesteps.numel()}, shift={getattr(self.config, 'shift', self.training_config.shift)}, "
            f"turbo={getattr(self.config, 'is_turbo', False)}"
        )

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        _device_type = self.device if isinstance(self.device, str) else self.device.type
        _autocast_dtype = torch.float16 if _device_type == "mps" else torch.bfloat16

        with torch.autocast(device_type=_device_type, dtype=_autocast_dtype):
            target_latents = batch["target_latents"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            encoder_hidden_states = batch["encoder_hidden_states"].to(self.device)
            encoder_attention_mask = batch["encoder_attention_mask"].to(self.device)
            context_latents = batch["context_latents"].to(self.device)

            bsz = target_latents.shape[0]
            if self.training_config.cfg_dropout_prob > 0:
                cfg_mask = torch.rand(bsz, device=self.device, dtype=encoder_hidden_states.dtype) < self.training_config.cfg_dropout_prob
                encoder_hidden_states = encoder_hidden_states.clone()
                encoder_attention_mask = encoder_attention_mask.clone()
                encoder_hidden_states[cfg_mask] = 0
                encoder_attention_mask[cfg_mask] = 0

            x1 = torch.randn_like(target_latents)
            x0 = target_latents

            use_continuous = self.training_config.use_continuous_timestep
            if use_continuous or not getattr(self.config, "is_turbo", False):
                t, r = _sample_flowmatching_t_r(
                    bsz=bsz,
                    device=self.device,
                    dtype=target_latents.dtype,
                    data_proportion=getattr(self.config, "data_proportion", 0.5),
                    timestep_mu=getattr(self.config, "timestep_mu", -0.4),
                    timestep_sigma=getattr(self.config, "timestep_sigma", 1.0),
                    use_meanflow=False,
                )
            else:
                if self.training_timesteps.dtype != target_latents.dtype:
                    self.training_timesteps = self.training_timesteps.to(dtype=target_latents.dtype)
                t, r = sample_discrete_timestep(bsz, self.training_timesteps)
            t_ = t.unsqueeze(-1).unsqueeze(-1)
            xt = t_ * x1 + (1.0 - t_) * x0

            decoder_outputs = self.model.decoder(
                hidden_states=xt,
                timestep=t,
                timestep_r=t,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
            )

            flow = x1 - x0
            diffusion_loss = F.mse_loss(decoder_outputs[0], flow)

        diffusion_loss = diffusion_loss.float()
        if self.training_config.use_grad_norm_sample_weighting and batch.get("grad_norms"):
            gn_list = batch["grad_norms"]
            weights = []
            for gn in gn_list:
                if gn and isinstance(gn, dict) and gn:
                    weights.append(sum(gn.values()) / len(gn))
                else:
                    weights.append(1.0)
            if weights:
                batch_w = sum(weights) / len(weights)
                batch_w = min(2.0, max(0.25, batch_w))
                diffusion_loss = diffusion_loss * batch_w
        self.training_losses.append(diffusion_loss.item())
        return diffusion_loss


class LoRATrainer:
    """High-level trainer for ACE-Step LoRA fine-tuning.
    
    Uses Lightning Fabric for distributed training and mixed precision.
    Supports training from preprocessed tensor directories.
    """
    
    def __init__(
        self,
        dit_handler,
        lora_config: LoRAConfig,
        training_config: TrainingConfig,
    ):
        """Initialize the trainer.
        
        Args:
            dit_handler: Initialized DiT handler (for model access)
            lora_config: LoRA configuration
            training_config: Training configuration
        """
        self.dit_handler = dit_handler
        self.lora_config = lora_config
        self.training_config = training_config
        
        self.module = None
        self.fabric = None
        self.is_training = False

    def _prepare_training_model(self) -> Optional[str]:
        """Ensure model is ready for LoRA injection/training."""
        if self.dit_handler.model is None:
            return "❌ Model not initialized. Please initialize the service first."

        warning_msg = None
        if _unwrap_stale_fabric_decoder(self.dit_handler.model):
            logger.info("Unwrapped stale Fabric decoder wrapper before LoRA training.")
        # torch.compile wraps the model; LoRA injection after compile is unreliable.
        if hasattr(self.dit_handler.model, "_orig_mod"):
            logger.warning("Model is torch.compile'd; disabling compile for LoRA training.")
            self.dit_handler.model = self.dit_handler.model._orig_mod
            if hasattr(self.dit_handler, "compiled"):
                self.dit_handler.compiled = False
            warning_msg = "⚠️ torch.compile disabled for LoRA training."

        # Ensure decoder is on the training device (basic loop uses it directly).
        try:
            self.dit_handler.model.decoder = (
                self.dit_handler.model.decoder.to(self.dit_handler.device).to(self.dit_handler.dtype)
            )
        except Exception as e:
            logger.warning(f"Could not move decoder to device: {e}")

        return warning_msg
    
    def train_from_preprocessed(
        self,
        tensor_dir: str,
        training_state: Optional[Dict] = None,
        resume_from: Optional[str] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Train LoRA adapters from preprocessed tensor files.

        This is the recommended training method for best performance.

        Args:
            tensor_dir: Directory containing preprocessed .pt files
            training_state: Optional state dict for stopping control
            resume_from: Optional path to checkpoint directory to resume from

        Yields:
            Tuples of (step, loss, status_message)
        """
        self.is_training = True
        
        try:
            warning_msg = self._prepare_training_model()
            if warning_msg:
                yield 0, 0.0, warning_msg

            # Validate tensor directory
            if not os.path.exists(tensor_dir):
                yield 0, 0.0, f"❌ Tensor directory not found: {tensor_dir}"
                return

            # Optionally derive LoRA target_modules from precomputed grad norms (fewer params = faster)
            effective_lora_config = self.lora_config
            if self.training_config.use_grad_norm_target_selection:
                try:
                    from acestep.training.grad_norm_utils import get_top_target_modules_by_grad_norm
                    dataset_for_agg = PreprocessedTensorDataset(tensor_dir)
                    agg = dataset_for_agg.aggregate_grad_norms()
                    if agg:
                        param_mean = {k: sum(v) / len(v) for k, v in agg.items()}
                        top_modules = get_top_target_modules_by_grad_norm(
                            param_mean,
                            top_k=self.training_config.grad_norm_target_top_k,
                        )
                        effective_lora_config = LoRAConfig(
                            r=self.lora_config.r,
                            alpha=self.lora_config.alpha,
                            dropout=self.lora_config.dropout,
                            target_modules=top_modules,
                            bias=self.lora_config.bias,
                        )
                        logger.info(
                            "Grad-norm target selection: using target_modules=%s (from precomputed grad norms)",
                            top_modules,
                        )
                        yield 0, 0.0, f"🎯 Grad-norm targets: {top_modules} (fewer params = faster)"
                except Exception as e:
                    logger.warning("Grad-norm target selection failed, using config targets: %s", e)

            # Create training module
            self.module = PreprocessedLoRAModule(
                model=self.dit_handler.model,
                lora_config=effective_lora_config,
                training_config=self.training_config,
                device=self.dit_handler.device,
                dtype=self.dit_handler.dtype,
            )
            
            # Create data module
            data_module = PreprocessedDataModule(
                tensor_dir=tensor_dir,
                batch_size=self.training_config.batch_size,
                num_workers=self.training_config.num_workers,
                pin_memory=self.training_config.pin_memory,
                prefetch_factor=self.training_config.prefetch_factor,
                persistent_workers=self.training_config.persistent_workers,
            )
            
            # Setup data
            data_module.setup('fit')
            
            if len(data_module.train_dataset) == 0:
                yield 0, 0.0, "❌ No valid samples found in tensor directory"
                return
            
            yield 0, 0.0, f"📂 Loaded {len(data_module.train_dataset)} preprocessed samples"

            if LIGHTNING_AVAILABLE:
                yield from self._train_with_fabric(data_module, training_state, resume_from)
            else:
                yield from self._train_basic(data_module, training_state)
                
        except Exception as e:
            logger.exception("Training failed")
            yield 0, 0.0, f"❌ Training failed: {str(e)}"
        finally:
            self.is_training = False
    
    def _train_with_fabric(
        self,
        data_module: PreprocessedDataModule,
        training_state: Optional[Dict],
        resume_from: Optional[str] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Train using Lightning Fabric."""
        # Create output directory
        os.makedirs(self.training_config.output_dir, exist_ok=True)
        
        # Force BFloat16 precision (only supported precision for this model)
        precision = "bf16-mixed"
        
        # Create TensorBoard logger
        tb_logger = TensorBoardLogger(
            root_dir=self.training_config.output_dir,
            name="logs"
        )
        
        # Initialize Fabric
        self.fabric = Fabric(
            accelerator="auto",
            devices=1,
            precision=precision,
            loggers=[tb_logger],
        )
        self.fabric.launch()
        
        yield 0, 0.0, f"🚀 Starting training (precision: {precision})..."
        
        # Get dataloader
        train_loader = data_module.train_dataloader()
        
        # Setup optimizer - only LoRA parameters
        trainable_params = [p for p in self.module.model.parameters() if p.requires_grad]
        
        if not trainable_params:
            yield 0, 0.0, "❌ No trainable parameters found!"
            return
        
        device_type = self.dit_handler.device if isinstance(self.dit_handler.device, str) else self.dit_handler.device.type
        if device_type == "mps":
            casted, total = _ensure_trainable_params_fp32(self.module.model)
            if casted:
                logger.info(f"Cast {casted}/{total} trainable params to fp32 for MPS stability.")

        yield 0, 0.0, f"🎯 Training {sum(p.numel() for p in trainable_params):,} parameters"
        
        optimizer = _select_optimizer(
            trainable_params,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            device_type=device_type,
        )
        param_name_lookup = _build_param_name_lookup(self.module.model, self.module.lycoris_net)
        
        # Calculate total steps
        total_steps = len(train_loader) * self.training_config.max_epochs // self.training_config.gradient_accumulation_steps
        warmup_steps = min(self.training_config.warmup_steps, max(1, total_steps // 10))
        
        # Scheduler
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, total_steps - warmup_steps),
            T_mult=1,
            eta_min=self.training_config.learning_rate * 0.01,
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )
        
        # Convert model to bfloat16 (entire model for consistent dtype)
        self.module.model = self.module.model.to(torch.bfloat16)

        # Setup with Fabric - only the decoder (which has LoRA)
        self.module.model.decoder, optimizer = self.fabric.setup(self.module.model.decoder, optimizer)

        if self.training_config.compile_decoder:
            try:
                torch._dynamo.config.allow_unspec_int_on_nn_module = True
                torch._dynamo.config.recompile_limit = 64
                torch._dynamo.config.suppress_errors = True
                self.module.model.decoder = torch.compile(self.module.model.decoder)
                yield 0, 0.0, "⚡ torch.compile enabled for decoder"
            except Exception as e:
                logger.warning(f"torch.compile failed, continuing without compile: {e}")
        train_loader = self.fabric.setup_dataloaders(train_loader)

        # Handle resume from checkpoint (load AFTER Fabric setup)
        start_epoch = 0
        global_step = 0
        checkpoint_info = None

        if resume_from and os.path.exists(resume_from):
            try:
                yield 0, 0.0, f"🔄 Loading checkpoint from {resume_from}..."

                # Load checkpoint using utility function
                checkpoint_info = load_lora_training_checkpoint_lycoris(
                    resume_from,
                    lycoris_net=self.module.lycoris_net,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=self.module.device,
                    param_id_to_name=param_name_lookup,
                )

                if checkpoint_info["weights_path"]:
                    start_epoch = checkpoint_info["epoch"]
                    global_step = checkpoint_info["global_step"]

                    status_parts = [f"✅ Resumed from epoch {start_epoch}, step {global_step}"]
                    if checkpoint_info["loaded_optimizer"]:
                        status_parts.append("optimizer ✓")
                    if checkpoint_info["loaded_scheduler"]:
                        status_parts.append("scheduler ✓")
                    yield 0, 0.0, ", ".join(status_parts)
                else:
                    yield 0, 0.0, f"⚠️ No valid checkpoint found in {resume_from}"

            except Exception as e:
                logger.exception("Failed to load checkpoint")
                yield 0, 0.0, f"⚠️ Failed to load checkpoint: {e}, starting fresh"
                start_epoch = 0
                global_step = 0
        elif resume_from:
            yield 0, 0.0, f"⚠️ Checkpoint path not found: {resume_from}, starting fresh"

        # Training loop
        accumulation_step = 0
        accumulated_loss = 0.0

        self.module.model.decoder.train()

        for epoch in range(start_epoch, self.training_config.max_epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_start_time = time.time()
            
            for batch_idx, batch in enumerate(train_loader):
                # Check for stop signal
                if training_state and training_state.get("should_stop", False):
                    checkpoint_dir = os.path.join(
                        self.training_config.output_dir,
                        "checkpoints",
                        f"paused_epoch_{epoch+1}_step_{global_step}",
                    )
                    extra_state = {}
                    if training_state is not None and "elapsed_seconds" in training_state:
                        extra_state["elapsed_seconds"] = training_state["elapsed_seconds"]
                    extra_state["optimizer_state_by_name"] = build_optimizer_state_by_name(optimizer, param_name_lookup)
                    save_lora_training_checkpoint_lycoris(
                        self.module.lycoris_net,
                        optimizer,
                        scheduler,
                        epoch + 1,
                        global_step,
                        checkpoint_dir,
                        lora_config=self.module.lora_config,
                        extra_state=extra_state,
                    )
                    if training_state is not None:
                        training_state["last_checkpoint_dir"] = checkpoint_dir
                    avg_loss = accumulated_loss / max(accumulation_step, 1)
                    yield global_step, avg_loss, f"⏸️ Paused. Checkpoint saved to {checkpoint_dir}"
                    return
                
                # Forward pass
                loss = self.module.training_step(batch)
                loss = loss / self.training_config.gradient_accumulation_steps
                
                # Backward pass
                self.fabric.backward(loss)
                accumulated_loss += loss.item()
                accumulation_step += 1
                
                # Optimizer step
                if accumulation_step >= self.training_config.gradient_accumulation_steps:
                    self.fabric.clip_gradients(
                        self.module.model.decoder,
                        optimizer,
                        max_norm=self.training_config.max_grad_norm,
                    )
                    
                    nonfinite, total_with_grad, details = _count_nonfinite_grads_detailed(
                        trainable_params, param_name_lookup
                    )
                    if nonfinite:
                        logger.warning(
                            f"Detected {nonfinite}/{total_with_grad} non-finite gradients; example: "
                            + "; ".join(details)
                        )

                    # Log per-parameter gradient norms (for rank/target selection) before step
                    next_step = global_step + 1
                    if self.training_config.log_gradient_norms_every > 0 and next_step % self.training_config.log_gradient_norms_every == 0:
                        for p in trainable_params:
                            if p.grad is not None and torch.isfinite(p.grad).all():
                                name = param_name_lookup.get(id(p), f"param_{id(p)}")
                                self.fabric.log(f"grad_norm/{name}", p.grad.norm().item(), step=next_step)

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Log
                    avg_loss = accumulated_loss / accumulation_step
                    self.fabric.log("train/loss", avg_loss, step=global_step)
                    self.fabric.log("train/lr", scheduler.get_last_lr()[0], step=global_step)
                    
                    if global_step % self.training_config.log_every_n_steps == 0:
                        yield global_step, avg_loss, f"Epoch {epoch+1}/{self.training_config.max_epochs}, Step {global_step}, Loss: {avg_loss:.4f}"
                    
                    epoch_loss += accumulated_loss
                    num_batches += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0
            
            # End of epoch
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            
            self.fabric.log("train/epoch_loss", avg_epoch_loss, step=epoch + 1)
            yield global_step, avg_epoch_loss, f"✅ Epoch {epoch+1}/{self.training_config.max_epochs} in {epoch_time:.1f}s, Loss: {avg_epoch_loss:.4f}"
            
            # Save checkpoint
            if (epoch + 1) % self.training_config.save_every_n_epochs == 0:
                checkpoint_dir = os.path.join(self.training_config.output_dir, "checkpoints", f"epoch_{epoch+1}")
                extra_state = {}
                if training_state is not None and "elapsed_seconds" in training_state:
                    extra_state["elapsed_seconds"] = training_state["elapsed_seconds"]
                extra_state["optimizer_state_by_name"] = build_optimizer_state_by_name(optimizer, param_name_lookup)
                save_lora_training_checkpoint_lycoris(
                    self.module.lycoris_net,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    global_step,
                    checkpoint_dir,
                    lora_config=self.module.lora_config,
                    extra_state=extra_state,
                )
                yield global_step, avg_epoch_loss, f"💾 Checkpoint saved at epoch {epoch+1}"

        # Save final model
        final_path = os.path.join(self.training_config.output_dir, "final")
        save_lora_weights_lycoris(self.module.lycoris_net, final_path)
        
        final_loss = self.module.training_losses[-1] if self.module.training_losses else 0.0
        yield global_step, final_loss, f"✅ Training complete! LoRA saved to {final_path}"
    
    def _train_basic(
        self,
        data_module: PreprocessedDataModule,
        training_state: Optional[Dict],
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Basic training loop without Fabric."""
        yield 0, 0.0, "🚀 Starting basic training loop..."
        
        os.makedirs(self.training_config.output_dir, exist_ok=True)
        
        train_loader = data_module.train_dataloader()
        
        trainable_params = [p for p in self.module.model.parameters() if p.requires_grad]
        
        if not trainable_params:
            yield 0, 0.0, "❌ No trainable parameters found!"
            return
        
        device_type = self.dit_handler.device if isinstance(self.dit_handler.device, str) else self.dit_handler.device.type
        if device_type == "mps":
            casted, total = _ensure_trainable_params_fp32(self.module.model)
            if casted:
                logger.info(f"Cast {casted}/{total} trainable params to fp32 for MPS stability.")

        optimizer = _select_optimizer(
            trainable_params,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            device_type=device_type,
        )
        param_name_lookup = _build_param_name_lookup(self.module.model, self.module.lycoris_net)
        
        total_steps = len(train_loader) * self.training_config.max_epochs // self.training_config.gradient_accumulation_steps
        warmup_steps = min(self.training_config.warmup_steps, max(1, total_steps // 10))
        
        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        main_scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=max(1, total_steps - warmup_steps), T_mult=1, eta_min=self.training_config.learning_rate * 0.01)
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])
        
        global_step = 0
        accumulation_step = 0
        accumulated_loss = 0.0
        
        self.module.model.decoder = self.module.model.decoder.to(self.module.device).to(self.module.dtype)
        self.module.model.decoder.train()
        
        for epoch in range(self.training_config.max_epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_start_time = time.time()
            
            for batch in train_loader:
                if training_state and training_state.get("should_stop", False):
                    checkpoint_dir = os.path.join(
                        self.training_config.output_dir,
                        "checkpoints",
                        f"paused_epoch_{epoch+1}_step_{global_step}",
                    )
                    extra_state = {}
                    if training_state is not None and "elapsed_seconds" in training_state:
                        extra_state["elapsed_seconds"] = training_state["elapsed_seconds"]
                    extra_state["optimizer_state_by_name"] = build_optimizer_state_by_name(optimizer, param_name_lookup)
                    save_lora_training_checkpoint_lycoris(
                        self.module.lycoris_net,
                        optimizer,
                        scheduler,
                        epoch + 1,
                        global_step,
                        checkpoint_dir,
                        lora_config=self.module.lora_config,
                        extra_state=extra_state,
                    )
                    if training_state is not None:
                        training_state["last_checkpoint_dir"] = checkpoint_dir
                    avg_loss = accumulated_loss / max(accumulation_step, 1)
                    yield global_step, avg_loss, f"⏸️ Paused. Checkpoint saved to {checkpoint_dir}"
                    return

                loss = self.module.training_step(batch)
                loss = loss / self.training_config.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()
                accumulation_step += 1
                
                if accumulation_step >= self.training_config.gradient_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.training_config.max_grad_norm)
                    nonfinite, total_with_grad, details = _count_nonfinite_grads_detailed(
                        trainable_params, param_name_lookup
                    )
                    if nonfinite:
                        logger.warning(
                            f"Detected {nonfinite}/{total_with_grad} non-finite gradients; example: "
                            + "; ".join(details)
                        )
                    next_step = global_step + 1
                    if self.training_config.log_gradient_norms_every > 0 and next_step % self.training_config.log_gradient_norms_every == 0:
                        for p in trainable_params:
                            if p.grad is not None and torch.isfinite(p.grad).all():
                                name = param_name_lookup.get(id(p), f"param_{id(p)}")
                                logger.debug(f"grad_norm/{name} = {p.grad.norm().item():.6f}")
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    
                    if global_step % self.training_config.log_every_n_steps == 0:
                        avg_loss = accumulated_loss / accumulation_step
                        yield global_step, avg_loss, f"Epoch {epoch+1}, Step {global_step}, Loss: {avg_loss:.4f}"
                    
                    epoch_loss += accumulated_loss
                    num_batches += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0
            
            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            yield global_step, avg_epoch_loss, f"✅ Epoch {epoch+1}/{self.training_config.max_epochs} in {epoch_time:.1f}s"
            
            if (epoch + 1) % self.training_config.save_every_n_epochs == 0:
                checkpoint_dir = os.path.join(self.training_config.output_dir, "checkpoints", f"epoch_{epoch+1}")
                save_lora_weights_lycoris(self.module.lycoris_net, checkpoint_dir)
                yield global_step, avg_epoch_loss, f"💾 Checkpoint saved"
        
        final_path = os.path.join(self.training_config.output_dir, "final")
        save_lora_weights_lycoris(self.module.lycoris_net, final_path)
        final_loss = self.module.training_losses[-1] if self.module.training_losses else 0.0
        yield global_step, final_loss, f"✅ Training complete! LoRA saved to {final_path}"
    
    def stop(self):
        """Stop training."""
        self.is_training = False


class LoKRTrainer:
    """High-level trainer for ACE-Step LoKR fine-tuning."""

    def __init__(
        self,
        dit_handler,
        lokr_config: LoKRConfig,
        training_config: TrainingConfig,
    ):
        self.dit_handler = dit_handler
        self.lokr_config = lokr_config
        self.training_config = training_config

        self.module = None
        self.fabric = None
        self.is_training = False

    def _prepare_training_model(self) -> Optional[str]:
        if self.dit_handler.model is None:
            return "❌ Model not initialized. Please initialize the service first."
        warning_msg = None
        if _unwrap_stale_fabric_decoder(self.dit_handler.model):
            logger.info("Unwrapped stale Fabric decoder wrapper before LoKR training.")
        if hasattr(self.dit_handler.model, "_orig_mod"):
            logger.warning("Model is torch.compile'd; disabling compile for LoKR training.")
            self.dit_handler.model = self.dit_handler.model._orig_mod
            if hasattr(self.dit_handler, "compiled"):
                self.dit_handler.compiled = False
            warning_msg = "⚠️ torch.compile disabled for LoKR training."
        try:
            self.dit_handler.model.decoder = (
                self.dit_handler.model.decoder.to(self.dit_handler.device).to(self.dit_handler.dtype)
            )
        except Exception as e:
            logger.warning(f"Could not move decoder to device: {e}")
        return warning_msg

    def _ensure_lokr_device(self) -> None:
        """Move LoKR (LyCORIS) modules to the training device/dtype."""
        if self.module is None:
            return
        try:
            if getattr(self.module, "lycoris_net", None) is not None:
                self.module.lycoris_net.to(self.dit_handler.device)
            if getattr(self.module, "model", None) is not None:
                self.module.model.decoder = (
                    self.module.model.decoder.to(self.dit_handler.device).to(self.dit_handler.dtype)
                )
        except Exception as e:
            logger.warning(f"Failed to move LoKR modules to device: {e}")

    def train_from_preprocessed(
        self,
        tensor_dir: str,
        training_state: Optional[Dict] = None,
        resume_from: Optional[str] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        self.is_training = True

        try:
            warning_msg = self._prepare_training_model()
            if warning_msg:
                yield 0, 0.0, warning_msg

            if not os.path.exists(tensor_dir):
                yield 0, 0.0, f"❌ Tensor directory not found: {tensor_dir}"
                return

            if not check_lycoris_available():
                yield 0, 0.0, "❌ LyCORIS not installed. Install lycoris-lora to train LoKR."
                return

            self.module = PreprocessedLoKRModule(
                model=self.dit_handler.model,
                lokr_config=self.lokr_config,
                training_config=self.training_config,
                device=self.dit_handler.device,
                dtype=self.dit_handler.dtype,
            )

            data_module = PreprocessedDataModule(
                tensor_dir=tensor_dir,
                batch_size=self.training_config.batch_size,
                num_workers=self.training_config.num_workers,
                pin_memory=self.training_config.pin_memory,
                prefetch_factor=self.training_config.prefetch_factor,
                persistent_workers=self.training_config.persistent_workers,
            )
            data_module.setup('fit')

            if len(data_module.train_dataset) == 0:
                yield 0, 0.0, "❌ No valid samples found in tensor directory"
                return

            yield 0, 0.0, f"📂 Loaded {len(data_module.train_dataset)} preprocessed samples"

            if LIGHTNING_AVAILABLE:
                yield from self._train_with_fabric(data_module, training_state, resume_from=resume_from)
            else:
                yield from self._train_basic(data_module, training_state, resume_from=resume_from)

        except Exception as e:
            logger.exception("LoKR training failed")
            yield 0, 0.0, f"❌ Training failed: {str(e)}"
        finally:
            self.is_training = False

    def _train_with_fabric(
        self,
        data_module: PreprocessedDataModule,
        training_state: Optional[Dict],
        resume_from: Optional[str] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        os.makedirs(self.training_config.output_dir, exist_ok=True)

        precision = "bf16-mixed"
        tb_logger = TensorBoardLogger(
            root_dir=self.training_config.output_dir,
            name="logs",
        )

        self.fabric = Fabric(
            accelerator="auto",
            devices=1,
            precision=precision,
            loggers=[tb_logger],
        )
        self.fabric.launch()

        yield 0, 0.0, f"🚀 Starting training (precision: {precision})..."

        train_loader = data_module.train_dataloader()

        trainable_params = [p for p in self.module.model.decoder.parameters() if p.requires_grad]
        if not trainable_params and getattr(self.module, "lycoris_net", None) is not None:
            trainable_params = [p for p in self.module.lycoris_net.parameters() if p.requires_grad]
        if not trainable_params:
            yield 0, 0.0, "❌ No trainable parameters found!"
            return

        device_type = self.dit_handler.device if isinstance(self.dit_handler.device, str) else self.dit_handler.device.type
        if device_type == "mps":
            casted, total = _ensure_trainable_params_fp32(self.module.model)
            if casted:
                logger.info(f"Cast {casted}/{total} trainable params to fp32 for MPS stability.")

        yield 0, 0.0, f"🎯 Training {sum(p.numel() for p in trainable_params):,} parameters"

        optimizer = _select_optimizer(
            trainable_params,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            device_type=device_type,
        )
        param_name_lookup = _build_param_name_lookup(self.module.model, self.module.lycoris_net)

        total_steps = len(train_loader) * self.training_config.max_epochs // self.training_config.gradient_accumulation_steps
        warmup_steps = min(self.training_config.warmup_steps, max(1, total_steps // 10))

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, total_steps - warmup_steps),
            T_mult=1,
            eta_min=self.training_config.learning_rate * 0.01,
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps],
        )

        self.module.model = self.module.model.to(torch.bfloat16)
        decoder_for_setup = self.module.model.decoder
        if hasattr(decoder_for_setup, "_forward_module"):
            decoder_for_setup = decoder_for_setup._forward_module
        self.module.model.decoder, optimizer = self.fabric.setup(decoder_for_setup, optimizer)
        train_loader = self.fabric.setup_dataloaders(train_loader)
        self._ensure_lokr_device()

        # Handle resume from checkpoint
        start_epoch = 0
        global_step = 0
        checkpoint_info = None

        if resume_from and os.path.exists(resume_from):
            try:
                yield 0, 0.0, f"🔄 Loading checkpoint from {resume_from}..."
                checkpoint_info = load_lokr_training_checkpoint(
                    resume_from,
                    lycoris_net=self.module.lycoris_net,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=self.module.device,
                    param_id_to_name=param_name_lookup,
                )
                self._ensure_lokr_device()
                if checkpoint_info.get("weights_path"):
                    start_epoch = checkpoint_info.get("epoch", 0)
                    global_step = checkpoint_info.get("global_step", 0)
                    status_parts = [f"✅ Resumed from epoch {start_epoch}, step {global_step}"]
                    if checkpoint_info.get("loaded_optimizer"):
                        status_parts.append("optimizer ✓")
                    if checkpoint_info.get("loaded_scheduler"):
                        status_parts.append("scheduler ✓")
                    yield 0, 0.0, ", ".join(status_parts)
                else:
                    yield 0, 0.0, f"⚠️ No valid LoKR weights found in {resume_from}"
            except Exception as e:
                logger.exception("Failed to load LoKR checkpoint")
                yield 0, 0.0, f"⚠️ Failed to load checkpoint: {e}, starting fresh"
                start_epoch = 0
                global_step = 0
        elif resume_from:
            yield 0, 0.0, f"⚠️ Checkpoint path not found: {resume_from}, starting fresh"

        accumulation_step = 0
        accumulated_loss = 0.0

        self.module.model.decoder.train()

        for epoch in range(start_epoch, self.training_config.max_epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_start_time = time.time()

            for _, batch in enumerate(train_loader):
                if training_state and training_state.get("should_stop", False):
                    checkpoint_dir = os.path.join(
                        self.training_config.output_dir,
                        "checkpoints",
                        f"paused_epoch_{epoch+1}_step_{global_step}",
                    )
                    extra_state = {}
                    if training_state is not None and "elapsed_seconds" in training_state:
                        extra_state["elapsed_seconds"] = training_state["elapsed_seconds"]
                    extra_state["optimizer_state_by_name"] = build_optimizer_state_by_name(optimizer, param_name_lookup)
                    save_lokr_training_checkpoint(
                        self.module.lycoris_net,
                        optimizer,
                        scheduler,
                        epoch + 1,
                        global_step,
                        checkpoint_dir,
                        lokr_config=self.lokr_config,
                        extra_state=extra_state,
                    )
                    if training_state is not None:
                        training_state["last_checkpoint_dir"] = checkpoint_dir
                    avg_loss = accumulated_loss / max(accumulation_step, 1)
                    yield global_step, avg_loss, f"⏸️ Paused. Checkpoint saved to {checkpoint_dir}"
                    return

                loss = self.module.training_step(batch)
                loss = loss / self.training_config.gradient_accumulation_steps

                self.fabric.backward(loss)
                accumulated_loss += loss.item()
                accumulation_step += 1

                if accumulation_step >= self.training_config.gradient_accumulation_steps:
                    self.fabric.clip_gradients(
                        self.module.model.decoder,
                        optimizer,
                        max_norm=self.training_config.max_grad_norm,
                    )

                    nonfinite, total_with_grad, details = _count_nonfinite_grads_detailed(
                        trainable_params, param_name_lookup
                    )
                    if nonfinite:
                        logger.warning(
                            f"Detected {nonfinite}/{total_with_grad} non-finite gradients; example: "
                            + "; ".join(details)
                        )

                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    global_step += 1

                    avg_loss = accumulated_loss / accumulation_step
                    self.fabric.log("train/loss", avg_loss, step=global_step)
                    self.fabric.log("train/lr", scheduler.get_last_lr()[0], step=global_step)

                    if global_step % self.training_config.log_every_n_steps == 0:
                        yield global_step, avg_loss, (
                            f"Epoch {epoch+1}/{self.training_config.max_epochs}, "
                            f"Step {global_step}, Loss: {avg_loss:.4f}"
                        )

                    epoch_loss += accumulated_loss
                    num_batches += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0

            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            self.fabric.log("train/epoch_loss", avg_epoch_loss, step=epoch + 1)
            yield global_step, avg_epoch_loss, (
                f"✅ Epoch {epoch+1}/{self.training_config.max_epochs} in {epoch_time:.1f}s, "
                f"Loss: {avg_epoch_loss:.4f}"
            )

            if (epoch + 1) % self.training_config.save_every_n_epochs == 0:
                checkpoint_dir = os.path.join(self.training_config.output_dir, "checkpoints", f"epoch_{epoch+1}")
                save_lokr_training_checkpoint(
                    self.module.lycoris_net,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    global_step,
                    checkpoint_dir,
                    lokr_config=self.lokr_config,
                    extra_state={"optimizer_state_by_name": build_optimizer_state_by_name(optimizer, param_name_lookup)},
                )
                yield global_step, avg_epoch_loss, f"💾 Checkpoint saved at epoch {epoch+1}"

        final_path = os.path.join(self.training_config.output_dir, "final")
        save_lokr_weights(self.module.lycoris_net, final_path)
        final_loss = self.module.training_losses[-1] if self.module.training_losses else 0.0
        yield global_step, final_loss, f"✅ Training complete! LoKR saved to {final_path}"

    def _train_basic(
        self,
        data_module: PreprocessedDataModule,
        training_state: Optional[Dict],
        resume_from: Optional[str] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        yield 0, 0.0, "🚀 Starting basic training loop..."

        os.makedirs(self.training_config.output_dir, exist_ok=True)

        train_loader = data_module.train_dataloader()

        trainable_params = [p for p in self.module.model.decoder.parameters() if p.requires_grad]
        if not trainable_params and getattr(self.module, "lycoris_net", None) is not None:
            trainable_params = [p for p in self.module.lycoris_net.parameters() if p.requires_grad]
        if not trainable_params:
            yield 0, 0.0, "❌ No trainable parameters found!"
            return

        device_type = self.dit_handler.device if isinstance(self.dit_handler.device, str) else self.dit_handler.device.type
        if device_type == "mps":
            casted, total = _ensure_trainable_params_fp32(self.module.model)
            if casted:
                logger.info(f"Cast {casted}/{total} trainable params to fp32 for MPS stability.")

        optimizer = _select_optimizer(
            trainable_params,
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            device_type=device_type,
        )
        param_name_lookup = _build_param_name_lookup(self.module.model, self.module.lycoris_net)

        total_steps = len(train_loader) * self.training_config.max_epochs // self.training_config.gradient_accumulation_steps
        warmup_steps = min(self.training_config.warmup_steps, max(1, total_steps // 10))

        warmup_scheduler = LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_steps)
        main_scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, total_steps - warmup_steps),
            T_mult=1,
            eta_min=self.training_config.learning_rate * 0.01,
        )
        scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps])

        global_step = 0
        accumulation_step = 0
        accumulated_loss = 0.0

        if resume_from and os.path.exists(resume_from):
            try:
                yield 0, 0.0, f"🔄 Loading checkpoint from {resume_from}..."
                checkpoint_info = load_lokr_training_checkpoint(
                    resume_from,
                    lycoris_net=self.module.lycoris_net,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    device=self.module.device,
                    param_id_to_name=param_name_lookup,
                )
                if checkpoint_info.get("weights_path"):
                    global_step = checkpoint_info.get("global_step", 0)
                    yield 0, 0.0, f"✅ Resumed from step {global_step}"
                else:
                    yield 0, 0.0, f"⚠️ No valid LoKR weights found in {resume_from}"
            except Exception as e:
                logger.exception("Failed to load LoKR checkpoint")
                yield 0, 0.0, f"⚠️ Failed to load checkpoint: {e}, starting fresh"
        elif resume_from:
            yield 0, 0.0, f"⚠️ Checkpoint path not found: {resume_from}, starting fresh"

        self.module.model.decoder = self.module.model.decoder.to(self.module.device).to(self.module.dtype)
        self.module.model.decoder.train()

        for epoch in range(self.training_config.max_epochs):
            epoch_loss = 0.0
            num_batches = 0
            epoch_start_time = time.time()

            for batch in train_loader:
                if training_state and training_state.get("should_stop", False):
                    checkpoint_dir = os.path.join(
                        self.training_config.output_dir,
                        "checkpoints",
                        f"paused_epoch_{epoch+1}_step_{global_step}",
                    )
                    extra_state = {}
                    if training_state is not None and "elapsed_seconds" in training_state:
                        extra_state["elapsed_seconds"] = training_state["elapsed_seconds"]
                    extra_state["optimizer_state_by_name"] = build_optimizer_state_by_name(optimizer, param_name_lookup)
                    save_lokr_training_checkpoint(
                        self.module.lycoris_net,
                        optimizer,
                        scheduler,
                        epoch + 1,
                        global_step,
                        checkpoint_dir,
                        lokr_config=self.lokr_config,
                        extra_state=extra_state,
                    )
                    if training_state is not None:
                        training_state["last_checkpoint_dir"] = checkpoint_dir
                    avg_loss = accumulated_loss / max(accumulation_step, 1)
                    yield global_step, avg_loss, f"⏸️ Paused. Checkpoint saved to {checkpoint_dir}"
                    return

                loss = self.module.training_step(batch)
                loss = loss / self.training_config.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()
                accumulation_step += 1

                if accumulation_step >= self.training_config.gradient_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(trainable_params, self.training_config.max_grad_norm)
                    nonfinite, total_with_grad, details = _count_nonfinite_grads_detailed(
                        trainable_params, param_name_lookup
                    )
                    if nonfinite:
                        logger.warning(
                            f"Detected {nonfinite}/{total_with_grad} non-finite gradients; example: "
                            + "; ".join(details)
                        )
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1

                    if global_step % self.training_config.log_every_n_steps == 0:
                        avg_loss = accumulated_loss / accumulation_step
                        yield global_step, avg_loss, f"Epoch {epoch+1}, Step {global_step}, Loss: {avg_loss:.4f}"

                    epoch_loss += accumulated_loss
                    num_batches += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0

            epoch_time = time.time() - epoch_start_time
            avg_epoch_loss = epoch_loss / max(num_batches, 1)
            yield global_step, avg_epoch_loss, f"✅ Epoch {epoch+1}/{self.training_config.max_epochs} in {epoch_time:.1f}s"

            if (epoch + 1) % self.training_config.save_every_n_epochs == 0:
                checkpoint_dir = os.path.join(self.training_config.output_dir, "checkpoints", f"epoch_{epoch+1}")
                save_lokr_training_checkpoint(
                    self.module.lycoris_net,
                    optimizer,
                    scheduler,
                    epoch + 1,
                    global_step,
                    checkpoint_dir,
                    lokr_config=self.lokr_config,
                    extra_state={"optimizer_state_by_name": build_optimizer_state_by_name(optimizer, param_name_lookup)},
                )
                yield global_step, avg_epoch_loss, "💾 Checkpoint saved"

        final_path = os.path.join(self.training_config.output_dir, "final")
        save_lokr_weights(self.module.lycoris_net, final_path)
        final_loss = self.module.training_losses[-1] if self.module.training_losses else 0.0
        yield global_step, final_loss, f"✅ Training complete! LoKR saved to {final_path}"

    def stop(self):
        self.is_training = False
