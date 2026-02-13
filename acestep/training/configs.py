"""
Training Configuration Classes

Contains dataclasses for LoRA and training configurations.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation) training.
    
    Attributes:
        r: LoRA rank (dimension of low-rank matrices)
        alpha: LoRA scaling factor (alpha/r determines the scaling)
        dropout: Dropout probability for LoRA layers
        target_modules: List of module names to apply LoRA to
        bias: Whether to train bias parameters ("none", "all", or "lora_only")
    """
    r: int = 8
    alpha: int = 16
    dropout: float = 0.1
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    bias: str = "none"
    
    def to_dict(self):
        """Convert to dictionary for PEFT config."""
        return {
            "r": self.r,
            "lora_alpha": self.alpha,
            "lora_dropout": self.dropout,
            "target_modules": self.target_modules,
            "bias": self.bias,
        }


@dataclass
class LoKRConfig:
    """Configuration for LoKR (Low-Rank Kronecker Product) training via LyCORIS."""
    linear_dim: int = 64
    linear_alpha: int = 128
    factor: int = -1
    decompose_both: bool = False
    use_tucker: bool = False
    use_scalar: bool = False
    weight_decompose: bool = False
    target_modules: List[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    full_matrix: bool = False
    bypass_mode: bool = False
    rs_lora: bool = False
    unbalanced_factorization: bool = False

    def to_dict(self) -> dict:
        return {
            "linear_dim": self.linear_dim,
            "linear_alpha": self.linear_alpha,
            "factor": self.factor,
            "decompose_both": self.decompose_both,
            "use_tucker": self.use_tucker,
            "use_scalar": self.use_scalar,
            "weight_decompose": self.weight_decompose,
            "target_modules": self.target_modules,
            "full_matrix": self.full_matrix,
            "bypass_mode": self.bypass_mode,
            "rs_lora": self.rs_lora,
            "unbalanced_factorization": self.unbalanced_factorization,
        }


@dataclass
class TrainingConfig:
    """Configuration for LoRA training process.
    
    Training uses:
    - BFloat16 precision (only supported precision)
    - Timesteps are derived from the loaded model config (turbo vs base).
    
    Attributes:
        shift: Timestep shift factor (fixed at 3.0 for turbo model)
        num_inference_steps: Number of inference steps (fixed at 8 for turbo)
        learning_rate: Initial learning rate
        batch_size: Training batch size
        gradient_accumulation_steps: Number of gradient accumulation steps
        max_epochs: Maximum number of training epochs
        save_every_n_epochs: Save checkpoint every N epochs
        warmup_steps: Number of warmup steps for learning rate scheduler
        weight_decay: Weight decay for optimizer
        max_grad_norm: Maximum gradient norm for clipping
        mixed_precision: Always "bf16" (only supported precision)
        seed: Random seed for reproducibility
        output_dir: Directory to save checkpoints and logs
    """
    # Defaults (overridden based on loaded model config)
    shift: float = 3.0
    num_inference_steps: int = 8
    learning_rate: float = 1e-4
    batch_size: int = 1
    gradient_accumulation_steps: int = 4
    max_epochs: int = 100
    save_every_n_epochs: int = 10
    warmup_steps: int = 100
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: str = "bf16"  # Fixed: only bf16 supported
    seed: int = 42
    output_dir: str = "./lora_output"
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    persistent_workers: bool = False

    # Math/precision controls
    matmul_precision: str = "high"  # torch.set_float32_matmul_precision
    allow_tf32: bool = True
    compile_decoder: bool = False
    
    # Logging
    log_every_n_steps: int = 10

    # Side-Step style options
    use_continuous_timestep: bool = False  # Logit-normal timestep sampling (full denoising range); if False, turbo uses discrete 8-step
    cfg_dropout_prob: float = 0.15  # Classifier-free guidance: prob of null condition (0 = disabled)
    log_gradient_norms_every: int = 0  # Log per-parameter gradient norms every N steps for rank/target selection (0 = off)

    # Precomputed grad norms (from preprocess with "Save grad norms")
    use_grad_norm_target_selection: bool = True  # Use precomputed grad norms to pick top LoRA target modules (fewer params = faster)
    grad_norm_target_top_k: int = 4  # Number of module types to keep (e.g. q_proj, k_proj, v_proj, o_proj -> 4)
    use_grad_norm_sample_weighting: bool = True  # Weight batch loss by sample importance from precomputed grad norms

    def to_dict(self):
        """Convert to dictionary."""
        return {
            "shift": self.shift,
            "num_inference_steps": self.num_inference_steps,
            "learning_rate": self.learning_rate,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "max_epochs": self.max_epochs,
            "save_every_n_epochs": self.save_every_n_epochs,
            "warmup_steps": self.warmup_steps,
            "weight_decay": self.weight_decay,
            "max_grad_norm": self.max_grad_norm,
            "mixed_precision": self.mixed_precision,
            "seed": self.seed,
            "output_dir": self.output_dir,
            "num_workers": self.num_workers,
            "pin_memory": self.pin_memory,
            "prefetch_factor": self.prefetch_factor,
            "persistent_workers": self.persistent_workers,
            "matmul_precision": self.matmul_precision,
            "allow_tf32": self.allow_tf32,
            "compile_decoder": self.compile_decoder,
            "log_every_n_steps": self.log_every_n_steps,
            "use_continuous_timestep": self.use_continuous_timestep,
            "cfg_dropout_prob": self.cfg_dropout_prob,
            "log_gradient_norms_every": self.log_gradient_norms_every,
            "use_grad_norm_target_selection": self.use_grad_norm_target_selection,
            "grad_norm_target_top_k": self.grad_norm_target_top_k,
            "use_grad_norm_sample_weighting": self.use_grad_norm_sample_weighting,
        }
