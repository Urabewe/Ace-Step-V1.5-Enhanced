import torch


# Audio length above which to use tiled encode (30s at 48kHz). Reduces VRAM for long clips.
TILED_ENCODE_THRESHOLD_SAMPLES = 48000 * 30


def vae_encode(vae, audio, dtype):
    """VAE encode audio to get target latents."""
    model_device = next(vae.parameters()).device
    if audio.device != model_device:
        audio = audio.to(model_device)

    latent = vae.encode(audio).latent_dist.sample()
    target_latents = latent.transpose(1, 2).to(dtype)
    return target_latents


def vae_encode_tiled(dit_handler, audio, dtype):
    """
    VAE encode long audio in chunks via handler's tiled_encode; returns same shape as vae_encode.
    Use when audio length > TILED_ENCODE_THRESHOLD_SAMPLES to avoid OOM.
    """
    latents = dit_handler.tiled_encode(audio, offload_latent_to_cpu=True)
    target_latents = latents.transpose(1, 2).to(dtype)
    return target_latents
