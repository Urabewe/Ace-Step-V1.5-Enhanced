import os
import time
from contextlib import ExitStack
from typing import List, Tuple

import torch
from loguru import logger

from acestep.training.grad_norm_utils import compute_decoder_grad_norms_for_sample
from .models import AudioSample
from .preprocess_audio import load_audio_stereo
from .preprocess_context import build_context_latents
from .preprocess_encoder import run_encoder
from .preprocess_lyrics import encode_lyrics
from .preprocess_manifest import save_manifest
from .preprocess_text import build_text_prompt, encode_text
from .preprocess_utils import select_genre_indices
from .preprocess_vae import TILED_ENCODE_THRESHOLD_SAMPLES, vae_encode, vae_encode_tiled
from acestep.debug_utils import (
    debug_log_for,
    debug_log_verbose_for,
    debug_start_verbose_for,
    debug_end_verbose_for,
)


class PreprocessMixin:
    """Preprocess labeled samples to tensor files."""

    def preprocess_to_tensors(
        self,
        dit_handler,
        output_dir: str,
        max_duration: float = 240.0,
        progress_callback=None,
        save_grad_norms: bool = False,
    ) -> Tuple[List[str], str]:
        """Preprocess all labeled samples to tensor files for efficient training.

        If save_grad_norms is True, runs one forward+backward per sample and saves
        per-parameter gradient norms in each .pt for use in rank/target selection.
        """
        debug_log_for("dataset", f"preprocess_to_tensors: output_dir='{output_dir}', max_duration={max_duration}")
        if not self.samples:
            return [], "❌ No samples to preprocess"

        labeled_samples = [s for s in self.samples if s.labeled]
        if not labeled_samples:
            return [], "❌ No labeled samples to preprocess"

        if dit_handler is None or dit_handler.model is None:
            return [], "❌ Model not initialized. Please initialize the service first."

        os.makedirs(output_dir, exist_ok=True)

        output_paths: List[str] = []
        success_count = 0
        fail_count = 0

        model = dit_handler.model
        vae = dit_handler.vae
        text_encoder = dit_handler.text_encoder
        text_tokenizer = dit_handler.text_tokenizer
        silence_latent = dit_handler.silence_latent
        device = dit_handler.device
        dtype = dit_handler.dtype

        target_sample_rate = 48000

        genre_indices = select_genre_indices(labeled_samples, self.metadata.genre_ratio)
        debug_log_verbose_for("dataset", f"selected genre indices: count={len(genre_indices)}")

        total_samples = len(labeled_samples)
        start_time = time.time()

        # Keep models on the target device while preprocessing to avoid device mismatches.
        with ExitStack() as stack:
            if hasattr(dit_handler, "_load_model_context"):
                stack.enter_context(dit_handler._load_model_context("vae"))
                stack.enter_context(dit_handler._load_model_context("text_encoder"))
                stack.enter_context(dit_handler._load_model_context("model"))

            for i, sample in enumerate(labeled_samples):
                file_start = time.time()
                try:
                    debug_log_verbose_for("dataset", f"sample[{i}] id={sample.id} file={sample.filename}")
                    if progress_callback:
                        try:
                            progress_callback(i + 1, total_samples, sample.filename, time.time() - start_time, time.time() - file_start)
                        except TypeError:
                            progress_callback(f"Preprocessing {i+1}/{total_samples}: {sample.filename}")

                    use_genre = i in genre_indices

                    t0 = debug_start_verbose_for("dataset", f"load_audio_stereo[{i}]")
                    audio, _ = load_audio_stereo(sample.audio_path, target_sample_rate, max_duration)
                    debug_end_verbose_for("dataset", f"load_audio_stereo[{i}]", t0)
                    debug_log_verbose_for("dataset", f"audio shape={tuple(audio.shape)} dtype={audio.dtype}")
                    audio = audio.unsqueeze(0).to(device).to(vae.dtype)
                    debug_log_verbose_for(
                        "dataset",
                        f"vae device={next(vae.parameters()).device} vae dtype={vae.dtype} "
                        f"audio device={audio.device} audio dtype={audio.dtype}",
                    )

                    with torch.no_grad():
                        t0 = debug_start_verbose_for("dataset", f"vae_encode[{i}]")
                        if audio.shape[-1] > TILED_ENCODE_THRESHOLD_SAMPLES and hasattr(dit_handler, "tiled_encode"):
                            target_latents = vae_encode_tiled(dit_handler, audio, dtype)
                        else:
                            target_latents = vae_encode(vae, audio, dtype)
                        debug_end_verbose_for("dataset", f"vae_encode[{i}]", t0)

                    latent_length = target_latents.shape[1]
                    attention_mask = torch.ones(1, latent_length, device=device, dtype=dtype)
                    debug_log_verbose_for(
                        "dataset",
                        f"target_latents shape={tuple(target_latents.shape)} latent_length={latent_length}",
                    )

                    caption = sample.get_training_prompt(self.metadata.tag_position, use_genre=use_genre)
                    text_prompt = build_text_prompt(sample, self.metadata.tag_position, use_genre)

                    if i == 0:
                        logger.info(f"\n{'='*70}")
                        logger.info("🔍 [DEBUG] DiT TEXT ENCODER INPUT (Training Preprocess)")
                        logger.info(f"{'='*70}")
                        logger.info(f"text_prompt:\n{text_prompt}")
                        logger.info(f"{'='*70}\n")

                    t0 = debug_start_verbose_for("dataset", f"encode_text[{i}]")
                    text_hidden_states, text_attention_mask = encode_text(
                        text_encoder, text_tokenizer, text_prompt, device, dtype
                    )
                    debug_end_verbose_for("dataset", f"encode_text[{i}]", t0)
                    debug_log_verbose_for(
                        "dataset",
                        f"text_hidden_states shape={tuple(text_hidden_states.shape)} "
                        f"text_attention_mask shape={tuple(text_attention_mask.shape)}",
                    )

                    lyrics = sample.lyrics if sample.lyrics else "[Instrumental]"
                    t0 = debug_start_verbose_for("dataset", f"encode_lyrics[{i}]")
                    lyric_hidden_states, lyric_attention_mask = encode_lyrics(
                        text_encoder, text_tokenizer, lyrics, device, dtype
                    )
                    debug_end_verbose_for("dataset", f"encode_lyrics[{i}]", t0)
                    debug_log_verbose_for(
                        "dataset",
                        f"lyric_hidden_states shape={tuple(lyric_hidden_states.shape)} "
                        f"lyric_attention_mask shape={tuple(lyric_attention_mask.shape)}",
                    )

                    t0 = debug_start_verbose_for("dataset", f"run_encoder[{i}]")
                    encoder_hidden_states, encoder_attention_mask = run_encoder(
                        model,
                        text_hidden_states=text_hidden_states,
                        text_attention_mask=text_attention_mask,
                        lyric_hidden_states=lyric_hidden_states,
                        lyric_attention_mask=lyric_attention_mask,
                        device=device,
                        dtype=dtype,
                    )
                    debug_end_verbose_for("dataset", f"run_encoder[{i}]", t0)
                    debug_log_verbose_for(
                        "dataset",
                        f"encoder_hidden_states shape={tuple(encoder_hidden_states.shape)} "
                        f"encoder_attention_mask shape={tuple(encoder_attention_mask.shape)}",
                    )

                    t0 = debug_start_verbose_for("dataset", f"build_context_latents[{i}]")
                    context_latents = build_context_latents(silence_latent, latent_length, device, dtype)
                    debug_end_verbose_for("dataset", f"build_context_latents[{i}]", t0)

                    output_data = {
                        "target_latents": target_latents.squeeze(0).cpu(),
                        "attention_mask": attention_mask.squeeze(0).cpu(),
                        "encoder_hidden_states": encoder_hidden_states.squeeze(0).cpu(),
                        "encoder_attention_mask": encoder_attention_mask.squeeze(0).cpu(),
                        "context_latents": context_latents.squeeze(0).cpu(),
                        "metadata": {
                            "audio_path": sample.audio_path,
                            "filename": sample.filename,
                            "caption": caption,
                            "lyrics": lyrics,
                            "duration": sample.duration,
                            "bpm": sample.bpm,
                            "keyscale": sample.keyscale,
                            "timesignature": sample.timesignature,
                            "language": sample.language,
                            "is_instrumental": sample.is_instrumental,
                        },
                    }
                    if save_grad_norms:
                        try:
                            batch_tensors = {
                                "target_latents": target_latents.squeeze(0),
                                "attention_mask": attention_mask.squeeze(0),
                                "encoder_hidden_states": encoder_hidden_states.squeeze(0),
                                "encoder_attention_mask": encoder_attention_mask.squeeze(0),
                                "context_latents": context_latents.squeeze(0),
                            }
                            output_data["grad_norms"] = compute_decoder_grad_norms_for_sample(
                                model, batch_tensors, device, dtype
                            )
                        except Exception as e:
                            logger.warning(f"Failed to compute grad norms for {sample.filename}: {e}")

                    output_path = os.path.join(output_dir, f"{sample.id}.pt")
                    t0 = debug_start_verbose_for("dataset", f"torch.save[{i}]")
                    torch.save(output_data, output_path)
                    debug_end_verbose_for("dataset", f"torch.save[{i}]", t0)
                    output_paths.append(output_path)
                    success_count += 1

                except Exception as e:
                    logger.exception(f"Error preprocessing {sample.filename}")
                    fail_count += 1
                    if progress_callback:
                        progress_callback(f"❌ Failed: {sample.filename}: {str(e)}")

        t0 = debug_start_verbose_for("dataset", "save_manifest")
        save_manifest(output_dir, self.metadata, output_paths)
        debug_end_verbose_for("dataset", "save_manifest", t0)

        status = f"✅ Preprocessed {success_count}/{len(labeled_samples)} samples to {output_dir}"
        if fail_count > 0:
            status += f" ({fail_count} failed)"

        return output_paths, status

    def preprocess_to_tensors_two_pass(
        self,
        dit_handler,
        output_dir: str,
        max_duration: float = 240.0,
        progress_callback=None,
        save_grad_norms: bool = False,
    ) -> Tuple[List[str], str]:
        """Preprocess in two passes to reduce peak VRAM.
        Pass 1: VAE + Text Encoder (~3 GB) → save target_latents and text/lyric hidden states.
        Pass 2: DiT encoder only (~6 GB) → run encoder, build context_latents, write final .pt.
        If save_grad_norms is True, Pass 2 also computes per-parameter gradient norms and saves them in each .pt.
        """
        debug_log_for("dataset", f"preprocess_to_tensors_two_pass: output_dir='{output_dir}'")
        if not self.samples:
            return [], "❌ No samples to preprocess"
        labeled_samples = [s for s in self.samples if s.labeled]
        if not labeled_samples:
            return [], "❌ No labeled samples to preprocess"
        if dit_handler is None or dit_handler.model is None:
            return [], "❌ Model not initialized. Please initialize the service first."
        if not hasattr(dit_handler, "_load_model_context"):
            return [], "❌ Handler does not support _load_model_context (two-pass requires offload)."

        os.makedirs(output_dir, exist_ok=True)
        device = dit_handler.device
        dtype = dit_handler.dtype
        vae = dit_handler.vae
        text_encoder = dit_handler.text_encoder
        text_tokenizer = dit_handler.text_tokenizer
        model = dit_handler.model
        silence_latent = dit_handler.silence_latent
        if silence_latent is None:
            return [], "❌ silence_latent not loaded (initialize service with checkpoint first)."

        target_sample_rate = 48000
        genre_indices = select_genre_indices(labeled_samples, self.metadata.genre_ratio)
        total_samples = len(labeled_samples)
        start_time = time.time()
        output_paths: List[str] = []
        success_count = 0
        fail_count = 0

        # -------- Pass 1: VAE + Text Encoder only --------
        with dit_handler._load_model_context("vae"), dit_handler._load_model_context("text_encoder"):
            for i, sample in enumerate(labeled_samples):
                file_start = time.time()
                try:
                    if progress_callback:
                        try:
                            progress_callback(i + 1, total_samples, f"[Pass 1] {sample.filename}", time.time() - start_time, time.time() - file_start)
                        except TypeError:
                            progress_callback(f"Pass 1: {i+1}/{total_samples}: {sample.filename}")

                    use_genre = i in genre_indices
                    audio, _ = load_audio_stereo(sample.audio_path, target_sample_rate, max_duration)
                    audio = audio.unsqueeze(0).to(device).to(vae.dtype)

                    with torch.no_grad():
                        if audio.shape[-1] > TILED_ENCODE_THRESHOLD_SAMPLES and hasattr(dit_handler, "tiled_encode"):
                            target_latents = vae_encode_tiled(dit_handler, audio, dtype)
                        else:
                            target_latents = vae_encode(vae, audio, dtype)
                    latent_length = target_latents.shape[1]
                    attention_mask = torch.ones(1, latent_length, device=device, dtype=dtype)

                    text_prompt = build_text_prompt(sample, self.metadata.tag_position, use_genre)
                    text_hidden_states, text_attention_mask = encode_text(
                        text_encoder, text_tokenizer, text_prompt, device, dtype
                    )
                    lyrics = sample.lyrics if sample.lyrics else "[Instrumental]"
                    lyric_hidden_states, lyric_attention_mask = encode_lyrics(
                        text_encoder, text_tokenizer, lyrics, device, dtype
                    )
                    caption = sample.get_training_prompt(self.metadata.tag_position, use_genre=use_genre)

                    pass1_data = {
                        "target_latents": target_latents.squeeze(0).cpu(),
                        "attention_mask": attention_mask.squeeze(0).cpu(),
                        "text_hidden_states": text_hidden_states.cpu(),
                        "text_attention_mask": text_attention_mask.cpu(),
                        "lyric_hidden_states": lyric_hidden_states.cpu(),
                        "lyric_attention_mask": lyric_attention_mask.cpu(),
                        "latent_length": int(latent_length),
                        "metadata": {
                            "audio_path": sample.audio_path,
                            "filename": sample.filename,
                            "caption": caption,
                            "lyrics": lyrics,
                            "duration": sample.duration,
                            "bpm": sample.bpm,
                            "keyscale": sample.keyscale,
                            "timesignature": sample.timesignature,
                            "language": sample.language,
                            "is_instrumental": sample.is_instrumental,
                        },
                    }
                    out_path = os.path.join(output_dir, f"{sample.id}.pt")
                    torch.save(pass1_data, out_path)
                    output_paths.append(out_path)
                    success_count += 1
                except Exception as e:
                    logger.exception(f"Pass 1 failed: {sample.filename}")
                    fail_count += 1
                    if progress_callback:
                        try:
                            progress_callback(i + 1, total_samples, f"❌ Pass 1 failed: {sample.filename}: {str(e)}", time.time() - start_time, time.time() - file_start)
                        except TypeError:
                            progress_callback(f"Pass 1 failed: {sample.filename}")

        # -------- Pass 2: DiT encoder only --------
        success_count = 0
        final_output_paths: List[str] = []
        with dit_handler._load_model_context("model"):
            for i, sample in enumerate(labeled_samples):
                file_start = time.time()
                try:
                    if progress_callback:
                        try:
                            progress_callback(i + 1, total_samples, f"[Pass 2] {sample.filename}", time.time() - start_time, time.time() - file_start)
                        except TypeError:
                            progress_callback(f"Pass 2: {i+1}/{total_samples}: {sample.filename}")

                    out_path = os.path.join(output_dir, f"{sample.id}.pt")
                    try:
                        pass1_data = torch.load(out_path, map_location="cpu", weights_only=False)
                    except TypeError:
                        pass1_data = torch.load(out_path, map_location="cpu")
                    # Pass 1 saved tensors already with batch dim [1, seq_len, ...]; do not unsqueeze again
                    text_hidden_states = pass1_data["text_hidden_states"].to(device).to(dtype)
                    text_attention_mask = pass1_data["text_attention_mask"].to(device).to(dtype)
                    lyric_hidden_states = pass1_data["lyric_hidden_states"].to(device).to(dtype)
                    lyric_attention_mask = pass1_data["lyric_attention_mask"].to(device).to(dtype)
                    latent_length = int(pass1_data["latent_length"])

                    encoder_hidden_states, encoder_attention_mask = run_encoder(
                        model,
                        text_hidden_states=text_hidden_states,
                        text_attention_mask=text_attention_mask,
                        lyric_hidden_states=lyric_hidden_states,
                        lyric_attention_mask=lyric_attention_mask,
                        device=device,
                        dtype=dtype,
                    )
                    context_latents = build_context_latents(silence_latent, latent_length, device, dtype)

                    output_data = {
                        "target_latents": pass1_data["target_latents"],
                        "attention_mask": pass1_data["attention_mask"],
                        "encoder_hidden_states": encoder_hidden_states.squeeze(0).cpu(),
                        "encoder_attention_mask": encoder_attention_mask.squeeze(0).cpu(),
                        "context_latents": context_latents.squeeze(0).cpu(),
                        "metadata": pass1_data["metadata"],
                    }
                    if save_grad_norms:
                        try:
                            batch_tensors = {
                                "target_latents": pass1_data["target_latents"].to(device).to(dtype),
                                "attention_mask": pass1_data["attention_mask"].to(device).to(dtype),
                                "encoder_hidden_states": encoder_hidden_states.squeeze(0),
                                "encoder_attention_mask": encoder_attention_mask.squeeze(0),
                                "context_latents": context_latents.squeeze(0),
                            }
                            output_data["grad_norms"] = compute_decoder_grad_norms_for_sample(
                                model, batch_tensors, device, dtype
                            )
                        except Exception as e:
                            logger.warning(f"Failed to compute grad norms for {sample.filename}: {e}")
                    torch.save(output_data, out_path)
                    success_count += 1
                    final_output_paths.append(out_path)
                except Exception as e:
                    logger.exception(f"Pass 2 failed: {sample.filename}")
                    fail_count += 1
                    if progress_callback:
                        try:
                            progress_callback(i + 1, total_samples, f"❌ Pass 2 failed: {sample.filename}: {str(e)}", time.time() - start_time, time.time() - file_start)
                        except TypeError:
                            progress_callback(f"Pass 2 failed: {sample.filename}")

        save_manifest(output_dir, self.metadata, final_output_paths)
        status = f"✅ Two-pass preprocessed {success_count}/{len(labeled_samples)} samples to {output_dir}"
        if fail_count > 0:
            status += f" ({fail_count} failed)"
        return output_paths, status
