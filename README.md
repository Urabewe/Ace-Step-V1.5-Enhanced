# ACE-Step 1.5

An enhanced and mostly bug fixed version of Ace-Step V1.5 Gradio UI. Includes all features including new exposed paramters, QOL enhancements and the ability to pause and resume training sessions.

Original repo here: (Ace-Step V1.5)[https://github.com/ace-step/ACE-Step-1.5]

## Quick start

Windows:
- Run `install_Ace-Step_WIN.bat` once.
- Start the UI with `start_gradio_ui.bat`.
- Start the API with `start_api_server.bat`.

Linux or macOS:
- Run `install_Ace-Step_LINUX.sh` or `install_Ace-Step_MAC.sh`.
- Start the UI with `python -m acestep.acestep_v15_pipeline`.

## Enhancements in this fork

### Training

- **Pause/resume training** with saved optimizer/scheduler state and elapsed time.
- **Optimizer resume by parameter name**: when the full optimizer state doesn’t match (e.g. you changed LoRA/LoKR rank), the trainer restores optimizer state from the checkpoint by parameter name so you can pause, change config, and resume without losing momentum. Works for both LoRA and LoKR (AdamW and AdamW8bit).
- **Precomputed gradient norms**: optional “Save grad norms (preprocess)” during dataset preprocessing runs one forward+backward per sample and stores per-parameter gradient norms in each `.pt` file.
- **Grad-norm target selection**: at training start, when preprocessed data includes grad norms, the trainer aggregates them and selects the top-k LoRA target module types (e.g. q_proj, k_proj, v_proj, o_proj) by mean gradient norm. Fewer trainable parameters for the same quality, so faster steps and lower VRAM. Config: “Use grad norms for target selection” and “Grad norm top-k modules” in the Train LoRA tab.
- **Grad-norm sample weighting**: when batches include precomputed grad norms, the loss is weighted by batch importance for faster convergence. Config: “Use grad norms for sample weighting”.
- **Two-pass preprocessing (low VRAM)**: optional “Low VRAM (two-pass)” preprocessing uses Pass 1 (VAE + text encoder) then Pass 2 (DiT encoder only) to reduce peak VRAM when building tensor datasets.
- **Tiled VAE** for long clips during preprocessing to limit VRAM and speed up encoding where applicable.
- **Training preprocessing progress** with per-file timing and ETA.
- **Side-Step-style options**: continuous (logit-normal) timestep sampling, CFG dropout, and optional per-parameter gradient norm logging to TensorBoard for analysis.
- **LyCORIS-based LoRA/LoKR training** with single-file `.safetensors` outputs.
- **LoKR training support** integrated alongside LoRA, with pause/resume checkpoints and device-safe resume.
- **Trainer stability helpers**: non-finite gradient diagnostics, MPS fp32 safety, and optional bitsandbytes 8-bit Adam.

### Generation & outputs

- **Autosave outputs** toggle with persistent settings and date-stamped folders.
- **MP3/FLAC metadata** embedded directly in files; no sidecar JSONs.
- **Source audio cache** toggle to avoid redundant encoding.
- **Faster source/reference encoding** when VRAM allows (larger chunks, no CPU offload).
- **UI responsiveness** improvements when batching generations.
- **Date/time filenames** for all audio outputs (MP3/FLAC/WAV/etc.) to keep files ordered.
- **LoRA/LoKR loading** accepts PEFT folders and LyCORIS `.safetensors` files; renamed weights are supported.

### Runtime & install

- **Preprocessed-tensor training** keeps only the DiT decoder on GPU; VAE and text/condition encoders are unloaded to free VRAM.
- **Cross-platform installers** that auto-detect Python/CUDA and install torch 2.9.1.
- **Optional `flash-attn` install** that never blocks the main setup flow.
- **vLLM Windows fixes** to avoid `torch.distributed`/Gloo hostname errors and improve stability.
- **LM memory safety**: caps vLLM VRAM usage based on free VRAM and cleans up on re-init to reduce OOM risk.

## Requirements

- Python 3.10+ for the installers.
- CUDA 11.8–13 on Windows/Linux for GPU builds (CPU and Apple Silicon are supported on macOS).

## Docs

- `docs/en/GRADIO_GUIDE.md` for the UI
- `docs/en/API.md` and `docs/en/Openrouter_API_DOC.md` for API usage
- `docs/en/CLI.md` for command line usage
- `docs/en/INFERENCE.md` and `docs/en/Tutorial.md` for workflows
- `docs/en/GPU_COMPATIBILITY.md` for hardware notes

## Project layout

- `acestep/` core package
- `checkpoints/` model weights
- `datasets/` training data and preprocessed tensors
- `lora_output/` training outputs and checkpoints
- `gradio_outputs/` generated audio outputs

## License

MIT. See `LICENSE`.
