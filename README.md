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

- Pause/resume training with saved optimizer/scheduler state and elapsed time.
- Autosave outputs toggle with persistent settings and date-stamped folders.
- MP3/FLAC metadata embedded directly in files; no sidecar JSONs.
- Source audio cache toggle to avoid redundant encoding.
- UI responsiveness improvements when batching generations.
- Training preprocessing progress with per-file timing and ETA.
- Cross-platform installers that auto-detect Python/CUDA and install torch 2.9.1.
- Optional `flash-attn` install that never blocks the main setup flow.

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
