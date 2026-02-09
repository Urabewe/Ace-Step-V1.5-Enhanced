import math
import os
import shutil
import subprocess
import tempfile

import numpy as np
import soundfile as sf
import torch
from scipy import signal


def _read_with_soundfile(path: str):
    data, sr = sf.read(path, always_2d=True, dtype="float32")
    audio = torch.from_numpy(data).transpose(0, 1)  # [channels, frames]
    return audio, sr


def _resample_audio(audio: torch.Tensor, orig_sr: int, target_sr: int) -> torch.Tensor:
    if orig_sr == target_sr:
        return audio
    audio_np = audio.cpu().numpy().astype(np.float32)
    g = math.gcd(orig_sr, target_sr)
    up = target_sr // g
    down = orig_sr // g
    resampled = []
    for ch in audio_np:
        resampled.append(signal.resample_poly(ch, up, down).astype(np.float32))
    resampled = np.stack(resampled, axis=0)
    return torch.from_numpy(resampled)


def _resolve_audio_path(audio_path: str) -> str:
    if os.path.exists(audio_path):
        return audio_path
    # Try to remap to current project root if path was saved on another machine/dir
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    if "datasets" in audio_path:
        rel = audio_path.split("datasets", 1)[-1].lstrip("\\/")  # keep subpath under datasets
        candidate = os.path.join(project_root, "datasets", rel)
        if os.path.exists(candidate):
            return candidate
    return audio_path


def load_audio_stereo(audio_path: str, target_sample_rate: int, max_duration: float):
    """Load audio, resample, convert to stereo, and truncate."""
    audio_path = _resolve_audio_path(str(audio_path))
    ext = os.path.splitext(audio_path)[1].lower()
    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path and ext in {".mp3", ".m4a", ".aac", ".ogg", ".opus", ".wma"}:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                [ffmpeg_path, "-y", "-i", str(audio_path), "-ac", "2", "-ar", str(target_sample_rate), tmp_path],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"ffmpeg failed ({result.returncode}): {result.stderr.strip()}")
            audio, sr = _read_with_soundfile(tmp_path)
        finally:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    else:
        audio, sr = _read_with_soundfile(str(audio_path))

    audio = audio.to(torch.float32)
    if sr != target_sample_rate:
        audio = _resample_audio(audio, sr, target_sample_rate)
        sr = target_sample_rate

    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2, :]

    max_samples = int(max_duration * target_sample_rate)
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]

    return audio, sr
