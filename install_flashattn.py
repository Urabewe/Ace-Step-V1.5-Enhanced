import sys
import subprocess


FLASH_ATTN_VERSION = "2.8.3"
FLASH_ATTN_RELEASE = "v0.7.11"


def main() -> int:
    try:
        if sys.platform != "win32":
            print("[flash-attn] Skipped (Windows-only wheel).")
            return 0

        try:
            import torch
        except Exception as exc:
            print(f"[flash-attn] Skipped (torch not available): {exc}")
            return 0

        cuda_version = torch.version.cuda or ""
        if not cuda_version.startswith("12.8"):
            print("[flash-attn] Skipped (no matching wheel found).")
            return 0

        py_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
        
        # Special cases for specific Python versions with CUDA 12.8
        if py_tag == "cp312":
            url = (
                "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/"
                "v0.7.6/"
                "flash_attn-2.8.3+cu128torch2.9-cp312-cp312-win_amd64.whl"
            )
        elif py_tag == "cp313":
            url = (
                "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/"
                "v0.7.13/"
                "flash_attn-2.8.3+cu128torch2.9-cp313-cp313-win_amd64.whl"
            )
        else:
            torch_version = torch.__version__.split("+", 1)[0]
            torch_major_minor = ".".join(torch_version.split(".")[:2])
            torch_tag = f"torch{torch_major_minor}"
            cuda_tag = "cu128"
            url = (
                "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/"
                f"{FLASH_ATTN_RELEASE}/"
                f"flash_attn-{FLASH_ATTN_VERSION}+{cuda_tag}{torch_tag}-"
                f"{py_tag}-{py_tag}-win_amd64.whl"
            )
        print(f"[flash-attn] Installing from: {url}")
        result = subprocess.run([sys.executable, "-m", "pip", "install", url])
        return result.returncode
    except Exception as exc:
        print(f"[flash-attn] Install failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
