#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${ROOT_DIR}/.venv"
PYTHON_EXE=""
TORCH_VERSION="2.9.1"
TORCHVISION_VERSION="0.24.1"
TORCHAUDIO_VERSION="2.9.1"
FLASH_ATTN_VERSION="2.8.3"
FLASH_ATTN_RELEASE="v0.7.11"

echo ""
echo "========================================"
echo "ACE-Step Venv Setup (Linux)"
echo "========================================"
echo ""

if [[ -x "${VENV_DIR}/bin/python" ]]; then
  echo "[Setup] .venv already exists."
  PYTHON_EXE="${VENV_DIR}/bin/python"
else
  echo "[Setup] Creating .venv..."
  if command -v python3 >/dev/null 2>&1; then
    python3 -m venv "${VENV_DIR}"
  else
    echo "[Error] python3 not found. Please install Python 3.10+ and retry."
    exit 1
  fi

  if [[ ! -x "${VENV_DIR}/bin/python" ]]; then
    echo "[Error] Failed to create .venv."
    exit 1
  fi
  PYTHON_EXE="${VENV_DIR}/bin/python"
fi

echo ""
echo "[Setup] Upgrading pip..."
"${PYTHON_EXE}" -m pip install --upgrade pip

CUDA_VERSION=""
if command -v nvidia-smi >/dev/null 2>&1; then
  CUDA_VERSION=$(nvidia-smi | grep -i "CUDA Version" | awk -F ":" '{print $2}' | awk '{print $1}' | head -n 1 || true)
fi

TORCH_CUDA="cpu"
if [[ -n "${CUDA_VERSION}" ]]; then
  CUDA_MAJOR="${CUDA_VERSION%%.*}"
  CUDA_MINOR="${CUDA_VERSION#*.}"
  CUDA_MINOR="${CUDA_MINOR%%.*}"

  if [[ "${CUDA_MAJOR}" == "11" ]]; then
    TORCH_CUDA="cu118"
  elif [[ "${CUDA_MAJOR}" == "12" ]]; then
    if [[ -z "${CUDA_MINOR}" ]]; then
      TORCH_CUDA="cu121"
    elif [[ "${CUDA_MINOR}" -lt 4 ]]; then
      TORCH_CUDA="cu121"
    elif [[ "${CUDA_MINOR}" -lt 6 ]]; then
      TORCH_CUDA="cu124"
    elif [[ "${CUDA_MINOR}" -lt 8 ]]; then
      TORCH_CUDA="cu126"
    else
      TORCH_CUDA="cu128"
    fi
  elif [[ "${CUDA_MAJOR}" -ge 13 ]]; then
    echo "[Warning] CUDA ${CUDA_VERSION} detected; using cu128 wheels for torch 2.9.1."
    TORCH_CUDA="cu128"
  else
    echo "[Warning] CUDA ${CUDA_VERSION} not supported; using CPU wheels."
    TORCH_CUDA="cpu"
  fi
else
  echo "[Info] nvidia-smi not found or no CUDA detected; using CPU wheels."
fi

if [[ "${TORCH_CUDA}" == "cpu" ]]; then
  TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
  TORCH_TAG="+cpu"
else
  TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA}"
  TORCH_TAG="+${TORCH_CUDA}"
fi

echo ""
echo "[Setup] Installing torch ${TORCH_VERSION}${TORCH_TAG} (${TORCH_CUDA})..."
"${PYTHON_EXE}" -m pip install \
  "torch==${TORCH_VERSION}${TORCH_TAG}" \
  "torchvision==${TORCHVISION_VERSION}${TORCH_TAG}" \
  "torchaudio==${TORCHAUDIO_VERSION}${TORCH_TAG}" \
  --index-url "${TORCH_INDEX_URL}"

echo ""
echo "[Setup] Installing flash-attn wheel (if available)..."
PY_TAG="$("${PYTHON_EXE}" -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")"
if [[ "${TORCH_CUDA}" == "cu128" ]]; then
  FLASH_ATTN_URL="https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/${FLASH_ATTN_RELEASE}/flash_attn-${FLASH_ATTN_VERSION}%2Bcu128torch2.9-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
  if ! "${PYTHON_EXE}" -m pip install "${FLASH_ATTN_URL}"; then
    echo "[Warning] flash-attn install failed; continuing without it."
  fi
else
  echo "[Info] flash-attn skipped (requires cu128 torch wheels)."
fi

echo ""
echo "[Setup] Installing remaining dependencies..."
"${PYTHON_EXE}" -m pip install -r "${ROOT_DIR}/requirements-base.txt"

echo ""
echo "========================================"
echo "Setup complete! You can now run:"
echo "  ./start_gradio_ui.sh (if you have one)"
echo "========================================"
echo ""
