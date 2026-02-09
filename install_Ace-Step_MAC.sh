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
echo "ACE-Step Venv Setup (macOS)"
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

echo ""
echo "[Setup] Installing torch ${TORCH_VERSION} (macOS wheels)..."
"${PYTHON_EXE}" -m pip install \
  "torch==${TORCH_VERSION}" \
  "torchvision==${TORCHVISION_VERSION}" \
  "torchaudio==${TORCHAUDIO_VERSION}"

echo ""
echo "[Setup] flash-attn not installed on macOS (no official wheels)."

echo ""
echo "[Setup] Installing remaining dependencies..."
"${PYTHON_EXE}" -m pip install -r "${ROOT_DIR}/requirements-base.txt"

echo ""
echo "========================================"
echo "Setup complete! You can now run:"
echo "  ./start_gradio_ui.sh (if you have one)"
echo "========================================"
echo ""
