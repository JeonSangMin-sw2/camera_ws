#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "==================================================="
echo "   Camera Calibrator - Linux Standalone Build"
echo "==================================================="

# 1. Virtual environment setup
if [ ! -f ".venv/bin/activate" ]; then
    echo "[1/4] Creating virtual environment (.venv)..."
    python3 -m venv .venv
fi

echo "[2/4] Activating virtual environment..."
source .venv/bin/activate

# 2. Dependencies
echo "[3/4] Installing dependencies..."
python3 -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller
pip install -e .

# 3. Build with PyInstaller
echo "[4/4] Building with PyInstaller..."
pyinstaller --clean --noconfirm camera_calibrator.spec

echo ""
echo "==================================================="
echo "   Build Successful! (Linux)"
echo "   Output binary: dist/camera_calibrator_$(uname -m)"
echo "==================================================="
