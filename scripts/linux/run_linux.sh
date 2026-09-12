#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

echo "==================================================="
echo "   Camera Calibrator - Linux Python Launcher"
echo "==================================================="

if [ -f ".venv/bin/activate" ]; then
    echo "[*] Activating virtual environment (.venv)..."
    source .venv/bin/activate
else
    echo "[!] Warning: .venv not found. Using system Python3..."
fi

if [ -z "$1" ]; then
    echo "[*] Starting main_ui.py..."
    echo "[*] Tip: Pass '--ui' for simulation / UI-only mode."
    python3 main_ui.py
else
    echo "[*] Starting main_ui.py with args: $@"
    python3 main_ui.py "$@"
fi
