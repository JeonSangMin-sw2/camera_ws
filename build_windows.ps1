# ===================================================
#   Camera Calibrator - Windows PowerShell Build Script
# ===================================================

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "   Camera Calibrator - Windows Standalone Build   " -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

# 0. Kill running instances
Write-Host "[*] Closing any running camera_calibrator instances..." -ForegroundColor Yellow
Get-Process -Name "camera_calibrator*" -ErrorAction SilentlyContinue | Stop-Process -Force

# 1. Find suitable Python (prefer 3.11, 3.10, 3.12)
$pyExe = "python"
$pyArgs = @()

foreach ($ver in @("3.11", "3.10", "3.12")) {
    $test = & py -$ver --version 2>$null
    if ($LASTEXITCODE -eq 0) {
        $pyExe = "py"
        $pyArgs = @("-$ver")
        Write-Host "[*] Found supported Python via py launcher: $test" -ForegroundColor Green
        break
    }
}

if ($pyExe -eq "python" -and $pyArgs.Count -eq 0) {
    try {
        $pythonVersion = python --version 2>&1
        Write-Host "[*] Found Python: $pythonVersion" -ForegroundColor Green
    } catch {
        Write-Host "[ERROR] Python 3.10, 3.11, or 3.12 is not installed or not in PATH!" -ForegroundColor Red
        Write-Host "Please install Python 3.11 and add it to PATH."
        Exit 1
    }
}

# 2. Virtual environment setup
if (-not (Test-Path ".venv\Scripts\Activate.ps1")) {
    Write-Host "[1/4] Creating virtual environment (.venv)..." -ForegroundColor Yellow
    if ($pyArgs.Count -gt 0) {
        & $pyExe @pyArgs -m venv .venv
    } else {
        & $pyExe -m venv .venv
    }
}


Write-Host "[2/4] Activating virtual environment..." -ForegroundColor Yellow
& .\.venv\Scripts\Activate.ps1

# 3. Install Dependencies
Write-Host "[3/4] Installing dependencies from requirements.txt..." -ForegroundColor Yellow
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install pyinstaller
pip install --no-deps -e .

# 4. PyInstaller Build
Write-Host "[4/4] Building standalone executable with PyInstaller..." -ForegroundColor Yellow
pyinstaller --clean --noconfirm camera_calibrator.spec

Write-Host ""
Write-Host "===================================================" -ForegroundColor Green
Write-Host "   Build Successful! (Windows)                     " -ForegroundColor Green
Write-Host "   Executable available in: .\dist\               " -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Green

