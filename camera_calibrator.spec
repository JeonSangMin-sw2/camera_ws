# -*- mode: python ; coding: utf-8 -*-
from PyInstaller.utils.hooks import collect_all
import platform

datas_rby, binaries_rby, hiddenimports_rby = collect_all('rby1_sdk')

datas = [
    ('config', 'config'),
    ('img', 'img'),
] + datas_rby

binaries = [] + binaries_rby
hiddenimports = [
    'osqp.ext_builtin',
    'mpl_toolkits.mplot3d',
    'mpl_toolkits.mplot3d.axes3d',
    'rby1_sdk',
    'rby1_sdk._bindings',
    'rby1_sdk._robot_command',
    'rby1_sdk.dynamics',
    'rby1_sdk.math',
    'rby1_sdk.upc',
] + hiddenimports_rby

a = Analysis(
    ['main_ui.py'],
    pathex=['core'],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

arch = platform.machine()  # 'x86_64' (Linux) or 'AMD64' (Windows) or 'aarch64' (Jetson)
exe_name = f'camera_calibrator_{arch}'

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name=exe_name,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

