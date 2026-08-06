# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main_ui.py'],
    pathex=['core'],
    binaries=[],
    datas=[
        ('config/i18n.yaml', 'config'),
        ('config/setting.yaml', 'config'),
        ('config/ready_poses.yaml', 'config'),
        ('config/camera_intrinsics.yaml', 'config'),
        ('img/*', 'img')
    ],
    hiddenimports=['osqp.ext_builtin'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

import platform
arch = platform.machine()  # 'x86_64' 또는 Jetson의 'aarch64'
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
