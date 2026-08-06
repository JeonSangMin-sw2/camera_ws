# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['main_ui.py'],
    pathex=['core'],
    binaries=[],
    datas=[
        ('config/i18n.yaml', 'config'),
        ('config/setting.yaml', 'config'),
        ('config/ready_poses.yaml', 'config'),
        ('config/camera_intrinsics.yaml', 'config')
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

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='camera_calibrator',
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
