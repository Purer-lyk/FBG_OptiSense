# -*- mode: python ; coding: utf-8 -*-
from pathlib import Path
import json
import runpy
import sys

from PyInstaller.utils.hooks import collect_submodules

repository = Path(SPECPATH).resolve().parents[1]
python_dir = repository / 'Python'
sys.path.insert(0, str(repository / 'deployment'))
from beta_client_resources import beta_client_datas

identity = runpy.run_path(str(python_dir / 'app_version.py'))
if identity['APP_RELEASE_CHANNEL'] != 'beta-offline':
    raise ValueError('Beta client requires an explicit beta-offline release identity')

hiddenimports = ['PyQt5.sip', 'OpenGL.platform.win32', 'matplotlib.backends.backend_agg']
hiddenimports += collect_submodules('sklearn')
a = Analysis(
    [str(python_dir / 'launch_fbg_studio_beta.pyw')],
    pathex=[str(python_dir)], binaries=[], datas=beta_client_datas(repository),
    hiddenimports=hiddenimports, hookspath=[], hooksconfig={}, runtime_hooks=[],
    excludes=['PySide2', 'PySide6', 'PyQt6', 'tkinter', 'pyocd'],
    noarchive=False, optimize=0,
)
pyz = PYZ(a.pure)
exe = EXE(
    pyz, a.scripts, [], exclude_binaries=True, name='FBG_OptiSense_Studio',
    debug=False, bootloader_ignore_signals=False, strip=False, upx=False,
    console=False, disable_windowed_traceback=False, argv_emulation=False,
    target_arch=None, codesign_identity=None, entitlements_file=None,
    icon=str(Path(SPECPATH) / 'fbg_optisense_studio.ico'),
    version=str(Path(SPECPATH) / 'version_info.txt'),
)
coll = COLLECT(exe, a.binaries, a.datas, strip=False, upx=False,
               upx_exclude=[], name='app_beta_build')
(Path(DISTPATH) / 'app_beta_build' / 'client_release.json').write_text(
    json.dumps({'version': identity['APP_VERSION'], 'channel': 'beta-offline',
                'beta': True, 'firmware_version': '1.0.118'}, indent=2),
    encoding='utf-8',
)
