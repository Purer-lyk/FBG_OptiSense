"""One executable containing exactly the signed Beta client and USB firmware.

Set FBG_OFFLINE_BUNDLE_DIR to a fresh directory from prepare_offline_release.py.
Only explicit manifest payload files enter the executable, never an entire
developer directory or signing key.
"""
import json
import os
from pathlib import Path
import sys

repository = Path(SPECPATH).resolve().parents[1]
python_dir = repository/'Python'
bundle_setting = os.environ.get('FBG_OFFLINE_BUNDLE_DIR', '').strip()
if not bundle_setting:
    raise RuntimeError('Set FBG_OFFLINE_BUNDLE_DIR to the signed Beta delivery directory')
bundle_dir = Path(bundle_setting).resolve()
sys.path.insert(0, str(python_dir))
from offline_update_helper import verify_bundle

verification = verify_bundle(bundle_dir)
manifest = json.loads((bundle_dir/'update_manifest.json').read_text(encoding='utf-8'))
names = ['update_manifest.json', manifest['client']['filename']]
names.extend(item['filename'] for item in manifest['firmware'])
if len(names) != len(set(names)):
    raise RuntimeError('Signed manifest contains overlapping payload names')
datas = [(str(bundle_dir/name), 'offline_payload') for name in names]

a = Analysis([str(python_dir/'offline_update_helper.py')], pathex=[str(python_dir)],
             datas=datas, binaries=[], hiddenimports=['PyQt5.sip'], hookspath=[],
             hooksconfig={}, runtime_hooks=[],
             excludes=['PySide6','PySide2','PyQt6','tkinter','matplotlib','numpy','pyqtgraph','sklearn'],
             noarchive=False,optimize=0)
pyz = PYZ(a.pure)
exe = EXE(pyz,a.scripts,a.binaries,a.datas,[],name='OptiSense_Beta_Update',debug=False,
          bootloader_ignore_signals=False,strip=False,upx=False,console=False,
          disable_windowed_traceback=False,
          icon=str(Path(SPECPATH)/'fbg_optisense_studio.ico'))
