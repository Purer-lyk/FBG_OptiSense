"""Initialize the active PyQt5 plugin paths before creating QApplication.

On this Windows host QLibraryInfo's compiled-in path loses Chinese characters.
Use the installed package's actual Unicode path; do not change rendering modes.
Standalone UI probes must call configure_qt() before constructing QApplication.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


def configure_qt() -> Path | None:
    candidates = []
    if getattr(sys, 'frozen', False):
        bundle = Path(getattr(sys, '_MEIPASS', Path(sys.executable).parent))
        candidates.extend((bundle/'PyQt5'/'Qt5'/'plugins', bundle/'PyQt5'/'Qt'/'plugins'))
    spec = importlib.util.find_spec('PyQt5')
    if spec is not None and spec.submodule_search_locations:
        for directory in spec.submodule_search_locations:
            candidates.extend((Path(directory)/'Qt5'/'plugins', Path(directory)/'Qt'/'plugins'))
    for root in candidates:
        if not (root/'platforms').is_dir():
            continue
        root = root.resolve()
        os.environ['QT_PLUGIN_PATH'] = str(root)
        os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(root/'platforms')
        from PyQt5.QtCore import QCoreApplication
        QCoreApplication.setLibraryPaths([str(root)])
        return root
    return None
