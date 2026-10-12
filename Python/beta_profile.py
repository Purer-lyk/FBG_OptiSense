"""Explicit Beta profile, with separate per-user data in frozen builds."""

import os
from pathlib import Path
import shutil
import sys

_allow_default_frozen_migration = False


def beta_enabled():
    return os.environ.get('FBG_STUDIO_BETA', '') == '1'


def enable_beta():
    """Call before importing launcher/acquisition modules and their paths."""
    global _allow_default_frozen_migration
    explicit_root = bool(os.environ.get('FBG_STUDIO_DATA_DIR', '').strip())
    _allow_default_frozen_migration = bool(getattr(sys, 'frozen', False)) and not explicit_root
    os.environ['FBG_STUDIO_BETA'] = '1'
    if not explicit_root:
        if getattr(sys, 'frozen', False):
            base = Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData' / 'Local')
            root = base / 'FBGOptiSenseStudioBeta'
        else:
            root = Path(__file__).resolve().parent / 'beta_data'
        os.environ['FBG_STUDIO_DATA_DIR'] = str(root)


def seed_beta_data():
    """Copy missing local records once; never overwrite or ship user data.

    A packaged Beta migrates only from the same user's formal AppData folder.
    An explicit data-root override remains isolated and does not trigger that
    migration. Source-checkout Beta keeps its original engineering seeding.
    """
    if not beta_enabled():
        return
    destination = Path(os.environ['FBG_STUDIO_DATA_DIR']).resolve()
    if getattr(sys, 'frozen', False):
        if not _allow_default_frozen_migration:
            return
        base = Path(os.environ.get('LOCALAPPDATA') or Path.home() / 'AppData' / 'Local')
        if destination != (base / 'FBGOptiSenseStudioBeta').resolve():
            return
        source = (base / 'FBGOptiSenseStudio').resolve()
        relative_paths = []
        for relative in (Path('outputs/preferences'), Path('outputs/device_profiles'),
                         Path('outputs/temporary_test')):
            branch = source / relative
            if branch.is_dir():
                relative_paths.extend(path.relative_to(source) for path in branch.rglob('*')
                                      if path.is_file() and path.suffix.lower() in ('.json', '.csv'))
        for number in range(1, 5):
            branch = source / f'outputs/machines/machine_{number}/temporary_test'
            if branch.is_dir():
                relative_paths.extend(path.relative_to(source) for path in branch.rglob('*')
                                      if path.is_file() and path.suffix.lower() in ('.json', '.csv'))
    else:
        source = Path(__file__).resolve().parent
        relative_paths = [
            Path('outputs/preferences/finger_stress_scan.json'),
            Path('outputs/device_profiles/ota_machine_profiles.json'),
            Path('outputs/device_profiles/machine_identity.json'),
        ]
        for number in range(1, 5):
            for name in ('last_45_session.json', 'last_finger_capture.json'):
                relative_paths.append(Path(f'outputs/machines/machine_{number}/temporary_test/{name}'))
    if destination == source:
        raise ValueError('Beta data must not use the normal source data directory')
    for relative in relative_paths:
        existing, copied = source / relative, destination / relative
        if (existing.is_file() and not copied.exists()
                and existing.resolve().is_relative_to(source.resolve())
                and copied.resolve().is_relative_to(destination)):
            copied.parent.mkdir(parents=True, exist_ok=True)
            try:
                with existing.open('rb') as reader, copied.open('xb') as writer:
                    shutil.copyfileobj(reader, writer)
            except FileExistsError:
                continue  # Another process/previous Beta record always wins.
            shutil.copystat(existing, copied)

