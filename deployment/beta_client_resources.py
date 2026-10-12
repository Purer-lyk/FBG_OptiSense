"""Explicit read-only resources for the customer Beta client.

No recursive engineering outputs, private settings, model snapshots or debug
firmware are collected. Writable factory defaults come from the approved
universal seed tree, never from the developer's active data directories.
"""

from pathlib import Path


RUNTIME_FILES = (
    'machine_registry.json', 'finger_sensor_layout.yaml', 't31_touch_calibration.json',
    'fullband_equal_power_operational_2001.json', 'fullband_equal_power_operational_2001.csv',
    'fullband_equal_power_operational_2001_transition_guards.json',
    'fullband_pdt_pdr_reference_2001.json', 'mode_tables_from_fullband_2001.json',
    'stress_high_power_135_final.json', 'stress_shape_power_dynamic_v3.json',
    'stress_shape_power_final_v2.json', 'wave_const.yaml', 'stress_wave_const.yaml',
    'temporary_test_default_plan.json',
    'calibration_backups/2号解调仪第一次校准/2号解调仪第一次校准_2001点.json',
    'machine_defaults/machine_3/machine_3_provisional_2001_20260927.json',
    'mach3_bridge.ps1', 'usb_firmware_catalog.json',
)


def beta_client_datas(repository):
    root = Path(repository).resolve()
    python_dir = root / 'Python'
    items = [(root / 'deployment/universal/machine_profile.json', '.')]
    for relative in RUNTIME_FILES:
        items.append((python_dir / relative, str(Path(relative).parent).replace('\\', '/')))
    for number in range(1, 5):
        relative = f'machine_defaults/machine_{number}/temporary_test_defaults.json'
        items.append((python_dir / relative, str(Path(relative).parent).replace('\\', '/')))
    items.extend((
        (root / 'deployment/universal/client_seed', 'client_seed'),
        (python_dir / 'assets', 'assets'),
        (root / 'outputs/ota/JDSU_F205RE_v1.0.106.fbgfw', 'firmware/usb'),
    ))
    for source, _ in items:
        if not source.exists():
            raise FileNotFoundError(source)
    return [(str(source), destination) for source, destination in items]
