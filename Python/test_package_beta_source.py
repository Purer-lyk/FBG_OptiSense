"""Offline-only checks for the editable Beta source exporter."""
import json
import sys
import tempfile
import unittest
from pathlib import Path


DEPLOYMENT = Path(__file__).resolve().parents[1] / "deployment"
sys.path.insert(0, str(DEPLOYMENT))
import package_beta_source as bundle


class BetaSourceExportTests(unittest.TestCase):
    def test_supported_beta_branch_is118_not_first119_macro(self):
        header = (bundle.REPO / "JDSU/Core/Inc/firmware_version.h").read_bytes()
        self.assertEqual(bundle.source_beta_firmware_version(header), "1.0.118")

    def test_exact_validated_firmware_package(self):
        report = bundle.verify_firmware_bytes(bundle.VERIFIED_FIRMWARE.read_bytes())
        self.assertEqual(report["version"], "1.0.118")
        self.assertEqual(report["image_crc32"], "1A951B61")
        self.assertEqual(report["image_bytes"], 163224)

    def test_changed_firmware_is_rejected(self):
        raw = bytearray(bundle.VERIFIED_FIRMWARE.read_bytes())
        raw[-1] ^= 1
        with self.assertRaises(ValueError):
            bundle.verify_firmware_bytes(bytes(raw))

    def test_private_generated_paths_forbidden(self):
        for path in ("Python/beta_data/outputs/test.json", "Python/.venv/a.py", "keys/private.pem",
                     "Python/outputs/machine4_PRIVATE/full_flash.bin", "build/latest/test.bin",
                     "Python/fbg_remote_config.yaml", "backups/test/a.py", "../escape.py",
                     "JDSU/MDK-ARM/JDSU_beta_generated.uvprojx", "Python/model.joblib"):
            self.assertTrue(bundle.forbidden(path), path)
        for path in ("Python/fbg_remote_config.yaml.example", "firmware/JDSU_F205RE_v1.0.118_beta.fbgfw",
                     "JDSU/Drivers/CMSIS/Lib/ARM/arm_cortexM3l_math.lib"):
            self.assertFalse(bundle.forbidden(path), path)

    def test_snapshot_selection_keeps_required_build_and_factory_pins(self):
        selected = bundle.select_payload()
        self.assertFalse(bundle.REQUIRED_SOURCE - selected.keys())
        for path in selected:
            self.assertFalse(bundle.forbidden(path), path)
        self.assertNotIn("Python/machine_defaults/machine_3/paused_auto_calibration_state_20260927.json", selected)
        self.assertNotIn("Python/outputs/machines/machine_4/cmsis_20261012/application_upgrade.json", selected)
        self.assertIn("Python/client_seed/outputs/device_profiles/ota_machine_profiles.json", selected)
        self.assertIn("Python/outputs/temporary_test/finger_captures/0号手指.json", selected)
        self.assertIn("Python/outputs/temporary_test/finger_captures/9个光栅最新通过.json", selected)
        for relative, source in selected.items():
            if relative.startswith("Python/outputs/"):
                self.assertTrue(source.is_relative_to(bundle.REPO / "deployment"), relative)

    def test_sensitive_json_is_sanitized(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "safe.json"
            source.write_text(json.dumps({"wifi_password": "not-for-export", "value": 3}), encoding="utf-8")
            data, removed = bundle.safe_payload("deployment/test/safe.json", source)
        self.assertEqual(json.loads(data)["wifi_password"], "")
        self.assertEqual(removed, ["wifi_password"])

    def test_source_version_is_read_without_running_module(self):
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "app_version.py"
            source.write_text('APP_VERSION = "1.0.8"\nAPP_RELEASE_CHANNEL = "beta-offline"\nraise RuntimeError()\n')
            self.assertEqual(bundle.literal_source_version(source), ("1.0.8", "beta-offline"))


if __name__ == "__main__":
    unittest.main()
