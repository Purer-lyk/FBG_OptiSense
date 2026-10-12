import builtins
import base64
import hashlib
import json
import os
from pathlib import Path
import tempfile
import sys
import unittest
import zipfile
from types import SimpleNamespace
from unittest.mock import patch, MagicMock

os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
plugin_path = Path(sys.prefix)/'Lib'/'site-packages'/'PyQt5'/'Qt5'/'plugins'/'platforms'
if plugin_path.exists():
    os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(plugin_path)
from PyQt5 import QtWidgets
import offline_update_helper as helper
from fbg_usb_ota import UsbFirmwareInfo
from fbg_ota_protocol import build_package
from offline_update_engine import canonical_manifest, UpdateError
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


class OfflineHelperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_updater_theme_does_not_import_plot_dependencies(self):
        original = builtins.__import__
        def blocked(name, *args, **kwargs):
            if name.split('.')[0] in ('numpy', 'pyqtgraph'):
                raise ImportError('Updater must not need plot libraries')
            return original(name, *args, **kwargs)
        with patch('builtins.__import__', blocked):
            helper.install_theme(self.app, configure_plots=False)

    def worker(self, serial='395736523033'):
        manifest = {'firmware': [{'usb_serials': ['395736523033'], 'version': '1.0.109'}]}
        worker = helper.UpdateWorker(Path('.'), manifest, 'firmware', ('COM16', serial))
        results, failures = [], []
        worker.result.connect(results.append)
        worker.failed.connect(failures.append)
        return worker, manifest, results, failures

    def test_latest_firmware_reads_version_without_reflashing(self):
        worker, manifest, results, failures = self.worker()
        package = SimpleNamespace(version_text='1.0.109', firmware_version=0x1006D)
        with patch.object(helper, 'load_manifest', return_value=manifest), \
             patch.object(helper, 'checked_payload', return_value=Path('firmware.fbgfw')), \
             patch.object(helper, 'load_package', return_value=package), \
             patch.object(helper, 'UsbOtaClient') as client:
            runtime = client.return_value.__enter__.return_value
            runtime.query_info.return_value = UsbFirmwareInfo('395736523033',0x1006D,99,229376,True)
            worker.run()
            runtime.install.assert_not_called()
        self.assertFalse(failures)
        self.assertEqual(results[0]['status'], 'latest')

    def test_other_machine_rejected_before_opening_usb(self):
        worker, manifest, results, failures = self.worker('OTHER_MACHINE')
        with patch.object(helper, 'load_manifest', return_value=manifest), \
             patch.object(helper, 'UsbOtaClient') as client:
            worker.run()
            client.assert_not_called()
        self.assertFalse(results)
        self.assertIn('未授权', failures[0])

    def test_firmware_update_uses_verified_usb_install_result(self):
        worker, manifest, results, failures = self.worker()
        package = SimpleNamespace(version_text='1.0.109', firmware_version=0x1006D)
        with patch.object(helper, 'load_manifest', return_value=manifest), \
             patch.object(helper, 'checked_payload', return_value=Path('firmware.fbgfw')), \
             patch.object(helper, 'load_package', return_value=package), \
             patch.object(helper, 'UsbOtaClient') as client:
            runtime = client.return_value.__enter__.return_value
            runtime.query_info.return_value = UsbFirmwareInfo('395736523033',0x1006C,99,229376,True)
            runtime.install.return_value = UsbFirmwareInfo('395736523033',0x1006D,100,229376,True)
            runtime.reboot_acknowledged = True
            worker.run()
            runtime.install.assert_called_once()
        self.assertFalse(failures)
        self.assertEqual(results[0]['runtime']['boot_id'], 100)
        self.assertTrue(results[0]['reboot_acknowledged'])

    def test_invalid_signature_disables_both_update_buttons(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(helper.list_ports, 'comports', return_value=[]):
            window = helper.OfflineUpdater(folder)
            self.assertFalse(window.client_button.isEnabled())
            self.assertFalse(window.firmware_button.isEnabled())
            window.close()

    def test_explicit_bundle_directory_takes_priority_over_embedded_files(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(helper.sys, 'frozen', True, create=True), \
             patch.object(helper.sys, '_MEIPASS', folder, create=True):
            embedded=Path(folder)/'offline_payload';embedded.mkdir()
            self.assertEqual(helper.bundle_root(Path(folder)/'chosen'),(Path(folder)/'chosen').resolve())

    def test_frozen_bundle_selects_embedded_directory_even_when_incomplete(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(helper.sys, 'frozen', True, create=True), \
             patch.object(helper.sys, '_MEIPASS', folder, create=True):
            embedded=Path(folder)/'offline_payload';embedded.mkdir()
            self.assertEqual(helper.bundle_root(),embedded.resolve())
            with self.assertRaises(FileNotFoundError):helper.verify_bundle(embedded)

    def test_frozen_sidecar_updater_remains_compatible(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(helper.sys, 'frozen', True, create=True), \
             patch.object(helper.sys, '_MEIPASS', folder, create=True), \
             patch.object(helper.sys, 'executable', str(Path(folder)/'Updater.exe')):
            self.assertEqual(helper.bundle_root(),Path(folder).resolve())

    def test_source_bundle_uses_source_directory(self):
        with patch.object(helper.sys,'frozen',False,create=True):
            self.assertEqual(helper.bundle_root(),Path(helper.__file__).resolve().parent)

    def signed_bundle(self, root, *, marker=None, serials=None, include_firmware=True):
        marker=marker or dict(version='1.0.8',channel='beta-offline',beta=True)
        archive=root/'client.zip'
        with zipfile.ZipFile(archive,'w') as stream:
            stream.writestr('FBG_OptiSense_Studio.exe',b'fixture-client')
            stream.writestr('_internal/runtime.dll',b'fixture-runtime')
            stream.writestr('client_release.json',json.dumps(marker))
        def item(path,version,**extra):
            return dict(filename=path.name,bytes=path.stat().st_size,
                        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),version=version,**extra)
        manifest=dict(schema='fbg_offline_update_v1',release='1.0.8',
                      client=item(archive,'1.0.8'),firmware=[])
        if include_firmware:
            source=root/'firmware.bin'
            source.write_bytes(bytes.fromhex('0000012021800008')+bytes(64))
            package=build_package(source,root/'firmware.fbgfw',0x10076)
            source.unlink()
            manifest['firmware'].append(item(package.path,'1.0.118',
                                            usb_serials=serials or ['207A33813232']))
        key=Ed25519PrivateKey.generate()
        manifest['signature']=base64.b64encode(key.sign(canonical_manifest(manifest))).decode()
        (root/'update_manifest.json').write_text(json.dumps(manifest),encoding='utf-8')
        public=key.public_key().public_bytes(serialization.Encoding.Raw,serialization.PublicFormat.Raw).hex()
        return public

    def test_complete_signed_beta_bundle_verification_does_not_enumerate_usb(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(helper.list_ports,'comports',side_effect=AssertionError('No USB reads')):
            root=Path(folder);public=self.signed_bundle(root)
            result=helper.verify_bundle(root,public)
            self.assertTrue(result['verified'])
            self.assertEqual(result['client']['channel'],'beta-offline')
            self.assertEqual(result['firmware'][0]['version'],'1.0.118')
            self.assertFalse(result['usb_devices_enumerated'])

    def test_bundle_verification_rejects_unsigned_manifest_and_modified_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);public=self.signed_bundle(root)
            with self.assertRaises(UpdateError):helper.verify_bundle(root,'00'*32)
            (root/'firmware.fbgfw').write_bytes(b'changed')
            with self.assertRaisesRegex(UpdateError,'SHA-256'):helper.verify_bundle(root,public)

    def test_bundle_verification_requires_firmware_and_beta_marker(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);public=self.signed_bundle(root,include_firmware=False)
            with self.assertRaisesRegex(UpdateError,'不完整'):helper.verify_bundle(root,public)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);public=self.signed_bundle(root,marker=dict(version='1.0.8',channel='customer-universal'))
            with self.assertRaisesRegex(UpdateError,'Beta启动配置'):helper.verify_bundle(root,public)

    def test_bundle_verification_rejects_client_version_and_duplicate_or_invalid_uid(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);public=self.signed_bundle(root,marker=dict(version='1.0.7',channel='beta-offline',beta=True))
            with self.assertRaisesRegex(UpdateError,'客户端版本标记'):helper.verify_bundle(root,public)
        for serials in (['207A33813232','207A33813232'],['UNKNOWN']):
            with tempfile.TemporaryDirectory() as folder:
                root=Path(folder);public=self.signed_bundle(root,serials=serials)
                with self.assertRaisesRegex(UpdateError,'USB设备序列号'):helper.verify_bundle(root,public)

    def test_no_hardware_preview_never_enumerates_or_updates_firmware(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(helper.list_ports,'comports',side_effect=AssertionError('No USB reads')):
            window=helper.OfflineUpdater(folder,hardware_enabled=False)
            self.assertFalse(window.firmware_button.isEnabled())
            window.refresh_ports()
            self.assertFalse(window.firmware_button.isEnabled())
            window.close()

    def test_verify_cli_does_not_create_qapplication_and_preserves_existing_result(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.object(helper.QtWidgets,'QApplication',side_effect=AssertionError('No Qt needed')), \
             patch.object(helper.list_ports,'comports',side_effect=AssertionError('No USB reads')):
            root=Path(folder);public=self.signed_bundle(root)
            result=root/'result.json'
            # The default public-key argument is evaluated at definition time;
            # inject only the test key, never a developer signing key.
            verify=helper.verify_bundle
            with patch.object(helper,'verify_bundle',side_effect=lambda path:verify(path,public)):
                arguments=['--verify-bundle','--bundle-dir',str(root),'--result-file',str(result)]
                self.assertEqual(helper.main(arguments),0)
                self.assertTrue(json.loads(result.read_text(encoding='utf-8'))['verified'])
                preserved=result.read_bytes()
                with self.assertRaises(SystemExit):helper.main(arguments)
                self.assertEqual(result.read_bytes(),preserved)
