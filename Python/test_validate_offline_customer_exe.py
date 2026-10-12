import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import validate_offline_customer_exe as validator


class FrozenAcceptanceArgumentTests(unittest.TestCase):
    def arguments(self,*extra):
        return ['--updater','helper.exe','--target','updater_cli_smoke/target',
                '--report-dir','acceptance',*extra]

    def test_embedded_expected_version_and_current_customer_fixture(self):
        _,args=validator.parse_args(self.arguments('--expected-version','1.0.8',
                                                  '--old-version','1.0.7'))
        self.assertEqual(args.expected_version,'1.0.8')
        self.assertEqual(args.old_version,'1.0.7')

    def test_sidecar_defaults_remain_compatible(self):
        _,args=validator.parse_args(self.arguments())
        self.assertIsNone(args.expected_version)
        self.assertEqual(args.old_version,'1.0.4')

    def test_invalid_version_metadata_is_rejected(self):
        with contextlib.redirect_stderr(io.StringIO()):
            for flag in ('--expected-version','--old-version'):
                with self.assertRaises(SystemExit):validator.parse_args(self.arguments(flag,'1.0.8-beta'))

    def test_subprocess_environment_uses_only_isolated_data_and_windows_path(self):
        with tempfile.TemporaryDirectory() as folder, patch.dict(os.environ,dict(
                SystemRoot='C:/Windows',PATH='C:/Python;C:/private',
                LOCALAPPDATA='C:/Users/operator/AppData/Local',
                FBG_STUDIO_DATA_DIR='C:/private/data',PYTHONPATH='C:/private/python',
                PYTHONHOME='C:/Python',VIRTUAL_ENV='C:/venv',QT_QPA_PLATFORM='offscreen'),clear=True):
            root=Path(folder)
            environment=validator.isolated_environment(root)
            self.assertEqual(environment['PATH'],'C:/Windows/System32;C:/Windows')
            self.assertEqual(environment['LOCALAPPDATA'],str(root/'isolated_local_app_data'))
            self.assertEqual(environment['FBG_STUDIO_DATA_DIR'],str(root/'client_data'))
            for name in ('PYTHONPATH','PYTHONHOME','VIRTUAL_ENV','QT_QPA_PLATFORM'):
                self.assertNotIn(name,environment)
            self.assertEqual(os.environ['FBG_STUDIO_DATA_DIR'],'C:/private/data')


if __name__=='__main__':unittest.main()
