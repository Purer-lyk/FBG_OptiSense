"""Customer Beta packaging/migration tests; no user's files or hardware."""

import ast
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import beta_profile

REPOSITORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY / 'deployment'))
from beta_client_resources import beta_client_datas


class BetaFrozenDataTests(unittest.TestCase):
    def test_frozen_enable_uses_separate_appdata(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, {'LOCALAPPDATA': folder}, clear=True), \
             patch.object(sys, 'frozen', True, create=True):
            beta_profile.enable_beta()
            self.assertEqual(Path(os.environ['FBG_STUDIO_DATA_DIR']),
                             Path(folder) / 'FBGOptiSenseStudioBeta')

    def test_copy_missing_same_user_settings_routes_and_reference_only(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, {'LOCALAPPDATA': folder}, clear=True), \
             patch.object(sys, 'frozen', True, create=True):
            formal = Path(folder) / 'FBGOptiSenseStudio'
            relative_paths = (
                'outputs/preferences/finger_stress_scan.json',
                'outputs/device_profiles/machine_identity.json',
                'outputs/machines/machine_2/temporary_test/last_45_session.json',
                'outputs/machines/machine_3/temporary_test/finger_captures/finger.json',
                'outputs/machines/machine_4/temporary_test/reference.csv',
                'outputs/temporary_test/legacy.json',
            )
            for relative in relative_paths:
                path = formal / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('{"owned_record": true}', encoding='utf-8')
            private = formal / 'outputs/logs/private.log'
            private.parent.mkdir(parents=True)
            private.write_text('not migrated', encoding='utf-8')
            beta_profile.enable_beta()
            beta_profile.seed_beta_data()
            beta = Path(os.environ['FBG_STUDIO_DATA_DIR'])
            for relative in relative_paths:
                self.assertEqual((beta / relative).read_bytes(), (formal / relative).read_bytes())
            self.assertFalse((beta / 'outputs/logs/private.log').exists())
            self.assertEqual(private.read_text(encoding='utf-8'), 'not migrated')

    def test_existing_beta_files_and_formal_records_never_overwritten(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, {'LOCALAPPDATA': folder}, clear=True), \
             patch.object(sys, 'frozen', True, create=True):
            relative = 'outputs/device_profiles/machine_identity.json'
            formal = Path(folder) / 'FBGOptiSenseStudio' / relative
            beta = Path(folder) / 'FBGOptiSenseStudioBeta' / relative
            for path, value in ((formal, 'formal'), (beta, 'beta')):
                path.parent.mkdir(parents=True)
                path.write_text(value, encoding='utf-8')
            beta_profile.enable_beta()
            beta_profile.seed_beta_data()
            beta_profile.seed_beta_data()
            self.assertEqual(formal.read_text(encoding='utf-8'), 'formal')
            self.assertEqual(beta.read_text(encoding='utf-8'), 'beta')

    def test_explicit_frozen_override_never_reads_formal_user_data(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, {'LOCALAPPDATA': str(Path(folder) / 'unread-user'),
                                    'FBG_STUDIO_DATA_DIR': str(Path(folder) / 'isolated')}, clear=True), \
             patch.object(sys, 'frozen', True, create=True), \
             patch.object(Path, 'rglob', side_effect=AssertionError('Unexpected user-data read')):
            beta_profile.enable_beta()
            beta_profile.seed_beta_data()
            self.assertFalse((Path(folder) / 'isolated').exists())

    def test_missing_formal_profile_starts_without_migration(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, {'LOCALAPPDATA': folder}, clear=True), \
             patch.object(sys, 'frozen', True, create=True):
            beta_profile.enable_beta()
            beta_profile.seed_beta_data()
            self.assertFalse((Path(folder) / 'FBGOptiSenseStudio').exists())

    def test_explicit_default_beta_path_is_still_isolated(self):
        with tempfile.TemporaryDirectory() as folder, \
             patch.dict(os.environ, {'LOCALAPPDATA': folder,
                                    'FBG_STUDIO_DATA_DIR': str(Path(folder) / 'FBGOptiSenseStudioBeta')}, clear=True), \
             patch.object(sys, 'frozen', True, create=True), \
             patch.object(Path, 'rglob', side_effect=AssertionError('Unexpected user-data read')):
            beta_profile.enable_beta()
            beta_profile.seed_beta_data()
            self.assertFalse((Path(folder) / 'FBGOptiSenseStudioBeta').exists())


class BetaPayloadTests(unittest.TestCase):
    def test_sources_are_explicit_and_no_runtime_outputs_or_private_models(self):
        for source, destination in beta_client_datas(REPOSITORY):
            relative = Path(source).relative_to(REPOSITORY).as_posix()
            self.assertNotIn('Python/outputs', relative)
            self.assertNotIn('beta_data', relative)
            self.assertNotIn('fbg_remote_config', relative)
            self.assertNotIn('.joblib', relative)
            self.assertNotIn('PRIVATE', relative)
            if destination == 'client_seed':
                self.assertEqual(relative, 'deployment/universal/client_seed')

    def test_pending_machine_four_defaults_have_no_borrowed_route(self):
        path = REPOSITORY / 'Python/machine_defaults/machine_4/temporary_test_defaults.json'
        payload = json.loads(path.read_text(encoding='utf-8'))
        self.assertEqual(payload['machine_id'], 'machine_4')
        self.assertIsNone(payload['selection_plan'])
        self.assertIsNone(payload['validated_route_indices'])

    def test_beta_spec_marker_entry_and_disabled_cmsis(self):
        source = (REPOSITORY / 'deployment/pyinstaller/fbg_optisense_studio_beta.spec').read_text('utf-8')
        ast.parse(source)
        for literal in ('launch_fbg_studio_beta.pyw', "name='app_beta_build'", "'beta': True",
                        "'channel': 'beta-offline'", "'firmware_version': '1.0.118'", "'pyocd'"):
            self.assertIn(literal, source)
        self.assertNotIn('app_universal_build', source)


if __name__ == '__main__':
    unittest.main()
