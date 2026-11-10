"""Isolate real Windows OpenGL/Qt regression from unittest discovery parents."""
import os
import subprocess
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent

class FingerDesktopGLSubprocessTests(unittest.TestCase):
    @unittest.skipUnless(sys.platform == 'win32', 'requires the Windows desktop GL platform')
    def test_real_mesh_compact_dots_no_numbers_and_projected_page_clicks(self):
        python = HERE / '.venv/Scripts/python.exe'
        self.assertTrue(python.is_file(), 'project Qt venv is required')
        env = dict(os.environ)
        env['QT_QPA_PLATFORM'] = 'windows'
        env['QT_OPENGL'] = 'desktop'
        env['PYTHONUTF8'] = '1'
        result = subprocess.run([str(python), str(HERE / 'finger_gl_context_probe.py'), '-v'],
            cwd=HERE, env=env, capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=60)
        log = result.stdout + result.stderr
        evidence = HERE.parent / 'migration_evidence/finger_compact_no_numbers'
        evidence.mkdir(parents=True, exist_ok=True)
        (evidence / 'green.log').write_text(log, encoding='utf-8')
        self.assertEqual(result.returncode, 0, log)
        self.assertNotIn('Traceback', log)
        self.assertNotIn('RuntimeWarning', log)
        print(log)

if __name__ == '__main__':
    unittest.main()
