"""Keep Torch-first tests out of the GUI unittest process."""
from pathlib import Path
import os
import subprocess
import sys
import unittest

class SensorTokenIsolatedTests(unittest.TestCase):
    def test_sensor_token_contract_in_torch_first_process(self):
        root=Path(__file__).resolve().parent
        result=subprocess.run([sys.executable,str(root/'sensor_token_subprocess_tests.py')],cwd=root,capture_output=True,text=True,encoding='utf-8',env={**os.environ,'PYTHONIOENCODING':'utf-8'},timeout=90)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertIn('Ran 7 tests',result.stderr)

if __name__=='__main__': unittest.main()
