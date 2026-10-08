"""Keep Torch dBm tests isolated from the Qt unittest process."""
import os
from pathlib import Path
import subprocess
import sys
import unittest

class DbmTrainingIsolatedTests(unittest.TestCase):
    def test_dbm_core_in_torch_first_child(self):
        root=Path(__file__).resolve().parent
        env=dict(os.environ,PYTHONIOENCODING='utf-8')
        scratch=r'C:\Users\Administrator\AppData\Local\hermes\cache\scratch'
        env.update(TEMP=scratch,TMP=scratch,TMPDIR=scratch)
        result=subprocess.run([sys.executable,str(root/'dbm_training_subprocess_tests.py')],cwd=root,env=env,capture_output=True,text=True,encoding='utf-8',timeout=60)
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        self.assertIn('Ran 4 tests',result.stderr)

if __name__=='__main__': unittest.main()
