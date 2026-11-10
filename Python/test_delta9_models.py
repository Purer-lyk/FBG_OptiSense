"""Torch-free unittest discovery wrapper; child alone imports/trains Torch."""
import subprocess,unittest
from pathlib import Path
from touch_worker_runtime import torch_worker_python
class Delta9ModelProcessTests(unittest.TestCase):
 def test_all_eight_real_models_in_clean_child(self):
  r=subprocess.run([torch_worker_python(),'delta9_models_child_tests.py','-v'],cwd=Path(__file__).resolve().parent,capture_output=True,text=True,encoding='utf-8',timeout=180)
  self.assertEqual(r.returncode,0,r.stdout+r.stderr)
  self.assertIn('OK',r.stderr)
if __name__=='__main__':unittest.main()
