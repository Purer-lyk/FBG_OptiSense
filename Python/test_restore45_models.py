"""Fresh Torch child; never load Torch into Qt unittest discovery."""
import unittest,subprocess,os,sys
from pathlib import Path
from touch_worker_runtime import torch_worker_python
class Measured45Child(unittest.TestCase):
 def test_real_torch_child(self):
  before='torch' in sys.modules
  result=subprocess.run([torch_worker_python(),'-m','unittest','restore45_models_child_tests','-v'],cwd=Path(__file__).resolve().parent,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=180)
  self.assertEqual(result.returncode,0,result.stdout+result.stderr)
  self.assertEqual('torch' in sys.modules,before)
if __name__=='__main__':unittest.main(verbosity=2)
