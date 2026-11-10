import importlib.util,unittest,subprocess,sys,os,tempfile
from pathlib import Path
class PipelineTests(unittest.TestCase):
    def test_worker_runtime_uses_installed_torch_without_importing_it_in_qt(self):
        self.assertIsNotNone(importlib.util.find_spec('touch_worker_runtime'))
        from touch_worker_runtime import torch_worker_python
        result=subprocess.run([torch_worker_python(),'-c','import torch;print(torch.__version__)'],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotIn('torch',sys.modules)

    def test_fresh_45_writer_training_and_saved_service_contract(self):
        self.assertIsNotNone(importlib.util.find_spec('touch_five_training'),'five-point production training entrypoint absent')
        probe=Path(__file__).with_name('pr5_touch_pipeline_probe.py')
        from touch_worker_runtime import torch_worker_python
        evidence=os.environ.get('PR5_EVIDENCE')
        temporary=None
        if evidence is None:
            scratch=Path.home()/'AppData/Local/hermes/cache/scratch'
            scratch.mkdir(parents=True,exist_ok=True)
            temporary=tempfile.TemporaryDirectory(dir=scratch)
            evidence=temporary.name
            replay=subprocess.run([sys.executable,str(probe.with_name('run_pr5_touch_qt_replay.py')),evidence],capture_output=True,text=True,encoding='utf-8')
            self.assertEqual(replay.returncode,0,replay.stdout+'\n'+replay.stderr)
        try:
            result=subprocess.run([torch_worker_python(),str(probe),evidence],capture_output=True,text=True,encoding='utf-8')
        finally:
            if temporary is not None:temporary.cleanup()
        self.assertEqual(result.returncode,0,result.stdout+'\n'+result.stderr)
        print(result.stdout)
if __name__=='__main__':unittest.main()
