"""Production Qt tests for independently loaded slots; no acquisition or hardware."""
import os,sys,tempfile,json,unittest
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
from PyQt5 import QtWidgets,QtTest
from manual_touch_training_page import ManualTouchTrainingPage
from touch_deployment_slots import register_group,save_pair
ROOT=Path(__file__).resolve().parent
class DualSlotQtTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
 def test_retired_nine_input_pin_does_not_load_into_fortyfive_workspace(self):
  with tempfile.TemporaryDirectory() as d:
   out=Path(d);save_pair(out/'touch_deployment_selection.json',register_group(ROOT/'outputs/delta9_groupcv_20261015_101629/selected_all680.joblib'))
   page=ManualTouchTrainingPage(default_dataset_dir=out,model_search_root=out)
   try:
    self.assertTrue(hasattr(page,'position_model_edit'),'missing independent position slot')
    self.assertIsNone(page.inference)
    self.assertIsNone(page._slot_loader)
    self.assertFalse(page._slots_ready)
    self.assertIn('等待45点',page.slot_warning_label.text())
    self.assertNotIn('torch',sys.modules)
   finally:page.shutdown();QtTest.QTest.qWait(100)
 def test_completion_waits_for_cleanup_then_loads_without_inference(self):
  from unittest.mock import patch
  with tempfile.TemporaryDirectory() as d:
   page=ManualTouchTrainingPage(default_dataset_dir=Path(d))
   pair=register_group(ROOT/'outputs/delta9_groupcv_20261015_101629/selected_all680.joblib')
   try:
    with patch('touch_deployment_ui.register_training',return_value=pair):
     page._training_completed(dict(model_path=pair['position']['model_path']))
     self.assertIsNone(page._slot_loader)
     page._training_finished()
     self.assertIsNone(page._slot_loader)
     self.assertFalse(page._slots_ready)
     self.assertIsNone(page.inference)
     self.assertIn('未载入',page.status_label.text())
   finally:page.shutdown();QtTest.QTest.qWait(100)
 def test_missing_pin_does_not_discover_models(self):
  with tempfile.TemporaryDirectory() as d:
   page=ManualTouchTrainingPage(default_dataset_dir=Path(d),model_search_root=Path(d))
   try:
    self.assertIsNone(page.slot_pair);self.assertIsNone(page._slot_loader);self.assertFalse(page.start_live_button.isEnabled())
    self.assertIn('无自动回退',page.slot_warning_label.text())
   finally:page.shutdown()
if __name__=='__main__':unittest.main()
