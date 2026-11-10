"""Restore measured45 without touching the independent optical heatmap."""
import os,sys,json,tempfile,unittest
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=str(Path(sys.prefix)/'Lib/site-packages/PyQt5/Qt5/plugins/platforms')
from test_manual_stress_capture import FixedGridCaptureTests,FakeTemporaryPage,temporary_frame
from manual_stress_capture import ManualStressCaptureWindow
from manual_touch_training_page import ManualTouchTrainingPage,TouchTrainingWorker

class Restore45(unittest.TestCase):
 @classmethod
 def setUpClass(cls):
  FixedGridCaptureTests.setUpClass();cls.app=FixedGridCaptureTests.app
 def test_default_writer_without_wavelength_baseline(self):
  with tempfile.TemporaryDirectory() as folder:
   source=FakeTemporaryPage();p=ManualStressCaptureWindow(source,wavelength_provider=lambda:source.reference_wavelengths,output_root=Path(folder)/'manual_touch_45')
   try:
    self.assertEqual(p.capture_contract,'legacy45')
    source.plan={'rows':[dict(index=i,measured_nm=w) for i,w in enumerate(source.reference_wavelengths)]}
    p._on_temporary_frame(temporary_frame());p.select_grid_point(7)
    self.assertIsNone(p.wavelength_state.baseline)
    self.assertTrue(p.capture_current_press(),p.capture_status_label.text())
    session=p.session_dir;p.close_session();r=json.loads((session/'frames.jsonl').read_text('utf-8'))
    self.assertEqual(r['quality']['stable_adc'],list(range(700,745)))
    self.assertEqual(len(r['quality']['stable_dbm']),45)
    self.assertEqual(r['quality'].get('route_indices'),list(range(45)))
    self.assertEqual(r['quality'].get('adc_channel'),1)
    self.assertEqual(session.parent.name,'manual_touch_45')
    self.assertTrue(hasattr(p,'wavelength_baseline_button'))
   finally:p.close()
 def test_live45_allows_reselection_but_keeps_order_channel_selector(self):
  from temporary_optical_units import adc_code_to_dbm
  from touch_45_data import validate_record
  from restore45_models_child_tests import record
  _,_,_,contract=validate_record(record())
  source=FakeTemporaryPage();source.plan={'rows':[dict(index=i,measured_nm=w) for i,w in enumerate(contract['route_wavelengths_nm'])]}
  with tempfile.TemporaryDirectory() as folder:
   p=ManualTouchTrainingPage(default_dataset_dir=Path(folder),frame_source=source)
   try:
    p.report={'schema':'fbg_touch45_multilabel_v1','input_contract':contract}
    f=temporary_frame();f['feedback_selector']=0
    self.assertTrue(hasattr(p,'_extract_live_dbm'),'Missing bound45 dBm live path')
    self.assertEqual(p._extract_live_dbm(f),adc_code_to_dbm(list(range(700,745)),0,floor_nonpositive=True).tolist())
    with self.assertRaises(ValueError):p._extract_live_dbm(dict(f,records=f['records'][:9]))
    with self.assertRaises(ValueError):p._extract_live_dbm(dict(f,feedback_selector=1))
    with self.assertRaises(ValueError):p._extract_live_dbm(dict(f,adc_channel=2))
    # Both indices and absolute wavelengths may change with temperature.
    source.plan={'rows':[dict(index=i+100,measured_nm=w+.2,group=i//5+1)
                         for i,w in enumerate(contract['route_wavelengths_nm'])]}
    self.assertEqual(p._extract_live_dbm(f),adc_code_to_dbm(list(range(700,745)),0,floor_nonpositive=True).tolist())
    source.plan['rows'][0]['group']=2
    with self.assertRaises(ValueError):p._extract_live_dbm(f)
    source.plan['rows'][0]['group']=1
    source.plan['rows'][0],source.plan['rows'][1]=source.plan['rows'][1],source.plan['rows'][0]
    with self.assertRaises(ValueError):p._extract_live_dbm(f)
    source.plan['rows']=source.plan['rows'][:9]
    with self.assertRaises(ValueError):p._extract_live_dbm(f)
   finally:p.close()
 def test_default_pin_never_auto_discovers_smoke_models(self):
  with tempfile.TemporaryDirectory() as folder:
   out=Path(folder);(out/'touch_deployment_selection.json').write_text(json.dumps(dict(schema='fbg_touch_independent_slots_v3',smoke_replay=True,position=dict(input_contract=dict(input_shape=[9,5])))),encoding='utf-8')
   p=ManualTouchTrainingPage(default_dataset_dir=out,model_search_root=out)
   try:
    self.assertIn('等待45点',p.slot_warning_label.text())
    self.assertIsNone(p._slot_loader);self.assertFalse(p.start_live_button.isEnabled())
   finally:p.close()
 def test_training_page_has_no_optical_baseline_but_force_zero_and_four_encoders(self):
  with tempfile.TemporaryDirectory() as folder:
   p=ManualTouchTrainingPage(default_dataset_dir=Path(folder))
   try:
    self.assertFalse(hasattr(p,'baseline_button'))
    self.assertFalse(hasattr(p,'reminder_timer'))
    self.assertIsNotNone(p.actual_force.zero_button)
    self.assertEqual(p.architecture_combo.count(),4)
    self.assertIn('touch_45_models',TouchTrainingWorker(folder).command())
    self.assertNotIn('torch',sys.modules)
   finally:p.close()
if __name__=='__main__':unittest.main(verbosity=2)
