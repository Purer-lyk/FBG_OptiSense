import unittest,tempfile,json,sys,os
from pathlib import Path
import numpy as np
class DeltaTrainingTests(unittest.TestCase):
 def test_reader_production_writer_and_multilabel(self):
  import touch_delta9_data as d
  from delta9_capture import Delta9SessionWriter,Delta9Frame
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as root:
   w=Delta9SessionWriter(root)
   ctx=dict(channel=1,feedback_selector=0,route_sha256='route',source_point_count=45,peak_to_position_labels=[25,15,5,3,13,23,21,11,1],fit_method='makima')
   for i in range(12):
    q=dict(annotation_mode='multi',position_labels=[2,22] if i%2 else [],position_multihot=[int(k in ([2,22] if i%2 else [])) for k in range(1,26)],baseline_id='explicit',smoke_only=True,cycle_start_us=i*100000,paxini_force=dict(fx_n=1.,fy_n=None,fz_n=3.,raw_six_axis=[9]*6))
    w.write_frame(Delta9Frame(i,i*100000000,tuple(1550+j+i*.001 for j in range(9)),tuple(1550+j for j in range(9)),ctx,'operator-set',q))
   w.flush();w.close()
   data=d.prepare_data(root)
   self.assertEqual(data['features'].shape,(12,9));self.assertEqual(data['labels'].shape,(12,25))
   self.assertEqual(data['labels'][1].sum(),2);self.assertTrue(np.isnan(data['forces'][:,1]).all())
   self.assertEqual(data['forces'][0,0],1.)
   split=d.freeze_split(data,Path(root)/'candidate');self.assertTrue(split['test'])
   stats=d.fit_statistics(data,split['train']);np.testing.assert_allclose(stats['feature_mean'],data['features'][split['train']].mean(0))
   self.assertFalse('torch' in sys.modules)
 def test_missing_force_axis_checkpoint_still_fails_closed_contract(self):
  from touch_delta9_data import validate_input_contract
  contract=dict(input_shape=[9],feature_kind='delta_wavelength9',feature_units='nm',peak_order=[f'P{i}' for i in range(1,10)],peak_to_position_labels=[25,15,5,3,13,23,21,11,1],peak_selection_method='makima_curve_global_max_dbm',baseline_semantics='current_same_frame_peak_nm-minus-explicit_unloaded_baseline_nm',channel=1,feedback_selector=0,source_point_count=45,route_sha256='synthetic',force_semantics='PX6D_host_tare_corrected_whole_contact_resultant_N')
  validate_input_contract(contract)
  for key in ('feature_kind','peak_order','peak_to_position_labels','baseline_semantics','route_sha256','force_semantics'):
   with self.subTest(field=key),self.assertRaises(ValueError):validate_input_contract(dict(contract,**{key:None}))
 def test_rejects_legacy_and_forged_difference(self):
  import touch_delta9_data as d
  from delta9_capture import Delta9Frame
  r=Delta9Frame(1,1,(1550.,)*9,(1550.,)*9,{},'time',dict(press_label=0,baseline_id='b')).to_record()
  with self.assertRaises(ValueError):d.validate_record(dict(r,values=[1.]*9))
  with self.assertRaises(ValueError):d.validate_record(dict(schema='legacy',values=[0.]*45))
if __name__=='__main__':unittest.main()
