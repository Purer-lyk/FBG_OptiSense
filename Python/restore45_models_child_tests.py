"""Torch-only measured45 model/reader contract; run outside Qt."""
import importlib.util,unittest,copy
import numpy as np
from temporary_optical_units import adc_code_to_dbm

def record(labels=None,label=7):
 adc=np.arange(700,745).tolist();waves=np.arange(45).tolist();dbm=adc_code_to_dbm(adc,0,floor_nonpositive=True).tolist()
 q=dict(stable_adc=adc,stable_dbm=dbm,stable_records=[dict(index=i,second_code=v) for i,v in enumerate(adc)],points_per_peak=5,point_count=45,feedback_selector=0,adc_channel=1,press_label=label,training_views=dict(stable_optical_power_dbm=dbm))
 if labels is not None:q.update(annotation_mode='multi',position_labels=labels,position_multihot=[int(i in labels) for i in range(1,26)])
 return dict(wavelengths_nm=waves,quality=q)
class Measured45Models(unittest.TestCase):
 def test_measured45_reader_and_all_encoder_contracts(self):
  self.assertIsNotNone(importlib.util.find_spec('touch_45_models'),'Missing actual45 models (not nine-feature repetition)')
  from touch_45_data import validate_record,validate_input_contract
  from touch_45_models import PositionModel,ForceModel,DEFAULT,build
  import torch
  torch.set_num_threads(2)
  for labels in (None,[],[7],[2,10,25]):
   x,y,f,c=validate_record(record(labels))
   self.assertEqual(x.shape,(45,));self.assertEqual(y.shape,(25,));self.assertTrue(np.isnan(f).all())
   self.assertEqual(np.flatnonzero(y).tolist(),[6] if labels is None else [i-1 for i in labels]);validate_input_contract(c)
  legacy=record([2,10])
  for row in legacy['quality']['stable_records']:row.pop('index')
  _,_,_,legacy_contract=validate_record(legacy)
  self.assertEqual(legacy_contract['route_binding'],'measured_wavelengths_legacy_no_indices')
  bad=record();bad['quality']['stable_dbm']=list(range(45))
  with self.assertRaises(ValueError):validate_record(bad)
  for kind in ('tcn','sensor_token','sparse_cnn','inception_1d'):
   for history in (False,True):
    config=dict(DEFAULT,architecture=kind,force_history=history);p=build(PositionModel,config);f=build(ForceModel,config)
    self.assertFalse({v.data_ptr() for v in p.parameters()} & {v.data_ptr() for v in f.parameters()})
    x=torch.randn(2,45,requires_grad=True);self.assertEqual(tuple(p(x).shape),(2,25));self.assertEqual(tuple(f(x)[0].shape),(2,3))
    with self.assertRaises(ValueError):p(torch.zeros(2,9))
    if kind=='sensor_token':self.assertEqual(p.encoder.embedding.in_features,5)
    if kind=='sparse_cnn':self.assertEqual(p.encoder.embedding.in_channels,6)
    if kind=='tcn':
     self.assertGreaterEqual(p.encoder.receptive_field,45)
     p.eval();g=torch.autograd.grad(p.encoder(x).sum(),x)[0]
     self.assertTrue((g[:,[0,44]].abs()>0).all())
if __name__=='__main__':unittest.main(verbosity=2)
