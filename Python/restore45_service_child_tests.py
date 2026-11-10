"""Actual45 checkpoint/independent slots/service parity using synthetic-only evidence."""
import unittest,json,os,subprocess,sys,copy
from pathlib import Path
import numpy as np
from touch_deployment_slots import register_training,save_pair,read_pair,validate_pair
class Measured45Service(unittest.TestCase):
 def test_explicit_smoke_dual_slots_match_frozen_validation(self):
  import inspect
  self.assertIn('allow_smoke',inspect.signature(register_training).parameters,'Missing explicit evidence-only45 adapter')
  out=Path(os.environ['RESTORE45_EVIDENCE']);ev=json.loads((out/'training_evidence.json').read_text())
  for item in ev['results']:
   path=Path(item['model_path'])
   with self.assertRaises(ValueError):register_training(path)
   pair=register_training(path,allow_smoke=True)
   snap=out/(path.parent.name+'_slots.json');snap.write_text(json.dumps(pair),encoding='utf-8')
   read_pair(snap)
   with self.assertRaises(ValueError):save_pair(out/'not_a_production_pin.json',pair)
   from touch_45_models import Measured45Bundle
   from manual_touch_cnn import load_touch_model
   self.assertIsInstance(load_touch_model(path),Measured45Bundle)
   from touch_45_data import prepare_data
   b=Measured45Bundle(path);data=prepare_data(out/'manual_touch_45');trace=np.load(path.parent/(path.stem+'_validation_predictions.npz'))
   requests=[]
   for n,(i,reset) in enumerate(zip(trace['indices'],trace['reset_flags'])):
    requests.append(dict(dbm=data['features'][i].tolist(),source_seq=int(data['source_line'][i]),timestamp_s=float(data['timestamp_s'][i]),session_id=int(data['session'][i]),reset=bool(reset),request_id=n,input_contract=data['input_contract']))
   requests += [dict(dbm=[0]*9),dict(delta_nm=[0]*9),dict(dbm=[0]*45,input_contract=dict(data['input_contract'],channel=3))]
   r=subprocess.run([sys.executable,'-m','touch_dual_service',str(snap)],input=''.join(json.dumps(x)+'\n' for x in requests),capture_output=True,text=True,encoding='utf-8',timeout=40)
   self.assertEqual(r.returncode,0,r.stderr+r.stdout);rows=[json.loads(v) for v in r.stdout.splitlines()]
   self.assertTrue(rows[0]['independent_parameter_storage']);pred=rows[1:-3]
   np.testing.assert_allclose([v['force'] for v in pred],trace['forces'],rtol=0,atol=2e-6)
   np.testing.assert_allclose([v['probabilities'] for v in pred],trace['probabilities'],rtol=0,atol=2e-6)
   self.assertTrue(all(v['type']=='error' for v in rows[-3:]))
   # Universal single-checkpoint CLI dispatch remains supported alongside dual slots.
   r=subprocess.run([sys.executable,'-m','manual_touch_cnn','--serve',str(path)],input=''.join(json.dumps(x)+'\n' for x in requests[:-3]),capture_output=True,text=True,encoding='utf-8',timeout=40)
   self.assertEqual(r.returncode,0,r.stderr);single=[json.loads(v) for v in r.stdout.splitlines()][1:]
   np.testing.assert_allclose([v['force'] for v in single],trace['forces'],rtol=0,atol=2e-6)
if __name__=='__main__':unittest.main(verbosity=2)
