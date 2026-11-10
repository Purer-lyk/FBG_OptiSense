import unittest, tempfile, json
from pathlib import Path
import touch_deployment_slots as slots
ROOT=Path(__file__).resolve().parent
class DeploymentSlotsTests(unittest.TestCase):
 def test_registered_group_winners_are_independent_and_fail_closed(self):
  pair=slots.register_group(ROOT/'outputs/delta9_groupcv_20261015_101629/selected_all680.joblib')
  self.assertEqual(pair['position']['component_id'],'rbf_g0.3_a0.1')
  self.assertEqual(pair['force']['component_id'],'small_mlp32_d2')
  self.assertFalse(pair['deployment_ready'])
  self.assertEqual(pair['position']['metric'][0],-0.09558823529411764)
  slots.validate_pair(pair)
  bad=json.loads(json.dumps(pair));bad['force']['checkpoint_sha256']='0'*64
  with self.assertRaises(ValueError):slots.validate_pair(bad)
  with tempfile.TemporaryDirectory() as d:
   pin=Path(d)/'pin.json';slots.save_pair(pin,pair);self.assertEqual(slots.read_pair(pin),pair)
   pin.write_text('{}')
   with self.assertRaises(ValueError):slots.read_pair(pin)
 def test_independent_validation_promotion_ignores_recency_and_test(self):
  with tempfile.TemporaryDirectory() as d:
   root=Path(d);contract=slots.register_group(ROOT/'outputs/delta9_groupcv_20261015_101629/selected_all680.joblib')['position']['input_contract']
   def fixture(name,p,f):
    folder=root/name;folder.mkdir();model=folder/'fixture.pt';model.write_bytes(b'UNIT TEST METADATA ONLY NOT A TORCH MODEL')
    (folder/'frozen_split.json').write_text(json.dumps(dict(split=dict(validation=[1,3],train=[0,2]),input_contract=contract)))
    sel=folder/'fixture_validation_selection.json';sel.write_text(json.dumps(dict(position=[p,-.3],force=[-f],source_sha256='test-fixture-source')))
    r=dict(schema='fbg_touch_delta9_multilabel_v1',input_contract=contract,model_path=str(model),checkpoint_sha256=slots.digest(model),validation_selection_sha256=slots.digest(sel),architecture=name,smoke_only=False,metrics=dict(test=dict(position_accuracy=999)))
    model.with_suffix('.report.json').write_text(json.dumps(r));return slots.register_training(model)
   current=fixture('old',.6,.3);candidate=fixture('new',.7,.4);selected=slots.promote(current,candidate)
   self.assertEqual(selected['position'],candidate['position']);self.assertEqual(selected['force'],current['force'])
   worse=fixture('newest',.5,.5);self.assertEqual(slots.promote(selected,worse),selected)
   bad=json.loads(json.dumps(selected));bad['force']['metric']=[0.0]
   with self.assertRaises(ValueError):slots.validate_pair(bad)
 def test_search_and_group_components_can_be_combined(self):
  pair=slots.register_group(ROOT/'outputs/delta9_groupcv_20261015_101629/selected_all680.joblib')
  other=slots.register_search(ROOT/'outputs/delta9_search_20261015_094820/selected_independent.pt')
  pair['force']=other['force'];pair['trusted_sources'].update(other['trusted_sources']);slots.validate_pair(pair)
  self.assertNotEqual(pair['position']['model_path'],pair['force']['model_path'])
  self.assertEqual(slots.promote(pair,other),pair,'different protocol must not be ranked')
 def test_real_child_loads_two_components_and_returns_same_source(self):
  import subprocess
  pair=slots.register_group(ROOT/'outputs/delta9_groupcv_20261015_101629/selected_all680.joblib')
  with tempfile.TemporaryDirectory() as d:
   p=Path(d)/'pair.json';slots.save_pair(p,pair)
   run=subprocess.run(['C:/lyk/defaultpy/Python312/python.exe','-m','touch_dual_service',str(p)],input=json.dumps(dict(delta_nm=[0.0]*9,source_seq=17,timestamp_s=2.5,session_id='test'))+'\n',cwd=ROOT,capture_output=True,text=True,encoding='utf-8',timeout=60)
   self.assertEqual(run.returncode,0,run.stderr)
   rows=[json.loads(x) for x in run.stdout.splitlines()];self.assertEqual(rows[0]['type'],'ready')
   self.assertEqual(rows[1]['source_seq'],17);self.assertEqual(len(rows[1]['force']),3)
   self.assertEqual(len(rows[1]['probabilities']),25)
   self.assertEqual(rows[0]['components'],['rbf_g0.3_a0.1','small_mlp32_d2'])
if __name__=='__main__':unittest.main()
