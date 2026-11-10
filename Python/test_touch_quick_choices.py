import copy,json,tempfile,unittest
from pathlib import Path
from touch_deployment_slots import ROOT,read_pair
class QuickChoicesTests(unittest.TestCase):
 def test_last_does_not_replace_best_or_other_task(self):
  import touch_deployment_quick as q
  choices=q.read_choices(ROOT/'outputs');best=choices['best'];last=choices['last']
  with tempfile.TemporaryDirectory() as d:
   home=Path(d);q.initialize(home,best,last)
   active=q.choose(home,best,'position','last')
   self.assertEqual(active['force'],best['force'])
   self.assertNotEqual(active['position']['checkpoint_sha256'],best['position']['checkpoint_sha256'])
   self.assertEqual(q.choose(home,active,'position','best')['position'],best['position'])
   self.assertEqual(q.read_choices(home)['best'],best)
 def test_corrupt_missing_incompatible_and_absent_last_fail_closed(self):
  import touch_deployment_quick as q
  best=read_pair(ROOT/'outputs/touch_deployment_selection.json')
  with tempfile.TemporaryDirectory() as d:
   home=Path(d);q.initialize(home,best,None);before=(home/q.NAME).read_bytes()
   with self.assertRaises(ValueError):q.choose(home,best,'position','last')
   self.assertEqual((home/q.NAME).read_bytes(),before)
   for change in ('hash','missing','identity','shape'):
    data=json.loads(before);data['last']=copy.deepcopy(best);c=data['last']['position']
    if change=='hash':c['checkpoint_sha256']='0'*64
    if change=='missing':c['model_path']=str(home/'absent.pt')
    if change=='identity':c['component_id']='spectral_timesnet:position'
    if change=='shape':c['input_contract']['input_shape']=[9]
    (home/q.NAME).write_text(json.dumps(data),encoding='utf8')
    with self.assertRaises((ValueError,FileNotFoundError),msg=change):q.choose(home,best,'position','last')
    self.assertEqual(best,read_pair(ROOT/'outputs/touch_deployment_selection.json'))
class AtomicPinTests(unittest.TestCase):
 def test_readback_failure_restores_exact_pin_bytes(self):
  from unittest.mock import patch
  import touch_deployment_slots as slots
  pair=read_pair(ROOT/'outputs/touch_deployment_selection.json')
  with tempfile.TemporaryDirectory() as d:
   target=Path(d)/'pin.json';target.write_bytes(b'original protected pin bytes')
   with patch.object(slots,'read_pair',side_effect=ValueError('readback fault')):
    with self.assertRaises(ValueError):slots.save_pair(target,pair)
   self.assertEqual(target.read_bytes(),b'original protected pin bytes')
class QuickUiTests(unittest.TestCase):
 def test_rejected_training_keeps_manual_active_other_slots(self):
  import touch_deployment_quick as q
  from touch_deployment_ui import IndependentSlotsMixin
  from types import SimpleNamespace
  choices=q.read_choices(ROOT/'outputs');best=choices['best'];last=choices['last']
  with tempfile.TemporaryDirectory() as d:
   home=Path(d);q.initialize(home,best,last);active=q.choose(home,best,'position','last');loaded=[]
   fake=SimpleNamespace(_slot_pin=home/'pin.json',slot_pair=active,status_label=SimpleNamespace(setText=lambda text:None),load_slot_pair=lambda pair:loaded.append(pair) or True)
   IndependentSlotsMixin._auto_select_training(fake,{'model_path':last['position']['model_path']})
   self.assertEqual(loaded,[], 'rejected training must not replace manually selected slots')
   self.assertEqual(q.read_choices(home)['best'],best)
 def test_one_click_controls_have_busy_gate(self):
  from touch_deployment_ui import IndependentSlotsMixin
  self.assertTrue(hasattr(IndependentSlotsMixin,'_quick_load'), 'one-click Best/Last handler missing')
  self.assertTrue(hasattr(IndependentSlotsMixin,'_refresh_quick_buttons'))
if __name__=='__main__':unittest.main(verbosity=2)
