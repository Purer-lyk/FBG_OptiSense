"""Torch-only child tests; never include in Qt-loaded discovery."""
import unittest,os,tempfile,sys,json,subprocess
from pathlib import Path
import numpy as np
import torch
class ModelTests(unittest.TestCase):
 def test_all_roster_train_reload_service(self):
  import touch_delta9_models as m
  from delta9_capture import Delta9Frame,Delta9SessionWriter
  with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as root:
   root=Path(root);w=Delta9SessionWriter(root/'data')
   ctx=dict(channel=1,feedback_selector=0,route_sha256='synthetic_route',source_point_count=45,fit_method='makima')
   for i in range(16):
    labels=[2,22] if i%2 else [];q=dict(annotation_mode='multi',position_labels=labels,position_multihot=[int(k in labels) for k in range(1,26)],baseline_id='explicit',smoke_only=True,cycle_start_us=i*100000,paxini_force=dict(fx_n=i*.1,fy_n=-i*.05 if i%3 else None,fz_n=i*.2))
    w.write_frame(Delta9Frame(i,i*100000000,tuple(1550+j+np.sin(i+j)*.02 for j in range(9)),tuple(1550+j for j in range(9)),ctx,'operator-set',q))
   w.flush();w.close()
   for arch in m.ARCHITECTURES:
    for history in (False,True):
     with self.subTest(architecture=arch,history=history):
      report=m.train(root/'data',root/'models',epochs=1,architecture=arch,force_history=history)
      bundle=m.Delta9Bundle(report['model_path']);self.assertEqual(bundle.input_contract['input_shape'],[9])
      x=np.arange(9,dtype=np.float32)*.001; a=bundle.predict_payload(x);bundle.reset_state();b=bundle.predict_payload(x)
      np.testing.assert_allclose(a['force'],b['force'],rtol=0,atol=0)
      self.assertEqual(len(a['probabilities']),25)
      with self.assertRaises(ValueError):bundle.predict_payload([0]*45)
      p,f=bundle.position_model,bundle.force_model
      self.assertFalse({q.data_ptr() for q in p.parameters()} & {q.data_ptr() for q in f.parameters()})
      logits=p(bundle.inputs(x)).detach();bundle.predict_payload(x+.1);self.assertTrue(torch.equal(logits,p(bundle.inputs(x)).detach()))
      if arch=='sparse_cnn':
       g,mask=p.encoder.sparse_input(torch.arange(1,10).float().reshape(1,9));self.assertEqual(g[0,0].tolist(),[[0,4,0],[9,5,3],[8,6,2],[7,0,1]])
      process=subprocess.run([sys.executable,'-m','manual_touch_cnn','--serve',report['model_path']],input=json.dumps(dict(delta_nm=x.tolist(),source_seq=1,timestamp_s=0,reset=True))+'\n',capture_output=True,text=True,encoding='utf-8',timeout=60)
      self.assertEqual(process.returncode,0,process.stderr);ready,result=[json.loads(s) for s in process.stdout.splitlines()]
      self.assertEqual(ready['input_contract'],bundle.input_contract);np.testing.assert_allclose(result['force'],a['force'],rtol=0,atol=1e-6)
      payload=torch.load(report['model_path'],weights_only=False,map_location='cpu');payload['input_contract']=dict(payload['input_contract'],peak_order=list(reversed(payload['input_contract']['peak_order'])))
      forged=root/'forged_contract.pt';torch.save(payload,forged)
      with self.assertRaises(ValueError):m.Delta9Bundle(forged)
      self.assertEqual(report['test_replays'],1)
      self.assertEqual(report['configuration']['chunk_length'],16)
      self.assertEqual(report['configuration']['batch_size'],64)
   # Paired history modes retrain exactly identical stateless position tensors.
   for arch in m.ARCHITECTURES:
    current=torch.load(root/'models'/(arch+'_current.pt'),weights_only=False,map_location='cpu')
    history=torch.load(root/'models'/(arch+'_history.pt'),weights_only=False,map_location='cpu')
    self.assertTrue(all(torch.equal(v,history['position_model_state'][k]) for k,v in current['position_model_state'].items()))
   records=[json.loads(s) for s in (root/'data/frames.jsonl').read_text('utf-8').splitlines()]
   for record in records:
    record['quality'].update(position_labels=[5],position_multihot=[int(i==5) for i in range(1,26)])
   (root/'data/frames.jsonl').write_text('\n'.join(json.dumps(r) for r in records)+'\n',encoding='utf-8')
   report=m.train(root/'data',root/'changed_labels_only',epochs=1,architecture='tcn',force_history=True)
   changed=torch.load(report['model_path'],weights_only=False,map_location='cpu');original=torch.load(root/'models/tcn_history.pt',weights_only=False,map_location='cpu')
   self.assertTrue(all(torch.equal(v,changed['force_model_state'][k]) for k,v in original['force_model_state'].items()))
if __name__=='__main__':unittest.main()
