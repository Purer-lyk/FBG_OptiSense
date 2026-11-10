"""Two independent components in one CPU child; predictions share one exact request."""
import json,sys
from pathlib import Path
import numpy as np
import torch
from touch_deployment_slots import read_pair,ROOT
torch.set_num_threads(2)
class Component:
 def __init__(self,c):
  self.c=c;self.task=c['task'];self.state=None;self.last=None
  if c['adapter']=='groupcv':
   import joblib
   self.payload=joblib.load(c['model_path']);self.entry=self.payload[self.task]
   if self.payload['input_contract']!=c['input_contract']:raise ValueError('Checkpoint contract mismatch')
   if self.entry['kind']=='neural':
    sys.path.insert(0,str(ROOT/'outputs/delta9_search_20261015_094820'))
    from selected_bundle import make_model
    self.networks=[]
    for a in self.entry[self.task]:
     n=make_model(a['config'],25 if self.task=='position' else 3,a['seed']);n.load_state_dict(a['state']);n.eval();self.networks.append((n,a))
  elif c['adapter']=='composition':
   import joblib,importlib.util
   source=ROOT/'outputs/delta9_composition_20261015_114020/composition_core.py'
   spec=importlib.util.spec_from_file_location('trusted_composition_runtime',source);self.composition=importlib.util.module_from_spec(spec);spec.loader.exec_module(self.composition)
   self.payload=joblib.load(c['model_path']);self.entry=self.payload[self.task]
   if self.payload['input_contract']!=c['input_contract'] or self.entry.get('task')!=self.task:raise ValueError('Composition component contract differs')
   if self.entry['kind']=='neural':
    a=self.entry;n=self.composition.make_net(a['config'],self.task,a['seed']);n.load_state_dict(a['state']);n.eval();self.networks=[(n,a)]
  elif c['adapter']=='search':
   sys.path.insert(0,str(ROOT/'outputs/delta9_search_20261015_094820'))
   from selected_bundle import make_model
   self.payload=torch.load(c['model_path'],map_location='cpu',weights_only=False)
   if self.payload['input_contract']!=c['input_contract']:raise ValueError('Search checkpoint input mismatch')
   self.entry=dict(kind='neural',**{self.task:self.payload[self.task]});self.networks=[]
   for a in self.entry[self.task]:
    n=make_model(a['config'],25 if self.task=='position' else 3,a['seed']);n.load_state_dict(a['state']);n.eval();self.networks.append((n,a))
  else:
   if c['adapter']=='measured45':
    from touch_45_models import Measured45Bundle
    self.bundle=Measured45Bundle(c['model_path'])
   else:
    from touch_delta9_models import Delta9Bundle
    self.bundle=Delta9Bundle(c['model_path'])
   if self.bundle.input_contract!=c['input_contract']:raise ValueError('Checkpoint contract mismatch')
 def predict(self,x,request):
  if self.c['adapter']=='composition':
   a=self.entry
   if a['kind']!='neural':return self.composition.predict(a,x)
   with torch.inference_mode():raw=self.networks[0][0](torch.from_numpy((x-a['scaler']['center'])/a['scaler']['scale']))[0]
   return raw.sigmoid().numpy() if self.task=='position' else raw.numpy()*a['force_scaler']['scale']+a['force_scaler']['center']
  if self.c['adapter'] in ('groupcv','search'):
   e=self.entry
   if e['kind']=='classical':
    z=((x-e['center'])/e['scale']).astype(np.float64)
    if self.task=='position':return 1/(1+np.exp(-np.clip(e['position'].predict(z),-60,60)))
    return np.stack([m.predict(z) for m in e['force']],1)*e['force_scale']+e['force_center']
   rows=[]
   for net,a in self.networks:
    with torch.inference_mode():y=net(torch.from_numpy((x-a['scaler']['center'])/a['scaler']['scale']))
    rows.append(y.sigmoid().numpy() if self.task=='position' else y.numpy()*a['force_scaler']['scale']+a['force_scaler']['center'])
   return np.mean(rows,0)
  b=self.bundle;clock=request.get('timestamp_s');seq=request.get('source_seq');session=request.get('session_id')
  if request.get('reset') or self.last is None or session!=self.last[0] or (seq is not None and self.last[1] is not None and seq!=self.last[1]+1) or (clock is not None and self.last[2] is not None and (clock<=self.last[2] or clock-self.last[2]>b.config.get('max_gap_s',2))):self.state=None
  self.last=(session,seq,clock)
  with torch.inference_mode():
   t=b.inputs(x)
   if self.task=='position':
    logits=b.position_model(t)
    return (logits.softmax(-1) if b.config.get('position_mode')=='single_label26_v1' else logits.sigmoid()).numpy()
   y,self.state=b.force_model(t,self.state);return y.numpy()*b.stats['force_std']+b.stats['force_mean']
class DualBundle:
 def __init__(self,pair):
  self.pair=pair;self.position=Component(pair['position']);self.force=Component(pair['force'])
  def storage(component):
   if hasattr(component,'networks'):nets=[n for n,_ in component.networks]
   elif hasattr(component,'bundle'):nets=[getattr(component.bundle,component.task+'_model')]
   else:nets=[]
   return {p.data_ptr() for net in nets for p in net.parameters()}
  self.independent_storage=not bool(storage(self.position)&storage(self.force))
  if not self.independent_storage:raise ValueError('Position and force share learned parameter storage')
 def predict(self,r):
  contract=self.pair['position']['input_contract'];measured=contract.get('input_shape')==[9,5]
  key='dbm' if measured else 'delta_nm';size=45 if measured else 9
  if measured and r.get('input_contract')!=contract:raise ValueError('Request route/channel/selector contract mismatch')
  if key not in r or ('delta_nm' in r if measured else 'dbm' in r):raise ValueError('Wrong input feature kind')
  x=np.asarray(r[key],np.float32)
  if x.shape!=(size,) or not np.isfinite(x).all():raise ValueError('Exactly '+str(size)+' finite measured features required')
  p=self.position.predict(x[None],r)[0];f=self.force.predict(x[None],r)[0]
  if not np.isfinite(p).all() or not np.isfinite(f).all():raise ValueError('Nonfinite prediction')
  if getattr(getattr(self.position,'bundle',None),'config',{}).get('position_mode')=='single_label26_v1':
   from touch_single26_contract import decode_single26
   decoded=decode_single26(p)
   return dict(type='prediction',request_id=r.get('request_id'),source_seq=r.get('source_seq'),timestamp_s=r.get('timestamp_s'),session_id=r.get('session_id'),force=f.tolist(),decision=decoded['decision_status'],**decoded)
  threshold=self.position.payload['threshold'] if self.pair['position']['adapter'] in ('groupcv','search','composition') else float(self.position.bundle.config.get('position_threshold',.5))
  labels=(np.flatnonzero(p>=threshold)+1).tolist();confidence=float(np.maximum(p,1-p).min())
  return dict(type='prediction',request_id=r.get('request_id'),source_seq=r.get('source_seq'),timestamp_s=r.get('timestamp_s'),session_id=r.get('session_id'),position_labels=labels,position=None,probabilities=p.tolist(),force=f.tolist(),confidence=confidence,decision='uncertain' if confidence<.65 else ('pressed' if labels else 'no_press'))
def main():
 try:
  pair=read_pair(sys.argv[1]);bundle=DualBundle(pair)
  print(json.dumps(dict(type='ready',independent_parameter_storage=bundle.independent_storage,input_contract=pair['position']['input_contract'],components=[pair[t]['component_id'] for t in ('position','force')])),flush=True)
  for line in sys.stdin:
   try:print(json.dumps(bundle.predict(json.loads(line)),allow_nan=False),flush=True)
   except Exception as e:print(json.dumps(dict(type='error',message=str(e))),flush=True)
 except Exception as e:print(json.dumps(dict(type='error',message=str(e))),flush=True);raise
if __name__=='__main__':main()
