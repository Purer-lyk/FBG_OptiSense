"""Strict Torch-free reader for explicit unloaded-baseline nine-peak nm deltas."""
import hashlib,json
from pathlib import Path
import numpy as np
from delta9_capture import SCHEMA,PEAK_TO_POSITION
MODEL_SCHEMA='fbg_touch_delta9_multilabel_v1'
ARCHITECTURES=('tcn','sensor_token','sparse_cnn','inception_1d')

def validate_input_contract(c):
 expected=dict(input_shape=[9],feature_kind='delta_wavelength9',feature_units='nm',peak_order=[f'P{i}' for i in range(1,10)],peak_to_position_labels=list(PEAK_TO_POSITION),peak_selection_method='makima_curve_global_max_dbm',baseline_semantics='current_same_frame_peak_nm-minus-explicit_unloaded_baseline_nm',source_point_count=45,force_semantics='PX6D_host_tare_corrected_whole_contact_resultant_N')
 if not isinstance(c,dict) or any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Invalid delta9 order/units/baseline/force semantics')
 if c.get('channel') not in (0,1,2,3) or type(c.get('feedback_selector'))!=int or not isinstance(c.get('route_sha256'),str) or not c['route_sha256']:raise ValueError('Missing channel/selector/ordered route binding')
 return c

def validate_record(r):
 if r.get('schema')!=SCHEMA or r.get('feature_kind')!='delta_wavelength9' or r.get('feature_units')!='nm' or r.get('input_shape')!=[9]:raise ValueError('Requires delta9 nm capture, never legacy45/crop/pad')
 if r.get('peak_order')!=[f'P{i}' for i in range(1,10)] or r.get('peak_to_position_labels')!=list(PEAK_TO_POSITION):raise ValueError('Wrong nine-peak order')
 arrays=[np.asarray(r.get(k),float) for k in ('values','delta_wavelength_nm','current_peak_wavelengths_nm','baseline_wavelengths_nm')]
 if any(a.shape!=(9,) or not np.isfinite(a).all() for a in arrays):raise ValueError('Nine finite wavelengths/deltas required')
 if not np.allclose(arrays[0],arrays[2]-arrays[3],rtol=0,atol=1e-10) or not np.array_equal(arrays[0],arrays[1]):raise ValueError('Delta must equal current minus explicit baseline')
 q=r.get('quality') or {};ctx=r.get('baseline_context') or {}
 if not q.get('baseline_id') or not r.get('baseline_time') or ctx.get('source_point_count')!=45 or ctx.get('fit_method')!='makima' or not ctx.get('route_sha256'):raise ValueError('Missing explicit baseline/route binding')
 if r.get('peak_selection_method')!='makima_curve_global_max_dbm':raise ValueError('Unsupported peak semantics')
 y=np.zeros(25,np.float32)
 if q.get('annotation_mode')=='multi':
  labels=q.get('position_labels');hot=q.get('position_multihot')
  if not isinstance(labels,list) or any(type(i)!=int or not 1<=i<=25 for i in labels) or labels!=sorted(set(labels)):raise ValueError('Invalid multi labels')
  y[[i-1 for i in labels]]=1
  if hot!=y.astype(int).tolist():raise ValueError('Multi-hot does not match labels')
 else:
  label=q.get('press_label')
  if type(label)!=int or not 0<=label<=25:raise ValueError('Invalid single label')
  if label:y[label-1]=1
 force=q.get('paxini_force') or {}
 f=np.array([force.get(k) for k in ('fx_n','fy_n','fz_n')],np.float32)
 contract=dict(input_shape=[9],feature_kind='delta_wavelength9',feature_units='nm',peak_order=r['peak_order'],peak_to_position_labels=r['peak_to_position_labels'],peak_selection_method=r['peak_selection_method'],baseline_semantics='current_same_frame_peak_nm-minus-explicit_unloaded_baseline_nm',channel=ctx.get('channel'),feedback_selector=ctx.get('feedback_selector'),route_sha256=ctx['route_sha256'],source_point_count=45,force_semantics='PX6D_host_tare_corrected_whole_contact_resultant_N')
 validate_input_contract(contract)
 return arrays[0].astype(np.float32),y,f,contract

def prepare_data(directory):
 directory=Path(directory)
 manifests=list(directory.rglob('manifest.json')) if not (directory/'frames.jsonl').exists() else [directory/'manifest.json']
 files=sorted(directory.rglob('frames.jsonl')) if not (directory/'frames.jsonl').exists() else [directory/'frames.jsonl']
 if not files:raise ValueError('No delta9 frames.jsonl')
 for m in manifests:
  if m.exists() and json.loads(m.read_text('utf-8')).get('schema')!=SCHEMA:raise ValueError('Mixed/legacy dataset manifest rejected')
 rows=[];contract=None;digest=hashlib.sha256();smoke=False
 for sid,path in enumerate(files):
  for line_no,raw in enumerate(path.read_bytes().splitlines(),1):
   digest.update(raw);r=json.loads(raw);x,y,f,c=validate_record(r)
   if contract is None:contract=c
   elif c!=contract:raise ValueError('Mixed route/channel/selector contracts rejected')
   q=r['quality'];t=q.get('cycle_start_us');t=float('nan') if t is None else float(t)/1e6
   rows.append((x,y,f,sid,line_no,t));smoke|=bool(q.get('smoke_only'))
 if not rows:raise ValueError('Empty delta9 dataset')
 return dict(features=np.stack([r[0] for r in rows]),labels=np.stack([r[1] for r in rows]),forces=np.stack([r[2] for r in rows]),valid=np.isfinite(np.stack([r[2] for r in rows])),session=np.array([r[3] for r in rows]),source_line=np.array([r[4] for r in rows]),timestamp_s=np.array([r[5] for r in rows]),input_contract=contract,source_sha256=digest.hexdigest(),smoke_only=smoke)

def freeze_split(data,output):
 n=len(data['features'])
 if n<6:raise ValueError('At least six frames for train/validation/test')
 ids=np.random.default_rng(20261014).permutation(n);a=max(1,int(n*.7));b=max(a+1,int(n*.85));b=min(b,n-1)
 split={k:sorted(v.tolist()) for k,v in zip(('train','validation','test'),(ids[:a],ids[a:b],ids[b:]))}
 payload=dict(source_sha256=data['source_sha256'],input_contract=data['input_contract'],split=split,architecture_roster=list(ARCHITECTURES),protocol='fixed seeded random frame 70/15/15; neighboring-frame leakage possible',selection_source='validation only',test_policy='one replay after validation selection per trial')
 path=Path(output)/'frozen_split.json';path.parent.mkdir(parents=True,exist_ok=True)
 if path.exists() and json.loads(path.read_text('utf-8'))!=payload:raise ValueError('Frozen split/source differs')
 if not path.exists():path.write_text(json.dumps(payload,indent=2),encoding='utf-8')
 return split

def fit_statistics(data,indices):
 x=data['features'][indices];f=data['forces'][indices];valid=data['valid'][indices]
 mean=x.mean(0);std=x.std(0);std[std<1e-6]=1
 fm=np.array([f[valid[:,a],a].mean() if valid[:,a].any() else 0 for a in range(3)],np.float32)
 fs=np.array([f[valid[:,a],a].std() if valid[:,a].any() else 1 for a in range(3)],np.float32);fs[fs<1e-5]=1
 return dict(feature_mean=mean,feature_std=std,force_mean=fm,force_std=fs)
