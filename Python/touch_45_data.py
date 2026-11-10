"""Strict real9x5 engineering-dBm input, single or operator multi-hot labels.
No wavelength baseline, interpolation or padding is used as model input.
"""
import hashlib,json,math
from pathlib import Path
import numpy as np
from temporary_optical_units import adc_code_to_dbm
from touch_delta9_data import ARCHITECTURES,freeze_split,fit_statistics
MODEL_SCHEMA='fbg_touch45_multilabel_v1'

def route_digest(indices,waves):
 return hashlib.sha256(json.dumps(list(zip(indices,waves)),separators=(',',':'),allow_nan=False).encode()).hexdigest()

def validate_input_contract(c):
 expected=dict(input_shape=[9,5],point_count=45,points_per_peak=5,feature_kind='stable_optical_power_dbm',feature_units='dBm',stable_sample_field='second_code',force_semantics='PX6D_host_tare_corrected_whole_contact_resultant_N')
 if not isinstance(c,dict) or any(c.get(k)!=v for k,v in expected.items()):raise ValueError('Requires45 real second_code dBm features')
 if c.get('channel') not in (0,1,2,3) or type(c.get('feedback_selector'))!=int or not 0<=c['feedback_selector']<=7:raise ValueError('Invalid channel/selector binding')
 if c['channel']>=2 and c['feedback_selector']!=0:raise ValueError('CH2/3 requires effective selector0')
 indices=c.get('route_indices');waves=c.get('route_wavelengths_nm')
 legacy=c.get('route_binding')=='measured_wavelengths_legacy_no_indices'
 if not isinstance(indices,list) or len(indices)!=45:raise ValueError('Requires45 ordered route entries')
 if legacy:
  if any(i is not None for i in indices):raise ValueError('Partial legacy route rejected')
 elif any(type(i)!=int for i in indices) or len(set(indices))!=45:raise ValueError('Requires45 ordered actual route indices')
 if not isinstance(waves,list) or len(waves)!=45 or not np.isfinite(waves).all() or c.get('route_sha256')!=route_digest(indices,waves):raise ValueError('Invalid measured route hash')
 return c

def validate_record(r):
 q=r.get('quality') or {}
 if r.get('schema')=='fbg-touch-delta9/v1' or q.get('point_count',45)!=45 or q.get('points_per_peak',5)!=5:raise ValueError('Not a45 measured point record')
 adc=np.asarray(q.get('stable_adc'),float);records=q.get('stable_records') or [];waves=r.get('wavelengths_nm');views=q.get('training_views') or {}
 if adc.shape!=(45,) or not np.isfinite(adc).all() or len(records)!=45 or not np.array_equal(adc,[v.get('second_code') for v in records]):raise ValueError('Requires45 finite actual second_code measurements')
 selector=q.get('feedback_selector');channel=q.get('adc_channel',1)
 c=dict(input_shape=[9,5],point_count=45,points_per_peak=5,feature_kind='stable_optical_power_dbm',feature_units='dBm',stable_sample_field='second_code',force_semantics='PX6D_host_tare_corrected_whole_contact_resultant_N',channel=channel,feedback_selector=selector,route_indices=q.get('route_indices') or [v.get('index') for v in records],route_wavelengths_nm=waves)
 c['route_binding']='measured_wavelengths_legacy_no_indices' if all(i is None for i in c['route_indices']) else 'ordered_indices_and_measured_wavelengths'
 c['route_sha256']=route_digest(c['route_indices'],waves);validate_input_contract(c)
 dbm=np.asarray(views.get('stable_optical_power_dbm',q.get('stable_dbm')),float)
 if dbm.shape!=(45,) or not np.isfinite(dbm).all() or not np.allclose(dbm,adc_code_to_dbm(adc,selector,floor_nonpositive=True),rtol=0,atol=1e-5):raise ValueError('Saved dBm must equal actual second_code conversion')
 if q.get('stable_dbm') is not None and not np.allclose(dbm,q['stable_dbm'],rtol=0,atol=1e-8):raise ValueError('Saved dBm aliases disagree')
 y=np.zeros(25,np.float32)
 if q.get('annotation_mode')=='multi':
  labels=q.get('position_labels')
  if not isinstance(labels,list) or any(type(i)!=int or not 1<=i<=25 for i in labels) or labels!=sorted(set(labels)):raise ValueError('Invalid contact set')
  y[[i-1 for i in labels]]=1
  if q.get('position_multihot')!=y.astype(int).tolist():raise ValueError('Invalid multi-hot target')
 else:
  label=q.get('press_label')
  if type(label)!=int or not 0<=label<=25:raise ValueError('Invalid single position (never silently relabel as empty)')
  if label:y[label-1]=1
 ref=q.get('paxini_force') or {};force=np.array([ref.get(k) for k in ('fx_n','fy_n','fz_n')],np.float32)
 return dbm.astype(np.float32),y,force,c

def prepare_data(directory):
 directory=Path(directory);files=[directory/'frames.jsonl'] if (directory/'frames.jsonl').exists() else sorted(directory.rglob('frames.jsonl'))
 if not files:raise ValueError('No45 point frames.jsonl')
 rows=[];contract=None;digest=hashlib.sha256();smoke=False
 for sid,path in enumerate(files):
  manifest=path.parent/'manifest.json'
  if manifest.exists() and json.loads(manifest.read_text('utf-8')).get('schema')=='fbg-touch-delta9/v1':raise ValueError('Delta9 sessions require their separate reader')
  with path.open('rb') as stream:
   for line_no,raw in enumerate(stream,1):
    digest.update(raw);r=json.loads(raw);x,y,f,c=validate_record(r)
    if contract is None:contract=c
    elif contract!=c:raise ValueError('Mixed channel/selector/ordered route contracts')
    q=r['quality'];clock=q.get('cycle_start_us');clock=float('nan') if clock is None else float(clock)/1e6
    rows.append((x,y,f,sid,line_no,clock));smoke|=bool(q.get('smoke_only')) or 'synthetic' in str(q.get('frame_provenance','')).lower()
 if not rows:raise ValueError('Empty45 point dataset')
 f=np.stack([r[2] for r in rows])
 return dict(features=np.stack([r[0] for r in rows]),labels=np.stack([r[1] for r in rows]),forces=f,valid=np.isfinite(f),session=np.array([r[3] for r in rows]),source_line=np.array([r[4] for r in rows]),timestamp_s=np.array([r[5] for r in rows]),input_contract=contract,source_sha256=digest.hexdigest(),smoke_only=smoke)
