"""Torch-first synthetic pipeline probe: real gradients, save/load and service."""
import os,sys,json,subprocess,time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent;os.chdir(ROOT);sys.path.insert(0,str(ROOT))
OUT=Path(sys.argv[1]);session=Path(json.loads((OUT/'qt_writer_replay.json').read_text('utf-8'))['session'])
from touch_five_training import prepare_data,train,ARCHITECTURES
import touch_temporal_models as models
from manual_touch_cnn import load_touch_model,prediction_payload
import torch
data,selector=prepare_data(session)
assert data['dbm'].shape==(52,45) and data['smoke_only']
assert not data['valid'][0].any() and np.isnan(data['forces'][0]).all()
rows=[]
run_id=str(time.time_ns())
for architecture in ARCHITECTURES:
 paired=[]
 for history in (False,True):
  output=OUT/'synthetic_models'/run_id/architecture/('history' if history else 'nohistory')
  report=train(session,output,epochs=1,architecture=architecture,force_history=history)
  path=Path(report['model_path']);bundle=load_touch_model(path)
  assert bundle.report['smoke_only'] and bundle.mean.shape==(9,5)
  assert len(set(p.data_ptr() for p in bundle.position_model.parameters()) & set(p.data_ptr() for p in bundle.force_model.parameters()))==0
  checkpoint=torch.load(path,map_location='cpu',weights_only=False)
  paired.append(checkpoint['position_model_state'])
  adc=np.asarray([r['second_code'] for r in json.loads((session/'frames.jsonl').read_text('utf-8').splitlines()[1])['quality']['stable_records']],np.float32)
  bundle.reset_state();expected=prediction_payload(bundle,adc)
  # Same current optical frame has stateless position despite force recurrence.
  again=prediction_payload(bundle,adc)
  np.testing.assert_array_equal(expected['probabilities'],again['probabilities'])
  requests=[dict(adc=adc.tolist(),request_id=1,source_seq=1,timestamp_s=1.,session_id='synthetic'),
      dict(adc=adc.tolist(),request_id=2,source_seq=3,timestamp_s=1.2,session_id='synthetic'),
      dict(adc=[1]*63,request_id=3,source_seq=4,timestamp_s=1.3,session_id='synthetic')]
  result=subprocess.run([sys.executable,'-m','manual_touch_cnn','--serve',str(path)],input=''.join(json.dumps(r)+'\n' for r in requests),capture_output=True,text=True,encoding='utf-8',timeout=90)
  assert result.returncode==0,result.stderr
  payloads=[json.loads(l) for l in result.stdout.splitlines()]
  assert payloads[0]['type']=='ready' and payloads[0]['input_shape']==[9,5]
  assert payloads[1]['type']=='prediction' and payloads[3]['type']=='error'
  np.testing.assert_allclose(payloads[1]['force'],expected['force'],rtol=0,atol=1e-6)
  assert payloads[2]['reset_reason']
  rows.append(dict(architecture=architecture,force_history=history,model_path=str(path),
      input_shape=[9,5],smoke_only=True,service_force_parity=True,position_stateless=True,independent_parameters=True,
      expected_adc=adc.tolist(),expected_prediction=expected))
 for name in paired[0]:assert torch.equal(paired[0][name],paired[1][name]),'history mode changed current-frame position weights'
(OUT/'pipeline_smoke.json').write_text(json.dumps(dict(smoke_only=True,hardware_used=False,formal_training=False,
    source_session=str(session),frames=52,trials=rows),ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(dict(smoke_only=True,actual_one_epoch_independent_trials=len(rows),all_four_architectures=True,paired_history=True,service_parity=True,missing_force_mask=True)))
