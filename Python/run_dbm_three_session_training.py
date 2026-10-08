"""Three-session saved-dBm search. Freeze holdout; select only on validation."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import numpy as np
import torch
from touch_dbm_training import prepare_data,random_split,train_trial,DbmModelBundle,SCHEMA
from manual_touch_sensor_token import metrics
ROOT=Path(__file__).resolve().parent
NAMES=['press_grid_20260927_113931','press_grid_20261008_151635','press_grid_20261008_162351']
DEFAULT_OUTPUT=ROOT/'outputs/model_comparisons/sensor_token26_dbm_three_sessions_random_v1'


def main(output=DEFAULT_OUTPUT,smoke=False):
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    if (output/'comparison.json').exists(): raise RuntimeError('Completed run exists; choose another output, do not overwrite')
    paths=[ROOT/'outputs/manual_touch_sequences'/name for name in NAMES]
    data,audit=prepare_data(paths,output)
    split=random_split(data['labels'],seed=20261008)
    split_path=output/'split.json'
    if split_path.exists(): raise RuntimeError('Split already exists; no silent re-splitting')
    split_path.write_text(json.dumps(split,indent=2),encoding='utf-8')
    base={'seed':20261008,'heads':4,'shared':128,'batch_size':64,'max_epochs':240,'patience':45}
    configs=[dict(base,id='token48_flat_regularized',width=48,layers=2,ffn=128,dropout=.05,lr=.001,weight_decay=.0001,force_weight=.2),dict(base,id='token64_flat_regularized',seed=20261009,width=64,layers=2,ffn=192,dropout=.05,lr=.0007,weight_decay=.0001,force_weight=.2),dict(base,id='token48_flat_light_regularization',seed=20261010,width=48,layers=1,ffn=128,dropout=0.,lr=.0015,weight_decay=.00001,force_weight=.1)]
    if smoke: configs=[dict(configs[0],max_epochs=2,patience=2,id='smoke')]
    # This file is committed before any model trains or test metrics are read.
    plan={'input_unit':'dBm','split_sha256':hashlib.sha256(split_path.read_bytes()).hexdigest(),'split_seed':20261008,'split_counts':{k:len(split[k]) for k in ['train','validation','test']},'target_test_accuracy':.80,'configs':configs,'selection':'highest validation accuracy, ties lowest validation cross entropy','test_policy':'Evaluate only selected trial after search; no retraining or repartitioning based on test score','warnings':['Merged randomized frame split permits same-press neighbors across partitions. Not independent-session accuracy.','Results cannot isolate effects of dBm from split, new data, or architecture changes.']}
    (output/'search_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    print(json.dumps({'type':'prepared','total':len(data['labels']),'valid_force':int(data['valid'].sum()),'split':plan['split_counts'],'dbm_conversion_error':audit['maximum_adc_to_saved_dbm_error']}),flush=True)
    trials=[]
    for config in configs:
        if (output/(config['id']+'.pt')).exists(): raise RuntimeError('Trial checkpoint exists; use fresh output')
        trials.append(train_trial(data,split,config,output,audit['feedback_selector']))
    selected=max(trials,key=lambda r:(r['best_validation_accuracy'],-r['best_validation_cross_entropy']))
    target=output/'sensor_token_dbm_selected.pt'
    shutil.copyfile(selected['model_path'],target)
    bundle=DbmModelBundle(target)
    # Only now do we use test labels to evaluate the single selected checkpoint.
    test=np.asarray(split['test'])
    labels,forces,prob=bundle.predict_dbm(data['dbm'][test])
    test_metrics=metrics(data['labels'][test],labels,data['forces'][test],forces,data['valid'][test])
    report=dict(selected)
    report['metrics']=dict(selected['metrics'],test=test_metrics)
    report.update(model_path=str(target.resolve()),test_evaluated_after_selection=True,input_contract='Online ADC second_code -> engineering dBm using saved front-end selector -> 45-dimension train normalization -> 9x5 tokens',feedback_selector=audit['feedback_selector'],validation_warning='三会话合并后按类别分层随机帧划分；同一次按压的邻帧可跨集合，不代表跨会话或独立按压泛化。',dataset_sources=[a['path'] for a in audit['sources']],target_test_accuracy=.80,target_met=test_metrics['position_accuracy']>=.8)
    report['checkpoint_reload_verified']=True
    online=bundle.predict(data['adc'][test])
    np.testing.assert_array_equal(online[0],labels)
    np.testing.assert_allclose(online[1],forces,rtol=2e-4,atol=2e-4)
    np.testing.assert_allclose(online[2],prob,rtol=2e-4,atol=2e-4)
    report['saved_dbm_vs_live_adc_conversion_verified']=True
    for name,idx in [(k,np.asarray(split[k])) for k in ('train','validation','test')]:
        y,pf,pr=bundle.predict_dbm(data['dbm'][idx])
        np.savez_compressed(output/(name+'_predictions.npz'),indices=idx,labels=data['labels'][idx],predicted_labels=y,forces=data['forces'][idx],predicted_forces=pf,force_valid=data['valid'][idx],probabilities=pr,session=data['session'][idx])
    sample=data['adc'][test[0]]
    for _ in range(10): bundle.predict(sample)
    timings=[]
    for _ in range(100):
        start=time.perf_counter(); bundle.predict(sample); timings.append((time.perf_counter()-start)*1000)
    report['cpu_single_frame_latency_ms']={'median':float(np.median(timings)),'p95':float(np.percentile(timings,95)),'threads':2}
    # Verify raw source hashes after all work; leave original JSONL untouched.
    for source in audit['sources']:
        digest=hashlib.sha256()
        with Path(source['path']).open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''): digest.update(chunk)
        if digest.hexdigest()!=source['sha256']: raise RuntimeError('Source changed during training')
    report['source_hashes_unchanged']=True
    payload=torch.load(target,map_location='cpu',weights_only=False); payload['report']=report; torch.save(payload,target)
    target.with_suffix('.report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    comparison={'schema':SCHEMA,'input_unit':'dBm','split_counts':plan['split_counts'],'split_description':split['description'],'selected_models':{'sensor_token':{'model_path':str(target.resolve()),'report_path':str(target.with_suffix('.report.json').resolve()),'criterion':plan['selection']}},'search_plan':plan,'trials':trials,'selected_report':report,'deployment_ready':False,'warnings':plan['warnings']}
    (output/'comparison.json').write_text(json.dumps(comparison,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'type':'completed','selected_trial':selected['configuration']['id'],'best_epoch':selected['best_epoch'],'test_metrics':test_metrics,'target_met':report['target_met'],'model_path':str(target.resolve())},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,default=DEFAULT_OUTPUT); parser.add_argument('--smoke',action='store_true'); args=parser.parse_args(); main(args.output,args.smoke)
