"""Torch-free four-source migration guard. No training, inference or pin writes."""
import json,copy
from pathlib import Path
import numpy as np

def promote_four45(current,folder):
 from touch_deployment_slots import ROOT,digest,register_training,validate_pair
 def load(p):return json.loads(Path(p).read_text('utf8'))
 manifest=folder/'matched_comparison_record.json'
 if not manifest.exists():raise ValueError('Four-source comparison manifest missing')
 record=load(manifest)
 if record.get('schema')!='fbg_matched45_four_source_v1':raise ValueError('Wrong four-source comparison manifest')
 required=['validation_selection_frozen.json','selection_identity_frozen.json','frozen_split.json','frozen_run_plan.json','pin_before.json','data_audit.json','trainval_arrays.npz','default_validation_predictions.npz','source_auxiliary_split.json','combo_auxiliary_split.json','auxiliary_progress.json','external_evaluation.json','default_external_predictions.npz','selected_external_predictions.npz','service_verification.json']
 if not set(required).issubset(record['evidence_sha256']):raise ValueError('Incomplete four-source manifest')
 for rel,h in record['evidence_sha256'].items():
  p=(folder/rel).resolve()
  if not p.is_relative_to(folder) or digest(p)!=h:raise ValueError('Four-source evidence changed: '+rel)
 def bound(p):
  p=Path(p).resolve()
  if not p.is_relative_to(folder) or record['evidence_sha256'].get(str(p.relative_to(folder)))!=digest(p):raise ValueError('Unbound four-source artifact')
  return p
 s=load(folder/'validation_selection_frozen.json');split=load(folder/'frozen_split.json');plan=load(folder/'frozen_run_plan.json');before=load(folder/'pin_before.json');rows=split['rows'];tr=split['train'];va=split['validation'];raw=np.load(folder/'trainval_arrays.npz')
 if s['plan_sha256']!=digest(folder/'frozen_run_plan.json') or s['split_sha256']!=digest(folder/'frozen_split.json') or plan['split_sha256']!=s['split_sha256']:raise ValueError('Four-source frozen plan/split mismatch')
 identity=load(folder/'selection_identity_frozen.json')
 if identity['selection_sha256']!=digest(folder/'validation_selection_frozen.json') or identity['external_labels_opened'] is not False:raise ValueError('Four-source component identity not frozen before external')
 for task in ['position','force']:
  component=identity['components'][task]
  if component['model_path']!=s[task] or component['checkpoint_sha256']!=digest(s[task]) or component['report_sha256']!=digest(Path(s[task]).with_suffix('.report.json')) or component['confirmation_sha256']!=digest(s['confirm'][task]):raise ValueError('Four-source component changed after freeze')
 if set(tr)&set(va) or {rows[i]['group'] for i in tr}&{rows[i]['group'] for i in va}:raise ValueError('Four-source split overlap')
 vh={rows[i]['row_sha256'] for i in va}
 for task in ['position','force']:
  if current[task]!=before[task] or record['default_identities'][task]!=current[task]['checkpoint_sha256']:raise ValueError('Four-source incumbent identity changed')
  checkpoint=Path(current[task]['model_path']);old=load(checkpoint.parent/'frozen_split.json');r=load(current[task]['report_path']);src=checkpoint.parent.parent/'trainval_arrays.npz'
  if digest(src)!=r['source_sha256'] or vh&{old['rows'][i]['row_sha256'] for i in old['train']}:raise ValueError('Four-source incumbent training overlap/source mismatch')
 scale=np.array([raw['forces'][tr,j][raw['valid'][tr,j]].std() for j in range(3)]);scale[scale<1e-5]=1
 def aligned(path,ix):
  z=np.load(bound(path))
  for key,target in [('indices',ix),('labels',raw['labels'][ix]),('true_forces',raw['forces'][ix]),('valid',raw['valid'][ix])]:
   if not np.array_equal(z[key],target,equal_nan=True):raise ValueError('Four-source reference misalignment '+key)
  return z
 def pm(a):
  y=a['labels'].astype(bool);p=a['probabilities']>=float(a['threshold']);same=(y==p).all(1);n=y.sum(1);tp=(y&p).sum();fp=(~y&p).sum();fn=(y&~p).sum()
  return dict(exact=float(same.mean()),f1=float(2*tp/max(2*tp+fp+fn,1)),single=float(same[n==1].mean()) if (n==1).any() else None,multi=float(same[n>1].mean()) if (n>1).any() else None,empty_fp=int(p[n==0].any(1).sum()),stress_exact=float(((a['stress_probabilities']>=float(a['threshold']))==y).all(1).mean()) if 'stress_probabilities' in a else None)
 def fm(a,group='all'):
  mask=np.ones(len(a['labels']),bool)
  if group=='empty':mask=a['labels'].sum(1)==0
  if group=='loaded':mask=np.linalg.norm(np.nan_to_num(a['true_forces']),axis=1)>=1
  if not mask.any():return np.full(3,np.nan)
  return np.array([np.abs(a['forces'][mask&a['valid'][:,j],j]-a['true_forces'][mask&a['valid'][:,j],j]).mean() for j in range(3)])
 def pg(a,b):
  a=pm(a);b=pm(b)
  return bool(b['exact']>=a['exact'] and b['f1']>=a['f1'] and (b['exact']>a['exact'] or b['f1']>a['f1']) and b['empty_fp']<=a['empty_fp'] and all(b[k]>=a[k] for k in ['single','multi','stress_exact'] if a[k] is not None))
 def fg(a,b):return bool(np.all(fm(b)<=fm(a)) and np.mean(fm(b)/scale)<=.99*np.mean(fm(a)/scale) and fm(b,'empty')[2]<min(.2,.8*fm(a,'empty')[2]) and np.all(fm(b,'loaded')<=fm(a,'loaded')))
 base=aligned(folder/'default_validation_predictions.npz',va);ext=load(folder/'external_evaluation.json');ea=np.load(bound(folder/'default_external_predictions.npz'));eb=np.load(bound(folder/'selected_external_predictions.npz'))
 if ext['selection_sha256']!=digest(folder/'validation_selection_frozen.json') or not ext['known_reused_test'] or not ext['no_post_test_search']:raise ValueError('Four-source external selection not frozen')
 for key in ['source_line','labels','true_forces','valid']:
  if len(ea[key])!=40 or not np.array_equal(ea[key],eb[key],equal_nan=True):raise ValueError('Four-source external reference misalignment')
 ax=pm(ea);bx=pm(eb);veto=dict(position=bx['exact']<ax['exact'] or bx['f1']<ax['f1'],force=bool(np.any(fm(eb)>fm(ea)) or np.any(fm(eb,'loaded')>fm(ea,'loaded')) or not fm(eb,'empty')[2]<fm(ea,'empty')[2]))
 if veto!=ext['safety_veto']:raise ValueError('Four-source strict external veto mismatch')
 out=copy.deepcopy(current);decisions={};auxreports=load(folder/'auxiliary_progress.json');control='mlp192_huber';selected_reports={t:load(Path(s[t]).with_suffix('.report.json')) for t in ['position','force']}
 for task in ['position','force']:
  r=selected_reports[task];p=bound(s[task]);bound(p.with_suffix('.report.json'));cp=bound(s['confirm'][task]);bound(cp.with_suffix('.report.json'));cr=load(cp.with_suffix('.report.json'));candidate=register_training(p);register_training(cp)
  if r['seed']==cr['seed'] or r['spec']!=cr['spec']:raise ValueError('Four-source distinct matched seeds required')
  traces=[aligned(q.parent/'validation_predictions.npz',va) for q in [p,cp]]
  for report,trace in zip([r,cr],traces):
   if report['statistics_fit_indices']!=tr or report['source_sha256']!=digest(folder/'trainval_arrays.npz'):raise ValueError('Four-source training source/statistics mismatch')
   met=report['metrics']['validation'];po=pm(trace)
   if not np.allclose([po['exact'],po['f1']],[met['position_exact_match'],met['position_micro_f1']],atol=1e-10,rtol=0) or not np.allclose(fm(trace),met['force_mae_n'],atol=1e-7,rtol=0):raise ValueError('Four-source report/prediction mismatch')
  gates={k:fun(base,z) for k,z in zip(['primary_gate','second_seed_gate'],traces) for fun in ([pg] if task=='position' else [fg])}
  auxpasses=[]
  for mode in ['source','combo']:
   sp=load(folder/(mode+'_auxiliary_split.json'));ix=sp['validation'];ms=[]
   if set(sp['train'])&set(ix):raise ValueError('Four-source auxiliary split overlap')
   if mode=='source' and any(rows[i]['session'].endswith('172418') for i in sp['train']):raise ValueError('Four-source session auxiliary overlap')
   if mode=='combo':
    trainsets={tuple(rows[i]['labels']) for i in sp['train']}
    if any(len(rows[i]['labels'])>1 and tuple(rows[i]['labels']) in trainsets for i in ix):raise ValueError('Four-source unseen-combo auxiliary overlap')
   for seed in plan['seeds']:
    found=[]
    for id in [control,r['spec']['id']]:
     rr=next((v for v in auxreports if v['prefix']=='aux_'+mode and v['seed']==seed and v['spec']['id']==id),None)
     if rr is None:found=[];break
     q=bound(rr['model_path']);bound(q.with_suffix('.report.json'));register_training(q)
     if rr['statistics_fit_indices']!=sp['train'] or load(q.parent/'frozen_split.json')!=sp:raise ValueError('Four-source auxiliary metadata mismatch')
     found.append(aligned(q.parent/'validation_predictions.npz',ix))
    ok=False
    if len(found)==2:
     a,b=found
     if task=='position':a,b=pm(a),pm(b);ok=b['exact']>=a['exact'] and b['f1']>=a['f1']
     else:ok=bool(np.all(fm(b)<=fm(a)))
    ms.append(ok)
   auxpasses+=ms
   if task=='position':gates[mode+'_auxiliary_gate']=all(ms)
  if task=='force':gates['auxiliary_gate']=all(auxpasses)
  for key,value in gates.items():
   if s['decisions'][task].get(key)!=value:raise ValueError('Four-source declared gate differs: '+task+'/'+key)
  allowed=all(gates.values()) and not veto[task];decisions[task]=dict(validation_gates=gates,external_safety_veto=veto[task],promoted=allowed)
  if allowed:out[task]=candidate[task];out['trusted_sources'].update(candidate['trusted_sources'])
 if any(d['promoted'] for d in decisions.values()):
  out['matched_comparison_record']=dict(path=str(manifest),sha256=digest(manifest));out['validation_description']='四源完整标注/tare块；实际旧默认训练行hash排除，双seed+整源/组合辅助+已知重复外测严格不退守门；非盲测';out['warning']='实验模型／泛化未达标；独立任务通过四源证据守门；不代表未来现场泛化，加载不启动硬件或推理。'
 validate_pair(out);return out,decisions
