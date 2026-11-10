"""Torch-free, hash-bound independent deployment selections. No directory recency ranking."""
import copy,hashlib,json,math,os,shutil,time
from pathlib import Path
ROOT=Path(__file__).resolve().parent
SCHEMA='fbg_touch_independent_slots_v3'
def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def _component(path,task,report,protocol,metric,identity,adapter):
 path=Path(path).resolve(); side=path.with_suffix('.report.json')
 component=dict(task=task,model_path=str(path),checkpoint_sha256=digest(path),report_path=str(side),report_sha256=digest(side),input_contract=report['input_contract'],protocol=protocol,metric=metric,component_id=identity,adapter=adapter,service_sha256=digest(ROOT/'touch_dual_service.py'),validation_metrics=report.get('selected_outer_oof') or (report.get('metrics') or {}).get('validation'),metric_contract=('position exact-set descending then secondary validation criterion' if task=='position' else ('masked mean axis MAE ascending (N)' if adapter in ('delta9','measured45') else 'frozen normalized axis MAE ascending')))
 if task=='position' and report.get('position_mode')=='single_label26_v1':
  m=report['detailed_metrics']['position'];cm=m['confusion'];tp=sum(cm[i][i] for i in range(1,26));true_positive_rows=sum(sum(row) for row in cm[1:]);predicted_positive_rows=sum(sum(row[1:]) for row in cm)
  component['validation_metrics']=dict(component['validation_metrics'],position_macro_f1=m['macro_f1'],position_micro_f1=2*tp/max(true_positive_rows+predicted_positive_rows,1),position_balanced_accuracy=m['balanced_accuracy'])
  component['metric_contract']='single26 balanced accuracy descending, macro F1 descending, accuracy descending; exclusive softmax argmax; report legacy micro field corrected from confusion'
 return component
def register_group(path):
 path=Path(path).resolve()
 approved=ROOT/'outputs/delta9_groupcv_20261015_101629/selected_all680.joblib'
 if path!=approved.resolve():raise ValueError('Only explicitly authorized local group artifact is registered')
 r=json.loads(path.with_suffix('.report.json').read_text('utf-8'))
 if r.get('status')!='completed' or r.get('checkpoint_sha256')!=digest(path):raise ValueError('Incomplete or changed group report')
 protocol='groupcv:'+r['selection']['plan_sha256']+':'+r['selection']['split_sha256']
 m=r['selected_outer_oof'];pair=dict(schema=SCHEMA,deployment_ready=False,warning='实验模型／泛化未达标；分组验证内位置/力独立选优；跨协议不直接排名。完整680行重拟合不等于OOF重放。',selection_explanation='分组验证协议内位置/力独立最优；跨协议不直接排名',position=_component(path,'position',r,protocol,[-m['position']['exact'],-m['position']['micro_f1']],r['selection']['position'],'groupcv'),force=_component(path,'force',r,protocol,[m['normalized_mae']],r['selection']['force'],'groupcv'))
 # Explicit local trust boundary; no arbitrary report-supplied Python import.
 sources=[path.parent/'verified_bundle.py',path.parent/'grouped_core.py',ROOT/'outputs/delta9_search_20261015_094820/selected_bundle.py']
 pair['trusted_sources']={str(p):digest(p) for p in sources}
 validate_pair(pair);return pair
def _composition_evidence(path):
 # Exact locally audited artifact; reports cannot authorize arbitrary pickle/Python paths.
 import numpy as np
 path=Path(path).resolve();approved=ROOT/'outputs/delta9_composition_20261015_114020/selected_all680.joblib'
 if path!=approved.resolve() or digest(path)!='e6e1a2e427738bc3e53a6f94dbdd453252a1abb9b2817d2171d85a0f3dcb84b3':raise ValueError('Unapproved composition artifact')
 r=json.loads(path.with_suffix('.report.json').read_text('utf-8'));folder=path.parent
 if r.get('status')!='completed' or r['checkpoint_sha256']!=digest(path):raise ValueError('Incomplete composition report')
 plan=folder/'run_plan.json';selection=folder/'frozen_selection.json';split=folder/'frozen_split.json';core=folder/'composition_core.py'
 if digest(plan)!=r['plan_sha256'] or digest(selection)!=r['selection_sha256'] or digest(split)!=r['split_sha256'] or digest(core)!=r['source_sha256']['composition_core.py']:raise ValueError('Composition frozen source/evidence changed')
 selected=json.loads(selection.read_text('utf-8'))
 if selected!=r['selection']:raise ValueError('Composition selection binding mismatch')
 prior=ROOT/'outputs/delta9_groupcv_20261015_101629';pr=json.loads((prior/'selected_all680.report.json').read_text('utf-8'));protocol='groupcv:'+pr['selection']['plan_sha256']+':'+pr['selection']['split_sha256']
 if digest(split)!=digest(prior/'frozen_split.json') or selected['protocol']!=protocol:raise ValueError('Composition validation protocol changed')
 frozen=json.loads(split.read_text('utf-8'));fold=np.array([a['fold'] for a in frozen['rows']]);baseline=np.load(prior/'small_mlp32_d2_oof.npz')
 for task in ('position','force'):
  if not selected[task+'_improved']:continue
  cid=selected[task];trace=folder/(cid+'_oof.npz');a=np.load(trace)
  if not np.array_equal(a['fold'],fold) or not np.array_equal(a['labels'],baseline['labels']) or not np.array_equal(a['true_forces'],baseline['true_forces']) or not np.array_equal(a['valid'],baseline['valid']):raise ValueError('Composition OOF reference alignment differs')
  if task=='position':
   z=a['probabilities']>=a['thresholds'];y=a['labels'].astype(bool);tp=(z&y).sum();fp=(z&~y).sum();fn=(~z&y).sum();score=[float((z==y).all(1).mean()),float(2*tp/max(2*tp+fp+fn,1))];declared=[r['selected_outer_oof']['position'][k] for k in ('exact','micro_f1')]
  else:
   true=a['true_forces'];pred=a['forces'];err=np.zeros_like(pred)
   for sp in frozen['splits']:
    va=sp['validation'];err[va]=np.abs(pred[va]-true[va])/np.maximum(true[sp['train']].std(0),1e-5)
   score=[float(err.mean())];declared=[r['selected_outer_oof']['normalized_mae']]
   if not np.allclose(np.abs(pred-true).mean(0),r['selected_outer_oof']['force_mae_n'],rtol=0,atol=1e-12):raise ValueError('Composition axis metrics differ')
  if not np.allclose(score,declared,rtol=0,atol=1e-12):raise ValueError('Composition metric differs from frozen OOF')
 return r

def register_composition(path):
 r=_composition_evidence(path);path=Path(path).resolve();pair=register_group(ROOT/'outputs/delta9_groupcv_20261015_101629/selected_all680.joblib')
 for task in ('position','force'):
  if not r['selection'][task+'_improved']:continue
  m=r['selected_outer_oof'];metric=[-m['position']['exact'],-m['position']['micro_f1']] if task=='position' else [m['normalized_mae']]
  pair[task]=_component(path,task,r,r['selection']['protocol'],metric,r['selection'][task],'composition')
 pair['trusted_sources'].update({str(p):digest(p) for p in (path.parent/'composition_core.py',path.parent/'verified_composition_bundle.py')})
 pair['warning']='实验模型／泛化未达标；相同冻结分组CV独立选优；位置保留优胜者，力归一化误差选优含轴间权衡；无新盲测'
 validate_pair(pair);return pair

def register_search(path):
 path=Path(path).resolve();approved=ROOT/'outputs/delta9_search_20261015_094820/selected_independent.pt'
 if path!=approved.resolve() or digest(path)!='8060fe63b3295b87755241c8d0ee4be9271fe94d21c79e8ff0513f01ecae1e70':raise ValueError('Unapproved search artifact')
 r=json.loads(path.with_suffix('.report.json').read_text('utf-8'));m=r['metrics']['validation']
 protocol='search-random:'+r['split_sha256']
 pair=dict(schema=SCHEMA,deployment_ready=False,warning='实验模型／泛化未达标；随机验证非分组CV，不能跨协议比较；外部会话仅6.67%描述性结果',trusted_sources={str(path.parent/'selected_bundle.py'):digest(path.parent/'selected_bundle.py')})
 for task in ('position','force'):
  selection=path.parent/'frozen_validation_selection.json'
  if digest(selection)!=r['selection_sha256']:raise ValueError('Search validation selection changed')
  metric=[-v for v in json.loads(selection.read_text('utf-8'))[task]['key']]
  pair[task]=_component(path,task,r,protocol,metric,'residual96x3_ensemble:'+task,'search')
 validate_pair(pair);return pair
def _training_protocol(adapter,split,source_sha256):
 frozen=json.loads(Path(split).read_text('utf-8'))
 grouped=adapter=='measured45' and frozen.get('protocol') in ('nine45-single26-append-shortempty-timeblocks-v1','eight45-single26-append-whole-class-tare-blocks-v1','seven45-single26-grouped-longempty-tail-v1','six45-incremental-best-unseen-annotation-tare-v1','six45-incremental-five45-source-actualfit-v1','six45-incremental-five45-combo-actualfit-v1','six45-entire-sixth-source-holdout-v1','current45-newsource-whole-annotation-holdout-v1','current45-three-source-default-unseen-row-whole-label-tare-v1','four45-default-unseen-annotation-tare-v1','four45-source-aux-v1','four45-combo-aux-v1','five45-best-unseen-annotation-tare-v1','five45-source-actualfit-v1','five45-combo-actualfit-v1')
 if adapter=='measured45' and frozen.get('protocol')=='single45-20261029-inherited-grouped-plus-longempty-timeholdout-v1':grouped=True
 return ('measured45-sourceholdout:' if grouped else adapter+'-random:')+digest(split)+':'+str(source_sha256)

def neural_adapter_sources(architecture):
 if str(architecture).startswith('tslib45_'):
  files=('touch_tslib45.py','touch_45_models.py','touch_timesnet45.py','touch_single26_contract.py')
  vendor=('models/iTransformer.py','models/TimeMixer.py','models/TSMixer.py','models/PatchTST.py','layers/Transformer_EncDec.py','layers/SelfAttention_Family.py','layers/Embed.py','layers/Autoformer_EncDec.py','layers/AutoCorrelation.py','layers/StandardNorm.py','utils/masking.py','LICENSE','provenance.json','__init__.py','models/__init__.py','layers/__init__.py','utils/__init__.py')
  return files+tuple('touch_tslib45_vendor/'+n for n in vendor)
 if str(architecture).startswith('mature45_'):return ('touch_mature45.py','touch_rtdl45_vendor.py','touch_45_models.py')
 return ('touch_neural45.py','touch_45_models.py')+(('touch_ensemble45.py',) if architecture=='neural45_hms_be' else ('touch_relation45.py',) if architecture=='neural45_relation' else ())
def register_training(path,*,allow_smoke=False):
 path=Path(path).resolve();r=json.loads(path.with_suffix('.report.json').read_text('utf-8'))
 adapter='measured45' if r.get('schema')=='fbg_touch45_multilabel_v1' else 'delta9'
 if r.get('schema') not in ('fbg_touch_delta9_multilabel_v1','fbg_touch45_multilabel_v1') or (r.get('smoke_only') and not allow_smoke):raise ValueError('Not a completed formal candidate')
 if allow_smoke and (adapter!='measured45' or not path.is_relative_to(ROOT/'migration_evidence')):raise ValueError('Smoke replay restricted to45 migration evidence')
 if r.get('checkpoint_sha256')!=digest(path):raise ValueError('Checkpoint changed')
 evidence=path.parent/(path.stem+'_validation_selection.json');s=json.loads(evidence.read_text('utf-8'))
 if digest(evidence)!=r['validation_selection_sha256']:raise ValueError('Validation selection changed')
 split=path.parent/'frozen_split.json'
 if not split.exists():raise ValueError('Missing frozen comparable validation split')
 protocol=_training_protocol(adapter,split,s['source_sha256'])
 pair=dict(schema=SCHEMA,deployment_ready=False,warning='实验模型／泛化未达标；随机帧验证可能同轮泄漏；跨协议不排名',trusted_sources={})
 for task in ('position','force'):
  key=s[task][1:4] if task=='position' and str(r.get('architecture','')).startswith(('neural45_','mature45_','tslib45_')) else s[task]
  pair[task]=_component(path,task,r,protocol,[-float(v) for v in key],r['architecture']+':'+task,adapter)
 if str(r.get('architecture','')).startswith(('neural45_','mature45_','tslib45_')):
  pair['trusted_sources']={str(ROOT/name):digest(ROOT/name) for name in neural_adapter_sources(r['architecture'])}
  pair['force']['metric_contract']='frozen eligibility then normalized masked MAE +0.25 empty Fz/train Fz std ascending; independent current45 neural backbone'
 if r.get('architecture')=='local45':
  pair['trusted_sources']={str(ROOT/name):digest(ROOT/name) for name in ('touch_local45.py','touch_45_models.py')}
  pair['force']['metric_contract']='frozen train-standardized normalized masked MAE ascending; direct optical local/kernel estimator, no residual'
 if r.get('architecture')=='anchored45':
  pair['trusted_sources']={str(ROOT/name):digest(ROOT/name) for name in ('touch_anchor45.py','touch_45_models.py','touch_spectral45_encoders.py')}
  pair['force']['metric_contract']='frozen train-standardized normalized masked axis MAE ascending; embedded task-specific anchor plus optical-only residual'
 if str(r.get('architecture','')).startswith('spectral_'):
  pair['trusted_sources']={str(ROOT/name):digest(ROOT/name) for name in ('touch_spectral45_encoders.py','touch_45_models.py')}
  if r['architecture']=='spectral_timesnet':pair['trusted_sources'][str(ROOT/'touch_timesnet45.py')]=digest(ROOT/'touch_timesnet45.py')
  pair['force']['metric_contract']='frozen train-standardized normalized masked axis MAE ascending'
  pair['validation_description']='标注连续片段内顺序80/20（邻帧泄漏）'
  if json.loads(split.read_text('utf-8')).get('protocol')=='current45-three-source-default-unseen-row-whole-label-tare-v1':
   pair['force']['metric_contract']='frozen eligibility first, then train-standardized normalized MAE +0.25*empty Fz MAE/train Fz std ascending; displayed normalized MAE is the actual report value'
   pair['validation_description']='三源完整标注/tare块分组留出；默认训练行hash无重叠；邻近块相关，旧验证非盲测'
 if json.loads(split.read_text('utf-8')).get('protocol','').startswith('four45-'):
  pair['validation_description']='四源实验：真实完整标注/tare块分组或明确的整源/未见组合辅助；非随机帧验证'
  pair['force']['metric_contract']='frozen eligibility then normalized masked MAE +0.25 empty Fz/train Fz std; displayed normalized MAE is report value'
 if json.loads(split.read_text('utf-8')).get('protocol','').startswith('five45-'):
  pair['validation_description']='五源实验：完整标注/tare块留出，或整源/未见组合辅助；Best训练行hash无重叠；旧验证非盲测'
  pair['warning']='实验模型／泛化未达标；五源整块验证含已研究旧验证，Last可能未通过晋级门禁；加载不启动硬件或推理'
  pair['force']['metric_contract']='frozen eligibility then normalized masked MAE +0.25 empty Fz/train Fz std; displayed normalized MAE is report value'
 if json.loads(split.read_text('utf-8')).get('protocol','').startswith('six45-'):
  pair['validation_description']='六源增量：旧划分原样保留、新标注/tare整块留出；同源相关、非盲测'
  pair['warning']='实验模型／泛化未达标；六源冻结增量验证；Last未必通过门禁；加载不启动硬件或推理'
  pair['force']['metric_contract']='frozen eligibility then normalized masked MAE +0.25 empty Fz/train Fz std; displayed normalized MAE is report value'
 if json.loads(split.read_text('utf-8')).get('protocol')=='seven45-single26-grouped-longempty-tail-v1':
  pair['validation_description']='七源单点26类：保留旧整块划分并排除多点；新增长空载前660训练、12帧60分钟purge、末171验证；同源相关、非盲测'
  pair['warning']='实验／泛化未达标；仅空载/单点，26类softmax CE；独立力不归零；加载不启动硬件或推理'
 if json.loads(split.read_text('utf-8')).get('protocol')=='eight45-single26-append-whole-class-tare-blocks-v1':
  pair['validation_description']='八源单点26类：旧整块划分不变；新增源完整类别/tare连续块按预声明类别步长留出；同会话相关，旧验证复用，非盲测'
  pair['warning']='实验／泛化未达标；仅空载/单点26类CE；力独立且不强制归零；加载不启动硬件或推理'
  pair['force']['metric_contract']='frozen loaded normalized masked axis MAE +0.25 empty normalized masked axis MAE ascending; independent corrected-N single-only regression'
 if json.loads(split.read_text('utf-8')).get('protocol')=='nine45-single26-append-shortempty-timeblocks-v1':
  pair['validation_description']='九源单点26类：旧八源成员不变；指定15帧空载前8训练、中2边界purge、末5验证；五分钟采样，同源时间相关，复用验证非盲测'
  pair['warning']='实验／泛化未达标；仅空载/单点26类CE；位置候选的力运输头不可部署；加载不启动硬件或推理'
 if json.loads(split.read_text('utf-8')).get('protocol')=='single45-20261029-inherited-grouped-plus-longempty-timeholdout-v1':
  pair['validation_description']='单位置26类及独立三轴力：继承标注/tare块划分，新增长空载时间留出及12帧purge；旧验证复用，非盲测'
  pair['warning']='实验模型；仅空载/单位置；9个位置无验证支持；保留原始质量标记；力不随位置强制归零；加载不启动硬件或推理'
  pair['position']['metric_contract']='single-position accuracy descending, overall accuracy descending, supported balanced accuracy descending; exclusive26 softmax argmax'
  pair['force']['metric_contract']='mean of masked per-axis MAE in N ascending; normalized MAE diagnostic only; independent single/empty corrected-force regression'
 pair['smoke_replay']=bool(r.get('smoke_only') and allow_smoke)
 validate_pair(pair);return pair
def validate_pair(pair):
 from touch_delta9_data import validate_input_contract
 if pair.get('schema')!=SCHEMA:raise ValueError('Invalid two-slot selection; no fallback')
 for task in ('position','force'):
  c=pair[task];p=Path(c['model_path']);rpath=Path(c['report_path'])
  if c['task']!=task or digest(p)!=c['checkpoint_sha256'] or digest(rpath)!=c['report_sha256']:raise ValueError('Component hash mismatch: '+task)
  r=json.loads(rpath.read_text('utf-8'))
  if Path(r['model_path']).resolve()!=p.resolve() or r['checkpoint_sha256']!=c['checkpoint_sha256'] or r['input_contract']!=c['input_contract']:raise ValueError('Component report binding mismatch')
  if c['adapter']=='measured45':
   from touch_45_data import validate_input_contract as validate45
   validate45(c['input_contract'])
  else:validate_input_contract(c['input_contract'])
  if c.get('service_sha256')!=digest(ROOT/'touch_dual_service.py'):raise ValueError('Runtime adapter hash mismatch')
  if c['adapter']=='groupcv':
   approved=ROOT/'outputs/delta9_groupcv_20261015_101629/selected_all680.joblib'
   if p.resolve()!=approved.resolve() or c['checkpoint_sha256']!='2991deb7dbd11bab5c1909d92111f8f1c8e1b3441be00f6ae16c4cd0d2bed09f':raise ValueError('Unapproved joblib artifact')
   m=r['selected_outer_oof'];expected=[-m['position']['exact'],-m['position']['micro_f1']] if task=='position' else [m['normalized_mae']]
   if c['metric']!=expected or c['component_id']!=r['selection'][task] or c['protocol']!='groupcv:'+r['selection']['plan_sha256']+':'+r['selection']['split_sha256']:raise ValueError('Validation metric/component/protocol identity differs from report')
   required=[p.parent/'verified_bundle.py',p.parent/'grouped_core.py',ROOT/'outputs/delta9_search_20261015_094820/selected_bundle.py']
   if any(str(q) not in pair.get('trusted_sources',{}) for q in required):raise ValueError('Missing trusted adapter hashes')
  if c['adapter'] in ('delta9','measured45'):
   adapter=c['adapter'];schema='fbg_touch45_multilabel_v1' if adapter=='measured45' else 'fbg_touch_delta9_multilabel_v1'
   if c['component_id']!=r['architecture']+':'+task:raise ValueError('Component architecture/task identity differs from bound report')
   smoke_allowed=pair.get('smoke_replay') and adapter=='measured45' and p.resolve().is_relative_to(ROOT/'migration_evidence')
   if r.get('schema')!=schema or (r.get('smoke_only') and not smoke_allowed):raise ValueError('Not a formal completed model')
   e=p.parent/(p.stem+'_validation_selection.json');split=p.parent/'frozen_split.json'
   if digest(e)!=r.get('validation_selection_sha256'):raise ValueError('Validation selection hash mismatch')
   s=json.loads(e.read_text('utf-8'));key=s[task][1:4] if task=='position' and str(r.get('architecture','')).startswith(('neural45_','mature45_','tslib45_')) else s[task];expected=[-float(v) for v in key]
   protocol=_training_protocol(adapter,split,s['source_sha256'])
   if c['metric']!=expected or c['protocol']!=protocol:raise ValueError('Validation metric/protocol binding differs')
   if str(r.get('architecture','')).startswith(('neural45_','mature45_','tslib45_')):
    required=[ROOT/name for name in neural_adapter_sources(r['architecture'])]
    if any(str(q) not in pair.get('trusted_sources',{}) for q in required):raise ValueError('Missing trusted current45 neural adapter hashes')
    if task=='force' and 'normalized' not in c.get('metric_contract',''):raise ValueError('Neural force normalized criterion required')
   if r.get('architecture')=='local45':
    required=[ROOT/'touch_local45.py',ROOT/'touch_45_models.py']
    if any(str(q) not in pair.get('trusted_sources',{}) for q in required):raise ValueError('Missing trusted direct local adapter hashes')
   if r.get('architecture')=='anchored45':
    required=[ROOT/name for name in ('touch_anchor45.py','touch_45_models.py','touch_spectral45_encoders.py')]
    if any(str(q) not in pair.get('trusted_sources',{}) for q in required):raise ValueError('Missing trusted anchor adapter source hashes')
   if str(r.get('architecture','')).startswith('spectral_'):
    required=[ROOT/'touch_spectral45_encoders.py',ROOT/'touch_45_models.py']
    if any(str(q) not in pair.get('trusted_sources',{}) for q in required):raise ValueError('Missing trusted spectral adapter source hashes')
    if r['architecture']=='spectral_timesnet' and str(ROOT/'touch_timesnet45.py') not in pair.get('trusted_sources',{}):raise ValueError('Missing trusted TimesNet adapter source hash')
    if task=='force' and 'normalized' not in c.get('metric_contract',''):raise ValueError('Spectral force normalized validation criterion required')
  if c['adapter']=='search':
   if p.resolve()!=(ROOT/'outputs/delta9_search_20261015_094820/selected_independent.pt').resolve() or digest(p)!='8060fe63b3295b87755241c8d0ee4be9271fe94d21c79e8ff0513f01ecae1e70':raise ValueError('Unapproved search checkpoint')
   if str(p.parent/'selected_bundle.py') not in pair.get('trusted_sources',{}):raise ValueError('Missing search adapter hash')
   e=p.parent/'frozen_validation_selection.json'
   if digest(e)!=r['selection_sha256'] or c['metric']!=[-v for v in json.loads(e.read_text('utf-8'))[task]['key']] or c['protocol']!='search-random:'+r['split_sha256']:raise ValueError('Search validation evidence mismatch')
  if c['adapter']=='composition':
   evidence=_composition_evidence(p);m=evidence['selected_outer_oof'];expected=[-m['position']['exact'],-m['position']['micro_f1']] if task=='position' else [m['normalized_mae']]
   if not evidence['selection'][task+'_improved'] or c['component_id']!=evidence['selection'][task] or c['metric']!=expected or c['protocol']!=evidence['selection']['protocol']:raise ValueError('Composition metric/component/protocol binding differs')
   if any(str(q) not in pair.get('trusted_sources',{}) for q in (p.parent/'composition_core.py',p.parent/'verified_composition_bundle.py')):raise ValueError('Missing composition adapter hashes')
  if c['adapter'] not in ('groupcv','delta9','measured45','search','composition') or not c['metric'] or not all(math.isfinite(v) for v in c['metric']):raise ValueError('Invalid adapter/validation metric')
 if pair['position']['input_contract']!=pair['force']['input_contract']:raise ValueError('Position/force measured-input contracts differ')
 for p,h in pair.get('trusted_sources',{}).items():
  if digest(p)!=h:raise ValueError('Trusted adapter source changed')
 if 'matched_comparison_record' in pair:
  binding=pair['matched_comparison_record'];record_path=Path(binding['path'])
  if not record_path.resolve().is_relative_to(ROOT/'outputs') or digest(record_path)!=binding['sha256']:raise ValueError('Matched comparison record changed')
  record=json.loads(record_path.read_text('utf-8'))
  for name,expected in record['evidence_sha256'].items():
   target=(record_path.parent/name).resolve()
   if not target.is_relative_to(record_path.parent.resolve()) or digest(target)!=expected:raise ValueError('Matched deployment evidence changed')
 return pair
def validate_runtime_contract(pair):
 """Check exact runtime artifacts/sources without opening optional historical comparisons.

 This is NOT a promotion/history audit. Default validate_pair/read_pair retain their
 full historical evidence checks. Use for guarded validation-only experiments and
 explicit metadata-only source reapproval; never claim historical evidence verified.
 """
 runtime=copy.deepcopy(pair);runtime.pop('matched_comparison_record',None)
 validate_pair(runtime)
 return pair

def position_promotion_gate(incumbent,candidate,evidence):
 # Explicit policy evidence; historical BCE ranking remains bound but cannot promote.
 if incumbent['protocol']!=candidate['protocol'] or tuple(candidate['metric'][:2])>=tuple(incumbent['metric'][:2]):return False
 if any(b>a for a,b in zip(incumbent['metric'][:2],candidate['metric'][:2])):return False
 if not isinstance(evidence,dict) or evidence.get('policy')!='position45-robust-v1':return False
 if evidence.get('incumbent_sha256')!=incumbent['checkpoint_sha256'] or evidence.get('candidate_sha256')!=candidate['checkpoint_sha256']:return False
 return all(evidence.get(k) is True for k in ('stress_nonregression','auxiliary_nonregression','multiseed_direction_consistent'))
def promote(current,candidate):
 validate_pair(candidate)
 if current is None:return candidate
 validate_pair(current);out=json.loads(json.dumps(current));changed=False
 for task in ('position','force'):
  a,b=current[task],candidate[task]
  better=(position_promotion_gate(a,b,candidate.get('position_promotion_evidence')) if task=='position' and current.get('position_promotion_policy')=='position45-robust-v1' else tuple(b['metric'][:2])<tuple(a['metric'][:2]) if task=='position' else tuple(b['metric'])<tuple(a['metric']))
  if a['protocol']==b['protocol'] and better:out[task]=b;changed=True
 if changed:out['trusted_sources'].update(candidate.get('trusted_sources',{}));validate_pair(out)
 return out
def _verify_three_source_training_exclusion(current,folder,split,audit):
 """Check the exact incumbent split/source hashes and actual source row bytes."""
 import numpy as np
 folder=Path(folder);rows=split['rows'];va=split['validation'];held={rows[i]['row_sha256'] for i in va}
 expected={'press_grid_20261015_161438','press_grid_20261016_100337','press_grid_20261016_155329'}
 if {r['session'] for r in rows}!=expected:raise ValueError('Unexpected three-source membership')
 for task in ('position','force'):
  c=current[task];r=json.loads(Path(c['report_path']).read_text('utf-8'));oldpath=Path(c['model_path']).parent/'frozen_split.json'
  if digest(oldpath)!=r['split_sha256']:raise ValueError('Incumbent original split hash changed')
  original=json.loads(oldpath.read_text('utf-8'));source=oldpath.parent.parent/'trainval_arrays.npz'
  if digest(source)!=r['source_sha256']:raise ValueError('Incumbent original training arrays changed')
  trained={original['rows'][i]['row_sha256'] for i in original['train']}
  if held&trained:raise ValueError('Validation overlaps actual incumbent training rows')
  proof=audit['incumbent_training_proof'][task]
  if proof['checkpoint_sha256']!=c['checkpoint_sha256'] or proof['declared_source_sha256']!=digest(source) or proof['actual_old_arrays_sha256']!=digest(source) or proof['trainrowhash_overlap']!=0:raise ValueError('Incumbent training proof does not bind current component')
 raw=np.load(folder/'trainval_arrays.npz');known={}
 from touch_45_data import validate_record
 for session in audit['sessions']:
  if session['name'] not in expected:continue
  for path,h in session['source_hashes'].items():
   if digest(path)!=h:raise ValueError('Measured source changed after freeze')
  source=ROOT/'outputs/manual_touch_45/manual_touch_multi45'/session['name']/'frames.jsonl';block=-1;last=None
  with source.open(encoding='utf-8') as stream:
   for line_no,line in enumerate(stream,1):
    record=json.loads(line);x,y,f,contract=validate_record(record);q=record['quality'];tare=json.dumps((q.get('paxini_force') or {}).get('tare'),sort_keys=True)
    if tare!=last:block+=1
    last=tare;known[(session['name'],line_no)]=(hashlib.sha256(line.encode()).hexdigest(),x,y,f,block,contract)
 for i,row in enumerate(rows):
  h,x,y,f,block,contract=known[(row['session'],row['line'])]
  if h!=row['row_sha256'] or contract!=split['input_contract'] or not np.array_equal(x,raw['features'][i]) or not np.array_equal(y,raw['labels'][i]) or not np.array_equal(f,raw['forces'][i],equal_nan=True):raise ValueError('Frozen training array or row identity changed')
  if row['session']=='press_grid_20261016_155329' and not y.any() and row['group']!=f"press_grid_20261016_155329:empty_tare_block:{block}":raise ValueError('Empty tare group identity changed')
 return True

def _matched_force_zero_gate(base,candidate,scale):
 """Three-source engineering gate; references are actual corrected forces, not0."""
 import numpy as np
 for k in ('labels','true_forces','valid'):
  if not np.array_equal(base[k],candidate[k],equal_nan=True):raise ValueError('Force gate references differ')
 empty=base['labels'].sum(1)==0;loaded=np.linalg.norm(np.nan_to_num(base['true_forces']),axis=1)>=1
 if not empty.any() or not loaded.any():return False
 def mae(a,mask):
  values=[]
  for j in range(3):
   v=a['valid'][:,j]&mask
   if not v.any():return np.full(3,np.inf)
   values.append(np.abs(a['forces'][v,j]-a['true_forces'][v,j]).mean())
  return np.asarray(values)
 allrows=np.ones(len(empty),bool);old=mae(base,allrows);new=mae(candidate,allrows)
 return bool(np.all(new<=old) and np.mean(new/scale)<=.99*np.mean(old/scale) and mae(candidate,empty)[2]<min(.2,.8*mae(base,empty)[2]) and np.all(mae(candidate,loaded)<=1.05*mae(base,loaded)+.005))

def promote_matched_validation(current,evidence_directory):
 """Migrate a validation protocol only after aligned cached incumbent replay.

 Does not write a pin, forward test, trust report-supplied code or relax legacy
 promote(). The caller still has to service-verify and atomically save/read back.
 """
 import numpy as np
 validate_pair(current);folder=Path(evidence_directory).resolve()
 if not folder.is_relative_to(ROOT/'outputs'):raise ValueError('Evidence outside project outputs')
 def load(name):return json.loads((folder/name).read_text('utf-8'))
 if load('frozen_split.json').get('protocol')=='four45-default-unseen-annotation-tare-v1':
  from touch_four45_promotion import promote_four45
  return promote_four45(current,folder)
 record=load('matched_comparison_record.json')
 if record.get('schema')!='fbg_matched45_comparison_v1':raise ValueError('Missing matched comparison schema')
 required={'validation_selection_frozen.json','frozen_split.json','frozen_run_plan.json','pin_before.json','data_audit.json','trainval_arrays.npz','default_validation_predictions.npz','auxiliary_split.json','auxiliary_progress.json','external_evaluation.json','default_external_predictions.npz','selected_external_predictions.npz'}
 if not required.issubset(record.get('evidence_sha256',{})):raise ValueError('Incomplete matched comparison hashes')
 for name,expected in record['evidence_sha256'].items():
  p=(folder/name).resolve()
  if not p.is_relative_to(folder) or digest(p)!=expected:raise ValueError('Matched evidence changed: '+name)
 selection=load('validation_selection_frozen.json');split=load('frozen_split.json');plan=load('frozen_run_plan.json');before=load('pin_before.json');audit=load('data_audit.json')
 if selection['split_sha256']!=digest(folder/'frozen_split.json') or selection['plan_sha256']!=digest(folder/'frozen_run_plan.json') or plan['split_sha256']!=selection['split_sha256']:raise ValueError('Frozen comparison protocol changed')
 three_source=split.get('protocol')=='current45-three-source-default-unseen-row-whole-label-tare-v1'
 if three_source:_verify_three_source_training_exclusion(current,folder,split,audit)
 elif split.get('protocol')!='current45-newsource-whole-annotation-holdout-v1' or audit.get('default_newvalidation_rowhash_overlap')!=0 or audit.get('default_training_source_hash_matches_old_session') is not True:raise ValueError('Incumbent training leakage not excluded')
 for task in ('position','force'):
  if current[task]!=before[task]:raise ValueError('Incumbent identity changed after comparison')
 raw=np.load(folder/'trainval_arrays.npz');tr=np.asarray(split['train']);va=np.asarray(split['validation']);rows=split['rows']
 if set(rows[i]['row_sha256'] for i in tr)&set(rows[i]['row_sha256'] for i in va):raise ValueError('Train/validation row overlap')
 if set(rows[i]['group'] for i in tr)&set(rows[i]['group'] for i in va):raise ValueError('Annotation group overlap')
 if not all(rows[i]['session'] in (('press_grid_20261016_100337','press_grid_20261016_155329') if three_source else ('press_grid_20261016_100337',)) for i in va):raise ValueError('Validation is not approved-source-only')
 if set(rows[i]['row_sha256'] for i in va)&set(r['row_sha256'] for r in rows if r['session']=='press_grid_20261015_161438'):raise ValueError('Default source overlap')
 scale=np.array([raw['forces'][tr,a][raw['valid'][tr,a]].std() for a in range(3)]);scale[scale<1e-5]=1
 def bound(path):
  path=Path(path).resolve()
  if not path.is_relative_to(folder):raise ValueError('Artifact outside matched comparison')
  name=str(path.relative_to(folder))
  if record['evidence_sha256'].get(name)!=digest(path):raise ValueError('Unbound matched artifact: '+name)
  return path
 def aligned(path,indices):
  a=np.load(bound(path))
  for k,v in [('indices',indices),('labels',raw['labels'][indices]),('true_forces',raw['forces'][indices]),('valid',raw['valid'][indices])]:
   if not np.array_equal(a[k],v,equal_nan=True):raise ValueError('Comparison references differ: '+k)
  return a
 def pm(a):
  y=a['labels'].astype(bool);p=a['probabilities']>=float(a['threshold']);same=(y==p).all(1);n=y.sum(1);tp=(y&p).sum();fp=(~y&p).sum();fn=(y&~p).sum()
  return dict(exact=float(same.mean()),f1=float(2*tp/max(2*tp+fp+fn,1)),single=float(same[n==1].mean()) if (n==1).any() else None,multi=float(same[n>1].mean()) if (n>1).any() else None,empty_fp=int(p[n==0].any(1).sum()),stress_exact=float(((a['stress_probabilities']>=float(a['threshold']))==y).all(1).mean()))
 def fm(a):
  return np.array([np.abs(a['forces'][a['valid'][:,j],j]-a['true_forces'][a['valid'][:,j],j]).mean() for j in range(3)])
 def pg(a,b):
  improvement=(b['exact']>=a['exact'] and b['f1']>=a['f1'] and (b['exact']>a['exact'] or b['f1']>a['f1'])) if three_source else (b['exact']>a['exact'] and b['f1']>=a['f1'])
  return improvement and b['empty_fp']<=a['empty_fp'] and all(b[k]>=a[k] for k in ('single','multi','stress_exact') if a[k] is not None)
 base=aligned(folder/'default_validation_predictions.npz',va);pa=pm(base);fa=fm(base);out=json.loads(json.dumps(current));decisions={};sources={};external=load('external_evaluation.json')
 if external['selection_sha256']!=digest(folder/'validation_selection_frozen.json') or external['count']!=40 or not external['no_post_test_search']:raise ValueError('Unfrozen external safety replay')
 ea=np.load(folder/'default_external_predictions.npz');eb=np.load(folder/'selected_external_predictions.npz')
 for key in ('labels','true_forces','valid'):
  if not np.array_equal(ea[key],eb[key],equal_nan=True) or len(ea[key])!=40:raise ValueError('External safety references differ')
 def ep(a):
  y=a['labels'].astype(bool);p=a['probabilities']>=float(a['threshold']);tp=(y&p).sum();fp=(~y&p).sum();fn=(y&~p).sum()
  return float((y==p).all(1).mean()),float(2*tp/max(2*tp+fp+fn,1))
 ax,af=ep(ea);bx,bf=ep(eb);efa=fm(ea);efb=fm(eb)
 veto=dict(position=bx<ax-.05 or bf<af-.02,force=bool(np.any((efb>1.10*efa)&((efb-efa)>.02))))
 if veto!=external['safety_veto']:raise ValueError('Declared safety veto disagrees with cached external results')
 for task in ('position','force'):
  path=Path(selection[task]).resolve()
  if not path.is_relative_to(folder):raise ValueError('Selected artifact outside comparison')
  bound(path);bound(path.with_suffix('.report.json'))
  candidate=register_training(path);report=json.loads(path.with_suffix('.report.json').read_text('utf-8'));a=aligned(path.parent/'validation_predictions.npz',va)
  confirm_path=Path(selection['confirm'][task]).resolve();bound(confirm_path);bound(confirm_path.with_suffix('.report.json'));confirm_report=json.loads(confirm_path.with_suffix('.report.json').read_text('utf-8'));register_training(confirm_path);b=aligned(confirm_path.parent/'validation_predictions.npz',va)
  if report['seed']==confirm_report['seed'] or report['spec']!=confirm_report['spec']:raise ValueError('Not matched distinct-seed confirmation')
  for r,trace in [(report,a),(confirm_report,b)]:
   if r['statistics_fit_indices']!=split['train'] or r['source_sha256']!=digest(folder/'trainval_arrays.npz'):raise ValueError('Train statistics/source mismatch')
   pmet=pm(trace);fmet=fm(trace);declared=r['metrics']['validation']
   if not np.allclose([pmet['exact'],pmet['f1']], [declared['position_exact_match'],declared['position_micro_f1']],atol=1e-10,rtol=0) or not np.allclose(fmet,declared['force_mae_n'],atol=1e-7,rtol=0):raise ValueError('Report metrics disagree with aligned predictions')
  if task=='position':
   gates=[pg(pa,pm(a)),pg(pa,pm(b))];aux=load('auxiliary_split.json');av=np.array(aux['validation']);auxreports=load('auxiliary_progress.json');auxpass=[]
   if set(aux['train'])&set(aux['validation']):raise ValueError('Auxiliary overlap')
   trainsets={tuple(rows[i]['labels']) for i in aux['train']}
   if any(tuple(rows[i]['labels']) in trainsets for i in av):raise ValueError('Auxiliary combinations are not unseen')
   for seed in plan['seeds']:
    rs=[next(q for q in auxreports if q['seed']==seed and q['spec']['id']==specid) for specid in (plan['configs'][0]['id'],report['spec']['id'])];mets=[]
    for r in rs:
     checkpoint=Path(r['model_path']);register_training(checkpoint)
     if r['statistics_fit_indices']!=aux['train']:raise ValueError('Auxiliary training split differs')
     mets.append(pm(aligned(checkpoint.parent/'validation_predictions.npz',av)))
    auxpass.append(mets[1]['exact']>=mets[0]['exact'] and mets[1]['f1']>=mets[0]['f1'])
   gates.append(all(auxpass))
  else:gates=[_matched_force_zero_gate(base,q,scale) if three_source else bool(np.all(fm(q)<=fa) and np.mean(fm(q)/scale)<=.99*np.mean(fa/scale)) for q in (a,b)]
  allowed=all(gates) and not external['safety_veto'][task];decisions[task]=dict(validation_gates=gates,external_safety_veto=external['safety_veto'][task],promoted=allowed)
  if allowed:out[task]=candidate[task];sources.update(candidate.get('trusted_sources',{}))
 if any(v['promoted'] for v in decisions.values()):
  out['trusted_sources'].update(sources);out['matched_comparison_record']=dict(path=str(folder/'matched_comparison_record.json'),sha256=digest(folder/'matched_comparison_record.json'));out['validation_description']=('三源分组验证：默认未训练过的100337行与155329完整标注/tare块；同样本重放，双默认训练行hash交集0；邻近tare块相关、旧验证非盲测' if three_source else '100337新来源完整标注段留出；旧默认与候选完全同样本重放，无旧训练行重叠；非新盲测')
  out['warning']='实验模型／泛化未达标；独立组件经同样本验证、多seed及已知外测安全否决；不代表未来会话泛化；加载不启动硬件或推理。';out['selection_explanation']='经哈希绑定同样本比较迁移验证协议；位置/力各自过门禁才替换，其余保留。'
 validate_pair(out);return out,decisions

def save_pair(path,pair):
 validate_pair(pair)
 if pair.get('smoke_replay'):raise ValueError('Smoke weights must never become a production deployment pin')
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 before=path.read_bytes() if path.exists() else None
 if before is not None:shutil.copy2(path,path.with_name(path.name+'.backup-'+str(time.time_ns())))
 tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(pair,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)
 try:
  if read_pair(path)!=pair:raise ValueError('Active deployment pin readback differs')
 except Exception:
  if before is None:path.unlink(missing_ok=True)
  else:tmp.write_bytes(before);os.replace(tmp,path)
  raise
def read_pair(path):return validate_pair(json.loads(Path(path).read_text('utf-8')))
