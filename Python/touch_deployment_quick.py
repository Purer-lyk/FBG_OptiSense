"""Explicit completed-candidate registry, separate from the mutable active pin. Torch free."""
import copy,json,os,shutil,time
from pathlib import Path
from touch_deployment_slots import validate_pair,digest,promote
NAME='touch_deployment_quick_choices.json'
TASKS=('position','force')
def _validate(data):
 if data.get('schema')!='fbg_touch_quick_choices_v1':raise ValueError('快捷模型索引格式错误')
 for kind in ('best','last'):
  pair=data.get(kind)
  if pair is not None:
   validate_pair(pair)
   if pair.get('smoke_replay') or pair['position']['input_contract'].get('input_shape')!=[9,5]:raise ValueError('快捷选择只允许真实45点模型')
 for path,h in data.get('evidence',{}).items():
  if digest(path)!=h:raise ValueError('完成实验冻结证据已改变')
 return data
def read_choices(home):return _validate(json.loads((Path(home)/NAME).read_text('utf-8')))
def _save(home,data):
 _validate(data);path=Path(home)/NAME;path.parent.mkdir(parents=True,exist_ok=True)
 before=path.read_bytes() if path.exists() else None
 if before is not None:shutil.copy2(path,path.with_name(path.name+'.backup-'+str(time.time_ns())))
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)
 try:
  if read_choices(home)!=data:raise ValueError('快捷索引回读不一致')
 except Exception:
  if before is None:path.unlink(missing_ok=True)
  else:tmp.write_bytes(before);os.replace(tmp,path)
  raise
 return data
def initialize(home,best,last=None,evidence=None):
 if (Path(home)/NAME).exists():return read_choices(home)
 return _save(home,dict(schema='fbg_touch_quick_choices_v1',best=copy.deepcopy(best),last=copy.deepcopy(last),evidence=evidence or {}))
def choose(home,active,task,kind):
 if task not in TASKS or kind not in ('best','last'):raise ValueError('未知模型槽/选择')
 data=read_choices(home);candidate=data.get(kind)
 if candidate is None:raise ValueError('没有已完成实验的Last候选；不回退Best')
 pair=copy.deepcopy(active);pair[task]=copy.deepcopy(candidate[task]);pair['trusted_sources'].update(candidate['trusted_sources'])
 pair.setdefault('quick_choices',{})[task]=dict(kind=kind,reason='manual_'+kind,checkpoint_sha256=pair[task]['checkpoint_sha256'])
 pair['selection_explanation']='手动独立槽选择；Best推荐身份不变；Last为最近完成实验候选，不是最后epoch'
 pair['warning']='实验／泛化未达标；Last最新候选可能更差、未必通过晋级门禁；Best是已验证推荐而非部署达标。加载不启动硬件或推理。'
 validate_pair(pair);return pair
def _validate_runtime_choices(data):
 """Runtime-only boundary: do not dereference historical comparison/evidence maps."""
 from touch_deployment_slots import validate_runtime_contract
 if data.get('schema')!='fbg_touch_quick_choices_v1':raise ValueError('快捷模型索引格式错误')
 for kind in ('best','last'):
  pair=data.get(kind)
  if pair is not None:
   validate_runtime_contract(pair)
   if pair.get('smoke_replay') or pair['position']['input_contract'].get('input_shape')!=[9,5]:raise ValueError('Only formal current45 runtime choices allowed')
 return data

def _atomic_runtime_write(path,data,validator):
 """Back up, atomically replace, validate exact target readback, byte-exact rollback."""
 path=Path(path);validator(data);before=path.read_bytes() if path.exists() else None
 if before is not None:shutil.copy2(path,path.with_name(path.name+'.backup-'+str(time.time_ns())))
 tmp=path.with_name(path.name+'.tmp');tmp.write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf-8');os.replace(tmp,path)
 try:
  actual=json.loads(path.read_text('utf-8'));validator(actual)
  if actual!=data:raise ValueError('Runtime target readback mismatch')
 except Exception:
  if before is None:path.unlink(missing_ok=True)
  else:tmp.write_bytes(before);os.replace(tmp,path)
  raise
 return data

def reapprove_runtime_sources(home,updates,*,active_path):
 """Explicit source-only reapproval, not promotion and not historical revalidation.

 Keep every checkpoint/report identity and all historical pointers byte-values.
 Ordinary read_choices/choose/completed_training remain fully history-validating.
 """
 from touch_deployment_slots import validate_runtime_contract
 path=Path(home)/NAME;active_path=Path(active_path);qb=path.read_bytes();ab=active_path.read_bytes();data=json.loads(qb);active=json.loads(ab);old=copy.deepcopy(data);olda=copy.deepcopy(active)
 for source,values in updates.items():
  if set(values)!={'previous','current'} or digest(source)!=values['current']:raise ValueError('Unapproved source reapproval hash')
  for pair in [data['best'],data.get('last'),active]:
   if pair is not None and source in pair.get('trusted_sources',{}):
    if pair['trusted_sources'][source]!=values['previous']:raise ValueError('Prior approved source mismatch')
    pair['trusted_sources'][source]=values['current']
 _validate_runtime_choices(data);validate_runtime_contract(active)
 if path.read_bytes()!=qb or active_path.read_bytes()!=ab:raise ValueError('Concurrent runtime selection change')
 try:
  _atomic_runtime_write(path,data,_validate_runtime_choices);_atomic_runtime_write(active_path,active,validate_runtime_contract)
 except Exception:
  for p,b in [(path,qb),(active_path,ab)]:
   tmp=p.with_name(p.name+'.rollback');tmp.write_bytes(b);os.replace(tmp,p)
  raise
 if any(data[k][t]!=old[k][t] for k in ('best','last') if old.get(k) for t in TASKS) or any(active[t]!=olda[t] for t in TASKS):raise AssertionError('Source reapproval changed component')
 return dict(source_updates=updates,historical_comparison_verified=False,scope='runtime checkpoint/report/contract/protocol/service/source only; historical pointers retained, not opened',active_model_identities_unchanged=True,registry_sha256=digest(path),active_sha256=digest(active_path))

def _completed_service_matches(service,validation_count,split_sha256):
 return service.get('rows_each_pair')==validation_count and len(service.get('pairs',[]))>=5 and service.get('split_sha256')==split_sha256

def complete_validation_only_last(home,candidate,*,selection_path):
 """Register a verified rejected candidate as Last without reading historical caches.

 No promotion is possible here. Require bound completed selection, training,
 service and real Qt evidence; keep Best and its historical pointers unchanged.
 """
 selection_path=Path(selection_path);bindings=candidate.get('completion_evidence',{})
 required=('validation_selection_frozen.json','training_manifest.json','service_verification.json','qt_verification.json')
 if not selection_path.exists() or any(str(selection_path.parent/name) not in bindings for name in required):raise ValueError('Missing completed validation-only experiment evidence')
 if 'matched_comparison_record' in candidate:raise ValueError('New validation-only Last must not bind historical comparisons')
 validate_pair(candidate)
 for p,h in bindings.items():
  if not Path(p).resolve().is_relative_to(selection_path.parent.resolve()) or digest(p)!=h:raise ValueError('Completed Last evidence hash mismatch')
 selection=json.loads(selection_path.read_text('utf-8'));service=json.loads((selection_path.parent/'service_verification.json').read_text('utf-8'));qt=json.loads((selection_path.parent/'qt_verification.json').read_text('utf-8'))
 if str(selection_path) not in bindings or selection.get('completed_primary_pairs',0)<1 or selection.get('test_model_forward_performed') is not False:raise ValueError('Incomplete validation-only selection')
 for task in TASKS:
  if selection[task]!=candidate[task]['model_path'] or selection['decisions'][task].get('promotion_allowed') is not False:raise ValueError('Validation-only API cannot promote or alter frozen candidate')
 split=selection_path.parent/'frozen_split.json';frozen=json.loads(split.read_text('utf-8'))
 expected_count=len(frozen['validation'])
 for task in TASKS:
  report=json.loads(Path(candidate[task]['report_path']).read_text('utf-8'))
  if report.get('validation_count')!=expected_count or report.get('split_sha256')!=digest(split):raise ValueError('Completed split/count binding differs from selected report')
 if not _completed_service_matches(service,expected_count,digest(split)) or qt.get('errors')!=[]:raise ValueError('Missing production service/Qt verification')
 path=Path(home)/NAME;before=path.read_bytes();data=_validate_runtime_choices(json.loads(before));best=copy.deepcopy(data['best']);data['last']=copy.deepcopy(candidate)
 _atomic_runtime_write(path,data,_validate_runtime_choices)
 actual=_validate_runtime_choices(json.loads(path.read_text('utf-8')))
 if actual['best']!=best or actual['last']!=candidate:raise ValueError('Validation-only Last registration readback mismatch')
 return actual

def completed_training(home,candidate):
 """Called only after the GUI training-completed signal and worker cleanup."""
 validate_pair(candidate);data=read_choices(home)
 data['last']=copy.deepcopy(candidate);data['best']=promote(data['best'],candidate)
 # Historical frozen run remains auditable; training sidecars bind future candidates.
 return _save(home,data)
