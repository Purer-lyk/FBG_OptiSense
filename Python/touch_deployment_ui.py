"""Qt-only independent model-slot controls; child holds two real loaded components."""
import copy,json,os,tempfile
from pathlib import Path
from PyQt5 import QtCore,QtWidgets
from touch_deployment_slots import read_pair,save_pair,validate_pair,register_group,register_search,register_training,promote,ROOT
class IndependentSlotsMixin:
 def _init_slots(self):
  self.slot_pair=None;self._slot_loader=None;self._slots_ready=False;self._pending_slots=None
  root=self.model_search_root
  self._slot_pin=Path(root)/'touch_deployment_selection.json' if root is not None else None
  if self._slot_pin is not None:
   try:
    raw=json.loads(self._slot_pin.read_text('utf-8'))
    if raw.get('state')=='disabled' or raw.get('smoke_replay') or (raw.get('position') or {}).get('input_contract',{}).get('input_shape')!=[9,5]:
     self.slot_warning_label.setText('等待45点模型训练/选择 · 旧9峰模型不兼容');self.start_live_button.setEnabled(False)
    else:self.load_slot_pair(read_pair(self._slot_pin),persist=False)
   except Exception as e:self.slot_warning_label.setText('默认双槽未加载（无自动回退）：'+str(e));self.start_live_button.setEnabled(False)
  self._quick_choices=None
  if self._slot_pin is not None:
   try:
    from touch_deployment_quick import read_choices
    self._quick_choices=read_choices(self._slot_pin.parent)
   except Exception as e:self.slot_warning_label.setToolTip('快捷选择不可用：'+str(e))
  self._quick_timer=QtCore.QTimer(self);self._quick_timer.timeout.connect(self._refresh_quick_buttons);self._quick_timer.start(150);self._refresh_quick_buttons()
 def _refresh_quick_buttons(self):
  busy=self.inference is not None or self.worker is not None or (self._slot_loader is not None and not self._slots_ready)
  for task in ('position','force'):
   for kind in ('best','last'):
    button=getattr(self,task+'_'+kind+'_button',None)
    if button is None:continue
    available=self._quick_choices is not None and self._quick_choices.get(kind) is not None
    button.setEnabled(not busy and available and self.slot_pair is not None)
    hint=('Best 已验证推荐；仍不代表泛化/部署达标。' if kind=='best' else 'Last 最近完成实验候选（实验性，可能更差）；不是最后epoch。')
    button.setToolTip(hint+('请先停止推理/等待训练或加载结束。' if busy else '' if available else '没有可验证的快捷模型索引；不会回退其他模型。'))
 def _quick_load(self,task,kind,persist=True):
  if self.inference is not None or self.worker is not None or (self._slot_loader is not None and not self._slots_ready):
   self.slot_warning_label.setText('请先停止推理并等待训练/模型加载结束后切换');return False
  try:
   from touch_deployment_quick import choose
   if self._slot_pin is None:raise ValueError('未配置持久化模型目录')
   pair=choose(self._slot_pin.parent,self.slot_pair,task,kind)
   return self.load_slot_pair(pair,persist=persist)
  except Exception as e:self.slot_warning_label.setText('拒绝快捷加载：'+str(e));return False
 def _browse_slot(self,task):
  p,_=QtWidgets.QFileDialog.getOpenFileName(self,'选择'+('位置' if task=='position' else '力')+'独立模型',str(ROOT/'outputs'),'模型 (*.pt *.joblib)')
  if p:self.load_component(task,p)
 def load_component(self,task,path):
  try:
   if str(path).endswith('.joblib'):
    from touch_deployment_slots import register_composition
    approved=ROOT/'outputs/delta9_composition_20261015_114020/selected_all680.joblib'
    candidate=register_composition(path) if Path(path).resolve()==approved.resolve() else register_group(path)
   else:candidate=register_search(path) if Path(path).name=='selected_independent.pt' else register_training(path)
   pair=copy.deepcopy(self.slot_pair or candidate);pair[task]=candidate[task];pair['trusted_sources'].update(candidate['trusted_sources']);validate_pair(pair)
   if pair['position']['protocol']!=pair['force']['protocol']:pair['warning']='实验模型／泛化未达标；位置与力来自不同验证协议的手动组合；跨协议不直接排名'
   return self.load_slot_pair(pair)
  except Exception as e:self.slot_warning_label.setText('拒绝载入：'+str(e));return False
 def load_slot_pair(self,pair,persist=True):
  from manual_touch_training_page import TouchInferenceWorker
  try:
   if pair['position']['input_contract'].get('input_shape')!=[9,5]:raise ValueError('当前部署只接收45点9×5模型；9峰模型已退役')
   if pair.get('smoke_replay') and persist:raise ValueError('合成smoke仅允许不持久化回放')
   validate_pair(pair)
  except Exception as e:self.slot_warning_label.setText('双槽校验失败：'+str(e));return False
  if self.inference is not None:
   self._pending_slots=(pair,persist);self.stop_live();return True
  if self._slot_loader is not None:
   self._pending_slots=(pair,persist);self._slot_loader.cancel();return True
  if persist and self._slot_pin is not None:
   try:
    save_pair(self._slot_pin,pair)  # Includes exact readback and byte-exact rollback on failure.
   except Exception as e:self.slot_warning_label.setText('持久化双槽失败：'+str(e));return False
  # Child snapshot is unique and immutable for this worker, not the mutable startup pin.
  scratch=Path(os.environ.get('TMPDIR',str(ROOT/'migration_evidence')));scratch.mkdir(parents=True,exist_ok=True)
  fd,name=tempfile.mkstemp(prefix='fbg-dual-',suffix='.json',dir=scratch);os.close(fd);Path(name).write_text(json.dumps(pair,ensure_ascii=False),encoding='utf-8')
  self.slot_pair=copy.deepcopy(pair);self._slots_ready=False
  self.model_path=Path(pair['position']['model_path']);self.report=dict(schema='fbg_touch45_multilabel_v1',input_contract=pair['position']['input_contract'])
  self.live_model_edit.setText(str(self.model_path))
  for task in ('position','force'):
   c=pair[task];getattr(self,task+'_model_edit').setText(c['model_path'])
   label=getattr(self,task+'_slot_status');label.setText(self._slot_summary(task,c,'加载中'))
   label.setToolTip(json.dumps(c,ensure_ascii=False,indent=2));getattr(self,task+'_model_edit').setToolTip(c['model_path'])
  self.slot_warning_label.setText(pair['warning']);self.slot_warning_label.setToolTip(pair['warning'])
  worker=TouchInferenceWorker(name,parent=self);worker.service_module='touch_dual_service';worker.snapshot_path=Path(name)
  self._slot_loader=worker
  worker.ready.connect(lambda payload,w=worker:self._slot_loaded(w,payload))
  worker.failed.connect(lambda error,w=worker:self._slot_failed(w,error))
  worker.finished.connect(lambda w=worker:self._slot_finished(w))
  self.start_live_button.setEnabled(False);self.live_status_label.setText('正在独立加载位置和力模型；不会启动采集或推理');worker.start();return True
 def _slot_summary(self,task,c,state):
  spectral=str(c.get('component_id','')).startswith('spectral_')
  protocol=('分组留出验证' if str(c.get('protocol','')).startswith('measured45-sourceholdout:') else '片段内顺序验证' if spectral else ('分组CV' if c['adapter'] in ('groupcv','composition') else '随机验证'))
  normalized='normalized' in c.get('metric_contract','') or c['adapter'] not in ('delta9','measured45')
  force_metric=(c.get('validation_metrics') or {}).get('force_normalized_mae',c['metric'][0]) if normalized else c['metric'][0]
  metric=('集合精确 '+format(-c['metric'][0],'.2%')) if task=='position' else (('归一MAE ' if normalized else 'MAE均值 ')+format(force_metric,'.4f'))
  title=({'position':'RBF γ0.3/α0.1','force':'MLP32×2'}[task] if c['adapter']=='groupcv' else ('Residual96×3集成' if c['adapter']=='search' else c['component_id']))
  if spectral:
   family=c['component_id'].split(':')[0]
   title={'spectral_patchtst':'PatchTST','spectral_itransformer':'iTransformer适配','spectral_tsmixer':'TSMixer','spectral_mlp':'MLP','spectral_timesnet':'TimesNet光谱'}.get(family,family)+(' 位置' if task=='position' else ' 力')
  if str(c.get('component_id','')).startswith('anchored45:'):
   report=json.loads(Path(c['report_path']).read_text('utf-8'))
   family=(report.get('spec') or {}).get('kind') if report.get('trained_task')==task else 'identity'
   title={'neural':'Best+残差','rbf':'Best+RBF','standalone':'ExtraTrees','identity':'Best锚点'}.get(family,'Anchor45')
  if str(c.get('component_id','')).startswith('local45:'):
   report=json.loads(Path(c['report_path']).read_text('utf-8'))
   family=report['task_specs'][task]['kind']
   title={'local':'局部线性','knn':'近邻','kernel':'直接核岭'}.get(family,'局部模型')
  if str(c.get('component_id','')).startswith('tslib45_'):
   family=c['component_id'].split(':')[0]
   title={'tslib45_itr':'iTr官方','tslib45_itr_query':'iTr-Q官方','tslib45_timemixer':'TimeMix官方','tslib45_patch':'Patch官方','tslib45_mixer':'TSMix官方','tslib45_timesnet':'Times适配'}.get(family,'TSLib适配')
  if str(c.get('component_id','')).startswith('mature45_'):
   family=c['component_id'].split(':')[0]
   title={'mature45_ftt':'FTT-CLS','mature45_resnet':'ResNet数值'}.get(family,'数值网络')
  if str(c.get('component_id','')).startswith('neural45_'):
   family=c['component_id'].split(':')[0]
   title={'neural45_hms':'HMS-Q','neural45_ft':'FT45-Q','neural45_geo':'GEO-Q','neural45_hms_be':'HMS-BE2','neural45_relation':'交互关系'}.get(family,'光谱注意力')
   if family=='neural45_relation' and c.get('report_path'):
    cfg=json.loads(Path(c['report_path']).read_text('utf-8'))['configuration']
    title=('轴专家'+str(cfg.get('experts',1))) if task=='force' else ('二阶关系' if cfg.get('relation_layers',0) else '二阶交互' if cfg.get('bilinear',False) else 'HMS基准')
  if task=='position' and c.get('report_path'):
   bound_report=json.loads(Path(c['report_path']).read_text('utf-8'))
   if bound_report.get('position_mode')=='single_label26_v1':
    title=title+'26类';metric='准确'+format(bound_report['metrics']['validation']['position_accuracy'],'.2%')
  choice=(getattr(self,'slot_pair',None) or {}).get('quick_choices',{}).get(task,{}).get('kind')
  badge=('Last实验' if choice=='last' else 'Best推荐' if choice=='best' else '')
  compact_protocol={'分组留出验证':'分组','片段内顺序验证':'顺序','随机验证':'随机'}.get(protocol,protocol)
  metric=metric.replace('集合精确 ','精确').replace('归一MAE ','归一MAE')
  title=title.replace(' 位置','').replace(' 力','')
  return title+' · '+compact_protocol+' '+metric+' · '+(badge+' · ' if badge else '')+state
 def _slot_loaded(self,worker,payload):
  if worker is not self._slot_loader:return
  if payload.get('input_contract')!=self.slot_pair['position']['input_contract']:
   self._slot_failed(worker,'报告和checkpoint输入契约不匹配');worker.cancel();return
  self._slots_ready=True
  for task in ('position','force'):
   c=self.slot_pair[task];text=self._slot_summary(task,c,'已加载')
   getattr(self,task+'_slot_status').setText(text)
  self.start_live_button.setEnabled(True);self.live_status_label.setText('双模型已加载（实验／泛化未达标）· 等待显式开始；未启动硬件')
 def _slot_failed(self,worker,error):
  if worker is not self._slot_loader:return
  self._slots_ready=False;self.start_live_button.setEnabled(False);self.live_status_label.setText('双模型加载／推理失败：'+error)
  for task in ('position','force'):getattr(self,task+'_slot_status').setText('加载失败：'+error)
 def _slot_finished(self,worker):
  if worker is not self._slot_loader:return
  if self.inference is worker:
   self.inference=None;self.live_render_timer.stop();self.stop_live_button.setEnabled(False)
  self._slot_loader=None;self._slots_ready=False;self.start_live_button.setEnabled(False)
  for task in ('position','force'):
   if self.slot_pair:getattr(self,task+'_slot_status').setText(self._slot_summary(task,self.slot_pair[task],'已停止／未加载'))
  worker.snapshot_path.unlink(missing_ok=True);worker.deleteLater()
  pending=self._pending_slots;self._pending_slots=None
  if pending:QtCore.QTimer.singleShot(0,lambda:self.load_slot_pair(*pending))
  elif self.slot_pair:self.live_status_label.setText('实时推理已停止；点击“重新加载”后可再次显式开始')
 def _start_slots(self):
  if not self._slots_ready or self._slot_loader is None:self.live_status_label.setText('请先加载两个模型并等待就绪');return False
  self.inference=self._slot_loader;self._live_time_origin=None;self._stream_reset_pending=True;self._last_dual_source=None
  self._live_prediction_count=0
  for rows in (self._live_samples,self._live_predicted,self._live_actual_force,self._live_raw_actual_force):rows.clear()
  for c in list(self.live_curves.predicted_curves.values())+list(self.live_curves.actual_curves.values()):c.setData([],[])
  self.live_render_timer.start();self.start_live_button.setEnabled(False);self.stop_live_button.setEnabled(True);self.live_status_label.setText('双模型实时推理已显式开始 · 实验／泛化未达标');return True
 def _auto_select_training(self,report):
  try:
   candidate=register_training(report['model_path'])
   if self._slot_pin is not None:
    from touch_deployment_quick import completed_training,read_choices
    previous_best=read_choices(self._slot_pin.parent)['best']
    self._quick_choices=completed_training(self._slot_pin.parent,candidate)
    recommended=self._quick_choices['best'];selected=copy.deepcopy(self.slot_pair or recommended)
    for task in ('position','force'):
     if recommended[task]!=previous_best[task]:
      selected[task]=copy.deepcopy(recommended[task]);selected.get('quick_choices',{}).pop(task,None)
      selected['trusted_sources'].update(recommended['trusted_sources'])
    validate_pair(selected)
   else:selected=promote(self.slot_pair,candidate)
   if selected==self.slot_pair:self.status_label.setText('候选已保存；验证协议不一致或无独立提升，保留当前双槽');return
   if not self.load_slot_pair(selected):
    self.status_label.setText('候选未载入：'+self.slot_warning_label.text());return
   self.status_label.setText('可信同协议验证选优完成；各自最佳模型自动加载，未自动开始')
  except Exception as e:self.status_label.setText('候选未自动加载：'+str(e))
 def _shutdown_slots(self):
  self._pending_slots=None
  w=self._slot_loader
  if w is not None:w.cancel();w.wait(5000)
