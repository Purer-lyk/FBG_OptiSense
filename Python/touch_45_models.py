"""Torch-first measured45 independent multilabel position / causal force models.
TCN measures45 real points within a scan; no optical-frame history for position.
"""
import argparse,copy,hashlib,json,time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from touch_45_data import MODEL_SCHEMA,ARCHITECTURES,prepare_data,freeze_split,fit_statistics,validate_input_contract
from touch_temporal_models import AttentionPool,GatedTCNBlock,InceptionBlock,SparseBlock
from temporal_sequence_audit import continuous_runs
DEFAULT=dict(width=48,layers=3,heads=4,ffn=144,dropout=.1,hidden=48,shared=96,force_history=False,architecture='sensor_token',max_gap_s=30.,seed=20261014,batch_size=64,chunk_length=16)
class Measured45Encoder(nn.Module):
 def __init__(self,config):
  super().__init__();self.config=c=dict(DEFAULT,**config);self.kind=c['architecture'];w=c['width']
  from touch_spectral45_encoders import KINDS,Spectral45Encoder
  if self.kind=='spectral_timesnet':
   from touch_timesnet45 import PeakTimesNet45
   self.spectral=PeakTimesNet45(c);return
  if self.kind in KINDS:self.spectral=Spectral45Encoder(c);return
  if self.kind not in ARCHITECTURES:raise ValueError('Unsupported measured45 encoder')
  self.pool=AttentionPool(w);self.output=nn.Sequential(nn.LayerNorm(w),nn.Linear(w,c['shared']),nn.GELU(),nn.Dropout(c['dropout']))
  if self.kind=='sensor_token':
   self.embedding=nn.Linear(5,w);self.peak_embedding=nn.Parameter(torch.randn(1,9,w)*.02)
   layer=nn.TransformerEncoderLayer(w,c['heads'],c['ffn'],c['dropout'],activation='gelu',batch_first=True,norm_first=True)
   self.blocks=nn.TransformerEncoder(layer,c['layers'],norm=nn.LayerNorm(w),enable_nested_tensor=False)
  elif self.kind=='sparse_cnn':
   self.embedding=nn.Conv2d(6,w,1);self.blocks=nn.ModuleList(SparseBlock(w,c['dropout']) for _ in range(c['layers']))
   self.register_buffer('spatial_mask',torch.tensor([[0,1,0],[1,1,1],[1,1,1],[1,0,1]],dtype=torch.float32).reshape(1,1,4,3))
  else:
   self.embedding=nn.Conv1d(1,w,1)
   if self.kind=='tcn':
    ds=[1,2,4,8];self.receptive_field=1+4*sum(ds);self.blocks=nn.Sequential(*[GatedTCNBlock(w,d,c['dropout']) for d in ds])
   else:self.blocks=nn.Sequential(*[InceptionBlock(w,c['dropout']) for _ in range(c['layers'])])
 def sparse_input(self,x):
  grid=x.new_zeros((len(x),5,4,3));x=x.reshape(-1,9,5)
  for row,sensors in enumerate(((0,4,0),(9,5,3),(8,6,2),(7,0,1))):
   for col,s in enumerate(sensors):
    if s:grid[:,:,row,col]=x[:,s-1]
  return grid,self.spatial_mask
 def forward(self,x):
  if hasattr(self,'spectral'):return self.spectral(x)
  if x.ndim!=2 or x.shape[1]!=45:raise ValueError('Exactly B x 45 engineering dBm features required')
  mask=None
  if self.kind=='sensor_token':tokens=self.blocks(self.embedding(x.reshape(-1,9,5))+self.peak_embedding)
  elif self.kind=='sparse_cnn':
   grid,m=self.sparse_input(x);h=self.embedding(torch.cat((grid,m.expand(len(x),-1,-1,-1)),1))*m
   for block in self.blocks:h=block(h,m)
   tokens=h.flatten(2).transpose(1,2);mask=m.flatten(2).squeeze(1).expand(len(x),-1)
  else:tokens=self.blocks(self.embedding(x.unsqueeze(1))).transpose(1,2)
  return self.output(self.pool(tokens,mask))
class PositionModel(nn.Module):
 def __init__(self,c):
  super().__init__();self.encoder=Measured45Encoder(c);self.head=nn.Linear(c['shared'],25)
 def forward(self,x):return self.head(self.encoder(x))
class ForceModel(nn.Module):
 def __init__(self,c):
  super().__init__();self.config=c;self.encoder=Measured45Encoder(c);w=c['shared']
  self.anchor=nn.Sequential(nn.Linear(w,w),nn.GELU(),nn.Dropout(c['dropout']),nn.Linear(w,3))
  if c['force_history']:self.gru=nn.GRUCell(w+4,c['hidden']);self.residual=nn.Linear(c['hidden'],3)
 def forward(self,x,state=None):
  z=self.encoder(x);y=self.anchor(z)
  if not self.config['force_history']:return y,None
  if state is None:state=(z.new_zeros(len(x),self.config['hidden']),z.new_zeros(len(x),3),z.new_zeros(len(x),1))
  h,previous,mask=state;h=self.gru(torch.cat((z,previous.detach()*mask,mask),1),h);y=y+self.residual(h)
  return y,(h,y.detach(),z.new_ones(len(x),1))
def build(cls,c):
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(c['seed'])
  if str(c.get('architecture','')).startswith('tslib45_'):
   from touch_tslib45 import TSLib45Task
   return TSLib45Task(c,'position' if cls is PositionModel else 'force')
  if str(c.get('architecture','')).startswith('mature45_'):
   from touch_mature45 import Mature45Task
   return Mature45Task(c,'position' if cls is PositionModel else 'force')
  if c.get('architecture')=='neural45_relation':
   from touch_relation45 import Relation45Task
   return Relation45Task(c,'position' if cls is PositionModel else 'force')
  if c.get('architecture')=='neural45_hms_be':
   from touch_ensemble45 import Ensemble45Task
   return Ensemble45Task(c,'position' if cls is PositionModel else 'force')
  if str(c.get('architecture','')).startswith('neural45_'):
   from touch_neural45 import Neural45Task
   return Neural45Task(c,'position' if cls is PositionModel else 'force')
  if c.get('architecture')=='local45':
   from touch_local45 import LocalTask
   task='position' if cls is PositionModel else 'force'
   return LocalTask(c[task+'_entry'],task)
  if c.get('architecture')=='anchored45':
   from touch_anchor45 import AnchoredTask
   task='position' if cls is PositionModel else 'force'
   return AnchoredTask(c[task+'_entry'],task)
  return cls(c)
def rollout(model,x,resets,state=None):
 rows=[]
 for i in range(len(x)):
  if resets[i]:state=None
  y,state=model(x[i:i+1],state);rows.append(y)
 return torch.cat(rows),state
class Measured45Bundle:
 def __init__(self,path):
  torch.set_num_threads(2);self.path=Path(path).resolve();p=torch.load(self.path,map_location='cpu',weights_only=False)
  if p.get('schema')!=MODEL_SCHEMA:raise ValueError('Wrong measured45 checkpoint schema')
  self.input_contract=p['input_contract'];self.report=p['report'];self.config=p['configuration'];self.kind='measured45_'+self.config['architecture']
  validate_input_contract(self.input_contract)
  if self.report.get('input_contract')!=self.input_contract:raise ValueError('Checkpoint report contract differs')
  self.position_model=build(PositionModel,self.config);self.force_model=build(ForceModel,self.config)
  self.position_model.load_state_dict(p['position_model_state']);self.force_model.load_state_dict(p['force_model_state']);self.position_model.eval();self.force_model.eval()
  self.stats={k:np.asarray(v,np.float32) for k,v in p['statistics'].items()}
  for key,shape in [('feature_mean',(45,)),('feature_std',(45,)),('force_mean',(3,)),('force_std',(3,))]:
   v=self.stats[key]
   if v.shape!=shape or not np.isfinite(v).all() or ('std' in key and (v<=0).any()):raise ValueError('Invalid measured45 scaler')
  self.reset_state()
 def reset_state(self):self._state=None
 def inputs(self,x):
  x=np.asarray(x,np.float32)
  if x.shape==(45,):x=x[None]
  if x.ndim!=2 or x.shape[1]!=45 or not np.isfinite(x).all():raise ValueError('Exactly N x 45 finite engineering dBm deltas required')
  return torch.from_numpy((x-self.stats['feature_mean'])/self.stats['feature_std'])
 def predict_payload(self,x):
  x=self.inputs(x)
  if len(x)!=1:raise ValueError('Streaming one frame only')
  with torch.inference_mode():
   raw=self.position_model(x);probs=(raw.softmax(-1) if self.config.get('position_mode')=='single_label26_v1' else raw.sigmoid())[0].numpy();f,self._state=self.force_model(x,self._state);force=f[0].numpy()*self.stats['force_std']+self.stats['force_mean']
  if self.config.get('position_mode')=='single_label26_v1':
   from touch_single26_contract import decode_single26
   return dict(decode_single26(probs),force=force.tolist())
  threshold=float(self.config.get('position_threshold',.5))
  labels=(np.flatnonzero(probs>=threshold)+1).tolist()
  return dict(position_labels=labels,position=labels[0] if len(labels)==1 else (0 if not labels else None),position_multihot=(probs>=threshold).astype(int).tolist(),probabilities=probs.tolist(),confidence=float(np.maximum(probs,1-probs).min()),decision_status='pressed' if labels else 'no_press',force=force.tolist())

def train(directory,output,*,epochs=180,architecture='sensor_token',force_history=False):
 torch.set_num_threads(2);data=prepare_data(directory);output=Path(output);split=freeze_split(data,output)
 c=dict(DEFAULT,architecture=architecture,force_history=bool(force_history));id=architecture+('_history' if force_history else '_current');path=output/(id+'.pt')
 if path.exists():raise FileExistsError('Candidate checkpoint already exists')
 if epochs<1:raise ValueError('Positive epochs required')
 stats=fit_statistics(data,split['train']);p=build(PositionModel,c);f=build(ForceModel,c)
 opts={k:torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001) for k,m in [('position',p),('force',f)]}
 support=data['labels'][split['train']].sum(0);posweight=np.ones(25,np.float32);np.divide(len(split['train'])-support,support,out=posweight,where=support>0);posweight=np.maximum(posweight,1)
 loss_position=nn.BCEWithLogitsLoss(pos_weight=torch.from_numpy(posweight))
 # Training never reads test values. Each split has isolated source-ordered runs.
 def subset(name):
  idx=split[name];runs=continuous_runs(data,np.array(idx),gap_s=c['max_gap_s']);order=[i for run in runs for i in run];resets=[j==0 for run in runs for j in range(len(run))]
  x=torch.from_numpy((data['features'][order]-stats['feature_mean'])/stats['feature_std']);y=torch.from_numpy(data['labels'][order]);true=data['forces'][order];valid=data['valid'][order]
  return order,x,y,true,valid,resets
 trainset=subset('train');valset=subset('validation')
 def force_loss(pred,true,valid):
  target=torch.from_numpy(np.nan_to_num((true-stats['force_mean'])/stats['force_std']));mask=torch.from_numpy(valid)
  if not mask.any():return pred.sum()*0
  return nn.functional.smooth_l1_loss(pred[mask],target[mask])
 def evaluate(s):
  order,x,y,true,valid,resets=s
  with torch.inference_mode():
   probs=torch.cat([p(x[i:i+1]).sigmoid() for i in range(len(x))]).numpy();pred,_=rollout(f,x,resets);pred=pred.numpy()*stats['force_std']+stats['force_mean']
  mae=[float(np.abs(pred[valid[:,a],a]-true[valid[:,a],a]).mean()) if valid[:,a].any() else None for a in range(3)]
  return dict(position_exact_match=float(((probs>=.5)==y.numpy()).all(1).mean()),position_accuracy=float(((probs>=.5)==y.numpy()).all(1).mean()),force_mae_n=mae,force_vector_mae_n=float(np.linalg.norm((pred-true)[valid.all(1)],axis=1).mean()) if valid.all(1).any() else None),probs,pred
 best={};history={};_,tx,ty,tf,tv,tr=trainset
 if not tv.any():raise ValueError('No valid corrected force training labels')
 for task,model in [('position',p),('force',f)]:
  bestkey=None;rows=[]
  for epoch in range(1,epochs+1):
   model.train();state=None
   if task=='position':
    generator=torch.Generator().manual_seed(c['seed']+epoch*1009)
    order=torch.randperm(len(tx),generator=generator)
    batches=[order[i:i+c['batch_size']] for i in range(0,len(tx),c['batch_size'])]
   else:batches=[slice(i,min(i+c['chunk_length'],len(tx))) for i in range(0,len(tx),c['chunk_length'])]
   for batch_no,index in enumerate(batches):
    opts[task].zero_grad(set_to_none=True)
    with torch.random.fork_rng(devices=[]):
     torch.manual_seed(c['seed']+epoch*1009+batch_no*2+(task=='force'))
     if task=='position':loss=loss_position(p(tx[index]),ty[index])
     else:pred,state=rollout(f,tx[index],tr[index],state);loss=force_loss(pred,tf[index],tv[index])
     loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(),1);opts[task].step()
    if state is not None:state=tuple(v.detach() for v in state)
   model.eval()
   _,vx,vy,vf,vv,vr=valset
   with torch.inference_mode():
    if task=='position':
     logits=p(vx);vl=float(loss_position(logits,vy));acc=float(((logits.sigmoid()>=.5)==vy.bool()).all(1).float().mean());key=(acc,-vl);row=dict(validation_accuracy=acc,validation_bce=vl)
    else:
     pred,_=rollout(f,vx,vr);fp=pred.numpy()*stats['force_std']+stats['force_mean'];mae=[float(np.abs(fp[vv[:,a],a]-vf[vv[:,a],a]).mean()) if vv[:,a].any() else None for a in range(3)];finite=[a for a in mae if a is not None]
     if not finite:raise ValueError('No force validation references')
     key=(-float(np.mean(finite)),);row=dict(validation_force_mae_n=mae)
   rows.append(dict(epoch=epoch,**row))
   if bestkey is None or key>bestkey:bestkey=key;best[task]=dict(epoch=epoch,state=copy.deepcopy(model.state_dict()),selection_key=list(key))
   print(json.dumps(dict(type='progress',task=task,epoch=epoch,total=epochs,metrics=dict(task=task,**row))),flush=True)
  model.load_state_dict(best[task]['state']);model.eval();history[task]=rows
 # Freeze independent validation-selected states before opening test exactly once.
 selection=dict(position=best['position']['selection_key'],force=best['force']['selection_key'],source_sha256=data['source_sha256'])
 selection_path=output/(id+'_validation_selection.json');selection_path.write_text(json.dumps(selection,indent=2),encoding='utf-8')
 metrics={};validation_trace=None
 for name in ('validation','test'):
  s=valset if name=='validation' else subset(name);met,probs,pred=evaluate(s);metrics[name]=met
  np.savez_compressed(output/(id+'_'+name+'_predictions.npz'),indices=s[0],reset_flags=s[5],true_forces=s[3],valid=s[4],probabilities=probs,forces=pred,labels=s[2].numpy(),timestamp_s=data['timestamp_s'][s[0]])
  if name=='validation':validation_trace=(s,pred)
 report=dict(schema=MODEL_SCHEMA,input_shape=[9,5],input_contract=data['input_contract'],architecture=architecture,configuration=c,input_unit='dBm',class_labels=list(range(1,26)),position_head='25 independent sigmoid multilabel; empty set=no press',normalization_source='train only',selection_source='validation only independent tasks',test_replays=1,validation_selection_sha256=hashlib.sha256(selection_path.read_bytes()).hexdigest(),force_history=force_history,teacher_forcing=False,force_semantics='corrected whole-contact resultant; no per-position allocation/no clamp/no tare prediction',train_count=len(split['train']),validation_count=len(split['validation']),test_count=len(split['test']),best_epoch={k:v['epoch'] for k,v in best.items()},metrics=metrics,history=history,training_class_support=support.tolist(),missing_training_positions=(np.flatnonzero(support==0)+1).tolist(),pos_weight_train_only=posweight.tolist(),force_training_axis_support=data['valid'][split['train']].sum(0).tolist(),smoke_only=data['smoke_only'],deployment_ready=False,validation_warning='随机邻帧存在泄漏；缺失位置无泛化证据。光学/PX6D近时采样非严格同步；合成smoke不代表真实精度。',model_path=str(path.resolve()),dataset_dir=str(Path(directory).resolve()),source_sha256=data['source_sha256'])
 torch.save(dict(schema=MODEL_SCHEMA,input_contract=data['input_contract'],configuration=c,position_model_state=p.state_dict(),force_model_state=f.state_dict(),statistics=stats,report=report),path)
 report['checkpoint_sha256']=hashlib.sha256(path.read_bytes()).hexdigest();path.with_suffix('.report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
 return report

def cli():
 parser=argparse.ArgumentParser();parser.add_argument('--train',type=Path,required=True);parser.add_argument('--output',type=Path);parser.add_argument('--epochs',type=int,default=180);parser.add_argument('--architecture',choices=ARCHITECTURES,default='sensor_token');parser.add_argument('--force-history',action='store_true');args=parser.parse_args()
 report=train(args.train,args.output or args.train/'models_measured45_candidates'/str(time.time_ns()),epochs=args.epochs,architecture=args.architecture,force_history=args.force_history)
 print(json.dumps(dict(type='completed',model_path=report['model_path'],report_path=str(Path(report['model_path']).with_suffix('.report.json')))),flush=True)
if __name__=='__main__':cli()
