"""Portable frozen single-task Best anchor; only real current45 optical inputs."""
import copy
import numpy as np
import torch
from torch import nn

def make_entry(path,task):
 from touch_45_models import Measured45Bundle
 b=Measured45Bundle(path)
 return dict(teacher_config=copy.deepcopy(b.config),teacher_state=copy.deepcopy(getattr(b,task+'_model').state_dict()),teacher_statistics=copy.deepcopy(b.stats),kind='identity')

class AxisEstimator:
 """Each force axis retains its own actual-target mask; preserve float64 RBF."""
 def __init__(self,models):self.models=models
 def predict(self,x):return np.column_stack([m.predict(np.asarray(x,np.float64)) for m in self.models])

class AnchoredTask(nn.Module):
 def __init__(self,entry,task):
  super().__init__()
  from touch_45_models import build,PositionModel,ForceModel
  self.entry=entry;self.task=task
  self.teacher=build(PositionModel if task=='position' else ForceModel,entry['teacher_config'])
  self.teacher.load_state_dict(entry['teacher_state']);self.teacher.requires_grad_(False);self.teacher.eval()
  s=entry['teacher_statistics']
  for k in ['feature_mean','feature_std','force_mean','force_std']:self.register_buffer(k,torch.as_tensor(s[k],dtype=torch.float32).clone())
  if entry['kind']=='neural':
   self.register_buffer('correction_center',torch.tensor(entry['center'],dtype=torch.float32));self.register_buffer('correction_scale',torch.tensor(entry['scale'],dtype=torch.float32))
   w=entry['width'];out=25 if task=='position' else 3
   self.delta=nn.Sequential(nn.Linear(45,w),nn.GELU(),nn.Linear(w,out))
   self.gate=nn.Sequential(nn.Linear(45,16),nn.GELU(),nn.Linear(16,1),nn.Sigmoid())
   nn.init.zeros_(self.delta[-1].weight);nn.init.zeros_(self.delta[-1].bias)
 def train(self,mode=True):
  super().train(mode);self.teacher.eval();return self
 def base(self,x):
  z=(x-self.feature_mean)/self.feature_std
  with torch.no_grad():y=self.teacher(z)
  return y if self.task=='position' else y[0]*self.force_std+self.force_mean
 def forward(self,x,state=None):
  y=self.base(x);e=self.entry
  if e['kind']=='neural':
   z=(x-self.correction_center)/self.correction_scale
   y=y+self.gate(z)*self.delta(z)
  elif e['kind']!='identity':
   z=(x.detach().cpu().numpy().astype(np.float64)-e['center'])/e['scale']
   pred=e['estimator'].predict(z)
   if self.task=='position' and e['kind']=='standalone':
    p=np.clip(pred,1e-6,1-1e-6);y=torch.as_tensor(np.log(p/(1-p)),dtype=x.dtype,device=x.device)
   elif e['kind']=='standalone':y=torch.as_tensor(pred,dtype=x.dtype,device=x.device)
   else:y=y+torch.as_tensor(pred*e['strength'],dtype=x.dtype,device=x.device)
  return y if self.task=='position' else (y,None)
