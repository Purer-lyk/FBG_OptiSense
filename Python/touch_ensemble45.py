"""HMS-Q with rank-one BatchEnsemble fast weights (TabM-inspired, not TabM).
The nine x five axis is one real scan, never time. Two members per independent task.
Official references and licenses pinned in five45_ensemble research artifacts.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F
from touch_neural45 import Neural45Task
class BatchEnsembleLinear(nn.Module):
 def __init__(self,linear,k,repeat=1):
  super().__init__();self.k=k;self.repeat=repeat
  self.weight=linear.weight
  self.r=nn.Parameter(torch.empty(k,linear.in_features).bernoulli_(.5).mul_(2).sub_(1))
  self.s=nn.Parameter(torch.ones(k,linear.out_features))
  self.bias=nn.Parameter(linear.bias.detach().expand(k,-1).clone()) if linear.bias is not None else None
 def forward(self,x):
  shape=x.shape;y=x.reshape(-1,self.k,self.repeat,*shape[1:]);extra=y.ndim-3
  r=self.r.reshape(1,self.k,*([1]*extra),-1);s=self.s.reshape(1,self.k,*([1]*extra),-1)
  z=F.linear(y*r,self.weight)*s
  if self.bias is not None:z=z+self.bias.reshape(1,self.k,*([1]*extra),-1)
  return z.reshape(*shape[:-1],-1)
class Ensemble45Task(nn.Module):
 def __init__(self,c,task):
  super().__init__();self.config=c;self.task=task;self.k=c.get('ensemble_members',2)
  if self.k!=2:raise ValueError('Frozen two-member ensemble required')
  base=dict(c,architecture='neural45_hms');self.base=Neural45Task(base,task)
  self.input_scale=nn.Parameter(torch.empty(self.k,45).bernoulli_(.5).mul_(.2).add_(.9))
  def convert(module,repeat=1):
   for n,ch in list(module.named_children()):
    if isinstance(ch,nn.Linear):setattr(module,n,BatchEnsembleLinear(ch,self.k,repeat))
    else:convert(ch,repeat)
  for name in ['absolute','blocks']:convert(getattr(self.base.encoder,name))
  convert(self.base.encoder.local,9);convert(self.base.decoder)
 @property
 def encoder(self):return self.base.encoder
 def load_pretrained(self,state):
  # Copy slow weights only; fast weights remain independently initialized.
  own=self.encoder.state_dict()
  for n,v in state.items():
   if n in own:
    if own[n].shape==v.shape:own[n].copy_(v)
    elif own[n].ndim==2 and v.ndim==1 and own[n].shape[1:]==v.shape:own[n].copy_(v[None].expand_as(own[n]))
  self.encoder.load_state_dict(own)
 def member_outputs(self,x,peak_ids=None):
  if x.ndim!=2 or x.shape[1]!=45:raise ValueError('Exactly B x45 required')
  scale=self.input_scale
  if peak_ids is not None:scale=scale.reshape(self.k,9,5)[:,peak_ids].flatten(1)
  z=(x[:,None,:]*scale[None]).reshape(-1,45);y=self.base(z,peak_ids=peak_ids)
  if self.task=='force':y=y[0]
  return y.reshape(len(x),self.k,-1)
 def forward(self,x,state=None,peak_ids=None):
  y=self.member_outputs(x,peak_ids)
  return torch.logit(y.sigmoid().mean(1).clamp(1e-7,1-1e-7)) if self.task=='position' else (y.mean(1),None)
def make_task(c,task):return Ensemble45Task(c,task) if c['architecture']=='neural45_hms_be' else Neural45Task(c,task)
def optimization_step(model,opt,closure,*,seed,rho=0.):
 """Two SAM passes use identical task-local dropout, restore exact weights before update."""
 opt.zero_grad(set_to_none=True)
 def backward():
  with torch.random.fork_rng(devices=[]):
   torch.manual_seed(seed);loss=closure();loss.backward()
  return loss
 loss=backward()
 if rho:
  params=[p for p in model.parameters() if p.grad is not None];norm=torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in params]));old=[p.detach().clone() for p in params]
  with torch.no_grad():
   for p in params:p.add_(p.grad,alpha=float(rho/(norm+1e-12)))
  opt.zero_grad(set_to_none=True)
  try:backward()
  finally:
   with torch.no_grad():
    for p,v in zip(params,old):p.copy_(v)
 nn.utils.clip_grad_norm_(model.parameters(),1.);opt.step();return float(loss.detach())
def ema_copy(model):return copy.deepcopy(model).requires_grad_(False)
def ema_update(ema,model,decay):
 with torch.no_grad():
  for e,p in zip(ema.parameters(),model.parameters()):e.lerp_(p,1-decay)
  for e,p in zip(ema.buffers(),model.buffers()):e.copy_(p)

