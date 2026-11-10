"""Bilinear peak interactions + explicit label relations; optical-only axis soft MoE.
Clean current-scan adaptation, not a C-Tran/ConvNeXtV2/SoftMoE reproduction.
Nine real identity-bearing peaks, five real points each. No true labels at inference.
"""
import torch
from torch import nn
from touch_neural45 import Neural45Backbone,Block,grid
class ResponseNorm(nn.Module):
 def __init__(self,w):
  super().__init__();self.gamma=nn.Parameter(torch.zeros(w));self.beta=nn.Parameter(torch.zeros(w))
 def forward(self,x):
  g=x.norm(p=2,dim=1,keepdim=True);return x+self.gamma*(x*g/(g.mean(-1,keepdim=True)+1e-6))+self.beta
class Relation45Task(nn.Module):
 def __init__(self,c,task):
  super().__init__();self.config=c;self.task=task;w=c['width'];d=c['dropout'];h=c.get('heads',4)
  self.encoder=Neural45Backbone(dict(c,architecture='neural45_hms'))
  self.response=ResponseNorm(w) if c.get('grn') else nn.Identity()
  self.bilinear=c.get('bilinear',True)
  if self.bilinear:self.pair=nn.Sequential(nn.Linear(w,w),nn.GELU(),nn.LayerNorm(w))
  if task=='position':
   self.queries=nn.Parameter(torch.randn(1,25,w)*.02);self.register_buffer('query_coords',grid(range(1,26)));self.query_coord=nn.Linear(2,w)
   self.cross=nn.ModuleList(Block(w,h,d) for _ in range(2));self.relations=nn.ModuleList(Block(w,h,d) for _ in range(c.get('relation_layers',1)))
   self.norm=nn.LayerNorm(w);self.weight=nn.Parameter(torch.randn(25,w)*.02);self.bias=nn.Parameter(torch.full((25,),-1.5))
  elif task=='force':
   self.expert_count=c.get('experts',2);e=self.expert_count;hidden=max(24,96//e);v=w+45
   self.base=nn.Sequential(nn.Linear(v,w),nn.GELU(),nn.Linear(w,3))
   self.experts=nn.ModuleList(nn.ModuleList(nn.Sequential(nn.Linear(v,hidden),nn.GELU(),nn.Dropout(d),nn.Linear(hidden,1)) for _ in range(e)) for a in range(3))
   self.gates=nn.ModuleList(nn.Sequential(nn.Linear(v,32),nn.GELU(),nn.Linear(32,e)) for _ in range(3))
  else:raise ValueError('Independent position or force required')
 def context(self,x,peak_ids=None):
  t,_=self.encoder(x,peak_ids=peak_ids);t=self.response(t)
  if self.bilinear:
   i,j=torch.triu_indices(9,9,offset=1,device=x.device);pairs=self.pair(t[:,i]*t[:,j]);return torch.cat((t,pairs),1)
  return t
 def force_parts(self,x,peak_ids=None):
  if self.task!='force':raise ValueError('Force model only')
  t=self.context(x,peak_ids);canonical=x if peak_ids is None else x.reshape(-1,9,5)[:,torch.argsort(peak_ids)].flatten(1)
  z=torch.cat((t.mean(1),canonical),-1);g=torch.stack([m(z).softmax(-1) for m in self.gates],1)
  residual=torch.stack([torch.cat([m(z) for m in axis],-1) for axis in self.experts],1)
  return self.base(z),g,residual
 def forward(self,x,state=None,peak_ids=None):
  if state is not None:raise ValueError('Current scan only; no force or label state inputs')
  if self.task=='force':
   base,g,r=self.force_parts(x,peak_ids);return base+(g*r).sum(-1),None
  t=self.context(x,peak_ids);q=self.queries.expand(len(x),-1,-1)+self.query_coord(self.query_coords)[None]
  for i,b in enumerate(self.cross):
   q=b(q,t)
   if i<len(self.relations):q=self.relations[i](q)
  for b in self.relations[len(self.cross):]:q=b(q)
  return (self.norm(q)*self.weight).sum(-1)+self.bias
 def regularization(self,x):
  if self.task!='force' or self.expert_count==1:return x.sum()*0
  _,g,_=self.force_parts(x);balance=(g.mean(0)-1/self.expert_count).square().mean()
  entropy=(g*g.clamp_min(1e-8).log()).sum(-1).mean()
  return .01*balance+.001*entropy
