"""Current-scan45 attention families; clean reimplementation, no upstream execution.
Research: FT-Transformer (Gorishniy et al. 2021), Set Transformer (Lee et al.
2019), Query2Label (Liu et al. 2021). Pinned reviewed sources/licenses and
adaptation decisions: outputs/five45_neural_20261019_161124/upstream_provenance.json.
Canonical coordinates are normalized label-grid identities, NOT calibrated mm.
"""
import torch
from torch import nn
from torch.nn import functional as F
KINDS=('neural45_hms','neural45_ft','neural45_geo')
ANCHORS=(25,15,5,3,13,23,21,11,1)
def grid(labels):
 t=torch.as_tensor(labels,dtype=torch.long)-1
 return torch.stack(((t%5).float()/4,(t//5).float()/4),-1)
class Attention(nn.Module):
 def __init__(self,w,heads,drop):
  super().__init__();self.heads=heads;self.drop=drop;self.q=nn.Linear(w,w);self.kv=nn.Linear(w,2*w);self.out=nn.Linear(w,w)
 def forward(self,q,kv,bias=None):
  b,n,w=q.shape;h=self.heads;d=w//h
  q=self.q(q).reshape(b,n,h,d).transpose(1,2);k,v=self.kv(kv).reshape(b,-1,2,h,d).permute(2,0,3,1,4).unbind(0)
  z=F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=self.drop if self.training else 0.)
  return self.out(z.transpose(1,2).reshape(b,n,w))
class Block(nn.Module):
 def __init__(self,w,h,drop):
  super().__init__();self.nq=nn.LayerNorm(w);self.nk=nn.LayerNorm(w);self.attn=Attention(w,h,drop);self.nf=nn.LayerNorm(w);self.up=nn.Linear(w,6*w);self.down=nn.Linear(3*w,w);self.drop=nn.Dropout(drop)
 def forward(self,x,context=None,bias=None):
  kv=x if context is None else context;x=x+self.drop(self.attn(self.nq(x),self.nk(kv),bias));a,g=self.up(self.nf(x)).chunk(2,-1)
  return x+self.drop(self.down(a*F.relu(g)))
class Neural45Backbone(nn.Module):
 def __init__(self,c):
  super().__init__();self.kind=c['architecture'];w=c['width'];h=c.get('heads',4);drop=c['dropout'];self.width=w
  if self.kind not in KINDS:raise ValueError('Unknown current45 neural family')
  self.register_buffer('sensor_coords',grid(ANCHORS));self.sensor_id=nn.Embedding(9,w);self.coord=nn.Sequential(nn.Linear(2,w),nn.GELU(),nn.Linear(w,w))
  self.absolute=nn.Sequential(nn.Linear(5,w),nn.GELU(),nn.Linear(w,w));self.mask_token=nn.Parameter(torch.zeros(1,1,w))
  if self.kind=='neural45_hms':
   self.convs=nn.ModuleList(nn.Conv1d(1,w//4,k,padding=k//2) for k in [1,3,5]);self.local=nn.Sequential(nn.Linear(3*(w//4)*5,w),nn.GELU(),nn.Linear(w,w))
  if self.kind=='neural45_ft':
   self.numeric_weight=nn.Parameter(torch.randn(45,w)*.05);self.numeric_bias=nn.Parameter(torch.randn(45,w)*.02);self.point_id=nn.Embedding(5,w)
  self.blocks=nn.ModuleList(Block(w,h,drop) for _ in range(c['layers']));self.norm=nn.LayerNorm(w)
  if self.kind=='neural45_geo':self.relative=nn.Sequential(nn.Linear(3,w//2),nn.GELU(),nn.Linear(w//2,h))
 def relation(self,q,k):
  d=q[:,None]-k[None];z=torch.cat((d,d.square().sum(-1,keepdim=True)),-1)
  return self.relative(z).permute(2,0,1)[None]
 def forward(self,x,peak_ids=None,unavailable=None):
  if x.ndim!=2 or x.shape[1]!=45:raise ValueError('Exactly B x45 real current-scan measurements required')
  b=len(x);p=x.reshape(b,9,5);ids=torch.arange(9,device=x.device) if peak_ids is None else peak_ids;coords=self.sensor_coords[ids]
  absolute=self.absolute(p);identity=self.sensor_id(ids)+self.coord(coords)
  if self.kind=='neural45_ft':
   t=x[:,:,None]*self.numeric_weight[None]+self.numeric_bias[None]
   t=t+identity.repeat_interleave(5,0)[None]+self.point_id(torch.arange(5,device=x.device)).repeat(9,1)[None]+absolute.repeat_interleave(5,1)
   if unavailable is not None:t=torch.where(unavailable.repeat_interleave(5,1)[:,:,None],self.mask_token+identity.repeat_interleave(5,0)[None],t)
  else:
   t=absolute
   if self.kind=='neural45_hms':
    shape=p-p.mean(-1,keepdim=True);shape=shape.reshape(b*9,1,5)
    t=t+self.local(torch.cat([F.gelu(conv(shape)).flatten(1) for conv in self.convs],-1)).reshape(b,9,-1)
   if unavailable is not None:t=torch.where(unavailable[:,:,None],self.mask_token,t)
   t=t+identity[None]
  bias=self.relation(coords,coords) if self.kind=='neural45_geo' else None
  for block in self.blocks:t=block(t,bias=bias)
  return self.norm(t),coords
def pretrain_encoder(c,x,train_indices,*,seed,epochs,batch_size=128):
 """Only explicit fold-training rows; unavailable peak never means physical zero."""
 with torch.random.fork_rng(devices=[]):
  torch.manual_seed(seed);model=Neural45Backbone(c);head=nn.Linear(c['width'],5);opt=torch.optim.AdamW(list(model.parameters())+list(head.parameters()),lr=.0007,weight_decay=.001)
  train=x[torch.as_tensor(train_indices)].clone()
  for ep in range(epochs):
   order=torch.randperm(len(train))
   for ids in order.split(batch_size):
    target=train[ids].reshape(-1,9,5);mask=torch.zeros(len(ids),9,dtype=torch.bool);mask[torch.arange(len(ids)),torch.randint(9,(len(ids),))]=True
    masked=target.clone();masked[mask]=0.;opt.zero_grad(set_to_none=True);tokens,_=model(masked.flatten(1),unavailable=mask)
    if tokens.shape[1]!=9:tokens=tokens.reshape(len(ids),9,5,-1).mean(2)
    loss=F.mse_loss(head(tokens)[mask],target[mask]);loss.backward();nn.utils.clip_grad_norm_(list(model.parameters())+list(head.parameters()),1);opt.step()
  return {k:v.detach().clone() for k,v in model.state_dict().items()}
class Neural45Task(nn.Module):
 def __init__(self,c,task):
  super().__init__();self.config=c;self.task=task;w=c['width'];h=c.get('heads',4);d=c['dropout'];self.encoder=Neural45Backbone(c);n=25 if task=='position' else 3
  self.queries=nn.Parameter(torch.randn(1,n,w)*.02);self.register_buffer('query_coords',grid(range(1,26)) if task=='position' else torch.zeros(3,2));self.query_coord=nn.Linear(2,w)
  self.decoder=nn.ModuleList(Block(w,h,d) for _ in range(2));self.norm=nn.LayerNorm(w);self.weight=nn.Parameter(torch.randn(n,w)*.02);self.bias=nn.Parameter(torch.full((n,),-1.5 if task=='position' else 0.))
 def forward(self,x,state=None,peak_ids=None):
  t,coords=self.encoder(x,peak_ids=peak_ids);q=self.queries.expand(len(x),-1,-1)
  if self.task=='position':q=q+self.query_coord(self.query_coords)[None]
  bias=None
  if self.config['architecture']=='neural45_geo' and self.task=='position':bias=self.encoder.relation(self.query_coords,coords)
  for block in self.decoder:q=block(q,t,bias)
  y=(self.norm(q)*self.weight).sum(-1)+self.bias
  return y if self.task=='position' else (y,None)
