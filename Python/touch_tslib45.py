"""Official MIT TSLib current-frame adapters, not temporal forecasting.
Pinned commit 4e938a1767106324dd753b2a44832bf870a0252e.
Vendor model/block methods unchanged; task head and physical input-axis mapping
are explicit below. Inputs are 45 real measurements; no generated time points.
"""
from types import SimpleNamespace
import torch
from torch import nn
from touch_tslib45_vendor.models.iTransformer import Model as ITransformer
class TSLib45Task(nn.Module):
 def __init__(self,c,task):
  super().__init__();self.config=c;self.task=task
  cfg=SimpleNamespace(task_name='classification',seq_len=5,pred_len=0,enc_in=9,d_model=c['width'],e_layers=c['layers'],n_heads=c['heads'],d_ff=c['width']*2,dropout=c['dropout'],factor=1,activation='gelu',embed='fixed',freq='h',num_class=(26 if c.get('position_mode')=='single_label26_v1' else 25) if task=='position' else 3)
  self.kind=c['architecture']
  if self.kind=='tslib45_timemixer':
   from touch_tslib45_vendor.models.TimeMixer import Model
   cfg.label_len=0;cfg.down_sampling_window=2;cfg.down_sampling_layers=1;cfg.down_sampling_method='avg';cfg.channel_independence=0;cfg.decomp_method='moving_avg';cfg.moving_avg=3;cfg.use_norm=0;cfg.c_out=9
   self.network=Model(cfg)
  elif self.kind=='tslib45_patch':
   from touch_tslib45_vendor.models.PatchTST import Model
   cfg.seq_len=45;cfg.enc_in=1;self.network=Model(cfg,patch_len=5,stride=5)
   self.network.patch_embedding.padding_patch_layer=nn.Identity();self.network.head_nf=9*c['width'];self.network.projection=nn.Linear(9*c['width'],cfg.num_class)
  elif self.kind=='tslib45_mixer':
   from touch_tslib45_vendor.models.TSMixer import ResBlock
   self.blocks=nn.ModuleList([ResBlock(cfg) for _ in range(c['layers'])]);self.head=nn.Sequential(nn.Flatten(),nn.Linear(45,c['width']),nn.GELU(),nn.Dropout(c['dropout']),nn.Linear(c['width'],cfg.num_class))
  elif self.kind=='tslib45_timesnet':
   from touch_timesnet45 import PeakTimesNet45
   self.encoder=PeakTimesNet45(dict(c,shared=c['width'],top_k=2,num_kernels=2));self.head=nn.Linear(c['width'],cfg.num_class)
  elif self.kind in ('tslib45_itr','tslib45_itr_query'):
   self.network=ITransformer(cfg)
   if self.kind=='tslib45_itr_query':
    w=c['width'];self.network.projection=nn.Identity();self.peak_identity=nn.Parameter(torch.randn(1,9,w)*.02)
    self.queries=nn.Parameter(torch.randn(1,cfg.num_class,w)*.02);self.cross=nn.MultiheadAttention(w,c['heads'],dropout=c['dropout'],batch_first=True)
    self.query_head=nn.Linear(w,1);self.absolute_head=nn.Linear(45,cfg.num_class)
  else:raise ValueError('Unknown official TSLib adapter')
 def forward(self,x,state=None):
  if x.ndim!=2 or x.shape[1]!=45:raise ValueError('Exactly B x45 current-frame actual dBm required')
  z=x.reshape(-1,9,5).transpose(1,2)
  if self.kind=='tslib45_itr_query':
   tokens=self.network.enc_embedding(z,None)+self.peak_identity;tokens,_=self.network.encoder(tokens,attn_mask=None)
   q,_=self.cross(self.queries.expand(len(x),-1,-1),tokens,tokens,need_weights=False)
   y=self.query_head(q).squeeze(-1)+self.absolute_head(x)
  elif self.kind=='tslib45_mixer':
   for block in self.blocks:z=block(z)
   y=self.head(z.transpose(1,2).reshape(-1,45))
  elif self.kind=='tslib45_timesnet':y=self.head(self.encoder(x))
  else:
   if self.kind=='tslib45_patch':z=x.unsqueeze(-1)
   y=self.network(z,z.new_ones((len(z),5)) if self.kind=='tslib45_timemixer' else None,None,None)
  return y if self.task=='position' else (y,None)
