"""Local task wrapper of unmodified Apache-2.0 official RTDL package 0.0.2.
Upstream e3ed46cac38568785289d8fa16b8cfa585bde27e, package/rtdl_revisiting_models.py.
Only width/depth/dropout and task output dimension changed; all45 actual dBm
continuous features retain identities. No frame normalization or neighbor memory.
"""
import torch
from torch import nn
from touch_rtdl45_vendor import FTTransformer,ResNet
KINDS=('mature45_ftt','mature45_resnet')
class Mature45Task(nn.Module):
 def __init__(self,c,task):
  super().__init__();self.task=task;self.config=c;kind=c['architecture'];w=c['width'];out=25 if task=='position' else 3
  if kind=='mature45_ftt':
   self.network=FTTransformer(n_cont_features=45,cat_cardinalities=[],d_out=out,n_blocks=c['layers'],d_block=w,attention_n_heads=c.get('heads',4),attention_dropout=c['dropout'],ffn_d_hidden=None,ffn_d_hidden_multiplier=4/3,ffn_dropout=c['dropout'],residual_dropout=0.)
  elif kind=='mature45_resnet':
   self.network=ResNet(d_in=45,d_out=out,n_blocks=c['layers'],d_block=w,d_hidden=None,d_hidden_multiplier=2.,dropout1=c['dropout'],dropout2=0.)
  else:raise ValueError('Unknown mature45 architecture')
 def forward(self,x,state=None):
  if x.ndim!=2 or x.shape[1]!=45:raise ValueError('Exactly B x 45 actual dBm required')
  y=self.network(x,None) if self.config['architecture']=='mature45_ftt' else self.network(x)
  return y if self.task=='position' else (y,None)
