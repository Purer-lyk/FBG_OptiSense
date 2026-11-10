"""Clean measurement-axis adaptations, not original temporal forecasting benchmarks.
Reference: thuml/Time-Series-Library models/{PatchTST,iTransformer,TSMixer}.py
Pinned reference source and MIT LICENSE retained in outputs/time_series45_20261015_172028.
45 values are nine real peaks x five nonuniform wavelength samples in ONE scan.
No cross-frame history, global FFT, interpolation or per-frame RevIN.
"""
import math
import torch
from torch import nn
KINDS=('spectral_patchtst','spectral_itransformer','spectral_tsmixer','spectral_mlp')
class AxisMixer(nn.Module):
 def __init__(self,w,drop):
  super().__init__();self.measurement=nn.Sequential(nn.Linear(5,w),nn.GELU(),nn.Linear(w,5),nn.Dropout(drop));self.sensor=nn.Sequential(nn.Linear(9,w),nn.GELU(),nn.Linear(w,9),nn.Dropout(drop))
 def forward(self,x):
  x=x+self.measurement(x);return x+self.sensor(x.transpose(1,2)).transpose(1,2)
class Spectral45Encoder(nn.Module):
 def __init__(self,c):
  super().__init__();self.kind=c['architecture'];w=c['width'];d=c['dropout'];layers=c['layers'];shared=c['shared']
  if self.kind not in KINDS:raise ValueError('Unknown spectral encoder')
  if self.kind=='spectral_patchtst':
   # Nonoverlapping length5/stride5 patches aligned with actual peak boundaries.
   self.patch=nn.Conv1d(1,w,kernel_size=5,stride=5)
   t=torch.arange(9).float()[:,None];k=torch.arange(0,w,2).float();pe=torch.zeros(9,w);pe[:,0::2]=torch.sin(t*torch.exp(-math.log(10000)*k/w));pe[:,1::2]=torch.cos(t*torch.exp(-math.log(10000)*k/w));self.register_buffer('patch_position',pe[None])
   block=nn.TransformerEncoderLayer(w,c['heads'],c['ffn'],d,activation='gelu',batch_first=True,norm_first=False)
   self.blocks=nn.TransformerEncoder(block,layers,enable_nested_tensor=False)
   self.readout=nn.Sequential(nn.Flatten(),nn.Linear(9*w,shared),nn.GELU(),nn.Dropout(d))
  elif self.kind=='spectral_itransformer':
   # Inverted variables=9 physical sensors, five morphology measurements per variable.
   self.variable=nn.Linear(5,w);self.sensor_identity=nn.Parameter(torch.randn(1,9,w)*.02)
   block=nn.TransformerEncoderLayer(w,c['heads'],c['ffn'],d,activation='gelu',batch_first=True,norm_first=True)
   self.blocks=nn.TransformerEncoder(block,layers,norm=nn.LayerNorm(w),enable_nested_tensor=False)
   self.readout=nn.Sequential(nn.Flatten(),nn.Linear(9*w,shared),nn.GELU(),nn.Dropout(d))
  elif self.kind=='spectral_tsmixer':
   self.blocks=nn.Sequential(*[AxisMixer(w,d) for _ in range(layers)]);self.readout=nn.Sequential(nn.Flatten(),nn.Linear(45,shared),nn.GELU(),nn.Dropout(d))
  else:self.mlp=nn.Sequential(nn.Linear(45,w),nn.GELU(),nn.Dropout(d),nn.Linear(w,w),nn.GELU(),nn.Dropout(d),nn.Linear(w,shared),nn.GELU())
 def forward(self,x):
  if x.ndim!=2 or x.shape[1]!=45:raise ValueError('Exactly B x45 current-frame real dBm required')
  if self.kind=='spectral_patchtst':return self.readout(self.blocks(self.patch(x[:,None]).transpose(1,2)+self.patch_position))
  if self.kind=='spectral_itransformer':return self.readout(self.blocks(self.variable(x.reshape(-1,9,5))+self.sensor_identity))
  if self.kind=='spectral_tsmixer':return self.readout(self.blocks(x.reshape(-1,9,5)))
  return self.mlp(x)
