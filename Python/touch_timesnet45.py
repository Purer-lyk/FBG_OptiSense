"""TSLib TimesNet spectral adapter (MIT, thuml commit 4e938a1767106324dd753b2a44832bf870a0252e).
Derived TimesBlock / Inception_Block_V1. Upstream originals and LICENSE retained at
outputs/four45_timesnet_20261016_174207/upstream. FFT is ONLY along each peak's five
point INDICES, not nonuniform wavelength frequency and not across-frame time.
Per-instance frequency selection removes upstream batch-dependent inference.
"""
import torch
from torch import nn
from torch.nn import functional as F

def small_spatial_conv(module,x):
 # At H=1, all kernel rows except the center multiply padding zeros. Elide only
 # coefficients that provably cannot overlap ANY real input; parameters retained.
 k=module.weight.shape[-1];r=k//2;rh=min(r,x.shape[-2]-1);rw=min(r,x.shape[-1]-1)
 w=module.weight[:,:,r-rh:r+rh+1,r-rw:r+rw+1]
 return F.conv2d(x,w,module.bias,padding=(rh,rw))

class Inception_Block_V1(nn.Module):
 def __init__(self,in_channels,out_channels,num_kernels=2):
  super().__init__();self.kernels=nn.ModuleList([nn.Conv2d(in_channels,out_channels,kernel_size=2*i+1,padding=i) for i in range(num_kernels)])
  for m in self.modules():
   if isinstance(m,nn.Conv2d):nn.init.kaiming_normal_(m.weight,mode='fan_out',nonlinearity='relu');nn.init.constant_(m.bias,0)
 def forward(self,x):return torch.stack([small_spatial_conv(m,x) for m in self.kernels],dim=-1).mean(-1)

class PeakTimesBlock(nn.Module):
 def __init__(self,width,top_k,kernels):
  super().__init__()
  if not 1<=top_k<=2:raise ValueError('Length5 allows exactly two non-DC rFFT bins')
  self.k=top_k;self.conv=nn.Sequential(Inception_Block_V1(width,width*2,kernels),nn.GELU(),Inception_Block_V1(width*2,width,kernels))
 def forward(self,x):
  b,t,n=x.shape
  if t!=5:raise ValueError('TimesNet adapter requires five actual points per peak')
  amplitudes=torch.fft.rfft(x,dim=1).abs().mean(-1)[:,1:]
  values,indices=amplitudes.topk(self.k,dim=1)
  weights=F.softmax(values,dim=1);out=x.new_zeros(x.shape)
  # Both legal periods are evaluated; instance-specific weights are batch invariant.
  for frequency,period in [(1,5),(2,2)]:
   length=((t+period-1)//period)*period;z=F.pad(x,(0,0,0,length-t))
   z=z.reshape(b,length//period,period,n).permute(0,3,1,2).contiguous()
   z=self.conv(z).permute(0,2,3,1).reshape(b,length,n)[:,:t]
   weight=(weights*(indices==frequency-1)).sum(1);out=out+z*weight[:,None,None]
  return out+x

class PeakTimesNet45(nn.Module):
 def __init__(self,c):
  super().__init__();w=c['width'];self.embed=nn.Linear(1,w);self.point_identity=nn.Parameter(torch.randn(1,5,w)*.02)
  self.blocks=nn.ModuleList([PeakTimesBlock(w,c.get('top_k',1 if c['layers']==1 else 2),c.get('num_kernels',2 if c['layers']==1 else 3)) for _ in range(c['layers'])]);self.norms=nn.ModuleList([nn.LayerNorm(w) for _ in self.blocks])
  self.peak_identity=nn.Parameter(torch.randn(1,9,w)*.02);self.local_readout=nn.Linear(5*w,w)
  self.readout=nn.Sequential(nn.Flatten(),nn.Linear(9*w+45,c['shared']),nn.GELU(),nn.Dropout(c['dropout']))
 def forward(self,x):
  if x.ndim!=2 or x.shape[1]!=45:raise ValueError('Exactly B x45 current-frame actual dBm required')
  z=self.embed(x.reshape(-1,5,1))+self.point_identity
  for block,norm in zip(self.blocks,self.norms):z=norm(block(z))
  z=self.local_readout(z.flatten(1)).reshape(-1,9,self.peak_identity.shape[-1])+self.peak_identity
  return self.readout(torch.cat([z.flatten(1),x],1))
