"""Direct local measured45 estimators. Derived features are not extra measurements."""
import numpy as np

def peak_features(x,wavelengths):
 x=np.asarray(x,dtype=np.float64);w=np.asarray(wavelengths,dtype=np.float64).reshape(9,5)
 if x.ndim!=2 or x.shape[1]!=45 or not np.isfinite(x).all():raise ValueError('Exactly finite Nx45 dBm required')
 spacing=np.diff(w,axis=1)
 if not np.isfinite(w).all() or (spacing<=0).any():raise ValueError('Ordered measured wavelengths required')
 p=x.reshape(-1,9,5);mean=p.mean(2,keepdims=True);shape=p-mean;diff=np.diff(p,axis=2)
 contrast=np.stack((p[:,:,2]-(p[:,:,0]+p[:,:,4])/2,p[:,:,4]-p[:,:,0]),axis=2)
 power=np.power(10.,np.clip(p,-300,100)/10.);ratio=power/np.maximum(power.sum(2,keepdims=True),1e-30)
 return np.concatenate((x,mean.reshape(-1,9),shape.reshape(-1,45),diff.reshape(-1,36),(diff/spacing).reshape(-1,36),contrast.reshape(-1,18),p.var(2),ratio.reshape(-1,45)),axis=1)

import torch
from torch import nn

class LocalTask(nn.Module):
 """Physical-N direct force / 25 independent contact probabilities, stateless."""
 def __init__(self,estimator,task):
  super().__init__();self.estimator=estimator;self.task=task
 def forward(self,x,state=None):
  pred=self.estimator.predict(x.detach().cpu().numpy().astype(np.float64))
  if self.task=='position':
   p=np.clip(pred,1e-6,1-1e-6);return torch.as_tensor(np.log(p/(1-p)),dtype=x.dtype,device=x.device)
  return torch.as_tensor(pred,dtype=x.dtype,device=x.device),None

class DirectLocal:
 """Train-fold-only optical metric and direct multioutput readout, no teacher."""
 def __init__(self,config,wavelengths):self.config=dict(config);self.wavelengths=np.array(wavelengths,dtype=np.float64)
 def features(self,x):
  x=np.asarray(x,np.float64)
  if x.ndim!=2 or x.shape[1]!=45 or not np.isfinite(x).all():raise ValueError('Exactly finite Nx45 engineering dBm')
  return peak_features(x,self.wavelengths) if self.config['features']=='peak' else x
 def transform(self,x):return ((self.features(x)-self.center)/self.scale)@self.projection
 def fit(self,x,y,valid,indices):
  from sklearn.decomposition import PCA
  f=self.features(x);self.center=f.mean(0);self.scale=np.maximum(f.std(0),1e-6);z=(f-self.center)/self.scale
  self.fit_indices=np.array(indices,dtype=np.int64);self.target=np.asarray(y,np.float64).copy();self.valid=np.asarray(valid,bool).copy()
  if len(f)!=len(self.fit_indices) or self.target.shape!=self.valid.shape or not np.isfinite(self.target[self.valid]).all():raise ValueError('Fit IDs/targets/masks differ')
  if self.config['features']=='peak':
   p=PCA(n_components=min(24,len(f)-1,z.shape[1]),svd_solver='full').fit(z);ev=p.explained_variance_;self.projection=p.components_.T/np.sqrt(.7*ev+.3*ev.mean()+1e-8)
  else:self.projection=np.eye(f.shape[1])
  self.train=z@self.projection
  if self.config['kind']=='kernel':
   from scipy.linalg import solve
   k=np.exp(-self.config['gamma']*self.distances(self.train));self.dual=np.zeros_like(self.target)
   maskgroups={}
   for axis in range(self.target.shape[1]):maskgroups.setdefault(self.valid[:,axis].tobytes(),[]).append(axis)
   self.kernel_system_count=len(maskgroups)
   for axes in maskgroups.values():
    ids=np.flatnonzero(self.valid[:,axes[0]]);ka=k[np.ix_(ids,ids)]+self.config['alpha']*np.eye(len(ids));self.dual[np.ix_(ids,axes)]=solve(ka,self.target[np.ix_(ids,axes)],assume_a='pos')
  return self
 def distances(self,z):
  # Explicit float64 differences avoid batch-dependent GEMM cancellation/ties.
  return np.array([np.einsum('ij,ij->i',self.train-row,self.train-row) for row in z])
 def predict(self,x):
  z=self.transform(x);d=self.distances(z);c=self.config
  if c['kind']=='kernel':return np.exp(-c['gamma']*d)@self.dual
  out=[]
  for row,dist in zip(z,d):
   ids=np.argsort(dist,kind='stable')[:min(c['k'],len(dist))];w=np.exp(-dist[ids]/max(float(dist[ids[-1]]),1e-12));ans=[]
   for a in range(self.target.shape[1]):
    use=self.valid[ids,a];ix=ids[use];weights=w[use]
    if not len(ix):raise ValueError('No valid local force supervision')
    if c['kind']=='local':
     design=np.column_stack((np.ones(len(ix)),self.train[ix]-row));penalty=np.eye(design.shape[1])*c['ridge'];penalty[0,0]=1e-8
     coef=np.linalg.solve(design.T@(weights[:,None]*design)+penalty,design.T@(weights*self.target[ix,a]));ans.append(coef[0])
    else:ans.append(np.sum(weights*self.target[ix,a])/weights.sum())
   out.append(ans)
  return np.asarray(out)

