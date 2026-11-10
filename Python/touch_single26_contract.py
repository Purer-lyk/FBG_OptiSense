"""Explicit single-only position contract; independent force is never contact-clamped."""
import numpy as np
MODE='single_label26_v1'
def class_from_multihot(y):
 y=np.asarray(y)
 if y.shape!=(25,) or not np.isin(y,[0,1]).all():raise ValueError('Expected exact25 binary targets')
 ids=np.flatnonzero(y)
 if len(ids)>1:raise ValueError('Multi-contact row excluded from single26 training and validation')
 return int(ids[0]+1) if len(ids) else 0
def decode_single26(p):
 p=np.asarray(p,float)
 if p.shape!=(26,) or not np.isfinite(p).all() or (p<0).any() or not np.isclose(p.sum(),1,atol=1e-5):raise ValueError('26 normalized softmax probabilities required')
 label=int(p.argmax());labels=[label] if label else [];hot=np.zeros(25,int)
 if label:hot[label-1]=1
 return dict(position_mode=MODE,position=label,position_labels=labels,position_multihot=hot.tolist(),probabilities=p[1:].tolist(),class_probabilities=p.tolist(),empty_probability=float(p[0]),confidence=float(p[label]),decision_status='pressed' if label else 'no_press')
