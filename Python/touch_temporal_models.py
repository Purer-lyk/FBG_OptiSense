"""Complex measured-axis encoders; independent tasks. Torch-first workers only.

Nine gratings x five real dBm measurements are never interpolated. Optical TCN
runs on measurement order within ONE frame, not on historical frames.
"""
import numpy as np
import torch
from torch import nn
import touch_separate_models as separate

SCHEMA = 'fbg_touch_temporal26_v1'
ARCHITECTURES = ('tcn', 'sensor_token', 'sparse_cnn', 'inception_1d')
DEFAULT_CONFIG = dict(separate.DEFAULT_CONFIG, width=96, layers=3, shared=192,
    ffn=288, heads=4, dropout=.1, hidden=96, chunk_length=16,
    max_epochs=160, patience=25, force_history=True, max_gap_s=30., seed=20261012)

def _config(config):
    c = dict(DEFAULT_CONFIG, **config)
    if c['architecture'] not in ARCHITECTURES: raise ValueError('Unknown temporal architecture')
    if c['width'] < 96 or c['layers'] < 3: raise ValueError('Complex encoder requires width >=96 and layers >=3')
    if c['width'] % c['heads']: raise ValueError('Token width must divide heads')
    if c['chunk_length'] < 1 or c['hidden'] < 1: raise ValueError('Positive chunk/hidden dimensions required')
    return c

class AttentionPool(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.score = nn.Sequential(nn.Linear(width,width//2),nn.Tanh(),nn.Linear(width//2,1))
    def forward(self, tokens, mask=None):
        scores = self.score(tokens).squeeze(-1)
        if mask is not None: scores = scores.masked_fill(~mask.bool(),float('-inf'))
        return (tokens * scores.softmax(-1).unsqueeze(-1)).sum(1)

class ChannelAttention(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(width,max(8,width//4)),nn.GELU(),nn.Linear(max(8,width//4),width),nn.Sigmoid())
    def forward(self, x, mask=None):
        if mask is None: pooled = x.flatten(2).mean(-1)
        else: pooled = (x*mask).flatten(2).sum(-1)/mask.sum().clamp_min(1)
        return x*self.net(pooled).reshape(x.shape[0],x.shape[1],*([1]*(x.ndim-2)))

class GatedTCNBlock(nn.Module):
    def __init__(self, width, dilation, dropout):
        super().__init__()
        self.left = 2*dilation
        self.conv1 = nn.Conv1d(width,2*width,3,dilation=dilation)
        self.conv2 = nn.Conv1d(width,2*width,3,dilation=dilation)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h = x
        for conv in (self.conv1,self.conv2):
            a,g = conv(nn.functional.pad(h,(self.left,0))).chunk(2,1)
            h = self.dropout(torch.tanh(a)*torch.sigmoid(g))
        return x+h

class SparseBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.conv1 = nn.Conv2d(width,width,3,padding=1)
        self.conv2 = nn.Conv2d(width,width,3,padding=1)
        self.se = ChannelAttention(width)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x, mask):
        h = nn.functional.gelu(self.conv1(x))*mask
        h = self.dropout(nn.functional.gelu(self.conv2(h)))*mask
        return (x+self.se(h,mask))*mask

class InceptionBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.branches = nn.ModuleList(nn.Conv1d(width,width,k,padding=k//2) for k in (1,3,5,7))
        self.mix = nn.Conv1d(4*width,width,1)
        self.se = ChannelAttention(width)
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        h = nn.functional.gelu(self.mix(torch.cat([b(x) for b in self.branches],1)))
        return x+self.dropout(self.se(h))

class OpticalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = c = _config(config)
        self.kind = c['architecture']; w = c['width']
        self.pool = AttentionPool(w)
        self.output = nn.Sequential(nn.LayerNorm(w),nn.Linear(w,c['shared']),nn.GELU(),nn.Dropout(c['dropout']))
        if self.kind == 'sensor_token':
            self.embedding = nn.Linear(5,w)
            self.measured_grating_embedding = nn.Parameter(torch.randn(1,9,w)*.02)
            layer = nn.TransformerEncoderLayer(w,c['heads'],c['ffn'],c['dropout'],activation='gelu',batch_first=True,norm_first=True)
            self.blocks = nn.TransformerEncoder(layer,c['layers'],norm=nn.LayerNorm(w),enable_nested_tensor=False)
        elif self.kind == 'tcn':
            self.embedding = nn.Conv1d(1,w,1)
            dilations = [2**i for i in range(max(4,c['layers']))]
            self.receptive_field = 1+4*sum(dilations)
            self.blocks = nn.Sequential(*[GatedTCNBlock(w,d,c['dropout']) for d in dilations])
        elif self.kind == 'inception_1d':
            self.embedding = nn.Conv1d(1,w,1)
            self.blocks = nn.Sequential(*[InceptionBlock(w,c['dropout']) for _ in range(c['layers'])])
        else:
            self.embedding = nn.Conv2d(5,w,1)
            self.blocks = nn.ModuleList(SparseBlock(w,c['dropout']) for _ in range(c['layers']))
            mask = torch.tensor([[0,1,0],[1,1,1],[1,1,1],[1,0,1]],dtype=torch.float32).reshape(1,1,4,3)
            self.register_buffer('spatial_mask',mask)
    def sparse_input(self, x):
        grid = x.new_zeros((x.shape[0],5,4,3))
        for row,sensors in enumerate(((0,4,0),(9,5,3),(8,6,2),(7,0,1))):
            for col,sensor in enumerate(sensors):
                if sensor: grid[:,:,row,col] = x[:,sensor-1,:]
        return grid,self.spatial_mask
    def spatial_states(self, x):
        grid,mask = self.sparse_input(x)
        h = self.embedding(grid)*mask
        states = [h]
        for block in self.blocks:
            h = block(h,mask); states.append(h)
        return states
    def forward(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (9,5): raise ValueError('Expected B x 9 x 5 measured values')
        if self.kind == 'sensor_token': tokens = self.blocks(self.embedding(x)+self.measured_grating_embedding); mask=None
        elif self.kind == 'sparse_cnn':
            tokens = self.spatial_states(x)[-1].flatten(2).transpose(1,2)
            mask = self.spatial_mask.flatten(2).squeeze(1).expand(x.shape[0],-1)
        else: tokens = self.blocks(self.embedding(x.reshape(-1,1,45))).transpose(1,2); mask=None
        return self.output(self.pool(tokens,mask))

class PositionModel(nn.Module):
    def __init__(self, config):
        super().__init__(); self.config=_config(config)
        self.encoder = OpticalEncoder(self.config)
        self.position_head = nn.Linear(self.config['shared'],26)
        self.contact_head = nn.Linear(self.config['shared'],1)
    def forward(self, x):
        z = self.encoder(x)
        return self.position_head(z),self.contact_head(z).squeeze(-1)

class ForceModel(nn.Module):
    def __init__(self, config):
        super().__init__(); self.config=_config(config)
        self.encoder = OpticalEncoder(self.config)
        w = self.config['shared']
        self.anchor = nn.Sequential(nn.Linear(w,w),nn.GELU(),nn.Dropout(self.config['dropout']),nn.Linear(w,3))
        if self.config['force_history']:
            self.gru = nn.GRUCell(w+4,self.config['hidden'])
            self.residual = nn.Linear(self.config['hidden'],3)
            self.residual_gate = nn.Linear(w+ self.config['hidden'],3)
    def forward(self, x, hidden=None, previous=None, available=None):
        z = self.encoder(x)
        anchor = self.anchor(z)
        if not self.config['force_history']: return anchor,None
        batch = len(x)
        if hidden is None: hidden = z.new_zeros(batch,self.config['hidden'])
        if previous is None: previous = z.new_zeros(batch,3)
        if available is None: available = z.new_zeros(batch,1)
        if hidden.shape != (batch,self.config['hidden']) or previous.shape != (batch,3) or available.shape != (batch,1):
            raise ValueError('Invalid force step state shapes')
        # Zero standardized history + mask=0 means unavailable, NOT physical force=0.
        hidden = self.gru(torch.cat([z,previous.detach()*available,available],-1),hidden)
        gate = torch.sigmoid(self.residual_gate(torch.cat([z,hidden],-1)))
        return anchor+gate*self.residual(hidden),hidden

def _build(cls, config):
    c = _config(config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(c['seed'])
        return cls(c)

def build_position(config): return _build(PositionModel,config)
def build_force(config): return _build(ForceModel,config)

def causal_rollout(model, x, reset_flags, state=None):
    """Free-running ordered T x 9 x 5 stream; returns predictions and state.

    No reference force/position argument exists. State feedback is detached;
    hidden remains connected within a training chunk. Call detach_state between
    TBPTT chunks. A True reset starts a new independent session/segment.
    """
    if x.ndim != 3 or tuple(x.shape[1:]) != (9,5): raise ValueError('Expected T x 9 x 5')
    resets = torch.as_tensor(reset_flags,dtype=torch.bool,device=x.device)
    if resets.shape != (len(x),): raise ValueError('Expected one reset flag per frame')
    if not model.config['force_history']:
        return torch.cat([model(x[i:i+1])[0] for i in range(len(x))],0) if len(x) else x.new_empty((0,3)),None
    current = state
    result = []
    for i in range(len(x)):
        if bool(resets[i]) or current is None:
            current = dict(hidden=x.new_zeros(1,model.config['hidden']),previous=x.new_zeros(1,3),available=x.new_zeros(1,1))
        y,h = model(x[i:i+1],current['hidden'],current['previous'],current['available'])
        result.append(y)
        current = dict(hidden=h,previous=y.detach(),available=x.new_ones(1,1))
    return torch.cat(result,0) if result else x.new_empty((0,3)),current

def detach_state(state):
    return None if state is None else {k:v.detach() for k,v in state.items()}

def save_bundle(path, position, force, stats, feedback_selector, calibration, report=None):
    """Save independent weights/configs; never serialize live recurrent state."""
    from pathlib import Path
    path = Path(path)
    if path.exists(): raise FileExistsError('Refusing checkpoint overwrite: '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    torch.save(dict(schema=SCHEMA,configuration=force.config,
        position_configuration=position.config,force_configuration=force.config,
        position_model_state=position.state_dict(),force_model_state=force.state_dict(),
        **stats,feedback_selector=int(feedback_selector),calibration=calibration,
        temporal_configuration=dict(force_history=force.config['force_history'],hidden=force.config['hidden'],chunk_length=force.config['chunk_length'],feedback='previous predicted standardized FxFyFz detached; never true force',cold_start='previous standardized zeros with available=0, not physical force zero'),
        input_unit='dBm',class_labels=list(range(26)),report=report or {}),path)

class TemporalModelBundle(separate.six.SixModelBundle):
    """Single-frame predict* is persistent; details_dbm is a fresh offline stream.

    Multi-frame details_dbm REQUIRES explicit reset_flags. Never infer continuity
    from labels or force. Caller resets live state at session/source/gap changes.
    """
    def __init__(self,path):
        from pathlib import Path
        torch.set_num_threads(2)
        self.path = Path(path).resolve()
        payload = torch.load(self.path,map_location='cpu',weights_only=False)
        if payload.get('schema') != SCHEMA: raise ValueError('Unsupported temporal schema')
        self.config=payload['configuration']
        self.position_config=payload['position_configuration']; self.force_config=payload['force_configuration']
        self.position_model=build_position(self.position_config); self.force_model=build_force(self.force_config)
        self.position_model.load_state_dict(payload['position_model_state']); self.force_model.load_state_dict(payload['force_model_state'])
        self.position_model.eval(); self.force_model.eval()
        self.stats={k:np.asarray(payload[k],np.float32) for k in ('feature_mean','feature_std','force_mean','force_std')}
        self.mean=self.stats['feature_mean']; self.std=self.stats['feature_std']
        self.force_mean=self.stats['force_mean']; self.force_std=self.stats['force_std']
        if self.mean.shape != (9,5) or self.std.shape != (9,5) or self.force_mean.shape != (3,) or self.force_std.shape != (3,):
            raise ValueError('Invalid normalization shapes')
        if not all(np.isfinite(v).all() for v in self.stats.values()) or (self.std<=0).any() or (self.force_std<=0).any():
            raise ValueError('Invalid normalization values')
        self.feedback_selector=int(payload['feedback_selector']); self.calibration=payload['calibration']; self.policy=self.calibration['policy']
        self.temporal_config=payload['temporal_configuration']; self.report=payload.get('report',{})
        self.kind='temporal_'+self.force_config['architecture']
        self.reset_state()
    def reset_state(self): self._state=None
    def _inputs(self,values):
        values=np.asarray(values,np.float32)
        if values.ndim==1: values=values[None]
        if values.ndim!=2 or values.shape[1]!=45 or not np.isfinite(values).all(): raise ValueError('Expected N x 45 finite dBm')
        return torch.from_numpy(((values.reshape(-1,9,5)-self.mean)/self.std).astype(np.float32))
    def _details(self,x,resets,state):
        with torch.inference_mode():
            # Framewise position execution matches streaming exactly too.
            rows=[self.position_model(x[i:i+1]) for i in range(len(x))]
            logits=torch.cat([v[0] for v in rows]); contact=torch.cat([v[1] for v in rows])
            force,state=causal_rollout(self.force_model,x,resets,state)
            probs=(logits/self.calibration['position_temperature']).softmax(-1).numpy()
            contacts=(contact/self.calibration['contact_temperature']).sigmoid().numpy()
            forces=force.numpy()*self.force_std+self.force_mean
        return (probs.argmax(-1),forces,probs,contacts),state
    def details_dbm(self,values,reset_flags=None):
        x=self._inputs(values)
        if not len(x): raise ValueError('Nonempty stream required')
        if reset_flags is None:
            if len(x)!=1: raise ValueError('Batch evaluation requires explicit continuity reset_flags')
            reset_flags=[True]
        return self._details(x,reset_flags,None)[0]
    def predict_dbm(self,values):
        x=self._inputs(values)
        if len(x)!=1: raise ValueError('Persistent prediction expects one frame; use evaluate_stream for batches')
        result,self._state=self._details(x,[False],self._state)
        return result[:3]
    def predict_payload(self,adc):
        from temporary_optical_units import adc_code_to_dbm
        from touch_decision import decide_contact
        x=self._inputs(adc_code_to_dbm(adc,self.feedback_selector,floor_nonpositive=True))
        if len(x)!=1: raise ValueError('Persistent prediction expects one frame')
        (_,forces,probs,contacts),self._state=self._details(x,[False],self._state)
        decision=decide_contact(probs[0],contacts[0],self.policy)
        # Contact never clamps, gates, zeros or rescales regression.
        return dict(decision,probabilities=probs[0].astype(float).tolist(),force=forces[0].astype(float).tolist())

def evaluate_stream(bundle,values,reset_flags):
    """Dedicated fresh free-running evaluation, never mutates bundle live state."""
    return bundle.details_dbm(values,reset_flags)

def train_trial(data, split, config, output, feedback_selector):
    """Independent task training; split-local chronological free-running force.

    Position is retrained reproducibly in each trial, including when the runner
    passes train_position=False. No cache savings are claimed by this module.
    Test index metadata is checked for overlap but test measurements never read.
    """
    import copy,json,time
    from pathlib import Path
    from temporal_sequence_audit import continuous_runs
    from touch_contact_guard import calibrate,decision_metrics
    six=separate.six
    c=_config(config)
    torch.set_num_threads(2)
    output=Path(output); path=output/(c['id']+'.pt')
    if path.exists(): raise FileExistsError('Refusing checkpoint overwrite: '+str(path))
    for key in ('session','source_line','timestamp_s'):
        if key not in data or len(data[key])!=len(data['dbm']): raise ValueError('Required sequence metadata missing: '+key)
    groups=[np.asarray(split.get(k,[]),np.int64) for k in ('train','validation','test')]
    joined=np.concatenate(groups)
    if len(joined)!=len(np.unique(joined)) or (joined<0).any() or (joined>=len(data['dbm'])).any(): raise ValueError('Invalid/overlapping split indices')
    if not len(groups[0]) or not len(groups[1]): raise ValueError('Nonempty train/validation required')
    if c['max_epochs']<1 or c['patience']<1: raise ValueError('Positive training budget required')
    gap=c.get('gap_s',c.get('max_gap_s',30.))
    runs={name:continuous_runs(data,idx,gap_s=gap) for name,idx in zip(('train','validation'),groups[:2])}
    ordered={name:np.array([i for run in rr for i in run],np.int64) for name,rr in runs.items()}
    def resets(name):
        return np.array([j==0 for run in runs[name] for j in range(len(run))],bool)
    def subset(name):
        idx=ordered[name]
        d=np.asarray(data['dbm'][idx],np.float32); y=np.asarray(data['labels'][idx],np.int64)
        f=np.asarray(data['forces'][idx],np.float32)
        if d.shape!=(len(idx),45) or not np.isfinite(d).all(): raise ValueError('Expected finite measured dBm')
        if y.shape!=(len(idx),) or (y<0).any() or (y>25).any(): raise ValueError('Expected labels 0..25')
        if f.shape!=(len(idx),3): raise ValueError('Expected N x 3 force')
        return d,y,f,separate.axis_mask(f,np.asarray(data['valid'])[idx])
    td,ty,tf,tv=subset('train'); vd,vy,vf,vv=subset('validation')
    if not tv.any(): raise ValueError('No valid training force references')
    mean=td.mean(0).reshape(9,5); std=td.std(0).reshape(9,5); std[std<1e-6]=1
    fm=np.array([tf[tv[:,a],a].mean() if tv[:,a].any() else 0. for a in range(3)],np.float32)
    fs=np.array([tf[tv[:,a],a].std() if tv[:,a].any() else 1. for a in range(3)],np.float32); fs[fs<1e-5]=1
    stats=dict(feature_mean=mean,feature_std=std,force_mean=fm,force_std=fs)
    bins=six.fit_force_bin_weights(tf,tv)
    def inputs(d): return torch.from_numpy(((d.reshape(-1,9,5)-mean)/std).astype(np.float32))
    tx,vx=inputs(td),inputs(vd); tr,vr=resets('train'),resets('validation')
    p=build_position(c); f=build_force(c)
    counts=np.bincount((ty>0).astype(int),minlength=2)
    def contact_weights(y):
        return torch.from_numpy((len(ty)/(np.count_nonzero(counts)*np.maximum(counts[(y>0).astype(int)],1))).astype(np.float32))
    tw,vw=contact_weights(ty),contact_weights(vy)
    opts={k:torch.optim.AdamW(m.parameters(),lr=c['lr'],weight_decay=c['weight_decay']) for k,m in [('position',p),('force',f)]}
    best={k:dict(key=None,state=None,epoch=0,row=None) for k in opts}
    history={k:[] for k in opts}; start=time.perf_counter()
    # Entire task phases have private RNG and optimizers; stopping or labels of
    # position cannot perturb force stochasticity, losses or validation keys.
    for task,model in [('position',p),('force',f)]:
        for epoch in range(1,c['max_epochs']+1):
            model.train(); total=0.; samples=0; state=None
            if task=='position':
                generator=torch.Generator().manual_seed(c['seed']+epoch*1009)
                sequence=torch.randperm(len(tx),generator=generator)
                batches=[sequence[i:i+c['batch_size']] for i in range(0,len(tx),c['batch_size'])]
            else: batches=[slice(i,min(i+c['chunk_length'],len(tx))) for i in range(0,len(tx),c['chunk_length'])]
            for batch_no,index in enumerate(batches):
                opts[task].zero_grad(set_to_none=True)
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(c['seed']+epoch*100000+batch_no*2+(task=='force'))
                    if task=='position':
                        logits,contact=model(tx[index])
                        loss=separate.position_loss(logits,contact,torch.from_numpy(ty)[index],tw[index],c)
                    else:
                        pred,state=causal_rollout(model,tx[index],tr[index],state=state)
                        loss=six.force_loss(pred,tf[index],tv[index],fm,fs,bins,c)
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opts[task].step()
                if task=='force': state=detach_state(state)
                size=len(tx[index]); total+=float(loss.detach())*size; samples+=size
            model.eval()
            with torch.inference_mode():
                if task=='position':
                    logits,contact=model(vx)
                    accuracy=float((logits.argmax(-1).numpy()==vy).mean())
                    pl=float(separate.position_loss(logits,contact,torch.from_numpy(vy),vw,dict(c,label_smoothing=0.)))
                    ce=float(nn.functional.cross_entropy(logits,torch.from_numpy(vy)))
                    key=separate.position_selection(accuracy,pl)
                    row=dict(epoch=epoch,validation_accuracy=accuracy,validation_cross_entropy=ce,validation_position_loss=pl)
                else:
                    pred,_=causal_rollout(model,vx,vr)
                    force=pred.numpy()*fs+fm
                    mae=six.axis_mae(vf,force,vv)
                    # No labels even in force checkpoint selection.
                    key=separate.force_selection(mae,None)
                    row=dict(epoch=epoch,validation_force_mae_n=mae)
            row.update(training_loss=total/samples,selection_key=list(key))
            history[task].append(row)
            if best[task]['key'] is None or key>best[task]['key']:
                best[task]=dict(key=key,state=copy.deepcopy(model.state_dict()),epoch=epoch,row=copy.deepcopy(row))
            if epoch==1 or epoch%10==0:
                print(json.dumps(dict(type='progress',trial=c['id'],task=task,**row),allow_nan=False),flush=True)
            if epoch-best[task]['epoch']>=c['patience']: break
        model.load_state_dict(best[task]['state']); model.eval()
    with torch.inference_mode(): vl,vc=p(vx)
    calibration=calibrate(vl.numpy(),vc.numpy(),vy)
    with torch.random.fork_rng(devices=[]):
        original_position,original_force=separate.build_models(dict(architecture=c['architecture']))
    original_parameters={k:sum(v.numel() for v in m.parameters()) for k,m in [('position',original_position),('force',original_force)]}
    current_parameters={k:sum(v.numel() for v in m.parameters()) for k,m in [('position',p),('force',f)]}
    del original_position,original_force
    report=dict(schema=SCHEMA,smoke_only=bool(data.get("smoke_only",False)),architecture=c['architecture'],configuration=c,config=c,
        position_configuration=p.config,force_configuration=f.config,input_unit='dBm',input_shape=[9,5],class_labels=list(range(26)),
        normalization_source='train only',selection_source='validation only',calibration_source='validation only',
        force_mask_granularity='per axis',force_conditioning='current optical + previous self-predicted standardized force and causal optical GRU; no labels or true-force history' if c['force_history'] else 'current optical only; same complex encoder and anchor; no temporal state',
        force_postprocessing='none: no clamp/contact gating/uncertain zeroing/test gain',
        position_best_epoch=best['position']['epoch'],force_best_epoch=best['force']['epoch'],
        best_epoch={k:v['epoch'] for k,v in best.items()},
        position_selection_key=list(best['position']['key']),force_selection_key=list(best['force']['key']),
        best_validation_accuracy=best['position']['row']['validation_accuracy'],best_validation_cross_entropy=best['position']['row']['validation_cross_entropy'],
        validation_force_mae_n=best['force']['row']['validation_force_mae_n'],
        epochs_completed={k:len(v) for k,v in history.items()},history=history,
        parameters=current_parameters,original_balanced_parameters=original_parameters,
        parameter_increments_over_original_balanced={k:current_parameters[k]-original_parameters[k] for k in current_parameters},
        optimizer='AdamW, independent per task',actual_position_training_runs=1,
        learning_rate_schedule='constant; no scheduler',
        train_count=len(td),validation_count=len(vd),test_count=len(groups[2]),train_seconds=time.perf_counter()-start,
        sequence_policy=dict(reset_inputs=['split','session','source_line','timestamp_s'],train_runs=runs['train'],validation_runs=runs['validation'],
            gap_s=gap,chunk_length=c['chunk_length'],teacher_forcing=False,feedback_detached=True,
            timestamp_fallback_frames=int(sum((~np.isfinite(np.asarray(data['timestamp_s'])[idx])).sum() for idx in ordered.values())),
            timestamp_fallback='source_line continuity only; unavailable timestamp not fabricated',cold_start='standardized previous zero with availability=0; NOT physical zero force'),
        force_bin_weights=bins.tolist(),calibration=calibration,metrics={},decision_metrics={},force_diagnostics={},
        deployment_ready=False,validation_warning='Split-local neighboring frames are not independent presses; recording time is not synchronized optical/force ground truth.',
        model_path=str(path.resolve()),position_reuse='deterministic matched retraining')
    output.mkdir(parents=True,exist_ok=True)
    # Identical framewise execution to saved bundle avoids batch-vs-stream numeric differences.
    for name,x,y,true,valid,rst in [('train',tx,ty,tf,tv,tr),('validation',vx,vy,vf,vv,vr)]:
        with torch.inference_mode():
            rows=[p(x[i:i+1]) for i in range(len(x))]
            logits=torch.cat([v[0] for v in rows]); contacts=torch.cat([v[1] for v in rows])
            pred,_=causal_rollout(f,x,rst)
            force=pred.numpy()*fs+fm
            probs=(logits/calibration['position_temperature']).softmax(-1).numpy()
            cp=(contacts/calibration['contact_temperature']).sigmoid().numpy()
        report['metrics'][name]=six.metrics(y,probs.argmax(-1),true,force,valid)
        report['decision_metrics'][name]=decision_metrics(y,probs,cp,calibration['policy'])
        report['force_diagnostics'][name]=six.force_diagnostics(y,true,force,valid)
        if name=='validation':
            np.savez_compressed(output/(c['id']+'_validation_predictions.npz'),indices=ordered[name],reset_flags=rst,
                labels=y,forces=force,true_forces=true,valid=valid,probabilities=probs,contact_probabilities=cp,logits=logits.numpy(),contact_logits=contacts.numpy())
    save_bundle(path,p,f,stats,feedback_selector,calibration,report)
    path.with_suffix('.report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return report
