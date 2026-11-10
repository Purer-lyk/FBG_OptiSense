"""Three measured-input backbones with a common physical-force objective.
Import only in Torch-first workers, never in the Qt parent process.
"""
import torch
import numpy as np
from torch import nn

SCHEMA = 'fbg_touch_force_compare26_v1'
DEFAULT_CONFIG = dict(architecture='sensor_token', seed=7, id='trial', width=64,
    heads=4, layers=2, ffn=192, shared=128, tcn_channels=32, kernel_size=3,
    dilations=[1,2,4,8], dropout=.05, lr=.0007, weight_decay=.0003,
    force_weight=.5, contact_weight=.2, label_smoothing=.02, batch_size=64,
    max_epochs=180, patience=25, mse_weight=.15, axis_scales=[.5,.5,1.])


def _config(config):
    return dict(DEFAULT_CONFIG, **config)


class CausalBlock(nn.Module):
    def __init__(self, incoming, channels, kernel, dilation, dropout):
        super().__init__()
        self.left = (kernel-1)*dilation
        self.conv1 = nn.Conv1d(incoming, channels, kernel, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel, dilation=dilation)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)
        self.skip = nn.Conv1d(incoming, channels, 1) if incoming != channels else nn.Identity()

    def forward(self, x):
        h = self.dropout(self.activation(self.conv1(nn.functional.pad(x,(self.left,0)))))
        h = self.dropout(self.activation(self.conv2(nn.functional.pad(h,(self.left,0)))))
        return h + self.skip(x)


class ForceComparisonModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = c = _config(config)
        self.kind = c['architecture']
        if self.kind == 'tcn':
            blocks=[]
            incoming=1
            for dilation in c['dilations']:
                blocks.append(CausalBlock(incoming,c['tcn_channels'],c['kernel_size'],dilation,c['dropout']))
                incoming=c['tcn_channels']
            self.backbone=nn.Sequential(*blocks)
            backbone_width=incoming
        elif self.kind == 'sensor_token':
            self.embedding=nn.Linear(5,c['width'])
            self.sensor_identity=nn.Parameter(torch.randn(1,9,c['width'])*.02)
            layer=nn.TransformerEncoderLayer(c['width'],c['heads'],c['ffn'],c['dropout'],
                activation='gelu',batch_first=True,norm_first=True)
            self.backbone=nn.TransformerEncoder(layer,c['layers'],norm=nn.LayerNorm(c['width']),enable_nested_tensor=False)
            backbone_width=9*c['width']
        elif self.kind == 'sparse_cnn':
            self.backbone=nn.Sequential(nn.Conv2d(6,32,3,padding=1),nn.GELU(),
                nn.Conv2d(32,64,3,padding=1),nn.GELU(),nn.Flatten(1))
            backbone_width=64*4*3
        else:
            raise ValueError('Unknown architecture: '+str(self.kind))
        self.shared=nn.Sequential(nn.Linear(backbone_width,c['shared']),nn.GELU(),nn.Dropout(c['dropout']))
        self.position_head=nn.Linear(c['shared'],26)
        self.force_head=nn.Sequential(nn.Linear(c['shared']+45,128),nn.GELU(),
            nn.Dropout(c['dropout']),nn.Linear(128,64),nn.GELU(),nn.Linear(64,3))
        self.force_residual=nn.Linear(45,3)
        nn.init.zeros_(self.force_residual.weight)
        nn.init.zeros_(self.force_residual.bias)
        self.contact_head=nn.Linear(c['shared'],1)

    def sparse_input(self, x):
        layout=((0,4,0),(9,5,3),(8,6,2),(7,0,1))
        grid=x.new_zeros((x.shape[0],6,4,3))
        for row, sensors in enumerate(layout):
            for col, sensor in enumerate(sensors):
                if sensor:
                    grid[:,:5,row,col]=x[:,sensor-1,:]
                    grid[:,5,row,col]=1
        return grid

    def last_feature(self, x):
        # These are 45 measured points in ONE frame, never 45 temporal frames.
        return self.backbone(x.reshape(-1,1,45))[:,:,-1]

    def features(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (9,5):
            raise ValueError('Expected B x 9 x 5 normalized measured points')
        if self.kind == 'tcn': h=self.last_feature(x)
        elif self.kind == 'sensor_token': h=self.backbone(self.embedding(x)+self.sensor_identity).flatten(1)
        else: h=self.backbone(self.sparse_input(x))
        return self.shared(h)

    def forward(self, x):
        features=self.features(x)
        flat=x.flatten(1)
        force=self.force_head(torch.cat((features,flat),1))+self.force_residual(flat)
        return self.position_head(features),force,self.contact_head(features).squeeze(-1)


def receptive_field(config):
    """Two left-causal convolutions per dilation; default 1+4*(1+2+4+8)."""
    c = _config(config)
    return 1 + 2*(c['kernel_size']-1)*sum(c['dilations'])


def build_model(config):
    c = _config(config)
    if c['architecture'] == 'tcn' and receptive_field(c) < 45:
        raise ValueError('TCN receptive field must cover all 45 measured points')
    torch.manual_seed(c['seed'])
    return ForceComparisonModel(c)


FORCE_BIN_EDGES = [0.,1.,3.,5.,10.,float('inf')]


def fit_force_bin_weights(train_forces, valid):
    """Fit per-axis inverse-sqrt frequency weights on supplied TRAIN rows only.
    Cap at 3 while maintaining empirical training mean weight of one.
    """
    force=np.asarray(train_forces,float)
    mask=np.asarray(valid,bool)
    if mask.ndim == 1: mask=np.broadcast_to(mask[:,None],force.shape)
    mask=mask & np.isfinite(force)
    result=np.ones((3,5),np.float32)
    for axis in range(3):
        values=force[mask[:,axis],axis]
        if not len(values): continue
        bins=np.searchsorted(FORCE_BIN_EDGES[1:-1],np.abs(values),side='right')
        counts=np.bincount(bins,minlength=5)
        raw=1/np.sqrt(np.maximum(counts,1))
        low,high=0.,float(np.sqrt(len(values))*4)
        for _ in range(64):
            mid=(low+high)/2
            if np.dot(counts,np.minimum(3.,mid*raw))/len(values)<1: low=mid
            else: high=mid
        result[axis]=np.minimum(3.,high*raw)
    return result


def force_loss(pred_norm, target_N, valid, force_mean, force_std, bin_weights, config):
    """Physical N error, fixed axis scales; mask BEFORE any target arithmetic."""
    c=_config(config)
    def tensor(value): return torch.as_tensor(value,dtype=pred_norm.dtype,device=pred_norm.device)
    target=tensor(target_N)
    mask=torch.as_tensor(valid,dtype=torch.bool,device=pred_norm.device)
    if mask.ndim == 1: mask=mask[:,None].expand_as(target)
    mask=mask & torch.isfinite(target)
    if not mask.any(): return pred_norm.sum()*0
    pred=pred_norm*tensor(force_std)+tensor(force_mean)
    scales=tensor(c['axis_scales']).expand_as(pred)
    error=(pred[mask]-target[mask])/scales[mask]
    axis=torch.arange(3,device=pred.device).expand_as(pred)[mask]
    bins=torch.bucketize(target[mask].abs().contiguous(),tensor(FORCE_BIN_EDGES[1:-1]),right=True)
    weights=tensor(bin_weights)[axis,bins]
    smooth=nn.functional.smooth_l1_loss(error,torch.zeros_like(error),reduction='none')
    return ((smooth+c['mse_weight']*error.square())*weights).mean()


def _gain(true, predicted):
    slopes=[]; intercepts=[]
    for axis in range(3):
        if len(true)<2 or np.var(true[:,axis])<1e-12:
            slopes.append(None); intercepts.append(None)
        else:
            x=true[:,axis]; y=predicted[:,axis]
            slope=float(np.dot(x-x.mean(),y-y.mean())/np.sum((x-x.mean())**2))
            slopes.append(slope); intercepts.append(float(y.mean()-slope*x.mean()))
    return {'slopes':slopes,'intercepts':intercepts}


def force_diagnostics(labels, true_forces, predicted_forces, valid):
    """Signed regression gain and within-position centered gain, never corrections."""
    y=np.asarray(true_forces,float); p=np.asarray(predicted_forces,float)
    mask=np.asarray(valid,bool)
    if mask.ndim == 2: mask=mask.all(1)
    mask=mask & np.isfinite(y).all(1) & np.isfinite(p).all(1)
    labels=np.asarray(labels)[mask]; y=y[mask]; p=p[mask]
    yc=y.copy(); pc=p.copy()
    for label in np.unique(labels):
        group=labels==label
        yc[group]-=y[group].mean(0); pc[group]-=p[group].mean(0)
    bins=[]
    for low,high in zip(FORCE_BIN_EDGES[:-1],FORCE_BIN_EDGES[1:]):
        group=(np.abs(y[:,2])>=low)&(np.abs(y[:,2])<high)
        count=int(group.sum())
        error=p[group]-y[group]
        amplitude=float(np.abs(y[group,2]).mean()) if count else 0.
        bins.append({'lower_n':low,'upper_n':high if np.isfinite(high) else None,
            'count':count,'mae_n':float(np.abs(error[:,2]).mean()) if count else None,
            'bias_n':float(error[:,2].mean()) if count else None,
            'amplitude_ratio':float(np.abs(p[group,2]).mean()/amplitude) if amplitude>1e-12 else None,
            'axis_mae_n':np.abs(error).mean(0).tolist() if count else None,
            'axis_bias_n':error.mean(0).tolist() if count else None})
    return {'valid_count':len(y),'force_gain':_gain(y,p),
        'within_label_gain':_gain(yc,pc),'abs_fz_bins':bins}


def validation_selection(accuracy, axis_mae, fz_slope, config):
    """Lexicographic feasible-first selection; no test statistics accepted."""
    c=_config(config)
    force_term=float(np.mean(np.asarray(axis_mae)/c['axis_scales'])) if axis_mae is not None else 0.
    objective=force_term+.2*(1-accuracy)+(.1*abs(fz_slope-1) if fz_slope is not None else 0.)
    feasible=accuracy>=.80
    key=(int(feasible),-objective,accuracy) if feasible else (0,accuracy,-objective)
    return key,objective

class ForceComparisonBundle:
    def __init__(self, path):
        from pathlib import Path
        torch.set_num_threads(2)
        self.path=Path(path).resolve()
        payload=torch.load(self.path,map_location='cpu',weights_only=False)
        if payload.get('schema') != SCHEMA:
            raise ValueError('Unsupported force comparison schema')
        self.config=payload['configuration']
        self.kind=self.config['architecture']
        self.model=build_model(self.config)
        self.model.load_state_dict(payload['model_state'])
        self.model.eval()
        self.mean=np.asarray(payload['feature_mean'],np.float32)
        self.std=np.asarray(payload['feature_std'],np.float32)
        self.force_mean=np.asarray(payload['force_mean'],np.float32)
        self.force_std=np.asarray(payload['force_std'],np.float32)
        self.stats={'feature_mean':self.mean,'feature_std':self.std,
            'force_mean':self.force_mean,'force_std':self.force_std}
        self.feedback_selector=int(payload['feedback_selector'])
        self.calibration=payload['calibration']
        self.policy=self.calibration['policy']
        self.report=payload.get('report',{})

    def details_dbm(self, values):
        values=np.asarray(values,np.float32)
        if values.ndim == 1: values=values[None]
        if values.ndim != 2 or values.shape[1] != 45 or not np.isfinite(values).all():
            raise ValueError('Expected N x 45 finite dBm values')
        x=((values.reshape(-1,9,5)-self.mean)/self.std).astype(np.float32)
        with torch.inference_mode():
            logits,force,contact=self.model(torch.from_numpy(x))
            probabilities=(logits/self.calibration['position_temperature']).softmax(-1).numpy()
            contacts=(contact/self.calibration['contact_temperature']).sigmoid().numpy()
            forces=force.numpy()*self.force_std+self.force_mean
        return probabilities.argmax(-1),forces,probabilities,contacts

    def predict_dbm(self, values):
        return self.details_dbm(values)[:3]

    def predict(self, adc):
        from temporary_optical_units import adc_code_to_dbm
        return self.predict_dbm(adc_code_to_dbm(adc,self.feedback_selector,floor_nonpositive=True))

    def predict_payload(self, adc):
        from temporary_optical_units import adc_code_to_dbm
        from touch_decision import decide_contact
        _,forces,probabilities,contacts=self.details_dbm(
            adc_code_to_dbm(adc,self.feedback_selector,floor_nonpositive=True))
        if len(forces) != 1: raise ValueError('predict_payload expects one frame')
        decision=decide_contact(probabilities[0],contacts[0],self.policy)
        return dict(decision,probabilities=probabilities[0].astype(float).tolist(),
            force=forces[0].astype(float).tolist())


def train_trial(data, split, config, output, feedback_selector):
    """Train/calibrate on frozen train/validation only; do not open test data."""
    import copy,json,time
    from pathlib import Path
    from torch.utils.data import TensorDataset,DataLoader
    from manual_touch_sensor_token import metrics
    from touch_contact_guard import calibrate,decision_metrics
    c=_config(config)
    torch.manual_seed(c['seed']); np.random.seed(c['seed']); torch.set_num_threads(2)
    train=np.asarray(split['train'],dtype=np.int64)
    val=np.asarray(split['validation'],dtype=np.int64)
    if not len(train) or not len(val): raise ValueError('Nonempty train and validation required')
    groups=[train,val,np.asarray(split.get('test',[]),dtype=np.int64)]
    joined=np.concatenate(groups)
    if len(np.unique(joined)) != len(joined): raise ValueError('Split indices must be disjoint and unique')
    if np.any(joined<0) or np.any(joined>=len(data['dbm'])): raise ValueError('Split index out of range')
    if c['max_epochs']<1: raise ValueError('max_epochs must be positive')
    # Slice BEFORE conversion/normalization: no test arrays, even transiently.
    def subset(indices):
        values=np.asarray(data['dbm'][indices],np.float32)
        labels=np.asarray(data['labels'][indices],np.int64)
        forces=np.asarray(data['forces'][indices],np.float32)
        valid=np.asarray(data['valid'][indices],bool)&np.isfinite(forces).all(1)
        if values.shape != (len(indices),45) or not np.isfinite(values).all():
            raise ValueError('Expected finite measured 45-dimensional dBm')
        return values,labels,forces,valid
    td,ty,tf,tv=subset(train); vd,vy,vf,vv=subset(val)
    if not tv.any(): raise ValueError('No valid training force references')
    mean=td.mean(0).reshape(9,5); std=td.std(0).reshape(9,5)
    std[std<1e-6]=1
    force_mean=tf[tv].mean(0); force_std=tf[tv].std(0)
    force_std[force_std<1e-5]=1
    bin_weights=fit_force_bin_weights(tf,tv)
    def inputs(values):
        return torch.from_numpy(((values.reshape(-1,9,5)-mean)/std).astype(np.float32))
    tx=inputs(td); vx=inputs(vd)
    target=(ty>0).astype(np.float32)
    counts=np.bincount(target.astype(int),minlength=2)
    states=int(np.count_nonzero(counts))
    contact_weights=torch.from_numpy((len(train)/(states*np.maximum(counts[target.astype(int)],1))).astype(np.float32))
    loader=DataLoader(TensorDataset(tx,torch.from_numpy(ty),torch.from_numpy(tf),
        torch.from_numpy(tv),contact_weights),batch_size=c['batch_size'],shuffle=True,
        generator=torch.Generator().manual_seed(c['seed']))
    model=build_model(c)
    optimizer=torch.optim.AdamW(model.parameters(),lr=c['lr'],weight_decay=c['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=c['max_epochs'],eta_min=1e-5)
    best_key=None; best_state=None; best_epoch=0; history=[]
    start=time.perf_counter()
    for epoch in range(1,c['max_epochs']+1):
        model.train(); totals=np.zeros(4); samples=0
        for bx,by,bf,bv,bw in loader:
            optimizer.zero_grad(set_to_none=True)
            logits,pf,contact=model(bx)
            classification=nn.functional.cross_entropy(logits,by,label_smoothing=c['label_smoothing'])
            regression=force_loss(pf,bf,bv,force_mean,force_std,bin_weights,c)
            binary=(nn.functional.binary_cross_entropy_with_logits(contact,(by>0).float(),reduction='none')*bw).mean()
            loss=classification+c['force_weight']*regression+c['contact_weight']*binary
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step()
            totals+=np.array([float(z.detach()) for z in (loss,classification,regression,binary)])*len(by)
            samples+=len(by)
        model.eval()
        with torch.inference_mode():
            vl,vpn,vc=model(vx)
            val_ce=float(nn.functional.cross_entropy(vl,torch.from_numpy(vy)))
            val_force_loss=float(force_loss(vpn,vf,vv,force_mean,force_std,bin_weights,c))
            pred_force=vpn.numpy()*force_std+force_mean
            val_pred=vl.argmax(-1).numpy()
            accuracy=float(np.mean(val_pred==vy))
        mae=np.abs(pred_force[vv]-vf[vv]).mean(0).tolist() if vv.any() else None
        diagnostics=force_diagnostics(vy,vf,pred_force,vv)
        slope=diagnostics['force_gain']['slopes'][2]
        key,objective=validation_selection(accuracy,mae,slope,c)
        row={'epoch':epoch,'training_loss':totals[0]/samples,
            'training_classification_loss':totals[1]/samples,'training_force_loss':totals[2]/samples,
            'training_contact_loss':totals[3]/samples,'validation_accuracy':accuracy,
            'validation_cross_entropy':val_ce,'validation_force_loss':val_force_loss,
            'validation_force_mae_n':mae,'validation_fz_slope':slope,
            'validation_selection_objective':objective,'validation_position_target_feasible':accuracy>=.80,
            'learning_rate':optimizer.param_groups[0]['lr'],'elapsed_seconds':time.perf_counter()-start}
        history.append(row)
        if best_key is None or key>best_key:
            best_key=key; best_epoch=epoch; best_state=copy.deepcopy(model.state_dict())
        scheduler.step()
        if epoch==1 or epoch%10==0:
            print(json.dumps({'type':'progress','trial':c['id'],**row},allow_nan=False),flush=True)
        if epoch-best_epoch>=c['patience']: break
    model.load_state_dict(best_state); model.eval()
    with torch.inference_mode(): vl,vpn,vc=model(vx)
    calibration=calibrate(vl.numpy(),vc.numpy(),vy)
    best_row=history[best_epoch-1]
    output=Path(output); output.mkdir(parents=True,exist_ok=True)
    path=output/(c['id']+'.pt')
    warning='随机邻帧可跨集合，不代表独立按压/会话泛化；逐点扫描与力采样/显示时间不对齐，不能作为同步动态真值。'
    if not vv.any(): warning+=' 验证集缺少有效力参考，无法验证力精度。'
    report={'schema':SCHEMA,'architecture':'force_compare_'+c['architecture'],
        'input_unit':'dBm','input_shape':[9,5],'configuration':c,'config':c,
        'train_count':len(train),'validation_count':len(val),'test_count':len(split.get('test',[])),
        'class_labels':list(range(26)),'best_epoch':best_epoch,'epochs_completed':len(history),
        'best_validation_accuracy':best_row['validation_accuracy'],
        'best_validation_cross_entropy':best_row['validation_cross_entropy'],
        'validation_selection_objective':best_row['validation_selection_objective'],
        'validation_position_target_feasible':best_row['validation_position_target_feasible'],
        'selection_source':'validation only','normalization_source':'train only',
        'force_bin_weights':bin_weights.tolist(),'force_bin_edges':[0,1,3,5,10,None],
        'history':history,'calibration':calibration,'parameters':sum(p.numel() for p in model.parameters()),
        'train_seconds':time.perf_counter()-start,'metrics':{},'decision_metrics':{},
        'force_diagnostics':{},'deployment_ready':False,'validation_warning':warning,
        'model_path':str(path.resolve())}
    if c['architecture']=='tcn':
        report.update(tcn_input_shape=[1,45],receptive_field=receptive_field(c))
    for name,x,labels,forces,valid in [('train',tx,ty,tf,tv),('validation',vx,vy,vf,vv)]:
        with torch.inference_mode():
            logits,pf,contact=model(x)
            probs=(logits/calibration['position_temperature']).softmax(-1).numpy()
            contacts=(contact/calibration['contact_temperature']).sigmoid().numpy()
            predicted_force=pf.numpy()*force_std+force_mean
        report['metrics'][name]=metrics(labels,probs.argmax(-1),forces,predicted_force,valid)
        report['decision_metrics'][name]=decision_metrics(labels,probs,contacts,calibration['policy'])
        report['force_diagnostics'][name]=force_diagnostics(labels,forces,predicted_force,valid)
        report['metrics'][name]['force_diagnostics']=report['force_diagnostics'][name]
        if name=='validation':
            np.savez_compressed(output/(c['id']+'_validation_predictions.npz'),indices=val,
                labels=labels,predicted_labels=probs.argmax(-1),forces=predicted_force,
                true_forces=forces,valid=valid,probabilities=probs,contact_probabilities=contacts,
                logits=logits.numpy(),contact_logits=contact.numpy())
    torch.save({'schema':SCHEMA,'architecture':c['architecture'],'configuration':c,
        'model_state':best_state,'feature_mean':mean,'feature_std':std,
        'force_mean':force_mean,'force_std':force_std,'force_bin_weights':bin_weights,
        'feedback_selector':int(feedback_selector),'calibration':calibration,
        'input_unit':'dBm','sensor_order':list(range(1,10)),'class_labels':list(range(26)),
        'report':report},path)
    path.with_suffix('.report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'type':'trial_completed','trial':c['id'],'best_epoch':best_epoch,
        'validation_accuracy':report['best_validation_accuracy'],
        'validation_selection_objective':report['validation_selection_objective'],
        'model_path':report['model_path']},ensure_ascii=False),flush=True)
    return report

