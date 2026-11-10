"""Six measured-dBm backbones. Torch-first subprocess imports only.

The original three backbones and common nonlinear + 45-point linear force
heads are reused unchanged. Training below is a local maintained extension
of touch_force_comparison.train_trial, not a mutation of that module.
"""
import numpy as np
import torch
from torch import nn
import touch_force_comparison as base

SCHEMA = 'fbg_touch_six26_v1'
ARCHITECTURES = ('tcn', 'sensor_token', 'sparse_cnn', 'residual_mlp', 'inception_1d', 'mlp_mixer')
DEFAULT_CONFIG = dict(base.DEFAULT_CONFIG, position_target=.95, force_mae_target=.1,
    max_epochs=240, patience=40, force_weight=1.)

def _config(config):
    c = dict(DEFAULT_CONFIG, **config)
    if c['architecture'] not in ARCHITECTURES:
        raise ValueError('Unknown architecture: ' + str(c['architecture']))
    scales = np.asarray(c['axis_scales'],float)
    if scales.shape != (3,) or not np.isfinite(scales).all() or (scales<=0).any():
        raise ValueError('Physical axis scales must be three finite positive constants')
    if c['position_target'] != .95 or c['force_mae_target'] != .1:
        raise ValueError('Validation targets are fixed at accuracy .95 and each-axis MAE .1 N')
    return c

force_loss = base.force_loss
fit_force_bin_weights = base.fit_force_bin_weights
def force_diagnostics(labels, true_forces, predicted_forces, valid):
    """Retain full-vector diagnostics and use all valid references per axis for gain."""
    result = base.force_diagnostics(labels,true_forces,predicted_forces,valid)
    y,p = np.asarray(true_forces,float),np.asarray(predicted_forces,float)
    mask = np.asarray(valid,bool)
    if mask.ndim == 1: mask = np.broadcast_to(mask[:,None],y.shape)
    mask = mask & np.isfinite(y) & np.isfinite(p)
    slopes,intercepts = [],[]
    centered_slopes,centered_intercepts = [],[]
    labels = np.asarray(labels)
    for a in range(3):
        group = mask[:,a]
        x,z = y[group,a],p[group,a]
        def gain(x,z):
            if len(x)<2 or np.var(x)<1e-12: return None,None
            slope = float(np.dot(x-x.mean(),z-z.mean())/np.sum((x-x.mean())**2))
            return slope,float(z.mean()-slope*x.mean())
        slope,intercept = gain(x,z)
        slopes.append(slope); intercepts.append(intercept)
        xc,zc = x.copy(),z.copy()
        for label in np.unique(labels[group]):
            selected = labels[group] == label
            xc[selected] -= xc[selected].mean(); zc[selected] -= zc[selected].mean()
        slope,intercept = gain(xc,zc)
        centered_slopes.append(slope); centered_intercepts.append(intercept)
    result['force_gain'] = dict(slopes=slopes,intercepts=intercepts)
    result['within_label_gain'] = dict(slopes=centered_slopes,intercepts=centered_intercepts)
    result['valid_axis_counts'] = mask.sum(0).astype(int).tolist()
    return result
receptive_field = base.receptive_field

def validation_selection(accuracy, axis_mae, fz_slope, config):
    """Feasible means ALL three force axes meet .1 N AND accuracy >= .95.

Unavailable force validation is never feasible. Within infeasible candidates,
    minimize normalized target violations, rather than classification alone.
    """
    c = _config(config)
    available = axis_mae is not None and np.shape(axis_mae) == (3,)
    available = available and all(v is not None and np.isfinite(v) for v in axis_mae)
    if available:
        ratios = np.asarray(axis_mae,float)/c['force_mae_target']
        force_term = float(np.mean(ratios))
        force_violation = float(np.maximum(ratios-1,0).sum())
    else:
        force_term, force_violation = 1e6, 1e6
    violation = max(0.,c['position_target']-accuracy)/.05 + force_violation
    objective = force_term + .2*(1-accuracy) + (.1*abs(fz_slope-1) if fz_slope is not None else 0.)
    feasible = accuracy >= c['position_target'] and available and bool(np.all(ratios <= 1))
    return (int(feasible),-violation,-objective,accuracy), objective

class ResidualMLPBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(width), nn.Linear(width,2*width),
            nn.GELU(), nn.Dropout(dropout), nn.Linear(2*width,width))
    def forward(self, x):
        return x + self.net(x)

class InceptionBlock(nn.Module):
    def __init__(self, incoming, width, dropout):
        super().__init__()
        self.branches = nn.ModuleList(nn.Conv1d(incoming,width,k,padding=k//2) for k in (1,3,5))
        self.mix = nn.Sequential(nn.GELU(), nn.Conv1d(3*width,width,1), nn.Dropout(dropout))
        self.skip = nn.Conv1d(incoming,width,1) if incoming != width else nn.Identity()
    def forward(self, x):
        return self.skip(x) + self.mix(torch.cat([branch(x) for branch in self.branches],1))

class MixerBlock(nn.Module):
    def __init__(self, width, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.token = nn.Sequential(nn.Linear(9,18),nn.GELU(),nn.Dropout(dropout),nn.Linear(18,9))
        self.channel = ResidualMLPBlock(width,dropout)
    def forward(self, x):
        x = x + self.token(self.norm(x).transpose(1,2)).transpose(1,2)
        return self.channel(x)

class SixModel(base.ForceComparisonModel):
    def __init__(self, config):
        c = _config(config)
        kind = c['architecture']
        # Initialize exactly the existing shared heads; replace only new backbones.
        super().__init__(dict(c, architecture=kind if kind in ARCHITECTURES[:3] else 'sparse_cnn'))
        self.config, self.kind = c, kind
        w, d = c['width'], c['dropout']
        if kind == 'residual_mlp':
            self.backbone = nn.Sequential(nn.Flatten(1),nn.Linear(45,w),nn.GELU(),
                *[ResidualMLPBlock(w,d) for _ in range(c['layers'])])
            incoming = w
        elif kind == 'inception_1d':
            self.backbone = nn.Sequential(InceptionBlock(1,w,d),
                *[InceptionBlock(w,w,d) for _ in range(max(0,c['layers']-1))],nn.Flatten(1))
            incoming = 45*w
        elif kind == 'mlp_mixer':
            self.embedding = nn.Linear(5,w)
            self.backbone = nn.Sequential(*[MixerBlock(w,d) for _ in range(c['layers'])],nn.LayerNorm(w),nn.Flatten(1))
            incoming = 9*w
        else:
            return
        self.shared = nn.Sequential(nn.Linear(incoming,c['shared']),nn.GELU(),nn.Dropout(d))

    def features(self, x):
        if x.ndim != 3 or tuple(x.shape[1:]) != (9,5):
            raise ValueError('Expected B x 9 x 5 normalized measured points')
        if self.kind == 'residual_mlp': h = self.backbone(x)
        elif self.kind == 'inception_1d': h = self.backbone(x.reshape(-1,1,45))
        elif self.kind == 'mlp_mixer': h = self.backbone(self.embedding(x))
        else: return super().features(x)
        return self.shared(h)

def build_model(config):
    c = _config(config)
    if c['architecture'] == 'tcn' and base.receptive_field(c) < 45:
        raise ValueError('TCN receptive field must cover all 45 measured points')
    torch.manual_seed(c['seed'])
    return SixModel(c)

def axis_mae(true, predicted, valid):
    return [float(np.abs(predicted[valid[:,a],a]-true[valid[:,a],a]).mean())
            if valid[:,a].any() else None for a in range(3)]

def force_feasible(mae, config):
    return mae is not None and all(v is not None and np.isfinite(v) and v <= config['force_mae_target'] for v in mae)

def metrics(labels, predicted, true, forces, valid):
    # Preserve legacy classification/confusion fields, then replace force metrics
    # with per-axis masked metrics (partial references are not dropped).
    from manual_touch_sensor_token import metrics as legacy_metrics
    result = legacy_metrics(labels,predicted,true,forces,valid.all(1))
    result['force_mae'] = axis_mae(true,forces,valid)
    result['force_mae_n'] = result['force_mae']
    result['force_valid_axis_counts'] = valid.sum(0).astype(int).tolist()
    rmse,r2,corr = [],[],[]
    for a in range(3):
        y,p = true[valid[:,a],a].astype(float),forces[valid[:,a],a].astype(float)
        if not len(y):
            rmse.append(None); r2.append(None); corr.append(None)
            continue
        error = p-y
        rmse.append(float(np.sqrt(np.mean(error**2))))
        denominator = float(np.sum((y-y.mean())**2))
        r2.append(float(1-np.sum(error**2)/denominator) if denominator>1e-12 else None)
        corr.append(float(np.corrcoef(y,p)[0,1]) if y.std()>1e-9 and p.std()>1e-9 else None)
    result.update(force_rmse_n=rmse,force_r2=r2,force_pearson=corr)
    return result

class SixModelBundle:
    def __init__(self, path):
        from pathlib import Path
        torch.set_num_threads(2)
        self.path=Path(path).resolve()
        payload=torch.load(self.path,map_location='cpu',weights_only=False)
        if payload.get('schema') != SCHEMA:
            raise ValueError('Unsupported six-model schema')
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
        valid=np.asarray(data['valid'][indices],bool)
        if valid.ndim == 1: valid=np.broadcast_to(valid[:,None],forces.shape).copy()
        if forces.shape != (len(indices),3) or valid.shape != forces.shape:
            raise ValueError('Expected forces and mask shaped N x 3 (or row mask)')
        valid=valid & np.isfinite(forces)
        if labels.shape != (len(indices),) or (labels<0).any() or (labels>25).any():
            raise ValueError('Expected class labels 0..25')
        if values.shape != (len(indices),45) or not np.isfinite(values).all():
            raise ValueError('Expected finite measured 45-dimensional dBm')
        return values,labels,forces,valid
    td,ty,tf,tv=subset(train); vd,vy,vf,vv=subset(val)
    if not tv.any(): raise ValueError('No valid training force references')
    mean=td.mean(0).reshape(9,5); std=td.std(0).reshape(9,5)
    std[std<1e-6]=1
    force_mean=np.array([tf[tv[:,a],a].mean() if tv[:,a].any() else 0. for a in range(3)],np.float32)
    force_std=np.array([tf[tv[:,a],a].std() if tv[:,a].any() else 1. for a in range(3)],np.float32)
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
        mae=axis_mae(vf,pred_force,vv)
        diagnostics=force_diagnostics(vy,vf,pred_force,vv)
        slope=diagnostics['force_gain']['slopes'][2]
        key,objective=validation_selection(accuracy,mae,slope,c)
        row={'epoch':epoch,'training_loss':totals[0]/samples,
            'training_classification_loss':totals[1]/samples,'training_force_loss':totals[2]/samples,
            'training_contact_loss':totals[3]/samples,'validation_accuracy':accuracy,
            'validation_cross_entropy':val_ce,'validation_force_loss':val_force_loss,
            'validation_force_mae_n':mae,'validation_fz_slope':slope,
            'validation_selection_objective':objective,'validation_position_target_feasible':accuracy>=c['position_target'],
            'validation_force_target_feasible':force_feasible(mae,c),
            'validation_joint_target_feasible':bool(key[0]),
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
    if not vv.any(0).all(): warning+=' 验证集缺少有效力参考，无法验证力精度。'
    report={'schema':SCHEMA,'architecture':c['architecture'],
        'input_unit':'dBm','input_shape':[9,5],'configuration':c,'config':c,
        'train_count':len(train),'validation_count':len(val),'test_count':len(split.get('test',[])),
        'class_labels':list(range(26)),'best_epoch':best_epoch,'epochs_completed':len(history),
        'best_validation_accuracy':best_row['validation_accuracy'],
        'best_validation_cross_entropy':best_row['validation_cross_entropy'],
        'validation_selection_objective':best_row['validation_selection_objective'],
        'validation_selection_key':list(best_key),
        'validation_position_target_feasible':best_row['validation_position_target_feasible'],
        'validation_force_target_feasible':best_row['validation_force_target_feasible'],
        'validation_joint_target_feasible':best_row['validation_joint_target_feasible'],
        'validation_force_mae_n':best_row['validation_force_mae_n'],
        'position_target':c['position_target'],'force_mae_target':c['force_mae_target'],
        'force_mask_granularity':'per axis','calibration_source':'validation only',
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

