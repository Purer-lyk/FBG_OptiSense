"""Controlled Encoder-depth model with explicit contact supervision and abstention.
Torch-first processes only. Raw 26-class metrics remain separate from decisions.
"""
import numpy as np
import torch
from touch_dbm_training import DbmSensorTokenModel
from touch_decision import decide_contact,DEFAULT_POLICY
SCHEMA='fbg_touch_dbm_guard26_v1'

class ContactGuardModel(DbmSensorTokenModel):
    def __init__(self,config):
        super().__init__(config)
        self.contact_head=torch.nn.Sequential(torch.nn.Linear(config['shared'],32),torch.nn.GELU(),torch.nn.Dropout(config['dropout']),torch.nn.Linear(32,1))

    def forward(self,values):
        features=self.shared(self.backbone(self.embedding(values)+self.sensor_identity))
        return self.position_head(features),self.force_head(features),self.contact_head(features).squeeze(-1)


def decision_metrics(labels,probabilities,contacts,policy):
    labels=np.asarray(labels)
    decisions=[decide_contact(p,c,policy) for p,c in zip(probabilities,contacts)]
    positions=np.array([-1 if d['position'] is None else d['position'] for d in decisions])
    accepted=positions>=0; zero=labels==0; pressed=labels>0
    return {'count':len(labels),'accepted_count':int(accepted.sum()),'uncertain_count':int((~accepted).sum()),'accepted_coverage':float(accepted.mean()),'accepted_accuracy':float(np.mean(positions[accepted]==labels[accepted])) if accepted.any() else None,'correct_and_accepted_fraction_all':float(np.mean(positions==labels)),'no_press_count':int(zero.sum()),'no_press_false_press_count':int(np.sum(zero&(positions>0))),'no_press_false_press_rate':float(np.mean(positions[zero]>0)) if zero.any() else None,'no_press_uncertain_count':int(np.sum(zero&(~accepted))),'no_press_uncertain_rate':float(np.mean(~accepted[zero])) if zero.any() else None,'no_press_confirmed_recall':float(np.mean(positions[zero]==0)) if zero.any() else None,'pressed_false_no_press_count':int(np.sum(pressed&(positions==0))),'pressed_uncertain_rate':float(np.mean(~accepted[pressed])) if pressed.any() else None}


def calibrate(logits,contact_logits,labels):
    y=torch.as_tensor(labels,dtype=torch.long); logits=torch.as_tensor(logits); contact_logits=torch.as_tensor(contact_logits); target=(y>0).float()
    temperatures=[.5,.75,1.,1.5,2.,3.]
    position_temperature=min(temperatures,key=lambda t:float(torch.nn.functional.cross_entropy(logits/t,y)))
    def binary_loss(t):
        losses=torch.nn.functional.binary_cross_entropy_with_logits(contact_logits/t,target,reduction='none')
        groups=[losses[target==state].mean() for state in [0,1] if torch.any(target==state)]
        return float(torch.stack(groups).mean())
    contact_temperature=min(temperatures,key=binary_loss)
    probability=(logits/position_temperature).softmax(-1).numpy(); contacts=(contact_logits/contact_temperature).sigmoid().numpy()
    candidates=[]
    for confidence in [.60,.65,.75,.85,.90]:
        for threshold in [.60,.75,.85,.90]:
            policy=dict(DEFAULT_POLICY,press_confidence=confidence,press_contact=threshold)
            m=decision_metrics(labels,probability,contacts,policy)
            if m['accepted_count']>0 and m['accepted_accuracy']>=.98 and m['no_press_false_press_count']==0:
                candidates.append((m['accepted_coverage'],m['accepted_accuracy'],confidence,threshold,policy))
    if candidates:
        policy=max(candidates,key=lambda item:(item[0],item[1],item[2],item[3]))[-1]; feasible=True
    else:
        policy=dict(DEFAULT_POLICY,press_confidence=.90,press_contact=.90); feasible=False
    return {'source':'validation only','position_temperature':position_temperature,'contact_temperature':contact_temperature,'policy':policy,'validation_target_accepted_accuracy':.98,'validation_target_feasible':feasible,'warning':'Only few independent no-press runs; validation zero-error is not a field guarantee'}


class ContactGuardBundle:
    def __init__(self,path):
        from pathlib import Path
        torch.set_num_threads(2)
        self.path=Path(path).resolve(); payload=torch.load(self.path,map_location='cpu',weights_only=False)
        if payload.get('schema')!=SCHEMA: raise ValueError('Unsupported contact guard schema')
        self.kind='sensor_token_dbm_guard'; self.config=payload['configuration']; self.report=payload.get('report',{})
        self.model=ContactGuardModel(self.config); self.model.load_state_dict(payload['model_state']); self.model.eval()
        self.mean=np.asarray(payload['feature_mean'],np.float32); self.std=np.asarray(payload['feature_std'],np.float32)
        self.force_mean=np.asarray(payload['force_mean'],np.float32); self.force_std=np.asarray(payload['force_std'],np.float32)
        self.feedback_selector=int(payload['feedback_selector']); self.calibration=payload['calibration']; self.policy=self.calibration['policy']

    def details_dbm(self,dbm):
        values=np.asarray(dbm,np.float32)
        if values.ndim==1: values=values[None]
        if values.ndim!=2 or values.shape[1]!=45 or not np.isfinite(values).all(): raise ValueError('Expected N x 45 finite dBm values')
        x=((values.reshape(-1,9,5)-self.mean)/self.std).astype(np.float32)
        with torch.inference_mode():
            logits,force,contact=self.model(torch.from_numpy(x))
            probability=(logits/self.calibration['position_temperature']).softmax(-1).numpy()
            contacts=(contact/self.calibration['contact_temperature']).sigmoid().numpy()
            force=force.numpy()*self.force_std+self.force_mean
        return probability.argmax(-1),force,probability,contacts

    def predict_dbm(self,dbm): return self.details_dbm(dbm)[:3]

    def predict(self,adc):
        from temporary_optical_units import adc_code_to_dbm
        return self.predict_dbm(adc_code_to_dbm(adc,self.feedback_selector,floor_nonpositive=True))

    def predict_payload(self,adc):
        from temporary_optical_units import adc_code_to_dbm
        _,forces,probabilities,contacts=self.details_dbm(adc_code_to_dbm(adc,self.feedback_selector,floor_nonpositive=True))
        decision=decide_contact(probabilities[0],contacts[0],self.policy)
        return dict(decision,probabilities=probabilities[0].astype(float).tolist(),force=forces[0].astype(float).tolist())


def train_guard(data,split,config,output,feedback_selector):
    import copy,json,time
    from torch.utils.data import TensorDataset,DataLoader
    from manual_touch_sensor_token import metrics
    torch.manual_seed(config['seed']); np.random.seed(config['seed']); torch.set_num_threads(2)
    train=np.asarray(split['train']); val=np.asarray(split['validation'])
    mean=data['dbm'][train].mean(0).reshape(9,5); std=data['dbm'][train].std(0).reshape(9,5); std[std<1e-6]=1
    valid_train=train[data['valid'][train]]
    if not len(valid_train): raise ValueError('No real force labels')
    force_mean=data['forces'][valid_train].mean(0); force_std=data['forces'][valid_train].std(0); force_std[force_std<1e-5]=1
    x=torch.from_numpy(((data['dbm'].reshape(-1,9,5)-mean)/std).astype(np.float32))
    y=torch.from_numpy(data['labels']); f=torch.from_numpy(np.nan_to_num((data['forces']-force_mean)/force_std,nan=0).astype(np.float32)); valid=torch.from_numpy(data['valid'])
    target=(y>0).float(); n0=int((y[train]==0).sum()); n1=len(train)-n0
    # Equal aggregate weight for contact/no-contact, not a majority-class shortcut.
    contact_weights=torch.where(target>0,len(train)/(2*max(n1,1)),len(train)/(2*max(n0,1)))
    loader=DataLoader(TensorDataset(x[train],y[train],f[train],valid[train],contact_weights[train]),batch_size=config['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(config['seed']))
    model=ContactGuardModel(config); optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['max_epochs'],eta_min=1e-5)
    counts=np.bincount(data['labels'][train],minlength=26); weights=torch.from_numpy((len(train)/(26*np.maximum(counts,1))).astype(np.float32))
    best_key=(-float('inf'),-float('inf')); best=None; best_epoch=0; history=[]; start=time.perf_counter()
    for epoch in range(1,config['max_epochs']+1):
        model.train(); losses=[]
        for bx,by,bf,bv,bw in loader:
            optimizer.zero_grad(set_to_none=True); logits,pf,contact=model(bx)
            cls=torch.nn.functional.cross_entropy(logits,by,weight=weights,label_smoothing=config['label_smoothing'])
            reg=torch.nn.functional.smooth_l1_loss(pf[bv],bf[bv]) if bv.any() else pf.sum()*0
            binary=(torch.nn.functional.binary_cross_entropy_with_logits(contact,(by>0).float(),reduction='none')*bw).mean()
            loss=cls+config['force_weight']*reg+config['contact_weight']*binary
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.); optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            logits,pf,contact=model(x[val]); pred=logits.argmax(-1).numpy(); val_ce=float(torch.nn.functional.cross_entropy(logits,y[val])); accuracy=float(np.mean(pred==data['labels'][val])); zero=data['labels'][val]==0
            false_press=float(np.mean(pred[zero]>0)) if zero.any() else 0.
        score=accuracy-.20*false_press; key=(score,-val_ce)
        history.append({'epoch':epoch,'training_loss':float(np.mean(losses)),'validation_accuracy':accuracy,'validation_no_press_false_press_rate_raw':false_press,'validation_cross_entropy':val_ce,'selection_score':score,'learning_rate':optimizer.param_groups[0]['lr']}); scheduler.step()
        if key>best_key: best_key=key; best_epoch=epoch; best=copy.deepcopy(model.state_dict())
        if epoch%10==0: print(json.dumps({'type':'progress','trial':config['id'],**history[-1]}),flush=True)
        if epoch-best_epoch>=config['patience']: break
    model.load_state_dict(best); model.eval()
    with torch.inference_mode(): vl,vf,vc=model(x[val])
    calibration=calibrate(vl.numpy(),vc.numpy(),data['labels'][val])
    report={'schema':SCHEMA,'architecture':'sensor_token_dbm_guard','input_unit':'dBm','input_shape':[9,5],'class_labels':list(range(26)),'train_count':len(train),'validation_count':len(val),'test_count':len(split.get('test',[])),'configuration':dict(config,optimizer='AdamW',scheduler='cosine to 1e-5',normalization='45 measured dimensions, fitted on train only',pooling='ordered flatten of nine real tokens',selection='validation accuracy - 0.20 * validation raw no-press false-press rate; ties validation CE'),'calibration':calibration,'best_epoch':best_epoch,'epochs_completed':len(history),'best_validation_selection_score':best_key[0],'best_validation_accuracy':history[best_epoch-1]['validation_accuracy'],'best_validation_cross_entropy':-best_key[1],'history':history,'parameters':sum(p.numel() for p in model.parameters()),'train_seconds':time.perf_counter()-start,'metrics':{},'decision_metrics':{},'deployment_ready':False,'validation_warning':'随机帧划分不代表独立按压/会话；低置信度或接触判定冲突显示不确定，不能算作0号正确预测。'}
    for name,idx in [('train',train),('validation',val)]:
        with torch.inference_mode(): logits,pf,contact=model(x[idx]); probs=(logits/calibration['position_temperature']).softmax(-1).numpy(); c=(contact/calibration['contact_temperature']).sigmoid().numpy(); pf=pf.numpy()*force_std+force_mean
        report['metrics'][name]=metrics(data['labels'][idx],probs.argmax(-1),data['forces'][idx],pf,data['valid'][idx])
        report['decision_metrics'][name]=decision_metrics(data['labels'][idx],probs,c,calibration['policy'])
    target_path=output/(config['id']+'.pt'); report['model_path']=str(target_path.resolve())
    torch.save({'schema':SCHEMA,'architecture':'sensor_token_dbm_guard','model_state':best,'configuration':config,'feature_mean':mean,'feature_std':std,'force_mean':force_mean,'force_std':force_std,'feedback_selector':feedback_selector,'calibration':calibration,'sensor_order':list(range(1,10)),'class_labels':list(range(26)),'report':report},target_path)
    target_path.with_suffix('.report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'type':'trial_completed','trial':config['id'],'best_epoch':best_epoch,'validation_accuracy':report['best_validation_accuracy'],'validation_decisions':report['decision_metrics']['validation']},ensure_ascii=False),flush=True)
    return report

