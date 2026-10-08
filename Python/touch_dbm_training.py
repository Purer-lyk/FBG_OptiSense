"""dBm Sensor Token training/inference. Import only in Torch-first children."""
import numpy as np
import torch

SCHEMA='fbg_touch_dbm26_v1'


def random_split(labels,seed=20261008):
    from sklearn.model_selection import train_test_split
    indices=np.arange(len(labels))
    train,holdout=train_test_split(indices,test_size=.30,random_state=seed,shuffle=True,stratify=labels)
    val,test=train_test_split(holdout,test_size=.50,random_state=seed+1,shuffle=True,stratify=np.asarray(labels)[holdout])
    return {'train':train.tolist(),'validation':val.tolist(),'test':test.tolist(),'seed':seed,'description':'Merged three sessions; shuffled stratified frame split 70/15/15. Adjacent frames can cross sets; not independent-session or independent-press performance.'}



def extract_dbm_sample(frame):
    q=frame.get('quality') or {}
    try:
        label=q.get('press_label')
        if label is None or int(label)!=label or not 0<=int(label)<=25:
            return None
        values=q.get('stable_dbm')
        if values is None:
            values=(q.get('training_views') or {}).get('stable_optical_power_dbm')
        values=np.asarray(values,np.float32)
        if values.shape!=(45,) or not np.isfinite(values).all():
            return None
    except (TypeError,ValueError):
        return None
    ref=q.get('paxini_force') or {}
    try:
        force=np.asarray([ref.get(k) for k in ('fx_n','fy_n','fz_n')],np.float32)
        valid=bool(np.isfinite(force).all())
    except (TypeError,ValueError):
        valid=False
    if not valid: force=np.full(3,np.nan,np.float32)
    return values,int(label),force,valid


class DbmSensorTokenModel(torch.nn.Module):
    def __init__(self,config):
        super().__init__()
        nn=torch.nn
        width=config['width']
        self.embedding=nn.Linear(5,width)
        self.sensor_identity=nn.Parameter(torch.randn(1,9,width)*.02)
        layer=nn.TransformerEncoderLayer(d_model=width,nhead=config['heads'],dim_feedforward=config['ffn'],dropout=config['dropout'],batch_first=True,norm_first=True,activation='gelu')
        self.backbone=nn.TransformerEncoder(layer,num_layers=config['layers'],norm=nn.LayerNorm(width),enable_nested_tensor=False)
        # Ordered flatten pooling preserves each real sensor's spatial identity.
        self.shared=nn.Sequential(nn.Flatten(1),nn.Linear(9*width,config['shared']),nn.GELU(),nn.Dropout(config['dropout']))
        self.position_head=nn.Linear(config['shared'],26)
        self.force_head=nn.Linear(config['shared'],3)

    def forward(self,values):
        features=self.shared(self.backbone(self.embedding(values)+self.sensor_identity))
        return self.position_head(features),self.force_head(features)


class DbmModelBundle:
    def __init__(self,path):
        from pathlib import Path
        torch.set_num_threads(2)
        self.path=Path(path).resolve()
        payload=torch.load(self.path,map_location='cpu',weights_only=False)
        if payload.get('schema')!=SCHEMA: raise ValueError('Unsupported dBm checkpoint')
        self.kind='sensor_token_dbm'
        self.config=payload['configuration']
        self.model=DbmSensorTokenModel(self.config)
        self.model.load_state_dict(payload['model_state']); self.model.eval()
        self.mean=np.asarray(payload['feature_mean'],np.float32)
        self.std=np.asarray(payload['feature_std'],np.float32)
        self.force_mean=np.asarray(payload['force_mean'],np.float32)
        self.force_std=np.asarray(payload['force_std'],np.float32)
        self.feedback_selector=int(payload['feedback_selector'])
        self.report=payload.get('report',{})

    def predict_dbm(self,dbm):
        values=np.asarray(dbm,np.float32).reshape(-1,9,5)
        if not np.isfinite(values).all(): raise ValueError('Nonfinite dBm input')
        values=(values-self.mean)/self.std
        with torch.inference_mode():
            logits,force=self.model(torch.from_numpy(values.astype(np.float32)))
            probability=logits.softmax(-1).numpy()
            force=force.numpy()*self.force_std+self.force_mean
        return probability.argmax(-1),force,probability

    def predict(self,adc):
        # Existing GUI passes measured ADC; convert exactly as acquisition did.
        from temporary_optical_units import adc_code_to_dbm
        return self.predict_dbm(adc_code_to_dbm(adc,self.feedback_selector,floor_nonpositive=True))


def prepare_data(paths,output):
    import json,hashlib
    from collections import Counter
    from temporary_optical_units import adc_code_to_dbm
    rows=[]; provenance=[]; sources=[]; routes=None; selectors=set(); max_error=0.
    for sid,session in enumerate(paths):
        source=session/'frames.jsonl'; digest=hashlib.sha256(); counts=Counter(); invalid=0; line_count=0; quality=Counter()
        with source.open('rb') as stream:
            for line_no,raw in enumerate(stream,1):
                digest.update(raw); line_count+=1; frame=json.loads(raw); q=frame.get('quality') or {}; views=q.get('training_views') or {}
                sample=extract_dbm_sample(frame)
                if sample is None: invalid+=1; continue
                values,label,force,valid=sample
                wavelengths=np.asarray(views.get('wavelengths_nm'),float)
                if wavelengths.shape!=(45,) or not np.isfinite(wavelengths).all(): raise ValueError('Missing measured wavelength route')
                if routes is None: routes=wavelengths
                elif not np.allclose(routes,wavelengths,rtol=0,atol=1e-6): raise ValueError('Wavelength routes differ')
                selector=q.get('feedback_selector',views.get('conversion',{}).get('feedback_selector'))
                if selector is None: raise ValueError('Missing feedback selector')
                selectors.add(int(selector))
                adc=q.get('stable_adc',views.get('stable_adc'))
                adc=np.asarray(adc,np.float32)
                if adc.shape!=(45,) or not np.isfinite(adc).all(): raise ValueError('Missing paired ADC needed for inference conversion verification')
                error=float(np.max(np.abs(adc_code_to_dbm(adc,selector,floor_nonpositive=True)-values)))
                max_error=max(max_error,error)
                if error>1e-4: raise ValueError('Saved dBm does not match live conversion')
                rows.append((values,label,force,valid,sid,adc)); counts[label]+=1
                flags=q.get('source_frame_quality') or {}
                quality[str(flags.get('stability_verified'))]+=1
                provenance.append({'index':len(rows)-1,'session':sid,'source_line':line_no,'label':label,'source_frame_sequence':q.get('source_frame_sequence')})
        sources.append({'path':str(source.resolve()),'sha256':digest.hexdigest(),'lines':line_count,'invalid_dbm_samples':invalid,'accepted':sum(counts.values()),'label_counts':dict(counts),'stability_verified_counts':dict(quality)})
    if not rows: raise ValueError('No valid saved dBm')
    if len(selectors)!=1: raise ValueError('Multiple front-end selectors; need per-frame conversion metadata support')
    data={'dbm':np.stack([r[0] for r in rows]),'labels':np.array([r[1] for r in rows],np.int64),'forces':np.stack([r[2] for r in rows]),'valid':np.array([r[3] for r in rows],bool),'session':np.array([r[4] for r in rows],np.int64),'adc':np.stack([r[5] for r in rows])}
    audit={'sources':sources,'samples':provenance,'total':len(rows),'valid_force':int(data['valid'].sum()),'feedback_selector':next(iter(selectors)),'maximum_adc_to_saved_dbm_error':max_error,'wavelength_route_nm':routes.tolist(),'input':'Saved measured quality.stable_dbm / training_views.stable_optical_power_dbm, not Makima fits','warnings':['Random frame holdout can contain neighbors of training frames','dBm is engineering conversion, not calibrated power-meter data']}
    np.savez_compressed(output/'dataset.npz',**data)
    (output/'data_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return data,audit


def train_trial(data,split,config,output,feedback_selector):
    import copy,json,time
    from torch.utils.data import TensorDataset,DataLoader
    from manual_touch_sensor_token import metrics
    torch.manual_seed(config['seed']); np.random.seed(config['seed']); torch.set_num_threads(2)
    train=np.asarray(split['train']); val=np.asarray(split['validation'])
    mean=data['dbm'][train].mean(0).reshape(9,5)
    std=data['dbm'][train].std(0).reshape(9,5); std[std<1e-6]=1
    valid_train=train[data['valid'][train]]
    if not len(valid_train): raise ValueError('No real force labels')
    force_mean=data['forces'][valid_train].mean(0)
    force_std=data['forces'][valid_train].std(0); force_std[force_std<1e-5]=1
    values=((data['dbm'].reshape(-1,9,5)-mean)/std).astype(np.float32)
    force=np.nan_to_num((data['forces']-force_mean)/force_std,nan=0).astype(np.float32)
    x=torch.from_numpy(values); y=torch.from_numpy(data['labels']); f=torch.from_numpy(force); valid=torch.from_numpy(data['valid'])
    loader=DataLoader(TensorDataset(x[train],y[train],f[train],valid[train]),batch_size=config['batch_size'],shuffle=True,generator=torch.Generator().manual_seed(config['seed']))
    model=DbmSensorTokenModel(config)
    optimizer=torch.optim.AdamW(model.parameters(),lr=config['lr'],weight_decay=config['weight_decay'])
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=config['max_epochs'],eta_min=1e-5)
    counts=np.bincount(data['labels'][train],minlength=26)
    weights=torch.from_numpy((len(train)/(26*np.maximum(counts,1))).astype(np.float32))
    best_key=(-1.,-float('inf')); best=None; best_epoch=0; history=[]; start=time.perf_counter()
    for epoch in range(1,config['max_epochs']+1):
        model.train(); losses=[]
        for bx,by,bf,bv in loader:
            optimizer.zero_grad(set_to_none=True)
            logits,pf=model(bx)
            cls=torch.nn.functional.cross_entropy(logits,by,weight=weights)
            reg=torch.nn.functional.smooth_l1_loss(pf[bv],bf[bv]) if bv.any() else pf.sum()*0
            loss=cls+config['force_weight']*reg
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
            optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            logits,pf=model(x[val]); pred=logits.argmax(-1).numpy()
            val_loss=float(torch.nn.functional.cross_entropy(logits,y[val]))
            accuracy=float(np.mean(pred==data['labels'][val]))
        lr=optimizer.param_groups[0]['lr']; scheduler.step()
        history.append({'epoch':epoch,'training_loss':float(np.mean(losses)),'validation_accuracy':accuracy,'validation_cross_entropy':val_loss,'learning_rate':lr})
        key=(accuracy,-val_loss)
        if key>best_key:
            best_key=key; best_epoch=epoch; best=copy.deepcopy(model.state_dict())
        if epoch%10==0:
            print(json.dumps({'type':'progress','trial':config['id'],**history[-1]}),flush=True)
        if epoch-best_epoch>=config['patience']: break
    model.load_state_dict(best); model.eval()
    result={'schema':SCHEMA,'architecture':'sensor_token_dbm','input_unit':'dBm','input_shape':[9,5],'class_labels':list(range(26)),'train_count':len(train),'validation_count':len(val),'test_count':len(split['test']),'configuration':dict(config,optimizer='AdamW',scheduler='cosine to 1e-5',normalization='45 measured dimensions individually; training-only statistics',pooling='ordered flatten, nine real tokens',selection='validation accuracy, tie break validation cross entropy'),'best_epoch':best_epoch,'epochs_completed':len(history),'best_validation_accuracy':best_key[0],'best_validation_cross_entropy':-best_key[1],'history':history,'train_seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters()),'deployment_ready':False,'validation_warning':split.get('description','Random frame split; not independent presses'),'metrics':{}}
    # Only train and validation are evaluated here. Holdout is opened after selection.
    for name,idx in [('train',train),('validation',val)]:
        with torch.inference_mode():
            logits,pf=model(x[idx]); pred=logits.argmax(-1).numpy(); pf=pf.numpy()*force_std+force_mean
        result['metrics'][name]=metrics(data['labels'][idx],pred,data['forces'][idx],pf,data['valid'][idx])
    target=output/(config['id']+'.pt'); result['model_path']=str(target.resolve())
    torch.save({'schema':SCHEMA,'architecture':'sensor_token_dbm','model_state':best,'configuration':config,'feature_mean':mean,'feature_std':std,'force_mean':force_mean,'force_std':force_std,'feedback_selector':feedback_selector,'sensor_order':list(range(1,10)),'class_labels':list(range(26)),'input_unit':'dBm','report':result},target)
    target.with_suffix('.report.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'type':'trial_completed','trial':config['id'],'validation_accuracy':best_key[0],'model_path':str(target.resolve())}),flush=True)
    return result

