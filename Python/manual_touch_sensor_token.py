"""Nine measured sensor tokens; shared attention backbone; 26-class + force heads.
Import only in a Torch-first process (never import from Qt GUI).
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from manual_touch_cnn import SparseTouchCNN, feature_statistics, spatialize_five_point_features

SCHEMA = 'fbg_touch26_v1'


def tokenize(adc, mean, std):
    values = np.asarray(adc, dtype=np.float32)
    if values.ndim == 1:
        values = values[None]
    if values.ndim != 2 or values.shape[1] != 45 or not np.isfinite(values).all():
        raise ValueError('Expected finite N x 45 measured ADC values')
    scale = np.asarray(std, np.float32).reshape(1, 1, 5)
    if np.any(scale <= 0):
        raise ValueError('Invalid standard deviation')
    return (values.reshape(-1, 9, 5) - np.asarray(mean, np.float32).reshape(1, 1, 5)) / scale


class SensorTokenModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Linear(5, 48)
        self.sensor_identity = nn.Parameter(torch.zeros(1, 9, 48))
        nn.init.normal_(self.sensor_identity, std=.02)
        # Identity follows fixed grating order 1..9, not nonexistent grid slots.
        layer = nn.TransformerEncoderLayer(d_model=48, nhead=4, dim_feedforward=128, dropout=.12, batch_first=True, norm_first=True, activation='gelu')
        self.backbone = nn.TransformerEncoder(layer, num_layers=2, norm=nn.LayerNorm(48), enable_nested_tensor=False)
        self.shared = nn.Sequential(nn.Linear(48, 128), nn.GELU(), nn.Dropout(.12))
        self.position_head = nn.Linear(128, 26)
        self.force_head = nn.Linear(128, 3)

    def forward(self, values):
        tokens = self.embedding(values) + self.sensor_identity
        features = self.shared(self.backbone(tokens).mean(dim=1))
        return self.position_head(features), self.force_head(features)


class SparseCNN26(SparseTouchCNN):
    """Original backbone unchanged; add class 0 to the classification head."""
    def __init__(self):
        super().__init__()
        self.position_head = nn.Linear(128, 26)


def build_model(kind):
    if kind == 'sensor_token':
        return SensorTokenModel()
    if kind == 'sparse_cnn26':
        return SparseCNN26()
    raise ValueError(f'Unknown architecture: {kind}')


def extract_sample(frame):
    q = frame.get('quality') or {}
    label = q.get('press_label')
    try:
        if label is None or int(label) != label or not 0 <= int(label) <= 25:
            return None
        stable = q.get('stable_adc')
        if stable is None:
            stable = (q.get('training_views') or {}).get('stable_adc')
        if stable is None:
            stable = (frame.get('raw_adc_codes') or {}).get('1')
        adc = np.asarray(stable, np.float32)
        if adc.shape != (45,) or not np.isfinite(adc).all():
            return None
    except (ValueError, TypeError):
        return None
    reference = q.get('paxini_force') or {}
    try:
        force = np.asarray([reference.get(k) for k in ('fx_n', 'fy_n', 'fz_n')], np.float32)
        valid = bool(np.isfinite(force).all())
    except (ValueError, TypeError):
        valid = False
    if not valid:
        force = np.full(3, np.nan, np.float32)
    return adc, int(label), force, valid


def multitask_loss(logits, predicted_force, labels, force, valid, class_weights):
    classification = nn.functional.cross_entropy(logits, labels, weight=class_weights)
    regression = nn.functional.smooth_l1_loss(predicted_force[valid], force[valid]) if valid.any() else predicted_force.sum() * 0
    return classification + regression


class ModelBundle:
    def __init__(self, path):
        self.path = Path(path).resolve()
        payload = torch.load(self.path, map_location='cpu', weights_only=False)
        if payload.get('schema') != SCHEMA:
            raise ValueError('Unsupported 26-class model schema')
        self.kind = payload['architecture']
        self.model = build_model(self.kind)
        self.model.load_state_dict(payload['model_state'])
        self.model.eval()
        self.mean = np.asarray(payload['feature_mean'], np.float32)
        self.std = np.asarray(payload['feature_std'], np.float32)
        self.force_mean = np.asarray(payload['force_mean'], np.float32)
        self.force_std = np.asarray(payload['force_std'], np.float32)
        self.report = payload.get('report', {})

    def predict(self, adc):
        if self.kind == 'sensor_token':
            values = tokenize(adc, self.mean, self.std)
        else:
            values = spatialize_five_point_features(adc, feature_mean=self.mean, feature_std=self.std)
            if values.ndim == 3:
                values = values[None]
        with torch.inference_mode():
            logits, force = self.model(torch.from_numpy(values.astype(np.float32)))
            probability = logits.softmax(-1).numpy()
            force = force.numpy() * self.force_std + self.force_mean
        return probability.argmax(-1), force, probability


def metrics(labels, predicted, forces, predicted_force, valid):
    from sklearn.metrics import confusion_matrix, f1_score, precision_recall_fscore_support
    labels = np.asarray(labels)
    predicted = np.asarray(predicted)
    precision, recall, f1, support = precision_recall_fscore_support(labels, predicted, labels=list(range(26)), zero_division=0)
    present = support > 0
    result = {'count': len(labels), 'position_accuracy': float(np.mean(labels == predicted)), 'macro_f1_present_classes': float(np.mean(f1[present])), 'balanced_accuracy_present_classes': float(np.mean(recall[present])), 'weighted_f1': float(f1_score(labels, predicted, average='weighted', zero_division=0)), 'pressed_only_accuracy': float(np.mean(labels[labels>0] == predicted[labels>0])) if np.any(labels>0) else None, 'no_press_recall': float(recall[0]) if support[0] else None, 'no_press_precision': float(precision[0]) if support[0] or np.any(predicted==0) else None, 'confusion_matrix': confusion_matrix(labels, predicted, labels=list(range(26))).tolist(), 'per_class': [{'label': i, 'support': int(support[i]), 'precision': float(precision[i]), 'recall': float(recall[i]), 'f1': float(f1[i])} for i in range(26)]}
    if np.any(valid):
        y = forces[valid].astype(float)
        error = predicted_force[valid].astype(float) - y
        denom = ((y-y.mean(0))**2).sum(0)
        r2 = np.where(denom>1e-12, 1-(error**2).sum(0)/np.maximum(denom, 1e-12), np.nan)
        corr=[]
        for k in range(3):
            corr.append(float(np.corrcoef(y[:,k], predicted_force[valid,k])[0,1]) if y[:,k].std()>1e-9 and predicted_force[valid,k].std()>1e-9 else None)
        result.update(force_reference_count=int(valid.sum()), force_mae_n=np.abs(error).mean(0).tolist(), force_rmse_n=np.sqrt((error**2).mean(0)).tolist(), force_r2=[float(v) if np.isfinite(v) else None for v in r2], force_pearson=corr, force_vector_mae_n=float(np.linalg.norm(error,axis=1).mean()), force_vector_rmse_n=float(np.sqrt((error**2).sum(1).mean())))
    else:
        result.update(force_reference_count=0, force_mae_n=None, force_rmse_n=None, force_r2=None, force_pearson=None, force_vector_mae_n=None, force_vector_rmse_n=None)
    return result


def load_sessions(paths, output):
    rows=[]; provenance=[]; audits=[]
    for session, root in enumerate(paths):
        path=Path(root)/'frames.jsonl'
        digest=hashlib.sha256(); line_count=0; invalid=0; group=-1; last_label=None; counts={}; reference_counts={}
        with path.open('rb') as stream:
            for number, raw in enumerate(stream,1):
                digest.update(raw); line_count+=1
                frame=json.loads(raw)
                sample=extract_sample(frame)
                if sample is None:
                    invalid+=1
                    continue
                adc,label,force,valid=sample
                if label != last_label:
                    group+=1; last_label=label
                rows.append((adc,label,force,valid,session,group))
                provenance.append({'session':session,'line':number,'label':label,'run':group,'sample_index':(frame.get('quality') or {}).get('sample_index')})
                counts[label]=counts.get(label,0)+1
                if valid: reference_counts[label]=reference_counts.get(label,0)+1
        audits.append({'path':str(path.resolve()),'sha256':digest.hexdigest(),'lines':line_count,'invalid_classification_samples':invalid,'counts':counts,'force_reference_counts':reference_counts})
        print(json.dumps({'type':'dataset_loaded','audit':audits[-1]},ensure_ascii=False),flush=True)
    if not rows: raise ValueError('No usable samples')
    data={'adc':np.stack([r[0] for r in rows]),'labels':np.array([r[1] for r in rows],np.int64),'forces':np.stack([r[2] for r in rows]),'valid':np.array([r[3] for r in rows],bool),'session':np.array([r[4] for r in rows],np.int64),'run':np.array([r[5] for r in rows],np.int64)}
    np.savez_compressed(Path(output)/'dataset.npz',**data)
    (Path(output)/'data_audit.json').write_text(json.dumps({'sources':audits,'samples':provenance},ensure_ascii=False,indent=2),encoding='utf-8')
    return data,audits


def blocked_split(labels):
    """Class-wise chronological 60/5-gap/15/5-gap/15 split; development only."""
    y = np.asarray(labels)
    split = {k: [] for k in ('train', 'validation', 'test', 'purged')}
    for label in range(26):
        indices = np.flatnonzero(y == label)
        n = len(indices)
        if n < 20:
            raise ValueError(f'Class {label} needs at least 20 examples for this split')
        ntrain = int(np.floor(n * .60))
        gap = max(1, int(round(n * .05)))
        nval = max(1, int(round(n * .15)))
        val_start = ntrain + gap
        test_start = val_start + nval + gap
        if test_start >= n:
            raise ValueError('Insufficient chronological holdout')
        split['train'].extend(indices[:ntrain].tolist())
        split['purged'].extend(indices[ntrain:val_start].tolist())
        split['validation'].extend(indices[val_start:val_start+nval].tolist())
        split['purged'].extend(indices[val_start+nval:test_start].tolist())
        split['test'].extend(indices[test_start:].tolist())
    for key in split:
        split[key] = sorted(split[key])
    split['description'] = 'Within each label: first 60% train, ~5% purge, ~15% validation, ~5% purge, remaining ~15% test. No random interleaving. Not independent-session/cycle generalization.'
    return split


def train_one(data, split, kind, seed, output, epochs=180, patience=30):
    torch.manual_seed(seed); np.random.seed(seed); torch.set_num_threads(2)
    train=np.asarray(split['train']); val=np.asarray(split['validation']); test=np.asarray(split['test'])
    mean,std=feature_statistics(data['adc'][train])
    valid_train=train[data['valid'][train]]
    if not len(valid_train): raise ValueError('No real force reference in training subset')
    force_mean=data['forces'][valid_train].mean(0); force_std=data['forces'][valid_train].std(0); force_std[force_std<1e-5]=1
    force=np.nan_to_num((data['forces']-force_mean)/force_std,nan=0).astype(np.float32)
    values=tokenize(data['adc'],mean,std) if kind=='sensor_token' else spatialize_five_point_features(data['adc'],feature_mean=mean,feature_std=std)
    tensors=[torch.from_numpy(v) for v in (values,data['labels'],force,data['valid'])]
    x,y,f,v=tensors
    generator=torch.Generator().manual_seed(seed)
    loader=DataLoader(TensorDataset(*[t[train] for t in tensors]),batch_size=64,shuffle=True,generator=generator)
    model=build_model(kind); optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    counts=np.bincount(data['labels'][train],minlength=26)
    weights=torch.from_numpy((len(train)/(26*np.maximum(counts,1))).astype(np.float32))
    best_score=-float('inf'); best=None; best_epoch=0; history=[]; start=time.perf_counter()
    for epoch in range(1,epochs+1):
        model.train(); losses=[]
        for bx,by,bf,bv in loader:
            optimizer.zero_grad(set_to_none=True)
            logits,predicted=model(bx)
            loss=multitask_loss(logits,predicted,by,bf,bv,weights)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),1.0); optimizer.step(); losses.append(float(loss.detach()))
        model.eval()
        with torch.inference_mode():
            logits,fp=model(x[val]); predicted=logits.argmax(-1).numpy(); force_pred=fp.numpy()*force_std+force_mean
        accuracy=float(np.mean(predicted==data['labels'][val]))
        mask=data['valid'][val]
        force_error=float(np.abs((force_pred[mask]-data['forces'][val][mask])/force_std).mean()) if mask.any() else 0
        score=accuracy-.05*force_error
        history.append({'epoch':epoch,'train_loss':float(np.mean(losses)),'validation_accuracy':accuracy,'validation_standardized_force_mae':force_error,'selection_score':score})
        if score>best_score+1e-5:
            best_score=score; best_epoch=epoch; best=copy.deepcopy(model.state_dict())
        if epoch%10==0: print(json.dumps({'type':'progress','architecture':kind,'seed':seed,**history[-1]}),flush=True)
        if epoch-best_epoch>=patience: break
    model.load_state_dict(best); model.eval(); results={}; predictions={}
    for name,indices in [('train',train),('validation',val),('test',test)]:
        with torch.inference_mode():
            logits,fp=model(x[indices]); probability=logits.softmax(-1).numpy(); pred=probability.argmax(-1); pf=fp.numpy()*force_std+force_mean
        results[name]=metrics(data['labels'][indices],pred,data['forces'][indices],pf,data['valid'][indices])
        predictions[name]={'indices':indices,'labels':data['labels'][indices],'predicted_labels':pred,'forces':data['forces'][indices],'predicted_forces':pf,'force_valid':data['valid'][indices],'probabilities':probability}
    report={'schema':SCHEMA,'architecture':kind,'seed':seed,'train_count':len(train),'validation_count':len(val),'test_count':len(test),'best_epoch':best_epoch,'epochs_completed':len(history),'train_seconds':time.perf_counter()-start,'parameters':sum(p.numel() for p in model.parameters()),'history':history,'metrics':results,'configuration':{'batch_size':64,'max_epochs':epochs,'patience':patience,'optimizer':'AdamW','learning_rate':.001,'weight_decay':.0001,'loss':'weighted cross entropy (26 mutually exclusive one-hot equivalent) + masked SmoothL1 on train-standardized Fx/Fy/Fz','selection_score':'validation accuracy - 0.05 * validation standardized force MAE','gradient_clip_norm':1.,'torch_threads':2,'signal_statistics':'five channels pooled across nine sensors, fit on training samples only','sensor_token':{'tokens':9,'measured_dimensions':5,'embedding_dimension':48,'learned_sensor_identity':True,'attention_heads':4,'encoder_layers':2,'ffn_dimension':128,'pooling':'mean','shared_feature_dimension':128,'dropout':.12}},'deployment_ready':False}
    target=Path(output)/f'{kind}_seed{seed}.pt'
    torch.save({'schema':SCHEMA,'architecture':kind,'model_state':best,'feature_mean':mean,'feature_std':std,'force_mean':force_mean,'force_std':force_std,'sensor_order':list(range(1,10)),'class_labels':list(range(26)),'report':report},target)
    bundle=ModelBundle(target)
    label_rt,force_rt,prob_rt=bundle.predict(data['adc'][test])
    np.testing.assert_array_equal(label_rt,predictions['test']['predicted_labels'])
    np.testing.assert_allclose(force_rt,predictions['test']['predicted_forces'],rtol=1e-5,atol=1e-5)
    # Single-frame CPU timing excludes file load and subprocess/GUI overhead.
    sample=data['adc'][test[0]]
    for _ in range(20): bundle.predict(sample)
    times=[]
    for _ in range(200):
        t=time.perf_counter(); bundle.predict(sample); times.append((time.perf_counter()-t)*1000)
    report['cpu_single_frame_latency_ms']={'median':float(np.median(times)),'p95':float(np.percentile(times,95)),'iterations':200,'threads':2}
    report['checkpoint_reload_verified']=True
    report['model_path']=str(target.resolve())
    for name,prediction in predictions.items(): np.savez_compressed(Path(output)/f'{kind}_seed{seed}_{name}_predictions.npz',**prediction)
    (target.with_suffix('.report.json')).write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'type':'completed','architecture':kind,'seed':seed,'best_epoch':best_epoch,'test_metrics':results['test'],'model_path':str(target.resolve())},ensure_ascii=False),flush=True)
    return report


def serve(path):
    torch.set_num_threads(2)
    bundle=ModelBundle(path)
    print(json.dumps({'type':'ready','model_path':str(bundle.path),'input_shape':[9,5] if bundle.kind=='sensor_token' else [6,4,3],'classes':26},ensure_ascii=False),flush=True)
    for line in __import__('sys').stdin:
        try:
            request=json.loads(line); labels,forces,p=bundle.predict(request['adc'])
            result={'type':'prediction','request_id':request.get('request_id'),'position':int(labels[0]),'confidence':float(p[0].max()),'probabilities':p[0].tolist(),'force':forces[0].tolist()}
        except Exception as exc:
            result={'type':'error','message':str(exc)}
        print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--serve',required=True); args=parser.parse_args(); serve(args.serve)
