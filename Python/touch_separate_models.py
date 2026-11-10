"""Independent position/contact and force networks, Torch-first subprocess only.

Legacy measured-feature implementations are reused, not their multitask forward.
Unused heads are deleted before optimizers are constructed. Force sees only the
45 measured dBm inputs; neither true nor predicted position is an input.
"""
import numpy as np
import torch
from torch import nn
import touch_six_models as six

SCHEMA = 'fbg_touch_separate26_v1'
ARCHITECTURES = ('tcn', 'sensor_token', 'sparse_cnn', 'inception_1d')
DEFAULT_CONFIG = dict(six.DEFAULT_CONFIG)
force_loss = six.force_loss


def _config(config):
    c = dict(DEFAULT_CONFIG, **config)
    if c['architecture'] not in ARCHITECTURES: raise ValueError('Unknown independent architecture: ' + str(c['architecture']))
    scales = np.asarray(c['axis_scales'], float)
    if scales.shape != (3,) or not np.isfinite(scales).all() or (scales <= 0).any(): raise ValueError('Expected three positive physical axis scales')
    if c['position_target'] != .95 or c['force_mae_target'] != .1: raise ValueError('Validation targets fixed at .95 accuracy and .1N each axis')
    return c


class FeatureModel(six.SixModel):
    def __init__(self, config):
        super().__init__(_config(config))


class PositionModel(FeatureModel):
    def __init__(self, config):
        super().__init__(config)
        del self.force_head
        del self.force_residual

    def forward(self, x):
        features = self.features(x)
        return self.position_head(features), self.contact_head(features).squeeze(-1)

class ForceModel(FeatureModel):
    def __init__(self, config):
        super().__init__(config)
        del self.position_head
        del self.contact_head

    def forward(self, x):
        flat = x.flatten(1)
        return self.force_head(torch.cat((self.features(x), flat), 1)) + self.force_residual(flat)

def build_models(position_config, force_config=None):
    pc = _config(position_config)
    fc = _config(force_config if force_config is not None else position_config)
    for c in (pc, fc):
        if c['architecture'] == 'tcn' and six.receptive_field(c) < 45:
            raise ValueError('TCN receptive field must cover all 45 measured points')
    torch.manual_seed(pc['seed'])
    position = PositionModel(pc)
    torch.manual_seed(fc['seed'])
    force = ForceModel(fc)
    return position, force

def position_loss(logits, contact, labels, contact_weights, config):
    ce = nn.functional.cross_entropy(logits, labels, label_smoothing=config['label_smoothing'])
    binary = nn.functional.binary_cross_entropy_with_logits(contact, (labels > 0).float(), reduction='none')
    return ce + config['contact_weight'] * (binary * contact_weights).mean()


def position_selection(accuracy, loss):
    """Accuracy first, then position/contact validation objective; no force."""
    return (float(accuracy), -float(loss))


def force_selection(mae, slope):
    """Feasible-first physical MAE and amplitude objective; no position."""
    values = np.array([v if v is not None and np.isfinite(v) else 1e6 for v in mae])
    ratios = values / .1
    violation = float(np.maximum(ratios - 1, 0).sum())
    objective = float(ratios.mean()) + (.1 * abs(slope - 1) if slope is not None else 0.)
    return (int(bool(np.all(values <= .1))), -violation, -objective)


class SeparateModelBundle(six.SixModelBundle):
    """Legacy six-model inference API, with independent weights/configurations."""
    def __init__(self, path):
        from pathlib import Path
        torch.set_num_threads(2)
        self.path = Path(path).resolve()
        payload = torch.load(self.path, map_location='cpu', weights_only=False)
        if payload.get('schema') != SCHEMA:
            raise ValueError('Unsupported independent position/force schema')
        self.position_config = payload['position_configuration']
        self.force_config = payload['force_configuration']
        self.config = payload['configuration']
        self.kind = 'separate_' + self.position_config['architecture'] + '_' + self.force_config['architecture']
        self.position_model, self.force_model = build_models(self.position_config, self.force_config)
        self.position_model.load_state_dict(payload['position_model_state'])
        self.force_model.load_state_dict(payload['force_model_state'])
        self.position_model.eval(); self.force_model.eval()
        self.mean = np.asarray(payload['feature_mean'], np.float32)
        self.std = np.asarray(payload['feature_std'], np.float32)
        self.force_mean = np.asarray(payload['force_mean'], np.float32)
        self.force_std = np.asarray(payload['force_std'], np.float32)
        self.stats = dict(feature_mean=self.mean, feature_std=self.std,
                          force_mean=self.force_mean, force_std=self.force_std)
        self.feedback_selector = int(payload['feedback_selector'])
        self.calibration = payload['calibration']; self.policy = self.calibration['policy']
        self.report = payload.get('report', {})

    def details_dbm(self, values):
        values = np.asarray(values, np.float32)
        if values.ndim == 1: values = values[None]
        if values.ndim != 2 or values.shape[1] != 45 or not np.isfinite(values).all():
            raise ValueError('Expected N x 45 finite dBm values')
        x = torch.from_numpy(((values.reshape(-1, 9, 5) - self.mean) / self.std).astype(np.float32))
        with torch.inference_mode():
            logits, contact = self.position_model(x)
            force = self.force_model(x)
            probabilities = (logits / self.calibration['position_temperature']).softmax(-1).numpy()
            contacts = (contact / self.calibration['contact_temperature']).sigmoid().numpy()
            forces = force.numpy() * self.force_std + self.force_mean
        return probabilities.argmax(-1), forces, probabilities, contacts


def axis_mask(forces, valid):
    valid = np.asarray(valid, bool)
    if valid.ndim == 1: valid = np.broadcast_to(valid[:, None], forces.shape).copy()
    if valid.shape != forces.shape: raise ValueError('Expected per-axis mask N x 3 or row mask')
    return valid & np.isfinite(forces)


def train_trial(data, split, config, output, feedback_selector):
    """Two optimizers, separate RNG streams and validation checkpoints; no test reads."""
    import copy, json, time
    from pathlib import Path
    from torch.utils.data import TensorDataset, DataLoader
    from touch_contact_guard import calibrate, decision_metrics
    c = _config(config)
    torch.set_num_threads(2)
    train = np.asarray(split['train'], np.int64); val = np.asarray(split['validation'], np.int64)
    indices = np.concatenate([train, val, np.asarray(split.get('test', []), np.int64)])
    if not len(train) or not len(val): raise ValueError('Nonempty train/validation required')
    if len(indices) != len(np.unique(indices)): raise ValueError('Split indices must be disjoint and unique')
    if (indices < 0).any() or (indices >= len(data['dbm'])).any(): raise ValueError('Split index out of range')
    if c['max_epochs'] < 1 or c['patience'] < 1: raise ValueError('Positive training budget required')
    def subset(idx):
        dbm = np.asarray(data['dbm'][idx], np.float32)
        labels = np.asarray(data['labels'][idx], np.int64)
        forces = np.asarray(data['forces'][idx], np.float32)
        if dbm.shape != (len(idx), 45) or not np.isfinite(dbm).all(): raise ValueError('Expected finite N x 45 measured dBm')
        if forces.shape != (len(idx), 3): raise ValueError('Expected N x 3 force')
        if labels.shape != (len(idx),) or (labels < 0).any() or (labels > 25).any(): raise ValueError('Expected labels 0..25')
        return dbm, labels, forces, axis_mask(forces, data['valid'][idx])
    td, ty, tf, tv = subset(train); vd, vy, vf, vv = subset(val)
    if not tv.any(): raise ValueError('No valid training force references')
    mean = td.mean(0).reshape(9, 5); std = td.std(0).reshape(9, 5); std[std < 1e-6] = 1
    fm = np.array([tf[tv[:, a], a].mean() if tv[:, a].any() else 0. for a in range(3)], np.float32)
    fs = np.array([tf[tv[:, a], a].std() if tv[:, a].any() else 1. for a in range(3)], np.float32); fs[fs < 1e-5] = 1
    bins = six.fit_force_bin_weights(tf, tv)
    def inputs(dbm): return torch.from_numpy(((dbm.reshape(-1, 9, 5) - mean) / std).astype(np.float32))
    tx, vx = inputs(td), inputs(vd)
    target = (ty > 0).astype(int); counts = np.bincount(target, minlength=2)
    cw = torch.from_numpy((len(train) / (np.count_nonzero(counts) * np.maximum(counts[target], 1))).astype(np.float32))
    val_target = (vy > 0).astype(int)
    vcw = torch.from_numpy((len(train) / (np.count_nonzero(counts) * np.maximum(counts[val_target], 1))).astype(np.float32))
    loader = DataLoader(TensorDataset(tx, torch.from_numpy(ty), torch.from_numpy(tf), torch.from_numpy(tv), cw),
                        batch_size=c['batch_size'], shuffle=True, generator=torch.Generator().manual_seed(c['seed']))
    pm, force = build_models(c)
    models = {'position': pm, 'force': force}
    opts = {k: torch.optim.AdamW(m.parameters(), lr=c['lr'], weight_decay=c['weight_decay']) for k, m in models.items()}
    schedulers = {k: torch.optim.lr_scheduler.CosineAnnealingLR(o, T_max=c['max_epochs'], eta_min=1e-5) for k, o in opts.items()}
    best = {k: dict(key=None, state=None, epoch=0, row=None) for k in models}
    active = dict(position=True, force=True); history = []; start = time.perf_counter()
    for epoch in range(1, c['max_epochs'] + 1):
        for m in models.values(): m.train()
        totals = dict(position=0., force=0.); samples = 0
        for batch_no, (bx, by, bf, bv, bw) in enumerate(loader):
            for kind, m in models.items():
                if not active[kind]: continue
                opts[kind].zero_grad(set_to_none=True)
                # Isolate dropout RNG too: position early stopping cannot alter force stochasticity.
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(c['seed'] + epoch * 100000 + batch_no * 2 + (kind == 'force'))
                    if kind == 'position':
                        logits, contact = m(bx)
                        loss = position_loss(logits, contact, by, bw, c)
                    else:
                        loss = force_loss(m(bx), bf, bv, fm, fs, bins, c)
                    loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 1.)
                opts[kind].step(); totals[kind] += float(loss.detach()) * len(by)
            samples += len(by)
        for m in models.values(): m.eval()
        with torch.inference_mode():
            logits, contact = pm(vx); pred = force(vx).numpy() * fs + fm
            accuracy = float((logits.argmax(-1).numpy() == vy).mean())
            ce = float(nn.functional.cross_entropy(logits, torch.from_numpy(vy)))
            pl = float(position_loss(logits, contact, torch.from_numpy(vy), vcw, dict(c, label_smoothing=0.)))
        mae = six.axis_mae(vf, pred, vv)
        diag = six.force_diagnostics(vy, vf, pred, vv)
        keys = dict(position=position_selection(accuracy, pl), force=force_selection(mae, diag['force_gain']['slopes'][2]))
        row = dict(epoch=epoch, validation_accuracy=accuracy, validation_cross_entropy=ce,
                   validation_position_loss=pl, validation_force_mae_n=mae,
                   validation_force_gain=diag['force_gain']['slopes'],
                   training_position_loss=totals['position']/samples, training_force_loss=totals['force']/samples,
                   position_active=active['position'], force_active=active['force'],
                   position_selection_key=list(keys['position']), force_selection_key=list(keys['force']))
        history.append(row)
        for kind, m in models.items():
            if not active[kind]: continue
            if best[kind]['key'] is None or keys[kind] > best[kind]['key']:
                best[kind] = dict(key=keys[kind], state=copy.deepcopy(m.state_dict()), epoch=epoch, row=copy.deepcopy(row))
            schedulers[kind].step()
            if epoch - best[kind]['epoch'] >= c['patience']: active[kind] = False
        if epoch == 1 or epoch % 10 == 0:
            print(json.dumps(dict(type='progress', trial=c['id'], **row), allow_nan=False), flush=True)
        if not any(active.values()): break
    for kind, m in models.items(): m.load_state_dict(best[kind]['state']); m.eval()
    with torch.inference_mode(): vl, vc = pm(vx)
    calibration = calibrate(vl.numpy(), vc.numpy(), vy)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    path = output / (c['id'] + '.pt')
    if path.exists(): raise RuntimeError('Refusing checkpoint overwrite: ' + str(path))
    report = dict(schema=SCHEMA, architecture=c['architecture'], configuration=c, config=c,
                  position_configuration=c.copy(), force_configuration=c.copy(),
                  class_labels=list(range(26)), input_unit='dBm', input_shape=[9,5],
                  normalization_source='train only', selection_source='validation only',
                  calibration_source='validation only', force_mask_granularity='per axis',
                  force_conditioning='45 measured dBm only; no true/predicted label',
                  force_postprocessing='none: no contact gating, clipping or test-fitted gain',
                  position_best_epoch=best['position']['epoch'], force_best_epoch=best['force']['epoch'],
                  best_epoch={'position':best['position']['epoch'], 'force':best['force']['epoch']},
                  epochs_completed=len(history), history=history,
                  position_selection_key=list(best['position']['key']), force_selection_key=list(best['force']['key']),
                  validation_force_mae_n=best['force']['row']['validation_force_mae_n'],
                  best_validation_accuracy=best['position']['row']['validation_accuracy'],
                  best_validation_cross_entropy=best['position']['row']['validation_cross_entropy'],
                  parameters={k:sum(p.numel() for p in m.parameters()) for k,m in models.items()},
                  train_count=len(train), validation_count=len(val), test_count=len(split.get('test', [])),
                  train_seconds=time.perf_counter()-start, force_bin_weights=bins.tolist(),
                  calibration=calibration, metrics={}, decision_metrics={}, force_diagnostics={},
                  deployment_ready=False, validation_warning=split.get('description','Random neighbors can cross sets; not independent-session generalization') + '; optical/force timestamps not synchronized.',
                  model_path=str(path.resolve()))
    for name, x, labels, true, valid in [('train',tx,ty,tf,tv),('validation',vx,vy,vf,vv)]:
        with torch.inference_mode():
            l, co = pm(x); f = force(x).numpy()*fs+fm
            probs = (l/calibration['position_temperature']).softmax(-1).numpy()
            contacts = (co/calibration['contact_temperature']).sigmoid().numpy()
        report['metrics'][name] = six.metrics(labels, probs.argmax(-1), true, f, valid)
        report['decision_metrics'][name] = decision_metrics(labels, probs, contacts, calibration['policy'])
        report['force_diagnostics'][name] = six.force_diagnostics(labels, true, f, valid)
        if name == 'validation':
            np.savez_compressed(output/(c['id']+'_validation_predictions.npz'), indices=val, labels=labels,
                                forces=f, true_forces=true, valid=valid, probabilities=probs,
                                contact_probabilities=contacts, logits=l.numpy(), contact_logits=co.numpy())
    payload = dict(schema=SCHEMA, architecture=c['architecture'], configuration=c,
                   position_configuration=c.copy(), force_configuration=c.copy(),
                   position_model_state=best['position']['state'], force_model_state=best['force']['state'],
                   feature_mean=mean, feature_std=std, force_mean=fm, force_std=fs,
                   force_bin_weights=bins, feedback_selector=int(feedback_selector), calibration=calibration,
                   input_unit='dBm', sensor_order=list(range(1,10)), class_labels=list(range(26)), report=report)
    torch.save(payload, path)
    path.with_suffix('.report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False), encoding='utf-8')
    return report
