"""Production 9x5 writer reader and explicit independent-task training CLI.
No Torch is imported by the Qt parent; no automatic deployment selection.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path
from touch_annotation_contract import require_single_position_annotation, require_single_position_session
import numpy as np
from temporary_optical_units import adc_code_to_dbm
ARCHITECTURES=('tcn','sensor_token','sparse_cnn','inception_1d')

def prepare_data(directory):
    require_single_position_session(directory)
    source=Path(directory)/'frames.jsonl'
    digest=hashlib.sha256();rows=[];contract=None;synthetic=False
    with source.open('rb') as stream:
        for line_number,raw in enumerate(stream,1):
            digest.update(raw)
            frame=json.loads(raw);q=frame.get('quality') or {}
            require_single_position_annotation(q)
            adc=np.asarray(q.get('stable_adc'),np.float32)
            records=q.get('stable_records') or []
            if adc.shape!=(45,) or q.get('points_per_peak',5)!=5 or q.get('point_count',45)!=45:
                raise ValueError('Expected exactly 9x5/45 measured second codes; no cropping/padding')
            if len(records)!=45 or not np.array_equal(adc,np.asarray([r['second_code'] for r in records],np.float32)):
                raise ValueError('45 complete second_code records required')
            selector=q.get('feedback_selector')
            if selector is None: raise ValueError('Missing acquisition feedback selector')
            waves=frame.get('wavelengths_nm')
            if len(waves or [])!=45: raise ValueError('Missing measured wavelength route')
            current=dict(input_shape=[9,5],point_count=45,points_per_peak=5,feedback_selector=int(selector),
                stable_sample_field='second_code',measurement_units='dBm',
                route_indices=[r.get('index') for r in records],route_wavelengths_nm=waves)
            if contract is None: contract=current
            elif current!=contract: raise ValueError('Route/selector changed within capture session')
            dbm=np.asarray(q.get('stable_dbm'),np.float32)
            if dbm.shape!=(45,) or not np.isfinite(dbm).all() or not np.allclose(dbm,adc_code_to_dbm(adc,selector,floor_nonpositive=True),rtol=0,atol=1e-4):
                raise ValueError('Saved 45 measured dBm differs from second-code conversion')
            label=q.get('press_label')
            if not isinstance(label,int) or not 0<=label<=25: raise ValueError('Expected annotated class 0..25')
            ref=q.get('paxini_force') or {}
            force=np.asarray([ref.get(k) for k in ('fx_n','fy_n','fz_n')],np.float32)
            clock=q.get('cycle_start_us')
            rows.append((dbm,label,force,line_number,float('nan') if clock is None else float(clock)/1e6))
            synthetic=synthetic or bool(q.get('smoke_only')) or 'synthetic' in str(q.get('frame_provenance','')).lower()
    if not rows: raise ValueError('No 45-point capture records')
    data=dict(dbm=np.stack([r[0] for r in rows]),labels=np.array([r[1] for r in rows]),
        forces=np.stack([r[2] for r in rows]),source_line=np.array([r[3] for r in rows]),
        timestamp_s=np.array([r[4] for r in rows]),session=np.zeros(len(rows),int),
        input_contract=contract,source_sha256=digest.hexdigest(),source_path=str(source.resolve()),smoke_only=synthetic)
    data['valid']=np.isfinite(data['forces'])
    return data,contract['feedback_selector']

def train(directory,output,*,epochs,architecture,force_history):
    import touch_temporal_models as models
    data,selector=prepare_data(directory)
    train_indices=[];validation=[]
    for label in np.unique(data['labels']):
        indices=np.flatnonzero(data['labels']==label)
        if len(indices)<2: raise ValueError('Each annotated class needs >=2 frames for train/validation')
        cut=max(1,int(len(indices)*.8))
        train_indices.extend(indices[:cut].tolist());validation.extend(indices[cut:].tolist())
    split=dict(train=sorted(train_indices),validation=sorted(validation),test=[])
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    frozen=dict(source_sha256=data['source_sha256'],input_contract=data['input_contract'],split=split,
        architecture_roster=list(ARCHITECTURES),selection_source='validation only',smoke_only=data['smoke_only'])
    path=output/'frozen_split.json'
    if path.exists() and json.loads(path.read_text('utf-8'))!=frozen: raise ValueError('Frozen source/split changed')
    path.write_text(json.dumps(frozen,indent=2),encoding='utf-8')
    report=models.train_trial(data,split,dict(id=architecture+('_history' if force_history else '_nohistory'),
        architecture=architecture,force_history=force_history,max_epochs=epochs,smoke_only=data['smoke_only']),output,selector)
    sidecar=Path(report['model_path']).with_suffix('.report.json')
    report.update(smoke_only=data['smoke_only'],input_contract=data['input_contract'],
        dataset_dir=str(Path(directory).resolve()),source_sha256=data['source_sha256'],
        checkpoint_sha256=hashlib.sha256(Path(report['model_path']).read_bytes()).hexdigest())
    sidecar.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    return report

def cli(argv=None):
    parser=argparse.ArgumentParser()
    parser.add_argument('--train',type=Path,required=True)
    parser.add_argument('--output',type=Path)
    parser.add_argument('--epochs',type=int,default=160)
    parser.add_argument('--architecture',choices=ARCHITECTURES,default='sensor_token')
    parser.add_argument('--force-history',action='store_true')
    args=parser.parse_args(argv)
    report=train(args.train,args.output or args.train/'models_45_candidates'/str(time.time_ns()),
        epochs=args.epochs,architecture=args.architecture,force_history=args.force_history)
    path=Path(report['model_path'])
    print(json.dumps(dict(type='completed',model_path=str(path),report_path=str(path.with_suffix('.report.json')))),flush=True)
if __name__=='__main__':cli()
