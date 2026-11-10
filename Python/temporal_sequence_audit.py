"""Torch-free sequence provenance. Never infer boundaries from supervision."""
import numpy as np
import json,hashlib
from pathlib import Path

def verify_source_hashes(audit):
    for source in audit['sources']:
        h=hashlib.sha256()
        with Path(source['path']).open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''): h.update(chunk)
        if h.hexdigest()!=source['sha256']: raise RuntimeError('Source hash mismatch: '+source['path'])

def attach_sequence_metadata(data,audit):
    verify_source_hashes(audit)
    n=len(data['session']); samples=audit['samples']
    if sorted(s['index'] for s in samples)!=list(range(n)): raise ValueError('Accepted mapping is not complete unique index coverage')
    data['source_line']=np.zeros(n,dtype=np.int64)
    data['timestamp_s']=np.full(n,np.nan)
    data['source_frame_sequence']=np.full(n,-1,dtype=np.int64)
    data['record_sequence']=np.full(n,-1,dtype=np.int64)
    data['forces']=np.full((n,3),np.nan,dtype=np.float32)
    reports=[]
    for sid,source in enumerate(audit['sources']):
        path=Path(source['path']); manifest_path=path.parent/'manifest.json'
        manifest=json.loads(manifest_path.read_text(encoding='utf-8')) if manifest_path.exists() else {}
        wanted={s['source_line']:s['index'] for s in samples if s['session']==sid}
        if len(wanted)!=source['accepted']: raise ValueError('Accepted source count/mapping mismatch')
        known_clock=manifest.get('timebase')=='time.monotonic_ns'
        raw_t=[]; raw_seq=[]; seen=set(); source_seq=[]
        with path.open(encoding='utf-8') as stream:
            for line_no,line in enumerate(stream,1):
                f=json.loads(line); q=f.get('quality') or {}
                try: t=float(f.get('monotonic_ns'))/1e9 if known_clock else np.nan
                except (ValueError,TypeError): t=np.nan
                raw_t.append(t); raw_seq.append(f.get('sequence'))
                if line_no not in wanted: continue
                i=wanted[line_no]; seen.add(line_no)
                if int(data['session'][i])!=sid: raise ValueError('Session mapping mismatch')
                if 'press_label' in q and int(data['labels'][i])!=q['press_label']: raise ValueError('Label mapping mismatch')
                views=q.get('training_views') or {}
                for field,reference in [('dbm',q.get('stable_dbm',views.get('stable_optical_power_dbm'))),('adc',q.get('stable_adc',views.get('stable_adc')))]:
                    if field in data:
                        real=np.asarray(reference,dtype=np.float32)
                        if real.shape!=(45,) or not np.isfinite(real).all() or not np.array_equal(np.asarray(data[field][i],dtype=np.float32),real):
                            raise ValueError('Cached optical input differs from accepted raw source: '+field)
                data['source_line'][i]=line_no; data['timestamp_s'][i]=t
                for field,key in [('source_frame_sequence',q.get('source_frame_sequence')),('record_sequence',f.get('sequence'))]:
                    if isinstance(key,int): data[field][i]=key
                source_seq.append(int(data['source_frame_sequence'][i]))
                ref=q.get('paxini_force') or {}
                for axis,key in enumerate(('fx_n','fy_n','fz_n')):
                    try: data['forces'][i,axis]=float(ref.get(key))
                    except (ValueError,TypeError): pass
        if seen!=set(wanted): raise ValueError('Accepted source lines missing')
        finite=np.asarray(raw_t); diffs=np.diff(finite); mask=np.isfinite(finite[:-1]) & np.isfinite(finite[1:])
        reports.append(dict(session=sid,path=str(path),manifest=manifest,
            manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest() if manifest_path.exists() else None,
            timestamp_provenance='frame.monotonic_ns (capture-record clock; not synchronized optical/force acquisition)' if known_clock else 'missing/unknown clock; NaN fallback to source-line order',
            raw_timestamp_missing=int((~np.isfinite(finite)).sum()),raw_timestamp_nonincreasing=int((diffs[mask]<=0).sum()),
            raw_timestamp_duplicate=int((diffs[mask]==0).sum()),raw_timestamp_reversed=int((diffs[mask]<0).sum()),
            positive_gap_seconds_percentiles=np.percentile(diffs[mask & (diffs>0)],[50,95,99,100]).tolist() if (mask & (diffs>0)).any() else [],
            accepted_source_sequence_nonincreasing=sum(b<=a for a,b in zip(source_seq,source_seq[1:]) if a>=0 and b>=0),
            record_sequence_nonincreasing=sum(b<=a for a,b in zip(raw_seq,raw_seq[1:]) if isinstance(a,int) and isinstance(b,int))))
    data['valid']=np.isfinite(data['forces'])
    return dict(sources=reports,boundary_inputs=['session','source_line','timestamp_s'],
        warnings=['Missing clocks use source-line order, never fabricated timestamps.',
            'Recorded monotonic timestamp is not proof of synchronized optical/force measurement.',
            'Label and true-force values are supervision only, never sequence boundaries.'])


def continuous_runs(data,indices,gap_s=30.):
    if not np.isfinite(gap_s) or gap_s<=0: raise ValueError('gap_s must be positive')
    indices=[int(i) for i in indices]
    if len(set(indices))!=len(indices): raise ValueError('Duplicate split index')
    if any(i<0 or i>=len(data['session']) for i in indices): raise ValueError('Split index out of bounds')
    ordered=sorted(indices,key=lambda i:(int(data['session'][i]),int(data['source_line'][i])))
    runs=[]
    for i in ordered:
        cut=not runs
        if runs:
            p=runs[-1][-1]
            cut=(data['session'][p]!=data['session'][i] or data['source_line'][i]!=data['source_line'][p]+1)
            t0,t1=data['timestamp_s'][p],data['timestamp_s'][i]
            if np.isfinite(t0) and np.isfinite(t1): cut=cut or t1<=t0 or t1-t0>gap_s
        if cut: runs.append([])
        runs[-1].append(i)
    return runs

def chronological_split(data,purge=16,history_length=16,gap_s=30.):
    if purge<history_length: raise ValueError('purge must cover history_length')
    result=dict(train=[],validation=[],test=[],purged=[],protocol='chronological',purge=purge,
                description='Source-ordered 70/15/15 within each continuous capture segment; first purge frames removed after each cut; not independent presses.')
    for run in continuous_runs(data,range(len(data['session'])),gap_s):
        a=int(len(run)*.7); b=int(len(run)*.85)
        result['train'].extend(run[:a])
        result['validation'].extend(run[min(a+purge,b):b])
        result['test'].extend(run[min(b+purge,len(run)):])
        result['purged'].extend(run[a:min(a+purge,b)]+run[b:min(b+purge,len(run))])
    return result

def split_sequence_audit(data,split,history_length=16,gap_s=30.):
    seen=set(); result={}
    for name in ('train','validation','test'):
        indices=split[name]
        if seen.intersection(indices): raise ValueError('Overlapping splits')
        seen.update(indices)
        runs=continuous_runs(data,indices,gap_s); lengths=[len(r) for r in runs]
        histories=[min(j,history_length) for r in runs for j in range(len(r))]
        count=len(indices); classes=sorted(set(int(data['labels'][i]) for i in indices))
        result[name]=dict(frames=count,empty=not count,runs=len(runs),run_length_histogram={str(n):lengths.count(n) for n in sorted(set(lengths))},
            max_run_length=max(lengths,default=0),cold_start_frames=len(runs),cold_start_fraction=len(runs)/count if count else None,
            full_history_frames=sum(h==history_length for h in histories),history_depth_histogram={str(n):histories.count(n) for n in sorted(set(histories))},
            classes=classes,missing_classes=sorted(set(range(26))-set(classes)),runs_indices=runs)
    return result
