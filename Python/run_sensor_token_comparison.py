"""Reproducible two-session benchmark; never alter source datasets or old models."""
import json
from pathlib import Path
import shutil
import numpy as np
import torch
from manual_touch_sensor_token import blocked_split, train_one
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/model_comparisons/sensor_token26_two_sessions_v1'

def main():
    torch.set_num_threads(2)
    cache=np.load(OUT/'dataset.npz',allow_pickle=False)
    data={k:cache[k] for k in cache.files}
    split=blocked_split(data['labels'])
    (OUT/'split.json').write_text(json.dumps(split,indent=2),encoding='utf-8')
    print(json.dumps({'type':'split','counts':{k:len(split[k]) for k in ['train','validation','test','purged']},'all_classes':np.unique(data['labels']).tolist()},ensure_ascii=False),flush=True)
    results=[]
    seeds=[20261008,20261009,20261010]
    for seed in seeds:
        for kind in ['sensor_token','sparse_cnn26']:
            report_path=OUT/f'{kind}_seed{seed}.report.json'
            if report_path.exists():
                raise RuntimeError(f'Report already exists: {report_path}; no silent reuse')
            results.append(train_one(data,split,kind,seed,OUT,epochs=180,patience=30))
    selection={}; aggregate={}
    for kind in ['sensor_token','sparse_cnn26']:
        reports=[r for r in results if r['architecture']==kind]
        best=max(reports,key=lambda r:r['history'][r['best_epoch']-1]['selection_score'])
        destination=OUT/f'{kind}_selected.pt'
        shutil.copy2(best['model_path'],destination)
        selected_report = dict(best)
        selected_report.update(model_path=str(destination.resolve()), input_shape=[9,5] if kind=='sensor_token' else [6,4,3], class_labels=list(range(26)), validation_warning='逐类连续分块开发测试；空载/按压来自不同会话，不代表独立会话泛化性能。')
        destination.with_suffix('.report.json').write_text(json.dumps(selected_report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
        selection[kind]={'seed':best['seed'],'criterion':'highest validation selection_score across three seeds, not test result','model_path':str(destination.resolve()),'report_path':str((OUT/f"{kind}_seed{best['seed']}.report.json").resolve())}
        aggregate[kind]={}
        for key in ['position_accuracy','macro_f1_present_classes','balanced_accuracy_present_classes','weighted_f1','pressed_only_accuracy','no_press_recall','no_press_precision','force_mae_n','force_rmse_n','force_r2','force_vector_mae_n','force_vector_rmse_n']:
            values=np.asarray([r['metrics']['test'][key] for r in reports],dtype=float)
            aggregate[kind][key]={'mean':values.mean(0).tolist(),'std_sample':values.std(0,ddof=1).tolist()}
    summary={'schema':'fbg_sensor_token_vs_sparse_cnn26_comparison_v1','seeds':seeds,'split_counts':{k:len(split[k]) for k in ['train','validation','test','purged']},'split_description':split['description'],'aggregate_test_metrics':aggregate,'selected_models':selection,'runs':results,'warnings':['Development chronological within-label holdout, not independent-cycle deployment accuracy.','Session 20260927 contains only pressed labels 1..25; session 20261008 contains only label 0. Session and contact state are confounded.','Historical CNN 25-class validation on 1936/484 differs from current test; no direct percentage-point claims are valid.','Never infer zero force from no-press label; use only measured force references.'],'deployment_ready':False}
    (OUT/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'type':'all_completed','output':str(OUT.resolve()),'aggregate_test_metrics':aggregate,'selected_models':selection},ensure_ascii=False),flush=True)

if __name__=='__main__': main()
