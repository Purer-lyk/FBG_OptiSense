"""Create report sidecars for selected checkpoints without retraining."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/model_comparisons/sensor_token26_two_sessions_v1'
comparison=json.loads((OUT/'comparison.json').read_text(encoding='utf-8'))
for kind,item in comparison['selected_models'].items():
    model=Path(item['model_path'])
    report=json.loads(Path(item['report_path']).read_text(encoding='utf-8'))
    report.update(model_path=str(model.resolve()),input_shape=[9,5] if kind=='sensor_token' else [6,4,3],class_labels=list(range(26)),validation_warning='逐类连续分块开发评估；不是独立会话测试。0号空载只来自10-08，1～25号只来自09-27，存在会话混淆与归零差异。')
    target=model.with_suffix('.report.json')
    if target.exists():
        raise RuntimeError(f'Refusing to overwrite existing sidecar {target}')
    target.write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps({'architecture':kind,'model':str(model),'report':str(target)},ensure_ascii=False))
