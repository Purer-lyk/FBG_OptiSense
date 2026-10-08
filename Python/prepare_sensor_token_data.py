import json
from pathlib import Path
from manual_touch_sensor_token import load_sessions
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs'/'model_comparisons'/'sensor_token26_two_sessions_v1'
OUT.mkdir(parents=True,exist_ok=True)
if (OUT/'dataset.npz').exists():
    raise RuntimeError('Compact experiment dataset already exists; do not overwrite it')
paths=[ROOT/'outputs/manual_touch_sequences'/name for name in ['press_grid_20260927_113931','press_grid_20261008_151635']]
data,audit=load_sessions(paths,OUT)
print(json.dumps({'output':str(OUT),'count':len(data['labels']),'sources':audit},ensure_ascii=False,indent=2))
