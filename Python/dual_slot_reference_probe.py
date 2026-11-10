"""Independent frozen adapters for synthetic validation-context parity; no external data."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'outputs/delta9_search_20261015_094820'))
import selected_bundle
sys.path.insert(0,str(ROOT/'outputs/delta9_groupcv_20261015_101629'))
from verified_bundle import Bundle
pair=json.loads(Path(sys.argv[1]).read_text('utf-8'));requests=json.loads(Path(sys.argv[2]).read_text('utf-8'))
models={t:(Bundle(pair[t]['model_path']) if pair[t]['adapter']=='groupcv' else selected_bundle.Bundle(pair[t]['model_path'])) for t in ('position','force')}
rows=[]
for x in requests:
 p=models['position'].predict(x);f=models['force'].predict(x)
 rows.append(dict(probabilities=p['probabilities'][0],position_labels=p['positions'][0],force=f['force_n'][0]))
print(json.dumps(rows))
