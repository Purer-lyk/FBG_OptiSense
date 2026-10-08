"""Exercise the saved models through the real persistent subprocess API."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile
import numpy as np
from manual_touch_sensor_token import ModelBundle
ROOT=Path(__file__).resolve().parent
OUT=ROOT/'outputs/model_comparisons/sensor_token26_two_sessions_v1'

def main():
    data=np.load(OUT/'dataset.npz',allow_pickle=False)
    comparison=json.loads((OUT/'comparison.json').read_text(encoding='utf-8'))
    ids=[int(np.flatnonzero(data['labels']==0)[-1]),int(np.flatnonzero(data['labels']==18)[-1])]
    verification=[]
    for kind in ['sensor_token','sparse_cnn26']:
        path=Path(comparison['selected_models'][kind]['model_path'])
        requests=[{'adc':data['adc'][index].tolist(),'request_id':i+1} for i,index in enumerate(ids)]
        result=subprocess.run([sys.executable,'-m','manual_touch_sensor_token','--serve',str(path)],input=''.join(json.dumps(r)+'\n' for r in requests),capture_output=True,text=True,encoding='utf-8',env={**os.environ,'PYTHONIOENCODING':'utf-8'},cwd=ROOT,timeout=90)
        assert result.returncode==0,result.stderr
        response=[json.loads(line) for line in result.stdout.splitlines()]
        assert len(response)==3 and response[0]['type']=='ready'
        bundle=ModelBundle(path)
        for request,res,index in zip(requests,response[1:],ids):
            assert res['type']=='prediction' and res['request_id']==request['request_id']
            assert len(res['probabilities'])==26 and 0<=res['position']<=25
            assert np.isfinite(res['force']).all()
            assert abs(sum(res['probabilities'])-1)<1e-5
            predicted,force,p=bundle.predict(data['adc'][index])
            assert int(predicted[0])==res['position']
            np.testing.assert_allclose(res['force'],force[0],rtol=1e-5,atol=1e-5)
        verification.append({'architecture':kind,'ready_verified':True,'requests':2,'responses':response[1:],'subprocess_exit_code':result.returncode})
    (OUT/'serve_verification.json').write_text(json.dumps(verification,ensure_ascii=False,indent=2),encoding='utf-8')
    old=ROOT/'outputs/manual_touch_sequences/press_grid_20260927_113931/models/manual_touch_sparse_cnn.report.json'
    shutil.copy2(old,OUT/'historical_sparse_cnn25.report.json')
    archive=OUT/'SensorToken26_模型与对比完整结果.zip'
    output_files=[p for p in OUT.rglob('*') if p.is_file() and p.suffix!='.zip']
    code=['manual_touch_sensor_token.py','manual_touch_cnn.py','prepare_sensor_token_data.py','run_sensor_token_comparison.py','sensor_token_subprocess_tests.py','test_manual_touch_sensor_token.py','build_sensor_token_report.py','verify_sensor_token_artifacts.py']
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for path in output_files: z.write(path,Path('results')/path.relative_to(OUT))
        for name in code: z.write(ROOT/name,Path('code')/name)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        names=z.namelist()
        for name in ['results/sensor_token_selected.pt','results/sparse_cnn26_selected.pt','results/comparison.json','results/split.json']:
            assert name in names
    print(json.dumps({'model_serving_verified':True,'archive':str(archive.resolve()),'archive_bytes':archive.stat().st_size,'archive_files':len(names),'docx':str(next(OUT.glob('*.docx')).resolve()),'selected_model':comparison['selected_models']['sensor_token']['model_path']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
