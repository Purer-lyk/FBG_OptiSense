"""Real unified Qt synthetic writer replay. Never opens serial/laser/CNC."""
import os,sys,json,time
from pathlib import Path
from unittest.mock import patch
ROOT=Path(__file__).resolve().parent;os.chdir(ROOT);sys.path.insert(0,str(ROOT))
OUT=Path(sys.argv[1]);OUT.mkdir(parents=True,exist_ok=True)
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=str(Path(sys.prefix)/'Lib/site-packages/PyQt5/Qt5/plugins/platforms')
from PyQt5 import QtCore,QtWidgets,QtGui
from test_fbg_unified_app import UnifiedStudioTests
from test_manual_stress_capture import temporary_frame
QtCore.QCoreApplication.setLibraryPaths([str(Path(sys.prefix)/'Lib/site-packages/PyQt5/Qt5/plugins')])
app=QtWidgets.QApplication([])
QtGui.QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc');app.setFont(QtGui.QFont('Microsoft YaHei',10))
errors=[];sys.excepthook=lambda *args:errors.append(str(args[1]))
fixture=UnifiedStudioTests();fixture.app=app
with patch('serial.Serial',side_effect=AssertionError('Hardware access forbidden')),patch('paxini_px6d.PaxiniForceSensor.start',side_effect=AssertionError('Sensor auto-start forbidden')):
 fixture.setUp();w=fixture.window;w.resize(1800,1100)
 try:
  for index,name in ((0,'main_navigation'),(5,'manual_capture'),(6,'training_disabled')):
   w.nav_buttons[index].click()
   for _ in range(12):app.processEvents()
   assert w.workspace_tabs.currentIndex()==index
   assert w.grab().save(str(OUT/(name+'.png')))
  capture=w.manual_capture;source=capture.frame_source;capture.output_root=OUT/'synthetic_capture'
  source.plan={'rows':[{'measured_nm':1530+i*.01,'index':i} for i in range(45)]}
  # Two instances of every class, real production writer and listener dispatch.
  for i in range(52):
   label=i%26
   capture.select_no_press() if label==0 else capture.select_grid_point(label)
   frame=temporary_frame(sequence=100+i,base=200+i)
   frame.update(provenance='synthetic smoke only NOT hardware',cycle_start_us=1_000_000+i*100_000)
   sensor=w.touch_force_sensor
   if i%7:
    sensor._latest=(i*.01,i*.02,-i*.03,0.,0.,0.);sensor._latest_ns=time.monotonic_ns();sensor.frames_received+=1
   else:sensor._latest=None;sensor._latest_ns=None
   source.update_frame(frame)
   assert capture.latest_frame is frame
   assert capture.capture_current_press(),capture.capture_status_label.text()
  folder=capture.session_dir;capture.close_session()
  manifest=folder/'manifest.json';payload=json.loads(manifest.read_text('utf-8'));payload['metadata'].update(synthetic=True,smoke_only=True,hardware_used=False);manifest.write_text(json.dumps(payload,ensure_ascii=False,indent=2),encoding='utf-8')
  records=[json.loads(l) for l in (folder/'frames.jsonl').read_text('utf-8').splitlines()]
  assert len(records)==52 and all(len(r['quality']['stable_records'])==45 for r in records)
  assert records[0]['quality']['paxini_force'] is None
  assert all('first_code' not in json.dumps(r) for r in records)
  assert not w.touch_force_sensor._running and 'torch' not in sys.modules
  report=dict(smoke_only=True,hardware_used=False,session=str(folder.resolve()),frames=52,input_shape=[9,5],
    parent_torch_free=True,shared_sensor_idle=True,default_model=None,screenshots=['main_navigation.png','manual_capture.png','training_disabled.png'])
  (OUT/'qt_writer_replay.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
 finally:
  fixture.tearDown();fixture.doCleanups()
 for _ in range(20):app.processEvents()
 assert not errors,errors
 assert not capture.force_timer.isActive() and not capture.link_timer.isActive()
 assert source.frame_listeners==[]
 print(json.dumps(report,ensure_ascii=False))
