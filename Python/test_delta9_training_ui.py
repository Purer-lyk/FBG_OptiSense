import os,sys,unittest,time,tempfile
from pathlib import Path
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=str(Path(sys.prefix)/'Lib/site-packages/PyQt5/Qt5/plugins/platforms')
from PyQt5 import QtWidgets,QtCore
from manual_touch_training_page import ManualTouchTrainingPage,ForceCurveComparison,TouchTrainingWorker,PerformanceFingerMap
class Delta9UiTests(unittest.TestCase):
 @classmethod
 def setUpClass(cls):cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
 def test_vertical_linked_compact_curves(self):
  c=ForceCurveComparison();c.resize(800,450);c.show();self.app.processEvents()
  plots=list(c.plots.values());self.assertTrue(plots[0].width()>700)
  self.assertEqual(plots[0].backgroundBrush().color().name(),'#15191f')
  self.assertFalse(plots[1].getAxis('left').autoSIPrefix)
  self.assertLess(plots[0].y(),plots[1].y());self.assertLess(plots[1].y(),plots[2].y());self.assertLess(c.minimumSizeHint().height(),490);c.close()
 def test_position_priority_and_tare_text_fits(self):
  from PyQt5 import QtTest
  from hyperos_theme import install_theme
  install_theme(self.app)
  p=ManualTouchTrainingPage(default_dataset_dir=Path('synthetic'));p.show()
  try:
   for width,height in ((1100,650),(1376,890)):
    p.resize(width,height);QtTest.QTest.qWait(100)
    sizes=p.results_splitter.sizes();self.assertGreaterEqual(sizes[0]/sum(sizes),.40)
    b=p.actual_force.zero_button
    self.assertGreaterEqual(b.width(),b.fontMetrics().horizontalAdvance(b.text())+24)
    self.assertGreaterEqual(b.height(),b.fontMetrics().height()+12)
    self.assertTrue(p.actual_force.rect().contains(b.geometry()))
  finally:p.shutdown();p.close()
 def test_prediction_caption_bounds_and_contact_states(self):
  p=ManualTouchTrainingPage(default_dataset_dir=Path('synthetic'))
  try:
   p.live_position_label.resize(300,30)
   p._set_position_caption('按压 · 位置集合 '+', '.join(map(str,range(1,26))), 'pressed')
   self.assertIn('25',p.live_position_label.toolTip());self.assertLessEqual(p.live_position_label.fontMetrics().horizontalAdvance(p.live_position_label.text()),300)
   p._set_position_caption('不确定／接触状态待确认','uncertain');self.assertIn('64748B',p.live_position_label.styleSheet())
  finally:p.shutdown();p.close()
 def test_position_camera_fits_tall_shape_after_widening(self):
  from PyQt5 import QtTest
  m=PerformanceFingerMap();m.resize(480,245);m.show();QtTest.QTest.qWait(100)
  try:
   points=[m._project_to_widget(v) for v in m.direction_vertices]
   ys=[v[1] for v in points if v is not None]
   self.assertGreaterEqual(min(ys),0);self.assertLessEqual(max(ys),m.height());self.assertGreaterEqual(max(ys)-min(ys),m.height()*.80)
  finally:m.close()
 def test_worker_routes_delta9(self):
  worker=TouchTrainingWorker('synthetic');self.assertIn('touch_delta9_models',worker.command())
 def test_page_baseline_reminder_does_not_set_baseline(self):
  p=ManualTouchTrainingPage(default_dataset_dir=Path('synthetic'))
  self.assertTrue(hasattr(p,'baseline_button'));self.assertFalse(p._set_deployment_baseline())
  before=p.deployment_wavelength_state.baseline;p._baseline_due=0;p._refresh_baseline_reminder()
  self.assertIs(p.deployment_wavelength_state.baseline,before);self.assertIn('空载',p.baseline_status.text())
  self.assertTrue(p.reminder_minutes.value()==5);self.assertFalse('torch' in sys.modules)
  p.shutdown();p.close()
 def test_baseline_rejects_replaced_snapshot_and_context(self):
  from test_delta9_capture import Delta9CaptureTests
  fixture=Delta9CaptureTests();fixture.setUp()
  p=ManualTouchTrainingPage(default_dataset_dir=Path(fixture.tmp.name),frame_source=fixture.source,capture_window=fixture.p)
  try:
   f=fixture.send(16);p._receive_live_frame(f);self.assertTrue(p._set_deployment_baseline())
   p._live_samples.append(2.);p._live_predicted.append((1.,2.,3.));p._live_actual_force.append((1.,2.,3.));p._live_raw_actual_force.append(None)
   self.assertTrue(p._set_deployment_baseline());self.assertFalse(p._live_samples)
   capture_before=fixture.p.wavelength_state.baseline
   fixture.source.makima_frame_snapshot=None
   self.assertFalse(p._set_deployment_baseline())
   self.assertEqual(fixture.p.wavelength_state.baseline,capture_before)
   f=fixture.send(17);p._receive_live_frame(f);fixture.source.plan['rows'][0]['index']+=1
   self.assertFalse(p._set_deployment_baseline());self.assertIsNone(p.deployment_wavelength_state.baseline)
  finally:p.shutdown();p.close();fixture.doCleanups()
 def test_compact_map_all25_projected_hit_targets(self):
  from PyQt5 import QtTest
  m=PerformanceFingerMap();m.setStyleSheet('QPushButton { min-height:28px; padding:6px; }');m.resize(220,250);m.show();QtTest.QTest.qWait(100);m._position_buttons()
  try:
   for i,point in enumerate(m.marker_positions,1):
    x,y=m._project_to_widget(point);child=m.childAt(QtCore.QPoint(round(x),round(y)))
    self.assertIs(child,m.group.button(i),i)
  finally:m.close()
 def test_prediction_halo_and_neutral_surface(self):
  m=PerformanceFingerMap()
  try:
   self.assertTrue(hasattr(m,'prediction_halo_item'))
   m.show_live_prediction([2,22]);self.assertEqual(len(m.prediction_halo_item.pos),2)
   colors=m.silicone_item.opts['meshdata'].vertexColors()[:,:3]
   self.assertTrue(((colors.max(axis=1)-colors.min(axis=1))<.12).all())
   m.show_live_prediction(None);self.assertEqual(len(m.prediction_halo_item.pos),0)
  finally:m.close()
 def test_map_multilabel_original25(self):
  m=PerformanceFingerMap();m.show_live_prediction([2,22]);self.assertEqual(m.live_selected_labels,[2,22]);self.assertFalse(m.number_labels);m.close()
if __name__=='__main__':unittest.main()
