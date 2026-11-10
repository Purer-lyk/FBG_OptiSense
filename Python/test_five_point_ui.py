"""Hardware-free Qt migration/disabled deployment regressions."""
import os,sys,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=str(Path(sys.prefix)/'Lib/site-packages/PyQt5/Qt5/plugins/platforms')
from PyQt5 import QtCore,QtWidgets
from manual_touch_training_page import ManualTouchTrainingPage
from temporary_test_widget import TemporaryTestWindow
import test_temporary_test_widget as existing
from test_five_point_migration import selected

class FiveUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        QtCore.QCoreApplication.setLibraryPaths([str(Path(sys.prefix)/'Lib/site-packages/PyQt5/Qt5/plugins')])
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_disabled_pin_blocks_latest_and_session_report_fallback(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); run=root/'model_comparisons/run';run.mkdir(parents=True)
            model=run/'old.pt';model.write_bytes(b'not a model, must never load')
            model.with_suffix('.report.json').write_text(json.dumps({'model_path':str(model)}))
            (run/'comparison.json').write_text(json.dumps({'selected_models':{'sensor_token':{'model_path':str(model)}}}))
            (root/'touch_deployment_selection.json').write_text(json.dumps({'schema':'fbg_touch_deployment_selection_v2','state':'disabled','required_input_shape':[9,5],'reason':'等待重新采集训练'}))
            with patch.object(ManualTouchTrainingPage,'_load_existing_report_if_present') as fallback:
                page=ManualTouchTrainingPage(default_dataset_dir=root/'data',model_search_root=root)
                try:
                    self.assertIsNone(page.model_path)
                    self.assertFalse(page.start_live_button.isEnabled())
                    self.assertIn('停用',page.live_status_label.text())
                    self.assertIn('9×5',page.live_status_label.text())
                    fallback.assert_not_called()
                finally: page.close()

    def test_real_temporary_page_dispatches_all_fortyfive_measured_records(self):
        from test_manual_stress_capture import temporary_frame
        sandbox=existing.WidgetTests();sandbox.setUp();self.addCleanup(sandbox.doCleanups)
        window=TemporaryTestWindow(None)
        try:
            window.plan=selected()[-1]
            received=[]
            self.assertTrue(hasattr(window,'frame_listeners'))
            window.frame_listeners.append(received.append)
            frame=temporary_frame();window.update_frame(frame)
            self.assertEqual(len(received),1)
            self.assertIs(received[0],frame)
            self.assertEqual(len(received[0]['records']),45)
            bad=dict(frame,records=frame['records']+[{'second_code':1}]*18)
            window.update_frame(bad)
            self.assertEqual(len(received),1)
        finally: window.close()

    def test_auto_selection_ui_displays_five_and_all_fortyfive_rows(self):
        table,x,y,plan=selected()
        with tempfile.TemporaryDirectory() as folder:
            sandbox=existing.WidgetTests(); sandbox.setUp()
            self.addCleanup(sandbox.doCleanups)
            with patch('app_JDSU.load_fullband_accuracy_table',return_value=table):
                window=TemporaryTestWindow(None)
                try:
                    window.dense_wavelengths=x;window.dense_values=y;window.dense_feedback_selector=2
                    self.assertTrue(window.auto_select_points())
                    self.assertEqual(window.point_count(),45)
                    self.assertEqual(window.points_per_peak(),5)
                    self.assertEqual(window.table.rowCount(),45)
                    self.assertIn('9峰×5点',window.status.text())
                    self.assertNotIn('63点',window.status.text())
                finally: window.close()

if __name__=='__main__': unittest.main(verbosity=2)
