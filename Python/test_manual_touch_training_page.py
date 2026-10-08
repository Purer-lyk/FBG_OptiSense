"""Tests for the fifth sparse-CNN training and evaluation workspace."""

import os
import sys
import tempfile
import unittest
import subprocess
import time
from unittest.mock import patch
from pathlib import Path

qt_platform_plugins = (
    Path(sys.prefix) / "Lib" / "site-packages" / "PyQt5" / "Qt5" / "plugins" / "platforms"
)
if qt_platform_plugins.is_dir():
    sys.path.append(str(qt_platform_plugins.parent))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5 import QtCore, QtWidgets  # noqa: E402

from manual_touch_training_page import ManualTouchTrainingPage, TouchInferenceWorker  # noqa: E402


class ManualTouchTrainingPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if qt_platform_plugins.is_dir():
            QtCore.QCoreApplication.setLibraryPaths([str(qt_platform_plugins.parent)])
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_dbm_model_description_matches_saved_configuration(self):
        from touch_model_catalog import model_description
        report={'architecture':'sensor_token_dbm','input_unit':'dBm','configuration':{'layers':1,'heads':4,'width':48}}
        description=model_description(report)
        self.assertIn('dBm',description)
        self.assertIn('26类',description)
        self.assertIn('1层',description)
        self.assertIn('ADC',description)

    def test_latest_token_model_autoloads_and_displays_test_metrics(self):
        import json
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            run=root/'model_comparisons'/'token_run'
            run.mkdir(parents=True)
            model=run/'sensor_token_selected.pt'
            model.write_bytes(b'test-model-placeholder')
            report={'schema':'fbg_touch26_v1','architecture':'sensor_token','train_count':1512,'validation_count':383,'test_count':383,'best_epoch':113,'metrics':{'test':{'position_accuracy':.6136,'force_mae_n':[.403,.362,1.161],'force_vector_mae_n':1.416}},'model_path':'wrong-seed.pt'}
            model.with_suffix('.report.json').write_text(json.dumps(report),encoding='utf-8')
            (run/'comparison.json').write_text(json.dumps({'selected_models':{'sensor_token':{'model_path':str(model)}}}),encoding='utf-8')
            page=ManualTouchTrainingPage(default_dataset_dir=root/'manual_touch_sequences',model_search_root=root)
            try:
                self.assertEqual(page.model_path,model.resolve())
                self.assertEqual(page.live_model_edit.text(),str(model.resolve()))
                self.assertTrue(page.start_live_button.isEnabled())
                self.assertIn('Sensor Token',page.input_shape_label.text())
                self.assertIn('26',page.input_shape_label.text())
                self.assertIn('测试集',page.overall_metrics_label.text())
                self.assertIn('61.36%',page.overall_metrics_label.text())
                self.assertFalse(page.live_render_timer.isActive())
                self.assertIsNone(page.inference)
            finally:
                page.close()

    def test_no_press_prediction_and_expected_state_are_distinct_from_no_comparison(self):
        with tempfile.TemporaryDirectory() as folder:
            page=ManualTouchTrainingPage(default_dataset_dir=Path(folder))
            try:
                self.assertIsNone(page.expected_position_combo.itemData(0))
                page.expected_position_combo.setCurrentIndex(page.expected_position_combo.findData(0))
                self.assertEqual(page._live_expected_label,0)
                page._live_prediction({'position':0,'confidence':.9,'force':[0.1,0.2,-.6]})
                self.assertIn('无按压',page.live_position_label.text())
                self.assertIn('期望',page.live_position_label.text())
            finally:
                page.close()

    def test_page_exposes_sparse_cnn_controls_and_25_point_results(self):
        with tempfile.TemporaryDirectory() as folder:
            page = ManualTouchTrainingPage(default_dataset_dir=Path(folder))
            try:
                self.assertIn("4×3", page.input_shape_label.text())
                self.assertIn("Mask", page.input_shape_label.text())
                self.assertEqual(page.performance_table.rowCount(), 25)
                self.assertEqual(page.actual_force.title, "真实平均三维力")
                self.assertEqual(page.predicted_force.title, "CNN预测三维力")
                self.assertFalse(page.cancel_button.isEnabled())
            finally:
                page.close()

    def test_report_colors_finger_points_and_displays_actual_vs_predicted_force(self):
        with tempfile.TemporaryDirectory() as folder:
            page = ManualTouchTrainingPage(default_dataset_dir=Path(folder))
            report = {
                "sample_count": 250,
                "train_count": 200,
                "validation_count": 50,
                "position_accuracy": 0.8,
                "force_mae_n": [0.1, 0.2, 0.3],
                "force_vector_mae_n": 0.4,
                "best_epoch": 12,
                "validation_warning": "仅有部分编号具备独立重复采集轮次，当前指标只用于开发评估。",
                "per_point": [
                    {
                        "label": label,
                        "support": 2,
                        "position_accuracy": label / 25.0,
                        "actual_force_mean": [label * 0.1, -0.2, -label * 0.3],
                        "predicted_force_mean": [label * 0.1 + 0.05, -0.1, -label * 0.3 + 0.1],
                        "force_mae": [0.05, 0.1, 0.1],
                        "force_trace": {
                            "sample_order": [1, 2, 3],
                            "actual": [
                                [label * 0.1, -0.2, -label * 0.3],
                                [label * 0.1 + 0.1, -0.1, -label * 0.3 - 0.2],
                                [label * 0.1 - 0.1, -0.3, -label * 0.3 + 0.2],
                            ],
                            "predicted": [
                                [label * 0.1 + 0.05, -0.1, -label * 0.3 + 0.1],
                                [label * 0.1 + 0.08, -0.12, -label * 0.3 - 0.1],
                                [label * 0.1 - 0.06, -0.25, -label * 0.3 + 0.15],
                            ],
                        },
                    }
                    for label in range(1, 26)
                ],
            }
            try:
                page.apply_report(report)
                page.select_result_point(17)
                self.assertIn("80.00%", page.overall_metrics_label.text())
                self.assertIn("开发评估", page.validation_warning_label.text())
                self.assertAlmostEqual(page.actual_force.force_xyz[0], 1.7)
                self.assertAlmostEqual(page.actual_force.force_xyz[1], -0.2)
                self.assertAlmostEqual(page.actual_force.force_xyz[2], -5.1)
                self.assertAlmostEqual(page.predicted_force.force_xyz[0], 1.75)
                self.assertAlmostEqual(page.predicted_force.force_xyz[1], -0.1)
                self.assertAlmostEqual(page.predicted_force.force_xyz[2], -5.0)
                self.assertIn("68.00%", page.selected_point_label.text())
                self.assertEqual(len(page.performance_map.performance_colors), 25)
                actual_x, actual_y = page.force_curves.actual_curves["Fx"].getData()
                predicted_x, predicted_y = page.force_curves.predicted_curves["Fx"].getData()
                self.assertEqual(actual_x.tolist(), [1, 2, 3])
                self.assertEqual(predicted_x.tolist(), [1, 2, 3])
                self.assertAlmostEqual(actual_y[0], 1.7)
                self.assertAlmostEqual(predicted_y[0], 1.75)
                self.assertFalse(page.actual_force.show_connection_status)
                self.assertFalse(page.predicted_force.show_connection_status)
            finally:
                page.close()

    def test_live_worker_reads_chinese_ready_payload_even_with_legacy_child_encoding(self):
        """Windows pipes may inherit GBK; IPC must force a single encoding."""
        real_popen = subprocess.Popen
        captured = {}
        script = (
            "import json,sys; "
            "print(json.dumps({'type':'ready','model_path':'三号机模型',"
            "'dataset_dir':'中文数据集','input_shape':[6,4,3]},"
            "ensure_ascii=False),flush=True); "
            "sys.stdin.readline()"
        )

        def start_fake_child(_args, **kwargs):
            captured['env'] = kwargs.get('env')
            # A real child whose inherited default is a legacy Chinese codepage.
            import os
            env = dict(os.environ)
            env['PYTHONIOENCODING'] = 'cp936'
            env.update(kwargs.get('env') or {})
            kwargs['env'] = env
            return real_popen([sys.executable, '-c', script], **kwargs)

        worker = TouchInferenceWorker('unused.pt')
        ready, errors = [], []
        worker.ready.connect(ready.append)
        worker.failed.connect(errors.append)
        try:
            with patch('manual_touch_training_page.subprocess.Popen', side_effect=start_fake_child):
                worker.start()
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline and not ready and not errors:
                    self.app.processEvents()
                    time.sleep(0.02)
            self.assertFalse(errors, errors)
            self.assertEqual(ready[0]['model_path'], '三号机模型')
            self.assertEqual(captured['env']['PYTHONIOENCODING'], 'utf-8')
        finally:
            worker.cancel()
            worker.wait(5000)
            self.app.processEvents()

    def test_visualization_is_live_only_and_position_is_prominent(self):
        with tempfile.TemporaryDirectory() as folder:
            page = ManualTouchTrainingPage(default_dataset_dir=Path(folder))
            try:
                self.assertTrue(page.force_curves.isHidden())
                self.assertTrue(page.actual_force.isHidden())
                self.assertTrue(page.predicted_force.isHidden())
                self.assertTrue(page.performance_table.isHidden())
                self.assertFalse(page.live_curves.isHidden())
                page._live_prediction({'position': 17, 'confidence': 0.87, 'force': [1, 2, -3]})
                self.assertIn('17', page.live_position_label.text())
                self.assertIn('87.0%', page.live_position_label.text())
            finally:
                page.close()

    def test_prediction_results_do_not_queue_gui_events(self):
        worker = TouchInferenceWorker('unused.pt')
        from PyQt5.QtTest import QSignalSpy
        spy = QSignalSpy(worker.predicted)
        for i in range(1000):
            worker.publish_prediction({'request_id': i})
        self.assertEqual(len(spy), 0)
        self.assertEqual(worker.take_latest_prediction()['request_id'], 999)
        self.assertIsNone(worker.take_latest_prediction())

    def test_unified_navigation_defines_training_as_fifth_page(self):
        from fbg_unified_app import PAGE_INFO, parse_args

        self.assertEqual(len(PAGE_INFO), 5)
        self.assertEqual(PAGE_INFO[4][0], "模型训练与评估")
        self.assertEqual(parse_args(["--view", "training"]).view, "training")

    def test_realtime_deployment_controls_and_45_point_frame_contract(self):
        class FrameSource:
            def __init__(self):
                self.frame_listeners = []

        source = FrameSource()
        with tempfile.TemporaryDirectory() as folder:
            page = ManualTouchTrainingPage(
                default_dataset_dir=Path(folder),
                frame_source=source,
                force_provider=lambda: (0.1, -0.2, -3.0),
            )
            try:
                self.assertFalse(page.start_live_button.isEnabled())
                self.assertEqual(page.expected_position_combo.count(), 27)
                frame = {
                    "records": [
                        {"first_code": index, "second_code": index + 100}
                        for index in range(45)
                    ]
                }
                self.assertEqual(
                    page._extract_live_adc(frame),
                    [index + 100 for index in range(45)],
                )
                page._attach_live_listener()
                self.assertIn(page._receive_live_frame, source.frame_listeners)
                page._detach_live_listener()
                self.assertNotIn(page._receive_live_frame, source.frame_listeners)
            finally:
                page.close()


if __name__ == "__main__":
    unittest.main()
