"""Deterministic Qt timer tests; no hardware, no five-minute wall-clock wait."""
import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from test_manual_stress_capture import FixedGridCaptureTests, FakeTemporaryPage, temporary_frame
from manual_stress_capture import ManualStressCaptureWindow


class TimedEmptyCaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        FixedGridCaptureTests.setUpClass()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=Path.home()/'AppData/Local/hermes/cache/scratch')
        self.source = FakeTemporaryPage()
        self.source.selected_channel = lambda: 1
        self.source.effective_feedback_selector = lambda channel: 2
        self.source.plan = {'rows': [dict(index=i, measured_nm=w) for i,w in enumerate(self.source.reference_wavelengths)]}
        self.page = ManualStressCaptureWindow(self.source, wavelength_provider=lambda: self.source.reference_wavelengths, output_root=Path(self.tmp.name))
        self.now = 0.
        self.page._empty_clock = lambda: self.now

    def tearDown(self):
        self.page.close()
        self.tmp.cleanup()

    def enable(self):
        with patch('manual_stress_capture.QtWidgets.QMessageBox.question', return_value=16384):
            self.page.timed_empty_button.click()

    def deliver(self, sequence=16):
        from test_px6d_capture_tare import feed_fixture
        feed_fixture(self.page.force_sensor, (2., -1., 4., .1, .2, .3))
        f = temporary_frame(sequence)
        f['provenance'] = 'SYNTHETIC timed empty smoke_only'
        self.page._on_temporary_frame(f)
        return f

    def rows(self):
        return [json.loads(x) for x in (self.page.timed_empty_session_dir/'frames.jsonl').read_text('utf-8').splitlines()]

    def test_explicit_enable_first_sample_after_300_seconds_only(self):
        p = self.page
        self.assertFalse(hasattr(p, 'timed_empty_button') and p.timed_empty_button.isChecked())
        self.enable()
        self.assertTrue(p.timed_empty_button.isChecked())
        self.assertEqual(p.timed_empty_interval_s, 300)
        self.now = 299.99; self.deliver(); p._tick_timed_empty()
        self.assertEqual(p.timed_empty_saved, 0)
        self.assertIn('00:01', p.timed_empty_status.text())
        self.now = 300.; p._tick_timed_empty()
        self.assertEqual(p.timed_empty_saved, 1)
        self.now = 600.; self.deliver(17); p._tick_timed_empty()
        self.assertEqual(len(self.rows()), 2)
        self.assertEqual([r['quality']['source_frame_sequence'] for r in self.rows()], [16,17])

    def test_due_tares_shared_fresh_raw_before_build_and_write_once(self):
        from paxini_px6d import PaxiniForceSensor
        from test_px6d_capture_tare import feed_fixture
        p = self.page
        sensor = p.force_sensor
        self.assertIsInstance(sensor, PaxiniForceSensor)
        feed_fixture(sensor, (2., -1., 4., .1, .2, .3))
        sensor.tare()
        self.enable(); self.deliver(); self.now = 300.
        feed_fixture(sensor, (3., -2., 5., .2, .3, .4))
        with patch.object(sensor, 'tare', wraps=sensor.tare) as tare:
            p._tick_timed_empty(); p._tick_timed_empty()
            self.assertEqual(tare.call_count, 1)
        q = self.rows()[0]['quality']['paxini_force']
        self.assertEqual(q['tare']['count'], 2)
        self.assertEqual(q['tare']['offset_six_axis'], q['raw_six_axis'])
        self.assertEqual(q['raw_six_axis'][:3], [3., -2., 5.])
        self.assertEqual([q[k] for k in ('fx_n', 'fy_n', 'fz_n')], [0., 0., 0.])

    def test_positive_label_blocks_enable_without_clearing_selection(self):
        self.page.select_grid_point(4)
        self.enable()
        self.assertFalse(self.page.timed_empty_button.isChecked())
        self.assertEqual(self.page.selected_points, {4})
        self.assertIsNone(self.page.timed_empty_writer)

    def test_positive_label_after_enable_skips_due_frame(self):
        self.enable(); self.page.select_grid_point(8)
        self.deliver(); self.now = 300.; self.page._tick_timed_empty()
        self.assertEqual(self.page.timed_empty_saved, 0)
        self.assertIn('按压标签', self.page.timed_empty_status.text())
        self.assertEqual(self.page.selected_point, 8)

    def test_malformed_latest_frame_never_reuses_previous_good_frame(self):
        self.enable(); self.deliver()
        self.page._on_temporary_frame({'kind': 'data', 'sequence':17, 'records':[]})
        self.now = 300.; self.page._tick_timed_empty()
        self.assertEqual(self.page.timed_empty_saved, 0)
        self.assertIn('已跳过', self.page.timed_empty_status.text())

    def test_source_context_change_is_rejected(self):
        self.enable(); self.deliver()
        self.source.wavelength_session_generation = 1
        self.now = 300.; self.page._tick_timed_empty()
        self.assertEqual(self.page.timed_empty_saved, 0)
        self.assertIn('上下文', self.page.timed_empty_status.text())

    def test_duplicate_sequence_is_not_reused_at_next_due(self):
        self.enable(); self.deliver(); self.now=300.; self.page._tick_timed_empty()
        self.deliver(); self.now=600.; self.page._tick_timed_empty()
        self.assertEqual(self.page.timed_empty_saved, 1)
        self.assertIn('重复', self.page.timed_empty_status.text())

    def test_stopped_acquisition_rejects_even_recent_frame(self):
        self.enable(); self.deliver(); self.source.peak_trend_stopped = True
        self.now=300.; self.page._tick_timed_empty()
        self.assertEqual(self.page.timed_empty_saved, 0)

    def test_current_channel_and_frame_channel_must_match(self):
        self.source.selected_channel = lambda: 2
        self.source.effective_feedback_selector = lambda channel: 0
        self.enable(); self.deliver(); self.now=300.; self.page._tick_timed_empty()
        self.assertEqual(self.page.timed_empty_saved, 0)

    def test_delay_never_catches_up_and_stop_cancels(self):
        self.enable(); self.deliver(); self.now=1500.; self.page._tick_timed_empty()
        self.assertEqual(self.page.timed_empty_saved, 1)
        self.page._tick_timed_empty()
        self.assertEqual(self.page.timed_empty_saved, 1)
        self.assertEqual(self.page._empty_due, 1800.)
        old_path=self.page.timed_empty_session_dir
        self.page.timed_empty_button.click()
        self.now=1800.; self.deliver(17); self.page._tick_timed_empty()
        self.assertEqual(self.page.timed_empty_saved, 1)
        self.assertFalse(self.page.timed_empty_timer.isActive())
        self.enable()
        self.assertNotEqual(old_path, self.page.timed_empty_session_dir)
        self.assertEqual(len([json.loads(x) for x in (old_path/'frames.jsonl').read_text('utf-8').splitlines()]), 1)

    def test_stale_skips_without_connecting_any_device(self):
        self.enable(); self.deliver()
        self.page.last_frame_monotonic=time.monotonic()-3
        self.now=300.; self.page._tick_timed_empty()
        self.assertEqual(self.page.timed_empty_saved, 0)
        self.assertTrue(self.page.timed_empty_button.isChecked())

    def test_cancel_enable_creates_no_session(self):
        with patch('manual_stress_capture.QtWidgets.QMessageBox.question', return_value=65536):
            self.page.timed_empty_button.click()
        self.assertIsNone(self.page.timed_empty_writer)
        self.assertEqual(list(Path(self.tmp.name).iterdir()), [])

    def test_fresh_force_saved_and_accepted_training_contract(self):
        from touch_45_data import validate_record, prepare_data
        self.enable(); self.deliver(); self.now=300.; self.page._tick_timed_empty()
        r=self.rows()[0]; q=r['quality']
        self.assertIsNotNone(q['paxini_force']); self.assertFalse(q['force_error'])
        self.assertEqual(q['position_labels'], [])
        self.assertEqual(q['position_multihot'], [0]*25)
        self.assertFalse(q['is_pressed'])
        self.assertEqual(q['annotation_source'], 'timer_empty_operator_confirmed')
        validate_record(r)
        data=prepare_data(self.page.timed_empty_session_dir)
        self.assertEqual(len(data['features']), 1)

    def test_manual_session_and_undo_are_independent(self):
        self.page.mode_combo.setCurrentIndex(1)
        self.deliver(); self.page.select_grid_point(7)
        self.assertTrue(self.page.capture_current_press())
        manual_path=self.page.session_dir
        self.page.select_no_press(); self.enable()
        self.now=300.; self.deliver(17); self.page._tick_timed_empty()
        self.assertEqual(len(self.page.capture_history), 1)
        self.assertEqual(self.page.combo_counts, {(7,):1})
        self.assertTrue(self.page.undo_last_annotation(confirm=False))
        self.assertEqual(self.page.timed_empty_saved, 1)
        self.assertEqual(len(self.rows()), 1)
        self.assertEqual((manual_path/'frames.jsonl').read_text('utf-8'), '')

    def test_write_failure_pauses_and_keeps_successful_count(self):
        self.enable(); self.deliver(); self.now=300.; self.page._tick_timed_empty()
        writer=self.page.timed_empty_writer
        with patch.object(writer, 'write_frame', side_effect=OSError('disk full')):
            self.now=600.; self.deliver(17); self.page._tick_timed_empty()
        self.assertEqual(self.page.timed_empty_saved, 1)
        self.assertFalse(self.page.timed_empty_button.isChecked())
        self.assertFalse(self.page.timed_empty_timer.isActive())
        self.assertIn('disk full', self.page.timed_empty_status.toolTip())
        self.assertIn('共享力归零已生效，无法回滚', self.page.timed_empty_status.toolTip())
        self.assertEqual(self.page.force_sensor.tare_count, 2)
        self.assertEqual(len(self.rows()), 1)
        events=[json.loads(x) for x in (self.page.timed_empty_session_dir/'events.jsonl').read_text('utf-8').splitlines()]
        self.assertEqual(sum(e['kind']=='press_sample' for e in events), 1)

    def test_counts_snapshot_matches_production_frames_and_events(self):
        self.enable(); self.deliver(); self.now=300.; self.page._tick_timed_empty()
        path=self.page.timed_empty_session_dir
        counts=json.loads((path/'combination_counts.json').read_text('utf-8'))
        self.assertEqual(counts['active_samples_count'], 1)
        self.assertEqual(counts['combinations'], [{'position_labels':[], 'count':1}])
        manifest=json.loads((path/'manifest.json').read_text('utf-8'))
        self.assertEqual(manifest['metadata']['interval_s'], 300)
        self.page.close()
        self.assertFalse(self.page.timed_empty_timer.isActive())
        events=[json.loads(x) for x in (path/'events.jsonl').read_text('utf-8').splitlines()]
        self.assertEqual(events[-1]['metadata']['total_samples'], 1)

    def test_stats_failure_keeps_saved_frame_and_reports_it(self):
        self.enable(); self.deliver(); self.now=300.
        with patch.object(self.page, '_persist_timed_empty_counts', side_effect=OSError('stats disk full')):
            self.page._tick_timed_empty()
        self.assertEqual(self.page.timed_empty_saved, 1)
        self.assertEqual(len(self.rows()), 1)
        self.assertIn('样本已保存', self.page.timed_empty_status.toolTip())

    def test_close_attempts_writer_cleanup_even_if_end_event_fails(self):
        self.enable(); writer=self.page.timed_empty_writer
        with patch.object(writer, 'write_event', side_effect=OSError('audit full')), patch.object(writer, 'close', wraps=writer.close) as close:
            self.page.timed_empty_button.click()
            self.assertEqual(close.call_count, 1)
        self.assertIn('audit full', self.page.timed_empty_status.toolTip())

    def test_source_read_failure_is_visible_without_crashing_qt_timer(self):
        self.enable(); self.deliver(); self.now=300.
        self.page.wavelength_provider=lambda: (_ for _ in ()).throw(RuntimeError('route unavailable'))
        self.page._tick_timed_empty()
        self.assertEqual(self.page.timed_empty_saved,0)
        self.assertIn('route unavailable', self.page.timed_empty_status.toolTip())
        self.assertTrue(self.page.timed_empty_timer.isActive())

    def test_enable_context_error_does_not_leave_a_writer_running(self):
        self.source.selected_channel=lambda: (_ for _ in ()).throw(RuntimeError('context unavailable'))
        with patch('manual_stress_capture.QtWidgets.QMessageBox.question', return_value=16384):
            self.page._toggle_timed_empty(True)
        self.assertIsNone(self.page.timed_empty_writer)
        self.assertFalse(self.page.timed_empty_timer.isActive())
        self.assertIn('context unavailable', self.page.timed_empty_status.toolTip())


class TimedEmptyTareGuardTests(unittest.TestCase):
    setUpClass = classmethod(lambda cls: FixedGridCaptureTests.setUpClass())
    setUp = TimedEmptyCaptureTests.setUp
    tearDown = TimedEmptyCaptureTests.tearDown
    enable = TimedEmptyCaptureTests.enable
    deliver = TimedEmptyCaptureTests.deliver
    rows = TimedEmptyCaptureTests.rows
    def test_precheck_tare_build_writer_order_and_next_real_sample_residual(self):
        from test_px6d_capture_tare import feed_fixture
        p = self.page; sensor = p.force_sensor; order = []
        self.enable(); self.deliver(); self.now = 300.
        original_check = p._timed_empty_skip_reason
        original_tare = sensor.tare
        original_build = p._build_45_capture
        original_write = p.timed_empty_writer.write_frame
        def check():
            order.append('check'); return original_check()
        def tare():
            order.append('tare'); result = original_tare()
            feed_fixture(sensor, (2.125, -1.25, 4.5, .1, .2, .3))
            return result
        def build(*args):
            order.append('build'); return original_build(*args)
        def write(record):
            order.append('write'); return original_write(record)
        with patch.object(p, '_timed_empty_skip_reason', side_effect=check), patch.object(sensor, 'tare', side_effect=tare), patch.object(p, '_build_45_capture', side_effect=build), patch.object(p.timed_empty_writer, 'write_frame', side_effect=write):
            p._tick_timed_empty()
        self.assertEqual(order, ['check', 'tare', 'build', 'write'])
        f = self.rows()[0]['quality']['paxini_force']
        self.assertEqual([f[k] for k in ('fx_n','fy_n','fz_n')], [.125, -.25, .5])
        self.assertEqual(f['raw_six_axis'][:3], [2.125, -1.25, 4.5])
        self.assertEqual(f['tare']['offset_six_axis'][:3], [2., -1., 4.])
        self.assertGreaterEqual(f['monotonic_ns'], f['tare']['source_sample_monotonic_ns'])

    def test_missing_stale_disconnected_disabled_and_tare_error_skip(self):
        p = self.page; sensor = p.force_sensor
        self.enable()
        for i, state in enumerate(('missing', 'stale', 'disconnected', 'disabled', 'error'), 1):
            self.deliver(16+i); self.now = i*300.
            p.force_sensor_enabled = state != 'disabled'
            if state == 'missing': sensor._latest = None
            if state == 'stale': sensor._latest_ns = time.monotonic_ns()-600_000_000
            if state == 'disconnected': sensor._running = False
            context = patch.object(sensor, 'tare', side_effect=RuntimeError('synthetic tare error')) if state == 'error' else patch.object(sensor, 'tare', wraps=sensor.tare)
            with context as tare:
                p._tick_timed_empty(); p._tick_timed_empty()
                self.assertLessEqual(tare.call_count, 1)
            self.assertEqual(p.timed_empty_saved, 0)
            self.assertIn('帕西尼力归零失败/无新鲜力数据，已跳过', p.timed_empty_status.text())
        self.assertEqual(sensor.tare_count, 0)

    def test_ineligible_optical_label_context_duplicate_never_tares(self):
        p = self.page; sensor = p.force_sensor; self.enable()
        for i, state in enumerate(('label', 'invalid', 'context', 'duplicate'), 1):
            p.select_no_press(); self.source.wavelength_session_generation = 0
            self.deliver(20+i); self.now = i*300.
            if state == 'label': p.select_grid_point(4)
            if state == 'invalid': p._empty_frame_valid = False
            if state == 'context': self.source.wavelength_session_generation = 1
            if state == 'duplicate': p._empty_last_sequence = 24
            with patch.object(sensor, 'tare', wraps=sensor.tare) as tare:
                p._tick_timed_empty()
                self.assertEqual(tare.call_count, 0)
            self.assertEqual(p.timed_empty_saved, 0)

    def test_force_disappears_after_tare_skips_without_rollback(self):
        p = self.page; sensor = p.force_sensor
        self.enable(); self.deliver(); self.now = 300.
        original = sensor.tare
        def disappear():
            result = original(); sensor._running = False; return result
        with patch.object(sensor, 'tare', side_effect=disappear): p._tick_timed_empty()
        self.assertEqual(sensor.tare_count, 1)
        self.assertEqual(p.timed_empty_saved, 0)
        self.assertIn('归零后无新鲜力数据，已跳过', p.timed_empty_status.text())

    def test_warning_explains_shared_host_tare_and_manual_stop_close_do_not_tare(self):
        p = self.page; sensor = p.force_sensor
        warning = '每次采集前自动进行帕西尼主机力归零（影响共享实际力参考）'
        self.assertIn(warning, p.timed_empty_button.toolTip())
        with patch('manual_stress_capture.QtWidgets.QMessageBox.question', return_value=16384) as question:
            p.timed_empty_button.click()
            self.assertIn(warning, question.call_args.args[2])
        self.deliver(); p._tick_timed_empty()
        self.assertEqual(sensor.tare_count, 0)
        p.timed_empty_button.click(); self.now = 1000.; p._tick_timed_empty(); p.close()
        self.assertEqual(sensor.tare_count, 0)
        self.assertIn(warning, p.timed_empty_status.toolTip())

    def test_manual_missing_force_still_null_without_tare(self):
        p = self.page; self.deliver(); p.force_sensor._latest = None
        p.select_no_press()
        self.assertTrue(p.capture_current_press())
        p.writer.flush()
        row = json.loads((p.session_dir/'frames.jsonl').read_text('utf-8').splitlines()[0])
        self.assertIsNone(row['quality']['paxini_force'])
        self.assertEqual(p.force_sensor.tare_count, 0)


if __name__ == '__main__':
    unittest.main()
