import unittest
import math

class TrendBufferTests(unittest.TestCase):
    def test_default_stability_window_rejects_startup_and_drift_then_freezes(self):
        from peak_dbm_trend import PeakWavelengthHistory
        h = PeakWavelengthHistory()
        for seq in range(16):
            h.append(seq, seq * 200000, [{'center_nm': 1520.}], 'r')
        self.assertIsNone(h.baselines_nm[0])
        for seq in range(16, 32):
            h.append(seq, seq * 200000,
                     [{'center_nm': 1530. + (seq-16)*.002}], 'r')
        self.assertIsNone(h.baselines_nm[0])
        for seq in range(32, 43):
            h.append(seq, seq * 200000,
                     [{'center_nm': 1540. + (seq % 2)*.0004}], 'r')
        self.assertAlmostEqual(h.baselines_nm[0], 1540. + 5*.0004/11)
        baseline = h.baselines_nm[0]
        h.append(43, 8600000, [{'center_nm': baseline + .02}], 'r')
        self.assertEqual(h.baselines_nm[0], baseline)
        self.assertAlmostEqual(h.samples[0][-1][1], 20.)

    def test_missing_data_and_gaps_restart_pending_stability(self):
        from peak_dbm_trend import PeakWavelengthHistory
        h = PeakWavelengthHistory(stable_s=2., stable_frames=3, warmup_frames=0)
        for seq in (0, 1):
            h.append(seq, seq*1000000, [{'center_nm': 1530.}], 'r')
        h.append(2, 2000000, [{}], 'r')
        for seq in (3, 4):
            h.append(seq, seq*1000000, [{'center_nm': 1530.}], 'r')
        self.assertIsNone(h.baselines_nm[0])
        h.append(6, 6000000, [{'center_nm': 1530.}], 'r')
        self.assertIsNone(h.baselines_nm[0])
        for seq in (7, 8):
            h.append(seq, seq*1000000, [{'center_nm': 1530.}], 'r')
        self.assertEqual(h.baselines_nm[0], 1530.)
        h.clear()
        self.assertIsNone(h.baselines_nm[0])

    def test_each_peak_baseline_missing_centers_and_signed_pm(self):
        from peak_dbm_trend import PeakWavelengthHistory
        h = PeakWavelengthHistory(stable_s=0., stable_frames=1, warmup_frames=0)
        h.append(0, 0, [{'center_nm': 1530., 'peak_dbm': -8}, {}], 'route')
        self.assertTrue(math.isnan(h.samples[1][-1][1]))
        h.append(1, 1000000, [{'center_nm': 1529.997, 'peak_dbm': -2},
                              {'center_nm': 1540.}], 'route')
        self.assertAlmostEqual(h.samples[0][-1][1], -3.)
        self.assertEqual(h.samples[1][-1][1], 0.)
        h.append(2, 2000000, [{'center_nm': 1529.997, 'peak_dbm': -10},
                              {'center_nm': 1540.004}], 'route')
        self.assertAlmostEqual(h.samples[0][-1][1], -3.)
        self.assertAlmostEqual(h.samples[1][-1][1], 4.)
        h.clear()
        h.append(3, 3000000, [{'center_nm': 1529.997}], 'route')
        self.assertEqual(h.samples[0][-1][1], 0.)

    def test_acquisition_clock_existing_peak_wavelength(self):
        from peak_dbm_trend import PeakWavelengthHistory
        h = PeakWavelengthHistory(stable_s=0., stable_frames=1, warmup_frames=0)
        h.append(16, 1000000, [{'center_nm': 1530.0}], ('route',))
        h.append(17, 1250000, [{'center_nm': 1530.002}], ('route',))
        self.assertEqual(h.samples[0][0], (0., 0.))
        self.assertAlmostEqual(h.samples[0][1][1], 2.)

    def test_discontinuities_invalid_values_bounded_and_reset(self):
        from peak_dbm_trend import PeakWavelengthHistory
        h = PeakWavelengthHistory(window_s=60, capacity=20, stable_s=0., stable_frames=1, warmup_frames=0)
        h.append(16, 1000000, [{'center_nm': -8}], ('r',))
        self.assertFalse(h.append(16, 1100000, [{'center_nm': -99}], ('r',)))
        self.assertFalse(h.append(15, 900000, [{'center_nm': -99}], ('r',)))
        h.append(18, 2000000, [{'center_nm': -7}], ('r',))
        self.assertTrue(math.isnan(h.samples[0][-2][1]))
        h.append(19, None, [{'center_nm': -6}], ('r',))
        h.append(20, 3000000, [{}], ('r',))
        self.assertTrue(math.isnan(h.samples[0][-1][1]))
        h.append(21, 4000000, [{'center_nm': -5}], ('r',))
        for seq in range(22, 102):
            h.append(seq, seq * 1000000, [{'center_nm': -4}], ('r',))
        self.assertLessEqual(len(h.samples[0]), 20)
        self.assertTrue(all(x >= 40 for x, y in h.samples[0]))
        h.append(102, 102000000, [{'center_nm': -3}], ('new route',))
        self.assertEqual(list(h.samples[0]), [(0., 0.)])
        h.clear()
        self.assertFalse(any(h.samples))

class TrendWidgetTests(unittest.TestCase):
    def setUp(self):
        from test_temporary_test_widget import WidgetTests
        WidgetTests.setUp(self)

    @classmethod
    def setUpClass(cls):
        from test_temporary_test_widget import WidgetTests
        WidgetTests.setUpClass.__func__(cls)

    def test_clear_after_stop_marks_waiting_not_stale_live_value(self):
        from temporary_test_widget import TemporaryTestWindow
        w = TemporaryTestWindow(None)
        self.addCleanup(w.close)
        w.peak_trend_history.stable_s = 0.
        w.peak_trend_history.stable_frames = 1
        w.peak_trend_history.warmup_frames = 0
        w.stop()
        w.peak_trend_clear.click()
        self.assertIn('等待', w.peak_trend_status.text())
        self.assertFalse(w.peak_trend_stopped)
        w.data_tabs.setCurrentIndex(2)
        w._flush_peak_trend()
        self.assertIn('等待', w.peak_trend_status.text())

    def test_missing_clock_breaks_without_fake_time_or_listener_loss(self):
        from temporary_test_widget import TemporaryTestWindow
        from test_temporary_test_mode import reference, frame
        from temporary_test_mode import select_points, decode, quality
        w = TemporaryTestWindow(None)
        self.addCleanup(w.close)
        w.peak_trend_history.stable_s = 0.
        w.peak_trend_history.stable_frames = 1
        w.peak_trend_history.warmup_frames = 0
        table, x, y = reference()
        w.plan = select_points(table, x, y)
        w.render_plan()
        heard = []
        w.frame_listeners.append(heard.append)
        for seq, clock in ((0, 1000000), (16, 2000000), (17, None), (19, 3000000)):
            f = decode(frame(), 123, 456)
            f['sequence'] = seq
            f['cycle_start_us'] = clock
            f['quality'] = quality(f)
            w.update_frame(f)
            if seq == 0:
                self.assertTrue(math.isfinite(w.peak_trend_history.samples[0][-1][1]))
        self.assertEqual(len(heard), 4)
        self.assertEqual(w.peak_trend_history.samples[0][-1][0], 2.)
        self.assertTrue(math.isnan(w.peak_trend_history.samples[0][-2][1]))
        w.data_tabs.setCurrentIndex(2)
        w._flush_peak_trend()
        self.assertGreater(len(w.peak_trend_curves[0].getData()[0]), 0)
        w.analog_gain.setCurrentIndex((w.analog_gain.currentIndex()+1) % w.analog_gain.count())
        self.assertFalse(any(w.peak_trend_history.samples))

    def test_bottom_tab_reuses_real_fit_without_extra_fits(self):
        from temporary_test_widget import TemporaryTestWindow
        from test_temporary_test_mode import reference, frame
        from temporary_test_mode import select_points, decode, quality
        from temporary_optical_units import adc_code_to_dbm
        from unittest.mock import patch
        import temporary_peak_fit_makima as fitter
        import numpy as np
        w = TemporaryTestWindow(None)
        self.addCleanup(w.close)
        w.peak_trend_history.stable_s = 0.
        w.peak_trend_history.stable_frames = 1
        w.peak_trend_history.warmup_frames = 0
        self.assertEqual(w.data_tabs.tabText(2), '峰值实时曲线')
        table, x, y = reference()
        w.plan = select_points(table, x, y)
        w.render_plan()
        w.data_tabs.setCurrentIndex(2)
        for seq in (16, 17):
            f = decode(frame(), 123, 456)
            f['sequence'] = seq
            f['cycle_start_us'] = 1000000 + (seq - 16) * 250000
            for i, rec in enumerate(f['records']):
                rec['second_code'] = [300, 750, 1150, 950, 400][i % 5] + (seq-16)*20
            f['quality'] = quality(f)
            with patch.object(fitter, 'fit_temporary_frame', wraps=fitter.fit_temporary_frame) as fit:
                w.update_frame(f)
                w._flush_peak_trend()
                self.assertEqual(fit.call_count, 1)
            self.assertAlmostEqual(w.peak_trend_history.samples[0][-1][1],
                (w.peak_fits[0]['center_nm'] - w.peak_trend_history.baselines_nm[0]) * 1000.)
            self.assertNotEqual(w.peak_fits[0]['peak_dbm'], float(adc_code_to_dbm([max(w.latest_values[:5])], w.latest_feedback_selector)[0]))
        xs, ys = w.peak_trend_curves[0].getData()
        np.testing.assert_allclose(xs, [0., .25])
        self.assertEqual(w.peak_trend_timer.interval(), 100)
        self.assertIn('pm', w.peak_trend_values.text())
        self.assertEqual(w.peak_trend_plot.getAxis('left').labelUnits, 'pm')
        self.assertIn('未经波长计精度验证', w.peak_trend_status.text())
        self.assertNotIn('质量', w.peak_trend_status.text())
        before = list(w.peak_trend_history.samples[0])
        w.stop()
        self.assertEqual(list(w.peak_trend_history.samples[0]), before)
        self.assertIn('已停采', w.peak_trend_status.text())
        w.render_plan()
        self.assertFalse(any(w.peak_trend_history.samples))

if __name__ == '__main__': unittest.main(verbosity=2)
