"""Capture UI rate shares the sensor, independent of GUI refresh and tare."""
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(Path(sys.prefix) / 'Lib/site-packages/PyQt5/Qt5/plugins/platforms')
from PyQt5 import QtWidgets
from manual_stress_capture import ManualStressCaptureWindow
from test_manual_stress_capture import FakeTemporaryPage
from test_px6d_sample_rate import Clock, replay, packet
from paxini_px6d import PaxiniForceSensor


class RateUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    def test_capture_label_tracks_shared_sensor_not_timer_or_tare(self):
        source = FakeTemporaryPage()
        sensor = PaxiniForceSensor()
        clock = Clock()
        with patch('serial.Serial', side_effect=AssertionError('no hardware')):
            page = ManualStressCaptureWindow(source, wavelength_provider=lambda: source.reference_wavelengths, force_sensor=sensor,
                output_root=Path(os.environ['TMPDIR']) / 'px6d_rate_ui_unused')
            try:
                self.assertTrue(hasattr(page, 'force_rate_label'), 'missing compact effective receipt Hz label')
                page._refresh_force_arrows()
                self.assertIn('— Hz', page.force_rate_label.text())
                with patch('paxini_px6d.time.monotonic_ns', clock):
                    replay(sensor, clock, [(clock.ns, packet())])
                    page._refresh_force_arrows()
                    self.assertIn('估计中', page.force_rate_label.text())
                    replay(sensor, clock, [(10_010_000_000 + i * 10_000_000, packet()) for i in range(100)])
                    page._refresh_force_arrows()
                    self.assertEqual(page.force_rate_label.text(), '有效接收：100.0 Hz')
                    for interval in (10, 100, 500):
                        page.force_timer.setInterval(interval)
                        for _ in range(100): page._refresh_force_arrows()
                        self.assertEqual(page.force_rate_label.text(), '有效接收：100.0 Hz')
                    page.zero_button.click()
                    self.assertEqual(page.force_vector.force_xyz, (0., 0., 0.))
                    self.assertEqual(page.force_rate_label.text(), '有效接收：100.0 Hz')
                    self.assertEqual(sensor.frames_received, 101)
                    clock.ns += 500_000_001
                    page._refresh_force_arrows()
                    self.assertIn('— Hz', page.force_rate_label.text())
                    self.assertFalse(page.force_vector.connected)
                    sensor.stop(); page._refresh_force_arrows()
                    self.assertIn('— Hz', page.force_rate_label.text())
            finally:
                sensor._ser = None
                page.close()


if __name__ == '__main__': unittest.main()
