"""Host tare integration; all serial inputs here are synthetic CRC protocol fixtures."""
import os
import sys
import struct
import time
import unittest
from pathlib import Path
plugins = Path(sys.prefix) / 'Lib/site-packages/PyQt5/Qt5/plugins'
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(plugins / 'platforms')
from PyQt5 import QtCore, QtWidgets, QtTest
from paxini_px6d import PaxiniForceSensor, build_rs485_frame


def feed_fixture(sensor, axes=(1., -2., 3., .1, .2, .3)):
    """Replay a synthetic cmd03 reply through the production CRC pump, no COM open."""
    packet = build_rs485_frame(3, struct.pack('<6f', *axes))
    class FixtureSerial:
        def write(self, data):
            assert data == bytes.fromhex('AA557F0501FA')
        def flush(self): pass
        @property
        def in_waiting(self): return len(packet)
        def read(self, count):
            sensor._running = False
            return packet
    sensor.protocol = 'rs485'
    sensor._running = True
    sensor._ser = FixtureSerial()
    sensor._pump_rs485()
    sensor._running = True


class HostTareTests(unittest.TestCase):
    def test_stop_invalidates_sample_before_any_reconnect(self):
        sensor = PaxiniForceSensor()
        feed_fixture(sensor)
        sensor._ser = None
        sensor.stop()
        self.assertIsNone(sensor._latest)
        self.assertIsNone(sensor._latest_ns)

    def test_start_is_single_reader_until_closed(self):
        from unittest.mock import patch
        sensor = PaxiniForceSensor()
        with patch.object(sensor, '_discover_port', return_value=None):
            sensor.start()
            thread = sensor._thread
            sensor.start()
            self.assertIs(sensor._thread, thread)
            sensor.close()
            self.assertFalse(sensor._running)
            self.assertFalse(thread.is_alive())

    def test_invalid_or_disconnected_samples_cannot_tare(self):
        sensor = PaxiniForceSensor()
        feed_fixture(sensor)
        sensor._latest = (1., 2., 3., 0., 0., float('nan'))
        with self.assertRaises(RuntimeError): sensor.tare()
        self.assertIsNone(sensor.measurement_snapshot())
        feed_fixture(sensor)
        sensor._running = False
        with self.assertRaises(RuntimeError): sensor.tare()
        self.assertIsNone(sensor.measurement_snapshot())

    def test_legacy_bad_checksum_does_not_publish_force_or_calibration_ack(self):
        sensor = PaxiniForceSensor('COM_FIXTURE')
        from paxini_px6d import build_frame
        for cmd, payload in [(3, struct.pack('<6f', 1, 2, 3, 4, 5, 6)), (0x10, bytes(8))]:
            packet = bytearray(build_frame(cmd, payload))
            packet[-1] ^= 1
            sensor._extract(packet)
        self.assertEqual(sensor.frames_received, 0)
        self.assertFalse(sensor._calibration_event.is_set())

    def test_auto_discovery_requires_crc_six_axis_not_legacy_ack(self):
        from unittest.mock import patch
        from types import SimpleNamespace
        from paxini_px6d import build_frame
        class FixtureSerial:
            def __init__(self, *args, **kwargs): self.data = bytearray()
            def write(self, data): self.data.extend(build_frame(1, bytes(8)))
            def flush(self): pass
            def close(self): pass
            @property
            def in_waiting(self): return len(self.data)
            def read(self, count):
                out = bytes(self.data[:count]); del self.data[:count]; return out
        sensor = PaxiniForceSensor()
        with patch('serial.tools.list_ports.comports', return_value=[SimpleNamespace(device='COM_FIXTURE')]), patch('serial.Serial', FixtureSerial):
            self.assertIsNone(sensor._discover_port(probe_timeout=.01))

    def test_tare_uses_fresh_six_axis_sample_and_preserves_raw(self):
        sensor = PaxiniForceSensor()
        feed_fixture(sensor)
        raw = sensor._latest
        self.assertTrue(hasattr(sensor, 'tare'), 'shared sensor lacks host tare')
        sensor.tare()
        snap = sensor.measurement_snapshot()
        self.assertEqual(snap['raw'], list(raw))
        self.assertEqual(snap['corrected'], [0.] * 6)
        self.assertEqual(snap['tare']['mode'], 'host_offset')
        self.assertEqual(snap['tare']['count'], 1)
        feed_fixture(sensor, (2., -1., 4., .2, .3, .4))
        self.assertEqual(sensor.measurement_snapshot()['corrected'][:3], [1., 1., 1.])
        sensor._latest_ns = time.monotonic_ns() - 600_000_000
        with self.assertRaises(RuntimeError): sensor.tare()
        self.assertIsNone(sensor.measurement_snapshot())
        self.assertEqual(sensor.tare_count, 1)


class SharedPageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        QtCore.QCoreApplication.setLibraryPaths([str(plugins)])
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_both_pages_tare_shared_capture_and_live_reference_not_predictions(self):
        import tempfile
        import json
        from manual_stress_capture import ManualStressCaptureWindow
        from manual_touch_training_page import ManualTouchTrainingPage
        from test_manual_stress_capture import FakeTemporaryPage, temporary_frame
        sensor = PaxiniForceSensor()
        source = FakeTemporaryPage()
        with tempfile.TemporaryDirectory() as folder:
            capture = ManualStressCaptureWindow(source, wavelength_provider=lambda: source.reference_wavelengths,
                output_root=Path(folder), force_sensor=sensor)
            self.assertTrue(hasattr(capture, 'force_disconnect_button'), 'missing explicit shared disconnect')
            training = ManualTouchTrainingPage(default_dataset_dir=Path(folder), force_sensor=sensor)
            try:
                capture.show(); training.show(); self.app.processEvents()
                self.assertFalse(sensor._running, 'page construction must not start acquisition')
                self.assertTrue(training.actual_force.isVisible())
                self.assertTrue(training.actual_force.zero_button.isVisible())
                self.assertTrue(training.force_connect_button.isVisible())
                self.assertFalse(training.predicted_force.zero_button.isVisible())
                feed_fixture(sensor)
                QtTest.QTest.mouseClick(capture.zero_button, QtCore.Qt.LeftButton)
                self.assertEqual(sensor.tare_count, 1)
                training._refresh_acquired_force()
                self.assertEqual(training.actual_force.force_xyz, (0., 0., 0.))
                feed_fixture(sensor, (2., -1., 4., .2, .3, .4))
                training._live_prediction({'position': 1, 'confidence': .99, 'force': [9, 8, 7]})
                self.assertEqual(training._live_actual_force[-1], (1., 1., 1.))
                self.assertEqual(training._live_predicted[-1], (9., 8., 7.))
                QtTest.QTest.mouseClick(training.actual_force.zero_button, QtCore.Qt.LeftButton)
                self.assertEqual(sensor.tare_count, 2)
                self.assertEqual(training._live_predicted[-1], (9., 8., 7.))
                self.assertEqual(training._live_actual_force[-1], (0., 0., 0.))
                actual_x, actual_y = training.live_curves.actual_curves['Fx'].getData()
                self.assertEqual(actual_y.tolist(), [0.])
                capture._refresh_force_arrows()
                self.assertEqual(capture.force_vector.force_xyz, (0., 0., 0.))
                capture._on_temporary_frame(temporary_frame())
                self.assertTrue(capture.capture_current_press())
                saved = capture.session_dir
                capture.close_session()
                record = json.loads((saved / 'frames.jsonl').read_text('utf-8').splitlines()[0])
                force = record['quality']['paxini_force']
                self.assertEqual([force[k] for k in ('fx_n','fy_n','fz_n')], [0.,0.,0.])
                self.assertEqual(force['raw_six_axis'][:3], [2.,-1.,4.])
                self.assertEqual(force['tare']['count'], 2)
                feed_fixture(sensor, (3., 0., 5., .3, .4, .5))
                training._live_prediction({'position': 1, 'confidence': .99, 'force': [9, 8, 7]})
                QtTest.QTest.mouseClick(capture.zero_button, QtCore.Qt.LeftButton)
                training._refresh_acquired_force()
                self.assertEqual(training._live_actual_force[-1], (0., 0., 0.))
                self.assertEqual(training._live_actual_force[0], (-1., -1., -1.))
                self.assertEqual(training._live_predicted[-1], (9., 8., 7.))
                sensor._latest_ns = time.monotonic_ns() - 600_000_000
                before = sensor.tare_count
                QtTest.QTest.mouseClick(training.actual_force.zero_button, QtCore.Qt.LeftButton)
                self.assertEqual(sensor.tare_count, before)
                capture._refresh_force_arrows()
                self.assertFalse(capture.force_vector.connected)
                self.assertIsNone(capture.force_vector.moments)
                self.assertIsNone(training.force_provider())
                capture._on_temporary_frame(temporary_frame(sequence=17))
                self.assertTrue(capture.capture_current_press(), 'missing force must not block optical capture')
                missing_session = capture.session_dir
                capture.close_session()
                missing = json.loads((missing_session / 'frames.jsonl').read_text('utf-8').splitlines()[0])
                self.assertIsNone(missing['quality']['paxini_force'])
                self.assertTrue(missing['quality']['force_error'])
            finally:
                sensor._running = False
                sensor._ser = None
                training.close(); capture.close()


if __name__ == '__main__': unittest.main()
