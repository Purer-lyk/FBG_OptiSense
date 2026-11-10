"""Effective host receipt rate: real protocol parser, deterministic clocks, no hardware."""
import struct
import unittest
from unittest.mock import patch
from paxini_px6d import PaxiniForceSensor, build_rs485_frame


class Clock:
    ns = 10_000_000_000
    def __call__(self): return self.ns


def replay(sensor, clock, chunks):
    """Each (ns, bytes) is one serial read, never a mocked sample publish."""
    pending = list(chunks)
    class Serial:
        def write(self, data):
            assert data == bytes.fromhex('AA557F0501FA')
        def flush(self): pass
        @property
        def in_waiting(self):
            if not pending:
                sensor._running = False
                return 0
            return len(pending[0][1])
        def read(self, count):
            clock.ns, data = pending.pop(0)
            return data
    sensor.protocol = 'rs485'
    sensor._running = True
    sensor._stop_event.clear()
    sensor._ser = Serial()
    with patch('paxini_px6d.time.monotonic_ns', clock), patch('paxini_px6d.time.monotonic', lambda: clock.ns / 1e9), patch.object(sensor._stop_event, 'wait', return_value=False):
        PaxiniForceSensor._pump_rs485(sensor)
    sensor._running = True
    sensor._ser = None


def packet(axes=(1., -2., 3., .1, .2, .3)):
    return build_rs485_frame(3, struct.pack('<6f', *axes))


class EffectiveRateTests(unittest.TestCase):
    def setUp(self):
        self.sensor = PaxiniForceSensor()
        self.clock = Clock()
        self.patcher = patch('paxini_px6d.time.monotonic_ns', self.clock)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
    def feed(self, stamps, data=None):
        replay(self.sensor, self.clock, [(ns, packet() if data is None else data) for ns in stamps])
    def test_rate_uses_valid_measurement_receipts_not_snapshot_polling(self):
        self.feed([10_000_000_000 + i * 10_000_000 for i in range(101)])
        snap = self.sensor.measurement_snapshot()
        self.assertIn('sample_rate_hz', snap, 'effective measurement receipt rate is missing')
        self.assertAlmostEqual(snap['sample_rate_hz'], 100.)
        self.assertEqual(snap['rate_sample_count'], 101)
        self.assertEqual(snap['rate_window_s'], 2.)
        for _ in range(100):
            self.assertEqual(self.sensor.measurement_snapshot()['sample_rate_hz'], 100.)
        self.assertEqual(self.sensor.frames_received, 101)


    def test_window_ages_during_snapshot_reads_without_counting_them(self):
        self.feed([10_000_000_000 + i * 10_000_000 for i in range(301)])
        self.assertEqual(self.sensor.measurement_snapshot()['rate_sample_count'], 201)
        self.clock.ns += 200_000_000
        snap = self.sensor.measurement_snapshot()
        self.assertEqual(snap['rate_sample_count'], 181)
        self.assertAlmostEqual(snap['sample_rate_hz'], 100.)

    def test_stop_and_reconnect_do_not_join_rate_sessions(self):
        self.feed([10_000_000_000, 10_010_000_000])
        self.sensor.stop()
        self.assertEqual(len(self.sensor._rate_receipts), 0)
        self.assertIsNone(self.sensor.measurement_snapshot())
        self.feed([10_200_000_000])
        self.assertIsNone(self.sensor.measurement_snapshot()['sample_rate_hz'])
        self.assertEqual(self.sensor.measurement_snapshot()['rate_sample_count'], 1)

    def test_automatic_reconnect_clears_before_and_after_each_connection(self):
        self.feed([10_000_000_000, 10_010_000_000])
        seen = []
        class Serial:
            def close(self): pass
        def pump():
            seen.append(self.sensor.measurement_snapshot())
            self.feed([10_100_000_000])
            self.sensor._stop_event.set()
        with patch.object(self.sensor, '_discover_port', return_value=Serial()), patch.object(self.sensor, '_pump_rs485', side_effect=pump):
            self.sensor._auto_pump()
        self.assertEqual(seen, [None])
        self.assertEqual(len(self.sensor._rate_receipts), 0)

    def test_invalid_crc_ack_nonfinite_fragmentation_count_complete_frame_once(self):
        good = packet()
        bad = bytearray(good); bad[-1] ^= 1
        ack = build_rs485_frame(1, bytes(8))
        calibration = build_rs485_frame(0x10, bytes(8))
        invalid = packet((1., 2., 3., 4., 5., float('nan')))
        replay(self.sensor, self.clock, [(10_000_000_000, ack + calibration + bad + invalid + good[:7]),
            (10_010_000_000, good[7:]), (10_020_000_000, good)])
        snap = self.sensor.measurement_snapshot()
        self.assertEqual(self.sensor.frames_received, 2)
        self.assertEqual(snap['rate_sample_count'], 2)
        self.assertAlmostEqual(snap['sample_rate_hz'], 100.)

    def test_startup_same_timestamp_and_stale_are_not_fake_zero(self):
        self.assertIsNone(self.sensor.measurement_snapshot())
        self.feed([10_000_000_000])
        self.assertIsNone(self.sensor.measurement_snapshot()['sample_rate_hz'])
        self.feed([10_000_000_000], packet() * 3)
        self.assertEqual(self.sensor.frames_received, 4)
        self.assertIsNone(self.sensor.measurement_snapshot()['sample_rate_hz'])
        self.clock.ns += 500_000_001
        self.assertIsNone(self.sensor.measurement_snapshot())

    def test_bounded_window_capacity_and_long_gap_pruning(self):
        self.feed([10_000_000_000 + i * 100_000 for i in range(5000)])
        snap = self.sensor.measurement_snapshot()
        self.assertEqual(snap['rate_sample_count'], 4096)
        self.assertAlmostEqual(snap['sample_rate_hz'], 10000.)
        self.feed([13_000_000_000])
        self.assertEqual(self.sensor.measurement_snapshot()['rate_sample_count'], 1)
        self.assertIsNone(self.sensor.measurement_snapshot()['sample_rate_hz'])

    def test_tare_and_legacy_checksum_parser_keep_receipt_semantics(self):
        from paxini_px6d import build_frame
        self.sensor._running = True
        for i in range(3):
            self.clock.ns = 10_000_000_000 + i * 20_000_000
            buffer = bytearray(build_frame(3, struct.pack('<6f', 1, -2, 3, 0, 0, 0)))
            self.sensor._extract(buffer)
            self.assertEqual(buffer, b'')
        before = self.sensor.measurement_snapshot()
        self.sensor.tare()
        after = self.sensor.measurement_snapshot()
        self.assertEqual(after['corrected'], [0.] * 6)
        self.assertEqual(after['sample_rate_hz'], 50.)
        self.assertEqual(after['rate_sample_count'], before['rate_sample_count'])


if __name__ == '__main__': unittest.main()
