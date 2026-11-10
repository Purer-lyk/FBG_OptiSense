"""Paxini PX6D six-axis force sensor reader (RS485/USB-CDC serial).

Hardware-verified on COM9 (streaming device id 0x01, baud 921600 8N1):
  - One-shot "get frame" (cmd 0x01) answers only a 13-byte ACK, no forces.
  - cmd 0x02 data 0x01 starts auto-push: first an ACK (cmd 0x02), then a
    continuous stream of cmd-0x03 frames whose 24 data bytes are six float32
    LE values: Fx Fy Fz (N) and Mx My Mz (N·m).
  - cmd 0x02 data 0x00 stops the stream.
Frame = AA 55 | device_id | cmd | data... | RC8 (XOR over id..data).
A missing/unplugged sensor must never block stress capture: ``read_six_axis``
raises and the caller records ``paxini_force`` as null with the reason.
"""

from __future__ import annotations

import struct
import math
import threading
import time
from collections import deque


HEADER = b"\xAA\x55"
DEFAULT_DEVICE_ID = 0x01
DEFAULT_BAUD = 921600
PUSH_DATA_LENGTH = {0x02: 8, 0x03: 24, 0x10: 8}


def rc8(body: bytes) -> int:
    checksum = 0
    for byte in body:
        checksum ^= byte
    return checksum & 0xFF


def build_frame(cmd: int, data: bytes, device_id: int = DEFAULT_DEVICE_ID) -> bytes:
    body = bytes((device_id, cmd)) + data
    return HEADER + body + bytes((rc8(body),))


def crc8(packet: bytes) -> int:
    value = 0
    for byte in packet:
        value ^= byte
        for _ in range(8):
            value = ((value << 1) ^ 0x07) & 0xFF if value & 0x80 else (value << 1) & 0xFF
    return value


def build_rs485_frame(cmd, data=b'\x01', device_id=0x7F):
    packet = HEADER + bytes((device_id, cmd)) + data
    return packet + bytes((crc8(packet),))


def parse_six_axis(payload: bytes):
    if len(payload) < 24:
        return None
    values = struct.unpack("<6f", payload[:24])
    if not all(value == value and abs(value) < 1e9 for value in values):
        return None
    return values


class PaxiniForceSensor:
    """Single-owner reader: AUTO uses CRC8/cmd05 RS485; explicit ports retain legacy."""

    def __init__(
        self,
        port: str | None = None,
        *,
        baud: int = DEFAULT_BAUD,
        device_id: int = DEFAULT_DEVICE_ID,
    ):
        self.auto_port = port is None or str(port).upper() == "AUTO"
        self.port = None if self.auto_port else port
        self._stop_event = threading.Event()
        self.protocol = "legacy"
        self.baud = int(baud)
        self.device_id = int(device_id)
        self.last_error: str | None = None
        self.frames_received = 0
        self._ser = None
        self._lock = threading.Lock()
        self._latest = None
        self._latest_ns = None
        self._thread: threading.Thread | None = None
        self._running = False
        self._tx_lock = threading.Lock()
        self._calibration_event = threading.Event()
        self._calibration_payload = None
        self.zero_count = 0
        self.last_zero_ns = None
        self._sample_lock = threading.RLock()
        self.tare_offset = (0.0,) * 6
        self.tare_count = 0
        self.last_tare_ns = None
        self.tare_source_ns = None
        self._rate_receipts = deque(maxlen=4096)
        self._rate_window_ns = 2_000_000_000

    def _publish_sample(self, values):
        with self._sample_lock:
            self._latest = tuple(values)
            self._latest_ns = time.monotonic_ns()
            self._rate_receipts.append(self._latest_ns)
            while self._rate_receipts and self._rate_receipts[0] < self._latest_ns - self._rate_window_ns:
                self._rate_receipts.popleft()
            self.frames_received += 1
            self.last_error = None

    def _clear_sample_session(self):
        """Invalidate both latest force and rate across transport sessions."""
        with self._sample_lock:
            self._latest = self._latest_ns = None
            self._rate_receipts.clear()

    def _measurement_snapshot_locked(self, max_age_ms=500.0):
        raw, stamp = self._latest, self._latest_ns
        if (not self._running or not isinstance(stamp, int) or raw is None
                or len(raw) != 6 or not all(math.isfinite(v) for v in raw)):
            return None
        now_ns = time.monotonic_ns()
        age = (now_ns - stamp) / 1e6
        if not 0 <= age <= max_age_ms:
            return None
        # Host effective receipt rate, not ADC timing or baud. Identical payloads
        # are distinct received frames: this protocol has no sample sequence ID.
        receipts = self._rate_receipts
        while receipts and receipts[0] < now_ns - self._rate_window_ns:
            receipts.popleft()
        span_ns = receipts[-1] - receipts[0] if len(receipts) >= 2 else 0
        rate_hz = (len(receipts) - 1) * 1e9 / span_ns if span_ns > 0 else None
        return {
            'sample_rate_hz': rate_hz, 'rate_window_s': self._rate_window_ns / 1e9,
            'rate_sample_count': len(receipts),
            'raw': list(raw),
            'corrected': [v - offset for v, offset in zip(raw, self.tare_offset)],
            'sample_monotonic_ns': stamp, 'age_ms': age,
            'port': self.port, 'protocol': self.protocol,
            'tare': {'mode': 'host_offset', 'offset_six_axis': list(self.tare_offset),
                     'count': self.tare_count, 'monotonic_ns': self.last_tare_ns,
                     'source_sample_monotonic_ns': self.tare_source_ns},
        }

    def measurement_snapshot(self, max_age_ms=500.0):
        """Atomic fresh raw/corrected sample. Missing/stale stays None; no I/O."""
        with self._sample_lock:
            return self._measurement_snapshot_locked(max_age_ms)

    def tare(self):
        """Host-only six-axis offset: never sends a device calibration command."""
        with self._sample_lock:
            sample = self._measurement_snapshot_locked()
            if sample is None:
                raise RuntimeError('PX6D未连接、数据过期或六轴无效，不能采集力归零')
            self.tare_offset = tuple(sample['raw'])
            self.tare_count += 1
            self.last_tare_ns = time.monotonic_ns()
            self.tare_source_ns = sample['sample_monotonic_ns']
            return self._measurement_snapshot_locked()

    # ---------------------------------------------------------------- lifecycle

    def start(self):
        """Open the port and start the auto-push pump thread."""

        with self._lock:
            if self._running:
                return
            self._clear_sample_session()
            self._stop_event.clear()
            if self.auto_port:
                self._running = True
                self._thread = threading.Thread(target=self._auto_pump, name="px6d-auto-port", daemon=True)
                self._thread.start()
                return
            import serial

            try:
                self._ser = serial.Serial(self.port, self.baud, timeout=0.05)
            except Exception as exc:
                self.last_error = str(exc)
                raise
            self._running = True
            self._thread = threading.Thread(
                target=self._pump, name="paxini-px6d", daemon=True
            )
            self._thread.start()

    def stop(self):
        self._stop_event.set()
        with self._lock:
            self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.5)
            self._thread = None
        if self._ser is not None:
            try:
                if self.protocol == "legacy":
                    self._ser.write(build_frame(0x02, b"\x00", self.device_id))
                    self._ser.flush()
                    time.sleep(0.05)
                self._ser.close()
            except Exception:
                pass
            self._ser = None
        self._clear_sample_session()

    def close(self):
        self.stop()

    @staticmethod
    def _is_px6d_response(data):
        """Require header, known response length and XOR; never trust a COM name."""
        for offset in range(max(0, len(data) - 4)):
            if data[offset:offset + 2] != HEADER:
                continue
            cmd = data[offset + 3]
            if cmd in (0x01, 0x03) and len(data) >= offset + 29:
                frame = bytes(data[offset:offset + 29])
                if crc8(frame[:-1]) == frame[-1] and parse_six_axis(frame[4:-1]) is not None:
                    return True
            length = {0x01: 8, 0x02: 8, 0x03: 24}.get(cmd)
            if length is None or len(data) < offset + 5 + length:
                continue
            body = data[offset + 2:offset + 4 + length]
            if rc8(body) != data[offset + 4 + length]:
                continue
            if cmd != 0x03 or parse_six_axis(bytes(data[offset + 4:offset + 4 + length])) is not None:
                return True
        return False

    @staticmethod
    def _is_rs485_force_response(data):
        for i in range(max(0, len(data) - 28)):
            frame = bytes(data[i:i + 29])
            if (frame[:2] == HEADER and frame[3] == 3
                    and crc8(frame[:-1]) == frame[-1]
                    and parse_six_axis(frame[4:-1]) is not None):
                return True
        return False

    def _discover_port(self, probe_timeout=0.35):
        import serial
        from serial.tools import list_ports
        ports = [item for item in list_ports.comports()
                 if not str(getattr(item, 'hwid', '')).upper().startswith('BTHENUM')
                 and not (getattr(item, 'vid', None) == 0x0483
                          and getattr(item, 'pid', None) == 0x5740)]
        # Never probe Bluetooth links or the workstation's STM32 control port.
        # Try the previous successful port first, but never assume it still belongs to PX6D.
        ports.sort(key=lambda item: item.device != self.port)
        for item in ports:
            if self._stop_event.is_set():
                break
            connection = None
            try:
                connection = serial.Serial(item.device, self.baud, timeout=0.03, write_timeout=0.1)
                # Read-only "get frame" query: no zeroing or settings changes.
                connection.write(build_rs485_frame(0x05))
                connection.flush()
                data = bytearray()
                deadline = time.monotonic() + probe_timeout
                while time.monotonic() < deadline and not self._stop_event.is_set():
                    waiting = connection.in_waiting
                    if waiting:
                        data.extend(connection.read(min(waiting, 4096)))
                        if self._is_rs485_force_response(data):
                            self.port = item.device
                            self.protocol = "rs485"
                            self.last_error = None
                            return connection
                    self._stop_event.wait(0.01)
            except Exception:
                pass  # Busy/foreign ports are not sensor failures.
            if connection is not None:
                connection.close()
        self.last_error = "未发现PX6D，正在自动搜索串口（请关闭官方PX6D软件）"
        return None

    def _auto_pump(self):
        while self._running and not self._stop_event.is_set():
            self._ser = self._discover_port()
            if self._ser is None:
                self._stop_event.wait(2.0)
                continue
            self._clear_sample_session()
            try:
                self._pump_rs485() if self.protocol == "rs485" else self._pump()
            finally:
                try:
                    self._ser.close()
                except Exception:
                    pass
                self._ser = None
                self._clear_sample_session()
            self._stop_event.wait(0.5)

    def _pump_rs485(self):
        """RS485 cmd05 polling; CRC-validated cmd03 contains six float32 LE values."""
        buffer = bytearray()
        started = time.monotonic()
        next_query = 0.0
        while self._running and not self._stop_event.is_set():
            try:
                if time.monotonic() >= next_query:
                    self._send(build_rs485_frame(0x05))
                    next_query = time.monotonic() + 0.04
                waiting = self._ser.in_waiting
                if waiting:
                    buffer.extend(self._ser.read(min(waiting, 4096)))
                while len(buffer) >= 5:
                    start = buffer.find(HEADER)
                    if start < 0:
                        del buffer[:-1]
                        break
                    if start:
                        del buffer[:start]
                    if len(buffer) < 5:
                        break
                    cmd = buffer[3]
                    size = 29 if cmd in (1, 3) else 13 if cmd == 0x10 else 0
                    if not size:
                        del buffer[:1]
                        continue
                    if len(buffer) < size:
                        break
                    frame = bytes(buffer[:size])
                    if crc8(frame[:-1]) != frame[-1]:
                        del buffer[:1]
                        continue
                    del buffer[:size]
                    if cmd in (1, 3):
                        values = parse_six_axis(frame[4:-1])
                        if values is not None:
                            self._publish_sample(values)
                    else:
                        self._calibration_payload = frame[4:-1]
                        self._calibration_event.set()
                if time.monotonic() - (self._latest_ns / 1e9 if self._latest_ns else started) > 3:
                    self.last_error = "PX6D RS485数据超时，正在重新搜索串口"
                    return
                self._stop_event.wait(0.004)
            except Exception as exc:
                self.last_error = str(exc)
                return

    # ------------------------------------------------------------------- pump

    def _send(self, packet: bytes):
        try:
            with self._tx_lock:
                self._ser.write(packet)
                self._ser.flush()
        except Exception as exc:
            self.last_error = str(exc)

    def _pump(self):
        buffer = bytearray()
        # 0 = stop any foreign stream, 1 = start ours; then just listen.
        phase = 0
        connected_at = time.monotonic()
        phase_deadline = time.monotonic() + 0.15
        while self._running:
            try:
                waiting = self._ser.in_waiting
                if waiting:
                    buffer.extend(self._ser.read(waiting))
                if time.monotonic() >= phase_deadline:
                    if phase == 0:
                        self._send(build_frame(0x02, b"\x00", self.device_id))
                        phase = 1
                    elif phase == 1:
                        self._send(build_frame(0x02, b"\x01", self.device_id))
                        phase = 2
                    buffer.clear()
                    phase_deadline = time.monotonic() + 0.15
                if self.auto_port and time.monotonic() - (self._latest_ns / 1e9 if self._latest_ns else connected_at) > 3.0:
                    self.last_error = "PX6D数据超时，正在重新搜索串口"
                    return
                consumed = self._extract(buffer)
                if not consumed:
                    time.sleep(0.004)
            except Exception as exc:
                self.last_error = str(exc)
                if self.auto_port:
                    return
                time.sleep(0.05)

    def _extract(self, buffer: bytearray) -> bool:
        start = buffer.find(HEADER)
        if start < 0:
            if len(buffer) > 512:
                del buffer[:-2]
            return False
        if start > 0:
            del buffer[:start]
            return True
        if len(buffer) < 5:
            return False
        cmd = buffer[3]
        expected = PUSH_DATA_LENGTH.get(cmd)
        if expected is None:
            del buffer[:1]
            return True
        if len(buffer) < 4 + expected + 1:
            return False
        frame = bytes(buffer[:4 + expected + 1])
        if rc8(frame[2:-1]) != frame[-1]:
            del buffer[:1]
            return True
        payload = frame[4:-1]
        del buffer[:4 + expected + 1]
        if cmd == 0x03:
            values = parse_six_axis(payload)
            if values is not None:
                self._publish_sample(values)
        elif cmd == 0x10:
            self._calibration_payload = payload
            self._calibration_event.set()
        return True

    # ------------------------------------------------------------------- read

    def calibrate_zero(self, timeout_s: float = 3.0):
        """Run the official PX6D recalibration/zero command (cmd 0x10).

        The official host sends this as a broadcast (device id 0x7F). Keep the
        sensor unloaded while it calibrates. Returns the 8-byte ACK payload.
        """

        if not self._running or self._ser is None:
            raise RuntimeError("PX6D未连接，无法归零")
        self._calibration_payload = None
        self._calibration_event.clear()
        self._send(
            build_rs485_frame(0x10) if self.protocol == "rs485"
            else build_frame(0x10, b"\x01", device_id=0x7F)
        )
        if not self._calibration_event.wait(timeout_s):
            raise TimeoutError("PX6D归零命令未确认")
        self.zero_count += 1
        self.last_zero_ns = time.monotonic_ns()
        return self._calibration_payload

    def read_six_axis(self, timeout_s: float = 1.0):
        """Legacy raw read (no host offset); GUI/capture use measurement_snapshot()."""

        deadline = time.monotonic() + timeout_s
        frames0 = self.frames_received
        while time.monotonic() < deadline:
            if self._latest is not None and self.frames_received > frames0:
                return self._latest
            if (
                self._latest is not None
                and self.frames_received == frames0
                and self._latest_ns is not None
                and (time.monotonic_ns() - self._latest_ns) / 1e6 < 500.0
            ):
                return self._latest
            time.sleep(0.01)
        raise TimeoutError(self.last_error or "PX6D 自动回传无数据")
