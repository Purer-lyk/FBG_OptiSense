"""Bounded Makima peak wavelength shifts in pm, relative to fixed stable-window means."""
from collections import deque
import math
from statistics import fmean


class PeakWavelengthHistory:
    def __init__(self, window_s=60., capacity=2000, stable_s=2.,
                 stable_frames=10, stable_range_pm=1., warmup_frames=16):
        self.window_s = window_s
        self.stable_s = stable_s
        self.stable_frames = stable_frames
        self.stable_range_pm = stable_range_pm
        self.warmup_frames = warmup_frames
        self.pending_centers = [deque(maxlen=capacity) for _ in range(9)]
        self.samples = [deque(maxlen=capacity) for _ in range(9)]
        self.clear()

    def clear(self):
        for samples in self.samples:
            samples.clear()
        for pending in self.pending_centers:
            pending.clear()
        self.baselines_nm = [None] * 9
        self.origin_us = self.last_clock_us = self.last_sequence = None
        self.identity = None
        self.discontinuous = False

    def append(self, sequence, clock_us, peak_fits, identity):
        if self.identity != identity:
            self.clear()
            self.identity = identity
        try:
            sequence = int(sequence)
            clock_us = float(clock_us)
        except (TypeError, ValueError, OverflowError):
            self.discontinuous = True
            return False
        if not math.isfinite(clock_us):
            self.discontinuous = True
            return False
        if ((self.last_sequence is not None and sequence <= self.last_sequence)
                or (self.last_clock_us is not None and clock_us <= self.last_clock_us)):
            self.discontinuous = True
            return False
        if self.origin_us is None:
            self.origin_us = clock_us
        t = (clock_us - self.origin_us) / 1e6
        gap = (self.discontinuous or (self.last_sequence is not None
               and sequence != self.last_sequence + 1)
               or (self.last_clock_us is not None
                   and clock_us - self.last_clock_us > self.window_s * 1e6))
        if gap:
            for pending in self.pending_centers:
                pending.clear()
        for i, samples in enumerate(self.samples):
            if gap and samples:
                samples.append((t, math.nan))
            result = peak_fits[i] if i < len(peak_fits) else {}
            try:
                value = float(result.get('center_nm'))
            except (TypeError, ValueError, OverflowError):
                value = math.nan
            pending = self.pending_centers[i]
            if not math.isfinite(value) or sequence < self.warmup_frames:
                pending.clear()
                value = math.nan
            else:
                if self.baselines_nm[i] is None:
                    pending.append((t, value))
                    # Retain the sample bracketing the start of the time window.
                    while len(pending) > 1 and pending[1][0] <= t - self.stable_s:
                        pending.popleft()
                    centers = [v for _, v in pending]
                    if (len(pending) >= self.stable_frames
                            and t - pending[0][0] >= self.stable_s
                            and (max(centers) - min(centers)) * 1000. <= self.stable_range_pm):
                        self.baselines_nm[i] = fmean(centers)
                        pending.clear()
                value = ((value - self.baselines_nm[i]) * 1000.
                         if self.baselines_nm[i] is not None else math.nan)
            samples.append((t, value))
            while samples and samples[0][0] < t - self.window_s:
                samples.popleft()
        self.last_clock_us, self.last_sequence = clock_us, sequence
        self.discontinuous = False
        return True
