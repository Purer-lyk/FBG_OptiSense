"""Torch-free signed wavelength display state. No measured/model inputs changed."""
import math
import time
import numpy as np

def finite(value):
    return isinstance(value, (int,float,np.number)) and math.isfinite(value)

PEAK_LABELS = (25, 15, 5, 3, 13, 23, 21, 11, 1)

class SignedWavelengthState:
    def __init__(self):
        self.context = None
        self.current = [None] * 9
        self.valid = [False] * 9
        self.baseline = None
        self.baseline_time = None
        self.sequence = None
        self.received = None

    def reset_context(self, context):
        if context != self.context:
            self.context = context
            self.baseline = self.baseline_time = None
            self.current = [None] * 9
            self.valid = [False] * 9
            self.sequence = self.received = None

    def update(self, fits, context, sequence, now):
        self.reset_context(context)
        if self.sequence is not None and sequence is not None and sequence <= self.sequence:
            self.baseline = self.baseline_time = None
        self.sequence = sequence
        self.received = now
        fits = list(fits or [])
        self.current = [None] * 9
        self.valid = [False] * 9
        if len(fits) == 9:
            for i, f in enumerate(fits):
                v = f.get('center_nm')
                self.current[i] = float(v) if finite(v) else None
                self.valid[i] = finite(v)  # Data validity only; legacy peak-quality flags are ignored.

    @property
    def deltas(self):
        if self.baseline is None:
            return [None] * 9
        return [v-b if v is not None else None for v,b in zip(self.current, self.baseline)]

    def stale(self, now):
        return self.received is None or now-self.received > 2.

    def set_baseline(self, now):
        if self.stale(now) or not all(self.valid):
            return False
        self.baseline = list(self.current)
        self.baseline_time = time.strftime('%Y-%m-%d %H:%M:%S')
        return True

def signed_colors(values, limit=.1):
    """Fixed symmetric blue/neutral/red; saturate color only."""
    if not finite(limit) or limit <= 0:
        raise ValueError('range must be positive')
    values = np.array([v if finite(v) else np.nan for v in values], dtype=float)
    neutral = np.array([.88,.89,.90])
    blue = np.array([.08,.30,.86])
    red = np.array([.86,.12,.16])
    t = np.clip(np.abs(values)/limit,0,1)[:,None]
    target = np.where((values<0)[:,None], blue, red)
    colors = np.column_stack((neutral*(1-t)+target*t, np.full(len(values),.93)))
    colors[~np.isfinite(values)] = (.62,.66,.70,.68)
    return colors

def spatial_field(points, anchors, values):
    """Display-only inverse-distance field; not extra measured sensors."""
    points, anchors = np.asarray(points), np.asarray(anchors)
    values = np.array([v if finite(v) else np.nan for v in values],dtype=float)
    d2 = np.sum((points[:,None,:]-anchors[None,:,:])**2,axis=2)
    weights = 1/np.maximum(d2,1e-12)
    weights[:,~np.isfinite(values)] = 0
    total = weights.sum(axis=1)
    result = np.full(len(points),np.nan)
    valid = total>0
    result[valid] = (weights@np.nan_to_num(values))[valid]/total[valid]
    rows, cols = np.where(d2 < 1e-12)
    result[rows] = values[cols]
    return result
