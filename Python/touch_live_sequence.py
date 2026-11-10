"""Torch-free live causal continuity. Single-slot drops break source continuity."""
import math

class LiveSequence:
    def __init__(self, bundle):
        self.bundle=bundle
        config=getattr(bundle,'force_config',{})
        self.gap_s=float(config.get('gap_s',config.get('max_gap_s',30.)))
        self.previous=None

    @staticmethod
    def _clock(value):
        try:
            value=float(value)
            return value if math.isfinite(value) else None
        except (ValueError,TypeError):
            return None

    def before(self, request):
        current=(request.get('session_id'), request.get('source_seq',request.get('request_id')), self._clock(request.get('timestamp_s')))
        previous=self.previous
        reason=None
        if request.get('reset'):
            reason='explicit'
        elif previous is None or current[0]!=previous[0]:
            reason='new_session'
        elif current[1] is None or previous[1] is None or current[1]!=previous[1]+1:
            reason='source_gap'
        elif current[2] is not None and previous[2] is not None:
            delta=current[2]-previous[2]
            if delta<=0: reason='clock_nonincreasing'
            elif delta>self.gap_s: reason='time_gap'
        if reason and hasattr(self.bundle,'reset_state'):
            self.bundle.reset_state()
        self.previous=current
        return reason

    def invalidate(self):
        self.previous=None
        if hasattr(self.bundle,'reset_state'): self.bundle.reset_state()
