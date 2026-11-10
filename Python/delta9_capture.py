"""Nine signed Makima peak deltas: separate schema/storage, never ADC features."""
import copy
import math
from dataclasses import dataclass
from finger_dataset import FingerSessionWriter

SCHEMA='fbg-touch-delta9/v1'
DATASET_KIND='manual_touch_delta9'
INPUT_CONTRACT='delta_wavelength9'
PEAK_TO_POSITION=(25,15,5,3,13,23,21,11,1)

@dataclass(frozen=True)
class Delta9Frame:
    sequence: int
    monotonic_ns: int
    current: tuple
    baseline: tuple
    baseline_context: dict
    baseline_time: str
    quality: dict

    def to_record(self):
        if len(self.current)!=9 or len(self.baseline)!=9:
            raise ValueError('delta9 requires nine current and baseline wavelengths')
        values=[float(c)-float(b) for c,b in zip(self.current,self.baseline)]
        if not all(math.isfinite(float(v)) for v in (*self.current,*self.baseline,*values)):
            raise ValueError('delta9 wavelengths must be finite')
        return copy.deepcopy(dict(schema=SCHEMA,feature_kind=INPUT_CONTRACT,feature_units='nm',input_shape=[9],
            sequence=self.sequence,monotonic_ns=self.monotonic_ns,values=values,delta_wavelength_nm=values,
            current_peak_wavelengths_nm=list(self.current),baseline_wavelengths_nm=list(self.baseline),
            peak_order=[f'P{i}' for i in range(1,10)],peak_to_position_labels=list(PEAK_TO_POSITION),
            baseline_context=self.baseline_context,baseline_time=self.baseline_time,
            peak_selection_method='makima_curve_global_max_dbm',quality=self.quality))

class Delta9SessionWriter(FingerSessionWriter):
    """Reuse loss-aware queue/LIFO/event storage without SpectrumFrame conversion."""
    def _write_manifest(self):
        self._manifest.update(schema=SCHEMA,dataset_kind=DATASET_KIND,input_contract=INPUT_CONTRACT,
            input_shape=[9],feature_units='nm',training_eligible=False,peak_order=[f'P{i}' for i in range(1,10)],
            peak_to_position_labels=list(PEAK_TO_POSITION))
        for field in ('adc','points_per_peak','digital_gain_policy','sensor_channel'):
            self._manifest.pop(field,None)
        super()._write_manifest()

    def write_frame(self,frame):
        if not isinstance(frame,Delta9Frame):
            raise TypeError('delta9 writer accepts only Delta9Frame, never raw spectrum')
        self._enqueue('frame',frame.to_record(),frame.monotonic_ns)
    append_frame=write_frame
    submit_frame=write_frame
