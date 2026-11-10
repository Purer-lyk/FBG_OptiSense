import os
import sys
import unittest
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(Path(sys.prefix)/'Lib/site-packages/PyQt5/Qt5/plugins/platforms')
from PyQt5 import QtCore,QtWidgets
from test_manual_stress_capture import FakeTemporaryPage, temporary_frame
from manual_stress_capture import ManualStressCaptureWindow
from signed_wavelength import PEAK_LABELS

class SignedCaptureQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        QtCore.QCoreApplication.setLibraryPaths([str(Path(sys.prefix)/'Lib/site-packages/PyQt5/Qt5/plugins')])
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_listener_baseline_mesh_and_same_frame_fit_binding(self):
        import numpy as np
        source = FakeTemporaryPage()
        source.plan = {'rows':[dict(index=i,measured_nm=1530+i*.01) for i in range(45)]}
        win = ManualStressCaptureWindow(source,wavelength_provider=lambda:source.reference_wavelengths,output_root=Path(os.environ['TMPDIR']))
        try:
            self.assertTrue(hasattr(win,'wavelength_baseline_button'),'explicit wavelength baseline UI missing')
            win.attach()
            f = temporary_frame()
            fits = [dict(center_nm=1530+i,valid=True) for i in range(9)]
            source.makima_frame_snapshot = (f, fits)
            source.frame_listeners[0](f)
            self.assertIs(win.latest_peak_fits, fits, 'must consume same computed fits, not refit')
            win.wavelength_baseline_button.click()
            self.assertIsNotNone(win.wavelength_state.baseline)
            positions = win.surface_map.marker_positions.copy()
            f = temporary_frame(sequence=17)
            fits = [dict(center_nm=1530+i+(.05 if i%2 else -.05),valid=True) for i in range(9)]
            source.makima_frame_snapshot=(f,fits)
            source.frame_listeners[0](f)
            win._refresh_wavelength_display(force=True)
            self.assertLess(win.wavelength_state.deltas[0],0)
            self.assertGreater(win.wavelength_state.deltas[1],0)
            np.testing.assert_equal(win.surface_map.marker_positions,positions)
            np.testing.assert_equal(win.surface_map.sensor_positions,positions[np.array(PEAK_LABELS)-1])
            self.assertGreater(np.ptp(win.surface_map.wavelength_vertex_colors[:,:3]),.1)
            self.assertEqual(len(win.grid_buttons),25)
            self.assertIn('-0.0500',win.wavelength_table.item(0,3).text())
            source.frame_listeners[0](temporary_frame(sequence=18))
            self.assertEqual(win.wavelength_state.deltas,[None]*9,'never reuse another frame fits')
            source.plan['rows'][0]['index']=999
            win._refresh_wavelength_display(force=True)
            self.assertIsNone(win.wavelength_state.baseline)
            # Legacy quality metadata must not gate a complete finite baseline.
            f = temporary_frame(sequence=19); f['quality'].update(clipped=True, unsettled=True, warmed_up=False)
            fits = [dict(center_nm=1530+i, valid=False, quality_warning=True, r_squared=0.) for i in range(9)]
            source.makima_frame_snapshot = (f, fits)
            source.frame_listeners[0](f)
            win.wavelength_baseline_button.click()
            self.assertIsNotNone(win.wavelength_state.baseline)
            self.assertNotIn('质量', win.wavelength_status.text())
            self.assertNotIn('⚠', win.wavelength_table.item(0,1).text())
            # Transport endpoint changes invalidate reference, even without a new frame.
            f = temporary_frame(sequence=20)
            source.wavelength_transport_context = ('USB','COM6')
            source.makima_frame_snapshot = (f, fits)
            source.frame_listeners[0](f)
            win.wavelength_baseline_button.click()
            self.assertIsNotNone(win.wavelength_state.baseline)
            source.wavelength_transport_context = ('LAN','192.0.2.10')
            win._refresh_wavelength_display(force=True)
            self.assertIsNone(win.wavelength_state.baseline)
            source.selected_channel = lambda: 0
            source.effective_feedback_selector = lambda channel: 2
            f = temporary_frame(sequence=21)  # CH1 frame must not become CH0 reference.
            source.makima_frame_snapshot=(f,fits)
            source.frame_listeners[0](f)
            self.assertEqual(win.wavelength_state.current,[None]*9)
        finally:
            win.close()

class BaselineValidityQtTests(SignedCaptureQtTests):
    def test_latest_invalid_frame_cannot_reuse_recent_valid_centers(self):
        from unittest.mock import patch
        import time
        source=FakeTemporaryPage()
        source.plan={'rows':[dict(index=i,measured_nm=1530+i*.01) for i in range(45)]}
        win=ManualStressCaptureWindow(source,wavelength_provider=lambda:source.reference_wavelengths,output_root=Path(os.environ['TMPDIR']))
        self.addCleanup(win.close)
        win.attach()
        fits=[dict(center_nm=1530+i,valid=False) for i in range(9)]
        good=temporary_frame();source.makima_frame_snapshot=(good,fits)
        source.frame_listeners[0](good)
        for value in (float('nan'),float('inf'),float('-inf')):
            with self.subTest(value=value):
                source.makima_frame_snapshot=(good,fits)
                source.frame_listeners[0](good)
                bad=temporary_frame(sequence=17);bad['records'][0]['second_code']=value
                source.makima_frame_snapshot=(bad,fits)
                source.frame_listeners[0](bad)
                win.wavelength_baseline_button.click()
                self.assertIsNone(win.wavelength_state.baseline,'new invalid frame must not reuse old centers')
        # Completeness/nonfinite/staleness remain explicit data-validity failures.
        for case in ('missing','incomplete','nonfinite','stale','stopped','bus'):
            with self.subTest(case=case):
                f=temporary_frame(sequence=18)
                current=[dict(center_nm=1530+i) for i in range(9)]
                if case=='incomplete': current.pop()
                if case=='nonfinite': current[2]['center_nm']=float('inf')
                source.makima_frame_snapshot=None if case=='missing' else (f,current)
                source.peak_trend_stopped=case=='stopped'
                if case=='bus': f['spi_errors']=1
                source.frame_listeners[0](f)
                if case=='stale': win.wavelength_state.received=time.monotonic()-3
                win.wavelength_baseline_button.click()
                self.assertIsNone(win.wavelength_state.baseline)
                self.assertIn('未设置',win.wavelength_status.text())

if __name__ == '__main__': unittest.main()
