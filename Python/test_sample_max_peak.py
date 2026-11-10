"""Hardware-free Makima curve-global-maximum peak/display/capture contract."""
import os
import sys
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(Path(sys.prefix) / 'Lib/site-packages/PyQt5/Qt5/plugins/platforms')
import json
import unittest
import numpy as np
from unittest.mock import patch
from scipy.interpolate import Akima1DInterpolator
from temporary_peak_fit_makima import fit_five_points, fit_temporary_frame
from temporary_optical_units import adc_code_to_dbm

def expected_peak(x, y, selector=0):
    grid = np.unique(np.r_[np.linspace(min(x), max(x), 401), x])
    values = Akima1DInterpolator(x, y, method="makima", extrapolate=False)(grid)
    dbm = adc_code_to_dbm(values, selector, floor_nonpositive=True)
    i = int(np.argmax(dbm))
    return float(grid[i]), float(values[i]), float(dbm[i])


class CurveMaxTests(unittest.TestCase):
    def test_grid_resolution_and_virtual_peak_provenance(self):
        x = [1534.6, 1534.79, 1535., 1535.21, 1535.4]
        y = [45, 250, 720, 500, 130]
        fit = fit_five_points(x, y)
        self.assertNotIn(fit['center_nm'], x)
        self.assertEqual(fit['peak_grid_uniform_point_count'], 401)
        self.assertAlmostEqual(fit['peak_grid_step_nm'], (x[-1]-x[0])/400)
        self.assertEqual(fit['peak_grid_point_count'], len(fit['curve_x_nm']))
        self.assertTrue(fit['peak_grid_approximation'])
        self.assertEqual(fit['peak_tie_rule'], 'first_in_wavelength_order')
        np.testing.assert_array_equal(fit['curve_x_nm'], np.unique(np.r_[np.linspace(x[0],x[-1],401),x]))

    def test_global_multipeak_and_endpoints_survive_shape_diagnostics(self):
        x = [1., 2., 3., 4., 5.]
        for y in ([100, 1000, 100, 1100, 100], [900,700,500,300,100], [100,300,500,700,900]):
            with self.subTest(y=y), patch('temporary_peak_fit_makima._gaussian_shape_diagnostics', create=True, side_effect=AssertionError('removed proxy called')):
                fit = fit_five_points(x,y)
                self.assertTrue(fit['fit_available'])
                self.assertNotIn('quality_warning', fit)
                self.assertEqual((fit['center_nm'],fit['peak_adc'],fit['peak_dbm']), expected_peak(x,y))

    def test_flat_and_floor_ties_use_first_curve_wavelength(self):
        for y in ([123.]*5, [0.]*5, [0.,.1,0.,.1,0.]):
            fit = fit_five_points([1.,2.,3.,4.,5.],y)
            self.assertEqual(fit['center_nm'],1.)
            self.assertEqual(fit['peak_curve_index'],0)
            powers = adc_code_to_dbm(fit['curve_adc'],0,floor_nonpositive=True)
            self.assertEqual(fit['peak_dbm'],float(np.max(powers)))

    def test_ties_edges_irregular_and_invalid_samples(self):
        x = [1534.6, 1534.8, 1535., 1535.2, 1535.4]
        for y in ([0, 720, 720, 500, 0], [720, 500, 400, 300, 0],
                  [0, 300, 400, 500, 720], [500, 2, 600, 4, 700]):
            with self.subTest(y=y):
                fit = fit_five_points(x, y)
                center, adc, dbm = expected_peak(x, y)
                self.assertTrue(fit['fit_available'])
                self.assertEqual(fit['center_nm'], center)
                self.assertEqual(fit['peak_adc'], adc)
        for xx, yy in ((x, [1, 2, float('nan'), 4, 5]),
                       (x, [1, 2, -1, 4, 5]), (x[:-1], [1] * 4),
                       ([1, 2, 2, 4, 5], [1] * 5)):
            with self.subTest(x=xx, y=yy):
                self.assertIsNone(fit_five_points(xx, yy)['center_nm'])

    def test_curve_max_does_not_require_spline_derivatives(self):
        with patch.object(Akima1DInterpolator, 'derivative', side_effect=AssertionError('no derivative')):
            fit = fit_five_points([1, 2, 3, 4, 5], [45, 250, 720, 500, 130])
        self.assertEqual((fit['center_nm'], fit['peak_adc'], fit['peak_dbm']), expected_peak([1,2,3,4,5], [45,250,720,500,130]))

    def test_dbm_floor_not_adc_argmax_and_actual_channel_selector(self):
        from temporary_optical_units import adc_code_to_dbm, transimpedance_ohm
        x = [1., 2., 3., 4., 5.]
        # A zero code uses the half-LSB dBm floor, above a positive 0.1 code.
        for channel in range(4):
            for selector in range(4):
                with self.subTest(channel=channel, selector=selector):
                    effective = selector if channel < 2 else 0
                    fit = fit_five_points(x, [0., .1, 0., .1, 0.],
                                          feedback_selector=selector, adc_channel=channel)
                    self.assertEqual(fit['center_nm'], x[0])
                    self.assertEqual(fit['peak_adc'], 0.)
                    self.assertEqual(fit['peak_dbm'], float(adc_code_to_dbm(0., effective, floor_nonpositive=True)))
                    self.assertEqual(fit['transimpedance_ohm'], transimpedance_ohm(effective))
                    self.assertEqual(fit['peak_selection_method'], 'makima_curve_global_max_dbm')

    def test_capture_records_same_curve_max_as_live_fitter(self):
        from manual_stress_capture import build_training_views
        from test_manual_stress_capture import temporary_frame
        x = [1525. + .02 * i for i in range(45)]
        frame = temporary_frame(sequence=16)
        for i, record in enumerate(frame['records']):
            record['second_code'] = [45, 250, 720, 500, 130][i % 5]
        views = build_training_views(frame, x)
        saved = json.loads(json.dumps(views))
        live = fit_temporary_frame([{'measured_nm': v} for v in x], views['stable_adc'], feedback_selector=frame['feedback_selector'])
        for i, fit in enumerate(saved['makima_peak_fits']):
            self.assertEqual((fit['center_nm'], fit['peak_adc'], fit['peak_dbm']), expected_peak(x[5*i:5*i+5], [45,250,720,500,130], frame['feedback_selector']))
            self.assertEqual(fit['peak_source'], 'interpolated')
            self.assertNotIn('peak_sample_index', fit)
            self.assertEqual(len(fit['input_adc']), 5)
            self.assertEqual(fit['peak_curve_index'], int(np.argmax(fit['curve_dbm'])))
            self.assertEqual(fit['fit_method'], live[i]['fit_method'])


class CurveMaxQtTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt5 import QtCore, QtWidgets
        QtCore.QCoreApplication.setLibraryPaths([str(Path(sys.prefix) / 'Lib/site-packages/PyQt5/Qt5/plugins')])
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_live_frame_dbm_marker_and_capture_use_actual_front_end(self):
        from temporary_test_widget import TemporaryTestWindow
        from temporary_test_mode import select_points, decode, quality
        from test_temporary_test_mode import reference, frame
        from manual_stress_capture import build_training_views
        from temporary_optical_units import adc_code_to_dbm
        import test_temporary_test_widget as existing
        sandbox = existing.WidgetTests(); sandbox.setUp()
        self.addCleanup(sandbox.doCleanups)
        window = TemporaryTestWindow(None)
        try:
            table, x, y = reference()
            window.plan = select_points(table, x, y)
            window.render_plan()
            window._set_display_mode('dbm')
            for channel in range(4):
                for selector in range(4):
                    with self.subTest(channel=channel, selector=selector):
                        data = decode(frame(), 123, 456)
                        data.update(adc_channel=channel, feedback_selector=selector,
                                    cycle_start_us=100000 + (channel * 4 + selector) * 70000)
                        for i, record in enumerate(data['records']):
                            record['second_code'] = [45, 250, 720, 500, 130][i % 5]
                        data['quality'] = quality(data)
                        window.update_frame(data)
                        self.assertIn('曲线', window.peak_fit_table.verticalHeaderItem(0).text())
                        self.assertIn('401', window.peak_fit_status.text())
                        self.assertNotIn('实测最大', window.peak_fit_status.text())
                        axis = [r['measured_nm'] for r in window.plan['rows']]
                        saved = build_training_views(data, axis)['makima_peak_fits']
                        effective = selector if channel < 2 else 0
                        center, peak_adc, expected = expected_peak(axis[:5], [45,250,720,500,130], effective)
                        self.assertEqual(window.latest_feedback_selector, effective)
                        for i, fit in enumerate(window.peak_fits):
                            self.assertEqual(fit['center_nm'], expected_peak(axis[i*5:i*5+5], [45,250,720,500,130], effective)[0])
                            self.assertEqual(window.fit_labels[i].pos().x(), fit['center_nm'])
                            self.assertEqual(fit['peak_dbm'], expected)
                            self.assertEqual(window.fit_labels[i].pos().y(), expected)
                            self.assertEqual(saved[i]['peak_dbm'], expected)
                            self.assertEqual(saved[i]['center_nm'], fit['center_nm'])
                        np.testing.assert_allclose(window.curves[0].getData()[1],
                            adc_code_to_dbm([45, 250, 720, 500, 130], effective))
        finally:
            window.close()

    def test_actual_label_uses_curve_peak_not_amplitude_fields(self):
        from temporary_test_widget import TemporaryTestWindow
        import test_temporary_test_widget as existing
        sandbox = existing.WidgetTests(); sandbox.setUp()
        self.addCleanup(sandbox.doCleanups)
        window = TemporaryTestWindow(None)
        try:
            window._set_display_mode('adc')
            fit = fit_five_points([1534.6, 1534.8, 1535., 1535.2, 1535.4], [45, 250, 720, 500, 130])
            # Drawing must honor curve argmax, not diagnostic amplitude fields.
            fit['amplitude_adc'] = 1000.
            window.peak_fits = [fit]
            window.redraw_peak_fits()
            self.assertEqual(window.fit_labels[0].pos().x(), fit['center_nm'])
            self.assertEqual(window.fit_labels[0].pos().y(), fit['peak_adc'])
            self.assertIn('Makima曲线最大', window.fit_labels[0].toPlainText())
        finally:
            window.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
