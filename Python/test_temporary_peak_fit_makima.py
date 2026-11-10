"""Regression checks for the optional, non-certified Makima display fitter."""

import unittest

import numpy as np

from temporary_peak_fit import fit_five_points as fit_gaussian
from temporary_peak_fit_makima import fit_five_points as fit_makima


class MakimaDisplayTests(unittest.TestCase):
    def test_interpolates_five_samples_without_mutating_input(self):
        x = 1535.0 + np.linspace(-0.4, 0.4, 5)
        y = np.array([45.0, 250.0, 720.0, 500.0, 130.0])
        original = y.copy()
        fit = fit_makima(x, y)
        self.assertTrue(fit['fit_available'])
        self.assertEqual(fit['fit_method'], 'makima_piecewise_cubic_5point')
        self.assertFalse(fit['optical_accuracy_verified'])
        self.assertTrue(x[0] <= fit['center_nm'] <= x[-1])
        np.testing.assert_allclose(
            np.interp(x, fit['curve_x_nm'], fit['curve_adc']), y, atol=0.3
        )
        np.testing.assert_array_equal(y, original)

    def test_peak_is_global_dbm_max_on_rendered_makima_curve(self):
        from temporary_optical_units import adc_code_to_dbm
        x = np.array([1534.6, 1534.8, 1535.0, 1535.2, 1535.4])
        y = np.array([45., 250., 720., 500., 130.])
        fit = fit_makima(x, y)
        powers = adc_code_to_dbm(fit['curve_adc'], 0, floor_nonpositive=True)
        index = int(np.argmax(powers))
        self.assertGreater(max(fit['curve_adc']), max(y))
        self.assertNotEqual(fit['curve_x_nm'][index], x[2])
        self.assertEqual(fit['center_nm'], fit['curve_x_nm'][index])
        self.assertEqual(fit['peak_adc'], fit['curve_adc'][index])
        self.assertEqual(fit['peak_dbm'], powers[index])

    def test_flat_max_uses_first_real_point(self):
        x = [1534.6, 1534.8, 1535.0, 1535.2, 1535.4]
        fit = fit_makima(x, [123.] * 5)
        self.assertEqual(fit['center_nm'], x[0])
        self.assertEqual(fit['peak_adc'], 123.)
        self.assertNotIn('quality_warning', fit)
        self.assertTrue(fit['fit_available'])

    def test_default_gaussian_remains_separate_and_more_accurate_on_gaussian(self):
        x = 1535.0 + np.linspace(-0.6, 0.6, 5)
        actual_center = 1534.78
        y = 30.0 + 700.0 * np.exp(-0.5 * ((x - actual_center) / 0.25) ** 2)
        gaussian = fit_gaussian(x, y)
        makima = fit_makima(x, y)
        self.assertAlmostEqual(gaussian['center_nm'], actual_center, delta=0.002)
        self.assertGreater(abs(makima['center_nm'] - actual_center), 0.01)
        self.assertFalse(makima['optical_accuracy_verified'])


if __name__ == '__main__':
    unittest.main()
