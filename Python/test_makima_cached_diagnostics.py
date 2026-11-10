"""Regression: Makima no longer has a Gaussian quality proxy."""
import unittest
from unittest.mock import patch
import numpy as np
import temporary_peak_fit_makima as m

class NoDiagnosticTests(unittest.TestCase):
    def test_no_gaussian_proxy_for_5_7_regular_irregular_and_noisy(self):
        rng=np.random.default_rng(7)
        self.assertFalse(hasattr(m,'_gaussian_shape_basis'))
        self.assertFalse(hasattr(m,'_gaussian_shape_diagnostics'))
        with patch.object(m,'_gaussian_shape_diagnostics',create=True,
                          side_effect=AssertionError('Gaussian proxy called')):
            for count in (5,7):
                for irregular in (False,True):
                    x=1530+np.arange(count)*.02
                    if irregular: x[1:-1]+=rng.uniform(-.004,.004,count-2)
                    for y in (np.zeros(count),np.full(count,40.),rng.uniform(0,4095,count)):
                        fit=m.fit_peak_points(x,y)
                        self.assertTrue(fit['fit_available'])
                        self.assertTrue(np.isfinite(fit['center_nm']))
                        self.assertNotIn('valid',fit)
                        self.assertNotIn('r_squared',fit)
                        self.assertNotIn('quality_warning',fit)
    def test_adc_dependent_results_remain_live(self):
        x=1530+np.arange(7)*.02
        y=np.array([20,200,1000,2000,1000,200,20])
        first=m.fit_peak_points(x,y)
        second=m.fit_peak_points(x,y+np.array([0,0,0,0,1400,0,0]))
        self.assertNotEqual(first['center_nm'],second['center_nm'])

if __name__=='__main__': unittest.main()
