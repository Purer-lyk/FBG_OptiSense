import unittest
from signed_wavelength import SignedWavelengthState

class NoPeakQualityTests(unittest.TestCase):
    def test_finite_complete_baseline_ignores_legacy_peak_flags(self):
        state = SignedWavelengthState()
        fits = [dict(center_nm=1530+i, valid=False, quality_warning=True,
                     quality_issues=['flat'], r_squared=0., fit_available=True) for i in range(9)]
        state.update(fits, ('machine',), 16, 10.)
        self.assertTrue(state.set_baseline(10.), 'finite same-frame centers must not be quality-gated')
        self.assertEqual(state.baseline, [1530+i for i in range(9)])

class MakimaNoQualityTests(unittest.TestCase):
    def test_finite_curves_have_no_peak_quality_diagnostics(self):
        import temporary_peak_fit_makima as m
        from unittest.mock import patch
        import numpy as np
        x = np.linspace(1530., 1530.8, 5)
        # Spy on the former diagnostic if present: default Makima must never call it.
        with patch.object(m, '_gaussian_shape_diagnostics', create=True,
                          side_effect=AssertionError('Gaussian quality proxy executed')):
            for y in ([40.]*5, [0,4095,30,4095,0], [45,250,720,500,130]):
                fit = m.fit_five_points(x,y)
                self.assertTrue(fit['fit_available'])
                self.assertTrue(np.isfinite(fit['center_nm']))
                for key in ('valid','quality_warning','quality_issues','r_squared','rmse_adc','quality_metric'):
                    self.assertNotIn(key,fit)
                self.assertNotIn('⚠',fit['reason'])
                self.assertNotIn('质量',fit['reason'])

class HiddenGaussianNoQualityTests(unittest.TestCase):
    def test_hidden_gaussian_keeps_residual_but_no_quality_verdict(self):
        from temporary_peak_fit import fit_five_points
        fit = fit_five_points([1.,2.,3.,4.,5.], [1.,1.,1.,1.,1.])
        self.assertTrue(fit['fit_available'])
        self.assertIn('rmse_adc',fit)
        for key in ('valid','quality_warning','quality_issues'):
            self.assertNotIn(key,fit)
        self.assertNotIn('质量',fit['reason'])
        self.assertNotIn('⚠',fit['reason'])

class WarmupNoQualityGateTests(unittest.TestCase):
    def test_complete_finite_frame_is_fitted_even_with_unverified_warmup(self):
        from temporary_peak_fit_makima import fit_temporary_frame as makima
        from temporary_peak_fit import fit_temporary_frame as gaussian
        rows = [dict(measured_nm=float(i)) for i in range(5)]
        for fitter in (makima,gaussian):
            fits = fitter(rows,[40.]*5,warmed_up=False)
            self.assertTrue(fits[0]['fit_available'])
            self.assertIsNotNone(fits[0]['center_nm'])
            self.assertNotIn('quality_warning',fits[0])

class OfflineNoPeakQualityTests(unittest.TestCase):
    def test_equal_finite_flat_waveform_has_no_peak_quality_veto(self):
        from test_temporary_route_balance_validation import capture
        from temporary_route_balance_validation import compare_full_route
        reports = [capture(),capture(850),capture()]
        for report in reports:
            for frame in report['frames']:
                for record in frame['records']: record['adc_code']=2
        result=compare_full_route(*reports)
        self.assertTrue(result['relative_accuracy_passed'])
        self.assertFalse(result['absolute_wavelength_accuracy_verified'])
        self.assertNotIn('reference_peak_shape_unresolved',result['reasons'])

class FrameNoPeakQualityTests(unittest.TestCase):
    def test_frame_metadata_does_not_compute_peak_edge_quality(self):
        from test_temporary_test_mode import frame
        from temporary_test_mode import decode,quality
        f=decode(frame(),123,456)
        metadata=quality(f)
        self.assertNotIn('peak_at_edge',metadata)
        self.assertIn('clipped',metadata)
        self.assertIn('unsettled',metadata)
        self.assertFalse(metadata['stability_verified'])

if __name__ == '__main__': unittest.main()
