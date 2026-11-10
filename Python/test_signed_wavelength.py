import unittest
import importlib.util

class SignedWavelengthTests(unittest.TestCase):
    def test_explicit_same_frame_baseline_signed_delta(self):
        self.assertIsNotNone(importlib.util.find_spec('signed_wavelength'), 'signed wavelength feature missing')
        from signed_wavelength import SignedWavelengthState, PEAK_LABELS
        self.assertEqual(PEAK_LABELS, (25,15,5,3,13,23,21,11,1))
        s = SignedWavelengthState()
        fits = [dict(center_nm=1530+i,valid=True) for i in range(9)]
        s.update(fits, ('machine1','ch1','route'), 16, 10.)
        self.assertEqual(s.deltas, [None]*9)
        self.assertTrue(s.set_baseline(10.))
        fits = [dict(center_nm=1530+i+(.025 if i%2 else -.05),valid=True) for i in range(9)]
        s.update(fits, ('machine1','ch1','route'), 17, 10.1)
        self.assertAlmostEqual(s.deltas[0], -.05)
        self.assertAlmostEqual(s.deltas[1], .025)

    def test_context_missing_quality_stale_and_saturation(self):
        from signed_wavelength import SignedWavelengthState
        s = SignedWavelengthState()
        fits = [dict(center_nm=1530+i,valid=True) for i in range(9)]
        s.update(fits, ('a',), 16, 10.)
        self.assertFalse(s.set_baseline(13.), 'stale frame cannot establish baseline')
        self.assertTrue(s.set_baseline(10.))
        s.update(fits, ('b',), 17, 10.1)
        self.assertIsNone(s.baseline)
        self.assertTrue(s.set_baseline(10.1))
        broken = list(fits); broken[3] = dict(center_nm=None, valid=False)
        s.update(broken, ('b',), 18, 10.2)
        self.assertIsNone(s.deltas[3])
        self.assertFalse(s.set_baseline(10.2))
        self.assertEqual(len(s.deltas),9)
        s.update(fits, ('b',), 0, 10.3)
        self.assertIsNone(s.baseline, 'sequence restart resets baseline')
        self.assertTrue(s.stale(13.))

    def test_fixed_diverging_colors_and_exact_spatial_anchors(self):
        import signed_wavelength as sw
        self.assertTrue(hasattr(sw, 'signed_colors'), 'diverging colors missing')
        import numpy as np
        colors = sw.signed_colors([-.1, 0, .1, None, .2], .1)
        self.assertGreater(colors[0,2], colors[0,0])
        self.assertGreater(colors[2,0], colors[2,2])
        np.testing.assert_equal(colors[2], colors[4])
        anchors = np.array([[0.,0.],[1.,0.],[0.,1.]])
        field = sw.spatial_field(anchors, anchors, [-.1,0,.1])
        np.testing.assert_allclose(field,[-.1,0,.1])
        field = sw.spatial_field(anchors, anchors, [None,0,.1])
        self.assertTrue(np.isnan(field[0]), 'missing anchor must remain missing')

if __name__ == '__main__': unittest.main()
