import unittest

class NoQualityUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from test_temporary_test_widget import WidgetTests
        WidgetTests.setUpClass.__func__(cls)

    def test_peak_table_trend_markers_never_judge_legacy_quality(self):
        from temporary_test_widget import TemporaryTestWindow
        from test_temporary_test_mode import reference, frame
        from temporary_test_mode import select_points, decode, quality
        from unittest.mock import patch
        import temporary_peak_fit_makima as fitter
        w = TemporaryTestWindow(None)
        self.addCleanup(w.close)
        self.assertEqual(w.peak_fit_table.rowCount(), 2, 'remove the quality/R2 result row')
        self.assertNotIn('质量', w.data_tabs.tabText(0))
        self.assertNotIn('R²', w.show_peak_fits.toolTip())
        table, x, y = reference()
        w.plan = select_points(table, x, y)
        w.render_plan()
        f = decode(frame(), 123, 456)
        f.update(sequence=16, cycle_start_us=1000000)
        f['quality'] = quality(f)
        f['quality']['peak_at_edge'] = True
        fits = fitter.fit_temporary_frame(w.plan['rows'], [r['second_code'] for r in f['records']])
        for fit in fits:
            fit.update(valid=False, quality_warning=True, r_squared=0., quality_issues=['flat'])
        w.data_tabs.setCurrentIndex(2)
        with patch.object(fitter, 'fit_temporary_frame', return_value=fits):
            for seq in (16,17,18,19):
                f = dict(f, sequence=seq, cycle_start_us=1000000+(seq-16)*100000)
                w.update_frame(f)
        w._flush_peak_trend()
        for text in (w.peak_fit_status.text(), w.peak_trend_status.text(), w.status.text(),
                     w.peak_fit_table.item(0,0).text(), w.fit_labels[0].toPlainText()):
            self.assertNotIn('质量', text)
            self.assertNotIn('⚠', text)
            self.assertNotIn('峰在窗口边缘', text)
        self.assertNotIn('⚠', w.show_peak_fits.toolTip())
        self.assertEqual(w.peak_trend_history.samples[0][-1][1], fits[0]['peak_dbm'])

class NoReferencePeakQualityTests(NoQualityUiTests):
    def test_reference_line_and_restore_do_not_assess_peak_quality(self):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace
        from temporary_test_widget import TemporaryTestWindow
        from test_temporary_test_mode import reference
        from temporary_test_mode import select_points
        from unittest.mock import patch
        table,x,y=reference()
        with tempfile.TemporaryDirectory() as folder, \
             patch('temporary_test_widget.LAST_SESSION_PATH',Path(folder)/'last.json'), \
             patch('app_JDSU.load_fullband_accuracy_table',return_value=table):
            w=TemporaryTestWindow(SimpleNamespace())
            self.addCleanup(w.close)
            w.plan=select_points(table,x,y);w.render_plan(ready=False)
            report=dict(complete=True,disarm_ack=True,shutter_ack=True,ch1_feedback_selector=2,
                rows=[dict(index=r['index'],dac_codes=r['codes'],stable=True,
                           ch1_saturated=False,ch1_adc_code=80) for r in w.plan['rows']])
            with patch('temporary_test_widget.assess_peak_shape',create=True,side_effect=AssertionError('live reference quality evaluated')):
                w.reference_line_ready(report)
            self.assertTrue(w.plan_ready, w.status.text())
            self.assertNotIn('live_reference_fit_checks',w.plan)
            self.assertNotIn('reference_shape_gate_bypassed',w.plan)
            self.assertNotIn('峰形',w.status.text())
            w.close()
            with patch('temporary_test_widget.assess_peak_shape',create=True,side_effect=AssertionError('restored quality evaluated')):
                restored=TemporaryTestWindow(SimpleNamespace())
            self.addCleanup(restored.close)
            self.assertTrue(restored.plan_ready,restored.status.text())

if __name__ == '__main__': unittest.main()
