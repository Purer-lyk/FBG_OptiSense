"""Archived explicit delta9 capture remains isolated from restored45 default."""
import os,sys,json,tempfile,unittest,time
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM','offscreen')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=str(Path(sys.prefix)/'Lib/site-packages/PyQt5/Qt5/plugins/platforms')
from PyQt5 import QtWidgets,QtCore
from manual_stress_capture import ManualStressCaptureWindow
from test_manual_stress_capture import FakeTemporaryPage,temporary_frame
from touch_annotation_contract import require_single_position_annotation,require_single_position_session

class Delta9CaptureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        QtCore.QCoreApplication.setLibraryPaths([str(Path(sys.prefix)/'Lib/site-packages/PyQt5/Qt5/plugins')])
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.source=FakeTemporaryPage();self.source.plan={'rows':[dict(index=i,measured_nm=1530+i*.01) for i in range(45)]}
        self.p=ManualStressCaptureWindow(self.source,wavelength_provider=lambda:self.source.reference_wavelengths,output_root=Path(self.tmp.name)/'manual_touch_45',capture_contract='delta9')
        self.addCleanup(self.p.close);self.p.attach()
    def send(self,seq,shift=0):
        frame=temporary_frame(sequence=seq);frame['provenance']='SYNTHETIC delta9 regression'
        fits=[dict(center_nm=1530+i+0.0037+shift*(1 if i%2 else -1),valid=False,quality_warning=True,peak_selection_method='makima_curve_global_max_dbm') for i in range(9)]
        self.source.makima_frame_snapshot=(frame,fits);self.p._on_temporary_frame(frame)
        return frame
    def baseline(self):
        self.send(16);self.p.wavelength_baseline_button.click();self.assertIsNotNone(self.p.wavelength_state.baseline)
    def test_requires_explicit_baseline_without_creating_session_or_increment(self):
        self.send(16);self.assertFalse(self.p.capture_current_press());self.assertIsNone(self.p.writer);self.assertEqual(self.p.capture_history,[])
    def test_default_writer_has_only_nine_delta_features_and_precise_virtual_centers(self):
        self.baseline();self.send(17,.021);self.p.select_grid_point(13)
        self.assertTrue(self.p.capture_current_press());self.p.writer.flush();directory=self.p.session_dir
        record=json.loads((directory/'frames.jsonl').read_text().strip());q=record['quality']
        self.assertEqual(record['schema'],'fbg-touch-delta9/v1')
        self.assertEqual(record['feature_kind'],'delta_wavelength9');self.assertEqual(len(record['values']),9)
        for i,d in enumerate(record['values']):self.assertAlmostEqual(d,.021*(1 if i%2 else -1),places=10)
        self.assertEqual(record['current_peak_wavelengths_nm'][0],1530.0037-.021)
        self.assertEqual(record['baseline_wavelengths_nm'][0],1530.0037)
        self.assertEqual(q['press_label'],13);self.assertFalse(q['training_eligible'])
        self.assertEqual(record['peak_to_position_labels'],[25,15,5,3,13,23,21,11,1])
        for field in ('stable_adc','stable_dbm','stable_voltage_v','training_views','stable_records','peak_fits','raw_adc_codes','adc_voltage_v'):
            self.assertNotIn(field,json.dumps(record))
        self.assertIn('manual_touch_delta9',str(directory));require_error=False
        with self.assertRaisesRegex(ValueError,'delta|Δλ'):require_single_position_session(directory)
        with self.assertRaisesRegex(ValueError,'delta|Δλ'):require_single_position_annotation(q)
    def test_invalid_context_stale_or_missing_same_frame_rejects(self):
        self.baseline()
        for kind in ('missing','stale','nonfinite','context','wrong_method'):
            self.send(17+len(self.p.capture_history))
            if kind=='missing':self.source.makima_frame_snapshot=None
            if kind=='stale':self.p.wavelength_state.received=time.monotonic()-3
            if kind=='nonfinite':self.p.wavelength_state.current[0]=float('nan')
            if kind=='context':self.source.plan['rows'][0]['index']+=1
            if kind=='wrong_method':self.p.latest_peak_fits[0]['peak_selection_method']='gaussian'
            self.assertFalse(self.p.capture_current_press(),kind);self.assertEqual(self.p.capture_history,[])
            self.baseline()
    def test_force_is_corrected_label_and_rejected_write_does_not_count(self):
        from unittest.mock import patch
        from test_px6d_capture_tare import feed_fixture
        self.baseline();self.send(17,.02)
        feed_fixture(self.p.force_sensor,(1.,-2.,3.,0.,0.,0.));self.p.force_sensor.tare()
        feed_fixture(self.p.force_sensor,(2.,-1.,4.,0.,0.,0.))
        self.assertTrue(self.p.capture_current_press());self.p.writer.flush()
        q=json.loads((self.p.session_dir/'frames.jsonl').read_text().strip())['quality']
        self.assertEqual([q['paxini_force'][k] for k in ('fx_n','fy_n','fz_n')],[1.,1.,1.])
        self.assertEqual(q['paxini_force']['raw_six_axis'],[2.,-1.,4.,0.,0.,0.])
        self.send(18,.03)
        with patch.object(self.p.writer,'write_frame',side_effect=OSError('disk fixture')):
            self.assertFalse(self.p.capture_current_press())
        self.assertEqual(len(self.p.capture_history),1);self.assertEqual(self.p.sample_index,1)
        self.assertEqual(self.p.capture_counts[0],1)

    def test_snapshot_failure_keeps_written_count_and_visible_warning(self):
        from unittest.mock import patch
        self.p.mode_combo.setCurrentIndex(1);self.baseline();self.send(17,.02)
        self.p.select_grid_point(22);self.assertTrue(self.p.capture_current_press())
        self.send(18,.03)
        with patch('manual_stress_capture.os.replace',side_effect=OSError('snapshot denied')):
            self.assertTrue(self.p.capture_current_press())
        self.assertEqual(self.p.combo_counts,{(22,):2})
        self.assertIn('统计保存失败',self.p.capture_status_label.text())
        old=json.loads((self.p.session_dir/'combination_counts.json').read_text())
        self.assertEqual(old['active_samples_count'],1)
        self.assertTrue(self.p._persist_combo_counts())
        self.assertEqual(json.loads((self.p.session_dir/'combination_counts.json').read_text())['active_samples_count'],2)

    def test_open_session_contract_cannot_be_switched_or_appended(self):
        from finger_dataset import FingerSessionWriter,SpectrumFrame
        self.baseline();self.send(17,.02);self.p.select_grid_point(13)
        self.assertTrue(self.p.capture_current_press());writer=self.p.writer
        events_before=(writer.session_dir/'events.jsonl').read_text('utf-8')
        self.p.capture_contract='legacy45'
        self.assertFalse(self.p.capture_current_press())
        self.assertIn('先保存关闭',self.p.capture_status_label.text())
        self.assertEqual((writer.session_dir/'events.jsonl').read_text('utf-8'),events_before)
        self.p.capture_contract='delta9';self.p.close_session()
        old=FingerSessionWriter(Path(self.tmp.name)/'old_manual_touch_45')
        old.write_frame(SpectrumFrame(sequence=9,wavelengths_nm=self.source.reference_wavelengths,raw_adc_codes=list(range(45))));old.flush();self.p.writer=old
        before=(old.session_dir/'frames.jsonl').read_text('utf-8')
        self.send(18,.03)
        self.assertFalse(self.p.capture_current_press())
        self.assertIn('先保存关闭',self.p.capture_status_label.text())
        self.assertEqual((old.session_dir/'frames.jsonl').read_text('utf-8'),before)
        self.p.writer=None;old.close()

    def test_multi_empty_counts_undo_snapshot_and_new_session(self):
        self.p.mode_combo.setCurrentIndex(1);self.baseline();self.send(17,.03)
        self.p.select_grid_point(23);self.p.select_grid_point(22);self.assertTrue(self.p.capture_current_press())
        self.p.select_no_press();self.send(18,.001);self.assertTrue(self.p.capture_current_press())
        first=self.p.session_dir;self.assertEqual(self.p.combo_counts,{(22,23):1,():1})
        self.assertTrue(self.p.undo_last_annotation(confirm=False));self.assertEqual(self.p.combo_counts,{(22,23):1})
        self.p.writer.flush();records=[json.loads(s) for s in (first/'frames.jsonl').read_text().splitlines()]
        self.assertEqual(records[0]['quality']['position_labels'],[22,23]);self.assertEqual(len(records[0]['values']),9)
        snapshot=json.loads((first/'combination_counts.json').read_text());self.assertEqual(snapshot['dataset_kind'],'manual_touch_delta9');self.assertEqual(snapshot['active_samples_count'],1)
        self.p.close_session();self.baseline();self.send(17,.001);self.assertTrue(self.p.capture_current_press());self.assertNotEqual(self.p.session_dir,first)

if __name__=='__main__':unittest.main()
