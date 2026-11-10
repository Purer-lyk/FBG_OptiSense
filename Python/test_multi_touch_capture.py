"""Multi-contact annotation: real Qt page and production session writer."""
import json
import os
import tempfile
import unittest
from pathlib import Path
import test_manual_stress_capture as legacy
from test_manual_stress_capture import FakeTemporaryPage, temporary_frame
from manual_stress_capture import ManualStressCaptureWindow, FINGER_POINT_BY_LABEL

class MultiTouchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        legacy.FixedGridCaptureTests.setUpClass()
        cls.app = legacy.FixedGridCaptureTests.app

    def setUp(self):
        scratch=Path.home()/'AppData/Local/hermes/cache/scratch'
        scratch.mkdir(parents=True,exist_ok=True)
        self.tmp = tempfile.TemporaryDirectory(dir=scratch)
        self.source = FakeTemporaryPage()
        self.page = ManualStressCaptureWindow(self.source,
            wavelength_provider=lambda: self.source.reference_wavelengths,
            output_root=Path(self.tmp.name))
        self.page.attach()
        self.page._on_temporary_frame(temporary_frame())

    def tearDown(self):
        self.page.close()
        self.tmp.cleanup()

    def test_explicit_multi_toggle_empty_and_single_default(self):
        p=self.page
        self.assertEqual(getattr(p,'annotation_mode',None),'single')
        p.grid_buttons[0].click();p.grid_buttons[4].click()
        self.assertEqual(p.selected_point,5)
        p.mode_combo.setCurrentIndex(1)
        for label in (1,13,25): p.grid_buttons[label-1].click()
        self.assertEqual(p.selected_points,{1,13,25})
        self.assertEqual(sum(b.isChecked() for b in p.grid_buttons),3)
        p.grid_buttons[12].click()
        self.assertEqual(p.selected_points,{1,25})
        p.no_press_button.click()
        self.assertEqual(p.selected_points,set())
        self.assertFalse(any(b.isChecked() for b in p.grid_buttons))
        p.mode_combo.setCurrentIndex(0)
        p.grid_buttons[2].click();p.grid_buttons[6].click()
        self.assertEqual(p.selected_point,7)
        self.assertEqual(sum(b.isChecked() for b in p.grid_buttons),1)

    def test_writer_roundtrip_multi_empty_and_one_keep_provenance(self):
        p=self.page
        p.mode_combo.setCurrentIndex(1)
        for label in (25,1,13): p.grid_buttons[label-1].click()
        self.assertTrue(p.capture_current_press())
        root=p.session_dir
        p.no_press_button.click(); self.assertTrue(p.capture_current_press())
        p.grid_buttons[2].click(); self.assertTrue(p.capture_current_press())
        p.close_session()
        records=[json.loads(line)['quality'] for line in (root/'frames.jsonl').read_text('utf-8').splitlines()]
        self.assertEqual(records[0].get('position_labels'),[1,13,25])
        self.assertEqual(records[0]['position_multihot'],[int(i in (1,13,25)) for i in range(1,26)])
        self.assertIsNone(records[0]['press_label'])
        for key in ('grid_row','grid_column','surface_x_normalized','surface_y_normalized','surface_region'):
            self.assertIsNone(records[0][key])
        self.assertEqual(records[0]['press_state'],'pressed')
        self.assertEqual(records[0]['position_coordinates'],[
            {'label':label,'x':FINGER_POINT_BY_LABEL[label][0],'y':FINGER_POINT_BY_LABEL[label][1]}
            for label in (1,13,25)])
        self.assertEqual(records[1]['position_labels'],[])
        self.assertEqual(records[1]['position_multihot'],[0]*25)
        self.assertEqual(records[1]['press_state'],'no_press')
        self.assertEqual(records[2]['position_labels'],[3])
        for q in records:
            self.assertEqual(q['annotation_mode'],'multi')
            self.assertEqual(q['position_label_schema'],'multi_hot25_v1')
            self.assertEqual(q['stable_adc'],list(range(700,745)))
            self.assertEqual(len(q['training_views']['makima_peak_fits']),9)
        manifest=json.loads((root/'manifest.json').read_text('utf-8'))['metadata']
        self.assertTrue(manifest['training_eligible'])
        self.assertEqual(manifest['dataset_kind'],'manual_touch_multi45')
        self.assertEqual(root.parent.name,'manual_touch_multi45')

    def test_multi_undo_restores_entire_set_and_sample_total(self):
        p=self.page; p.mode_combo.setCurrentIndex(1)
        for i in (1,13,25): p.grid_buttons[i-1].click()
        p.capture_current_press(); p.capture_current_press()
        p.no_press_button.click(); p.capture_current_press()
        self.assertTrue(p.undo_last_annotation(confirm=False))
        self.assertEqual(p.capture_counts[0],0)
        self.assertTrue(p.undo_last_annotation(confirm=False))
        self.assertEqual([p.capture_counts[i] for i in (1,13,25)],[1,1,1])
        self.assertIn('总样本 1',p.counts_label.text())
        root=p.session_dir; p.close_session()
        self.assertEqual(len((root/'frames.jsonl').read_text('utf-8').splitlines()),1)

    def test_mode_locked_until_close_and_next_session_counts_start_fresh(self):
        p=self.page; p.select_grid_point(5); p.capture_current_press()
        single=p.session_dir
        p.mode_combo.setCurrentIndex(1)
        self.assertEqual(p.annotation_mode,'single')
        self.assertIn('关闭',p.capture_status_label.text())
        p.close_session(); p.mode_combo.setCurrentIndex(1)
        p.grid_buttons[12].click(); p.capture_current_press()
        self.assertEqual(p.sample_index,1)
        self.assertEqual(p.capture_counts[5],0)
        multi=p.session_dir; p.close_session()
        self.assertNotEqual(single,multi)
        self.assertEqual(json.loads((single/'frames.jsonl').read_text('utf-8'))['quality']['press_label'],5)

    def test_multi_force_keeps_shared_tare_corrected_resultant(self):
        from test_px6d_capture_tare import feed_fixture
        p=self.page; p.mode_combo.setCurrentIndex(1)
        feed_fixture(p.force_sensor,(1.,-2.,3.,0.,0.,0.))
        p.force_sensor.tare()
        feed_fixture(p.force_sensor,(2.,-1.,4.,0.,0.,0.))
        for i in (2,8,22): p.grid_buttons[i-1].click()
        self.assertTrue(p.capture_current_press()); root=p.session_dir; p.close_session()
        q=json.loads((root/'frames.jsonl').read_text('utf-8'))['quality']
        self.assertEqual([q['paxini_force'][k] for k in ('fx_n','fy_n','fz_n')],[1.,1.,1.])
        self.assertEqual(q['paxini_force']['raw_six_axis'][:3],[2.,-1.,4.])
        self.assertEqual(q['paxini_force']['tare']['count'],1)
        self.assertEqual(q['force_semantics'],'whole_contact_resultant_not_per_position')
        self.assertFalse(any('force' in c for c in q['position_coordinates']))
        p.force_sensor._ser=None

    def test_all25_contact_set_keeps_canonical_coordinates_and_missing_force_null(self):
        p=self.page; p.mode_combo.setCurrentIndex(1)
        for i in range(25): p.grid_buttons[i].click()
        self.assertTrue(p.capture_current_press()); root=p.session_dir
        p.close_session()
        q=json.loads((root/'frames.jsonl').read_text('utf-8'))['quality']
        self.assertEqual(q['position_labels'],list(range(1,26)))
        self.assertEqual(q['position_multihot'],[1]*25)
        self.assertEqual(q['position_coordinates'],[
            dict(label=i,x=FINGER_POINT_BY_LABEL[i][0],y=FINGER_POINT_BY_LABEL[i][1]) for i in range(1,26)])
        self.assertIsNone(q['paxini_force'])
        self.assertIsNone(q['press_label'])
        self.assertFalse(q['source_frame_quality']['stability_verified'])
        end=json.loads((root/'events.jsonl').read_text('utf-8').splitlines()[-1])['metadata']
        self.assertEqual(end['total_samples'],1)
        self.assertEqual(sum(end['counts_by_label'].values()),25)

class MultiTouchLoaderIsolationTests(unittest.TestCase):
    def test_single_label_readers_fail_closed_in_torch_child(self):
        import subprocess
        import sys
        from touch_worker_runtime import torch_worker_python
        self.assertNotIn('torch',sys.modules)
        root=Path(__file__).resolve().parent
        result=subprocess.run([torch_worker_python(),str(root/'multi_touch_loader_child.py'),'-v'],
            cwd=root,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=90)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)
        self.assertNotIn('torch',sys.modules)

if __name__=='__main__': unittest.main()
