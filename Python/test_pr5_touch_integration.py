"""Synthetic/offscreen integration tests; no hardware access."""
import os,sys,unittest
from unittest.mock import patch
from pathlib import Path
os.environ['QT_QPA_PLATFORM']='offscreen'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=str(Path(sys.prefix)/'Lib/site-packages/PyQt5/Qt5/plugins/platforms')
import test_fbg_unified_app as fixture
from fbg_unified_app import PAGE_INFO

class TouchIntegrationTests(fixture.UnifiedStudioTests):
    def test_writer_keeps_all_second_codes_no_press_and_missing_force_reason(self):
        import json
        from test_manual_stress_capture import temporary_frame
        capture=self.window.manual_capture
        capture.output_root=Path(self._ota_tempdir.name)/'synthetic'
        source=capture.frame_source
        source.plan={'rows':[{'measured_nm':1530+i*.01,'index':i} for i in range(45)]}
        frame=temporary_frame(sequence=77,base=200)
        frame['provenance']='synthetic smoke only, NOT hardware'
        source.update_frame(frame)
        self.assertTrue(capture.capture_current_press())
        folder=capture.session_dir
        capture.close_session()
        row=json.loads((folder/'frames.jsonl').read_text('utf-8').splitlines()[0])
        self.assertEqual(row['quality']['press_label'],0)
        self.assertIsNone(row['quality']['paxini_force'])
        self.assertTrue(row['quality']['force_error'])
        self.assertEqual([r['second_code'] for r in row['quality']['stable_records']],list(range(700,745)))
        self.assertTrue(row['quality']['smoke_only'])
        self.assertNotIn('first_code',json.dumps(row))

    def test_training_roster_and_force_history_are_independent_45_point_controls(self):
        page=self.window.training_page
        self.assertTrue(hasattr(page,'architecture_combo'),'restored architecture selector missing')
        self.assertEqual([page.architecture_combo.itemData(i) for i in range(page.architecture_combo.count())],
            ['tcn','sensor_token','sparse_cnn','inception_1d'])
        self.assertTrue(hasattr(page,'force_history_check'))
        self.assertIn('当前45',page.input_shape_label.text())
        self.assertNotIn('共用',page.input_shape_label.text())
        from manual_touch_training_page import TouchTrainingWorker
        worker=TouchTrainingWorker(Path(self._ota_tempdir.name),epochs=1,architecture='tcn',force_history=True)
        command=worker.command()
        self.assertIn('touch_five_training',command)
        self.assertIn('--force-history',command)
        self.assertEqual(command[command.index('--architecture')+1],'tcn')
        for name in ('touch_temporal_models','touch_separate_models','touch_six_models','touch_live_sequence'):
            self.assertTrue((Path(__file__).parent/(name+'.py')).is_file(),name+' missing')

    def test_independent_task_reports_progress_and_candidates_do_not_activate(self):
        page=self.window.training_page
        page._training_progress(1,2,{'task':'position','validation_accuracy':.5})
        self.assertIn('position',page.status_label.text())
        model=Path(self._ota_tempdir.name)/'synthetic.pt';model.write_bytes(b'synthetic NOT checkpoint')
        report={'schema':'fbg_touch_temporal26_v1','model_path':str(model),
            'input_shape':[9,5],'best_epoch':{'position':1,'force':2},'metrics':{'validation':{'position_accuracy':.5}},
            'config':{'architecture':'tcn','force_history':False}}
        page._training_completed(report)
        self.assertIsNone(page.model_path,'training must save a candidate, not deploy it')
        self.assertFalse(page.start_live_button.isEnabled())
        self.assertIn('位置1',page.overall_metrics_label.text())
        self.assertIn('力2',page.overall_metrics_label.text())

    def test_training_force_reference_uses_same_cached_sensor_without_serial_read(self):
        from test_px6d_capture_tare import feed_fixture
        sensor=self.window.touch_force_sensor
        self.assertIs(sensor, self.window.training_page.force_sensor)
        feed_fixture(sensor, (1.,2.,3.,4.,5.,6.))
        self.assertEqual(self.window.training_page.force_provider(),(1.,2.,3.))
        self.window.manual_capture.zero_button.click()
        self.window.training_page._refresh_acquired_force()
        self.assertEqual(self.window.training_page.force_provider(),(0.,0.,0.))
        self.assertEqual(self.window.training_page.actual_force.force_xyz,(0.,0.,0.))
        sensor._latest_ns=None
        self.assertIsNone(self.window.training_page.force_provider())
        sensor._running=False; sensor._ser=None

    def test_uncertain_prediction_clears_marker_without_clamping_force(self):
        page=self.window.training_page
        page._live_prediction({'position':None,'decision_status':'uncertain','confidence':.2,'force':[1.,2.,3.]})
        self.assertIn('不确定',page.live_position_label.text())
        self.assertEqual(page._live_predicted[-1],(1.,2.,3.))

    def test_temporal_description_is_current_frame_position_and_independent_force(self):
        from touch_model_catalog import model_description
        text=model_description({'schema':'fbg_touch_temporal26_v1','architecture':'tcn','config':{'force_history':True}})
        self.assertIn('独立',text)
        self.assertIn('当前45',text)
        self.assertIn('TCN',text)

    def test_load_and_listener_reject_retired_shape_and_mismatched_selector(self):
        import json
        from test_manual_stress_capture import temporary_frame
        page=self.window.training_page
        model=Path(self._ota_tempdir.name)/'retired63.pt';model.write_bytes(b'retired')
        model.with_suffix('.report.json').write_text(json.dumps({'input_shape':[9,7],'schema':'fbg_touch_seven_temporal26_v1'}),'utf-8')
        self.assertFalse(page.load_model(model))
        page.report={'input_contract':{'input_shape':[9,5],'feedback_selector':0}}
        with self.assertRaises(ValueError):page._extract_live_adc(temporary_frame())

    def test_training_restart_waits_for_queued_finished_to_clear_worker(self):
        from types import SimpleNamespace
        page=self.window.training_page
        folder=Path(self._ota_tempdir.name);(folder/'frames.jsonl').write_text('{}\n','utf-8')
        page._set_dataset_path(folder)
        page.worker=SimpleNamespace(isRunning=lambda:False)
        try:
            with patch('manual_touch_training_page.TouchTrainingWorker',side_effect=AssertionError('must wait for finished')):
                self.assertFalse(page.start_training())
        finally:page.worker=None

    def test_selecting_dataset_does_not_resurrect_old_session_model(self):
        import json
        page=self.window.training_page
        folder=Path(self._ota_tempdir.name);model=folder/'models'/'manual_touch_sparse_cnn.pt';model.parent.mkdir()
        model.write_bytes(b'old session placeholder')
        model.with_suffix('.report.json').write_text(json.dumps({'best_epoch':1}),'utf-8')
        page._set_dataset_path(folder);page._load_existing_report_if_present()
        self.assertIsNone(page.model_path)
        self.assertFalse(page.start_live_button.isEnabled())

    def test_restored_pages_have_visible_navigation_and_idle_shared_sensor(self):
        w=self.window
        self.assertEqual(w.workspace_tabs.count(),7)
        self.assertEqual(len(w.nav_buttons),7)
        self.assertIs(w.manual_capture.frame_source,w.local_controller.page_temporary_test)
        self.assertIs(w.training_page.frame_source,w.local_controller.page_temporary_test)
        self.assertIs(w.manual_capture.force_sensor,w.touch_force_sensor)
        self.assertFalse(w.touch_force_sensor._running)
        self.assertIsNone(w.training_page.model_path)
        self.assertIsNone(w.training_page.inference)
        self.assertIsNone(w.training_page.worker)
        for index in (5,6):
            self.assertFalse(w.nav_buttons[index].isHidden())
            w.nav_buttons[index].click();self.app.processEvents()
            self.assertEqual(w.workspace_tabs.currentIndex(),index)
            self.assertEqual(w.page_title.text(),PAGE_INFO[index][0])
        self.assertNotIn('torch',sys.modules)
        w.close();self.app.processEvents()
        self.assertFalse(w.manual_capture.link_timer.isActive())
        self.assertFalse(w.manual_capture.force_timer.isActive())
        self.assertEqual(w.local_controller.page_temporary_test.frame_listeners,[])

# Don't inherit the entire historical suite into this focused module.
for name in list(vars(fixture.UnifiedStudioTests)):
    if name.startswith('test_'):
        setattr(TouchIntegrationTests,name,None)
if __name__=='__main__': unittest.main()
