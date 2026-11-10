"""Fresh Torch-first child: reject delta9 before any old 45-point cache/reader."""
import json,unittest,tempfile
from pathlib import Path
from manual_touch_cnn import _extract_frame,load_touch_dataset
from manual_touch_sensor_token import extract_sample,load_sessions
from touch_dbm_training import extract_dbm_sample,prepare_data as dbm_data
from touch_five_training import prepare_data as five_data

class Delta9LoaderContractTests(unittest.TestCase):
    def test_single_empty_multi_and_one_contact_delta9_never_enter_old_extractors(self):
        for mode,label in (('single',0),('single',13),('multi',0),('multi',1),('multi',None)):
            q=dict(dataset_kind='manual_touch_delta9',input_contract='delta_wavelength9',training_eligible=False,
                annotation_mode=mode,press_label=label,delta_wavelength_nm=[.01]*9)
            for extractor in (_extract_frame,extract_sample,extract_dbm_sample):
                with self.subTest(mode=mode,label=label,extractor=extractor.__name__):
                    with self.assertRaisesRegex(ValueError,'delta|Δλ'):
                        extractor(dict(schema='fbg-touch-delta9/v1',values=[.01]*9,quality=q))
    def test_actual_delta9_writer_session_is_rejected_by_old_readers(self):
        import os
        directory=os.environ.get('DELTA9_REPLAY_SESSION')
        if not directory:self.skipTest('production replay session supplied by verification runner')
        p=Path(directory)
        records=[json.loads(line) for line in (p/'frames.jsonl').read_text('utf-8').splitlines()]
        self.assertTrue(records)
        for record in records:
            self.assertEqual(len(record['values']),9)
            for extractor in (_extract_frame,extract_sample,extract_dbm_sample):
                with self.assertRaisesRegex(ValueError,'delta|Δλ'):extractor(record)
        for loader in (lambda:load_touch_dataset(p),lambda:load_sessions([p],p),lambda:dbm_data([p],p),lambda:five_data(p)):
            with self.assertRaisesRegex(ValueError,'delta|Δλ'):loader()

    def test_all_loaders_fail_before_cache_for_delta9_manifest(self):
        scratch=Path.home()/'AppData/Local/hermes/cache/scratch'
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            p=Path(folder)
            (p/'manifest.json').write_text(json.dumps(dict(schema='fbg-touch-delta9/v1',metadata=dict(dataset_kind='manual_touch_delta9',input_contract='delta_wavelength9',training_eligible=False))))
            (p/'frames.jsonl').touch()
            for loader in (lambda:load_touch_dataset(p),lambda:load_sessions([p],p),lambda:dbm_data([p],p),lambda:five_data(p)):
                with self.assertRaisesRegex(ValueError,'delta|Δλ'):loader()
if __name__=='__main__':unittest.main()
