"""Run only in a fresh Torch-first child, never Qt unittest discovery."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from manual_touch_cnn import _extract_frame, load_touch_dataset
from manual_touch_sensor_token import extract_sample, load_sessions
from touch_dbm_training import extract_dbm_sample, prepare_data as dbm_data
from touch_five_training import prepare_data as five_data

class LoaderContractTests(unittest.TestCase):
    def test_all_extractors_reject_multi_even_empty_or_one_label(self):
        for extractor in (_extract_frame,extract_sample,extract_dbm_sample):
            for label,positions in ((0,[]),(1,[1]),(None,[1,13,25])):
                with self.subTest(extractor=extractor.__name__,positions=positions):
                    q=dict(annotation_mode='multi',position_label_schema='multi_hot25_v1',
                        position_labels=positions,position_multihot=[int(i in positions) for i in range(1,26)],
                        press_label=label,stable_adc=[100]*45,stable_dbm=[-20.]*45,
                        paxini_force=dict(fx_n=1.,fy_n=2.,fz_n=3.))
                    with self.assertRaisesRegex(ValueError,'multi'):
                        extractor({'quality':q})

    def test_all_loaders_reject_multi_manifest_before_frames_or_cache(self):
        scratch=Path.home()/'AppData/Local/hermes/cache/scratch'
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            p=Path(folder)
            (p/'manifest.json').write_text(json.dumps({'metadata':dict(dataset_kind='manual_touch_multi45',annotation_mode='multi',position_label_schema='multi_hot25_v1',training_eligible=False)}))
            (p/'frames.jsonl').touch()
            for loader in (lambda:load_touch_dataset(p),lambda:load_sessions([p],p),lambda:dbm_data([p],p),lambda:five_data(p)):
                with self.subTest(loader=loader):
                    with self.assertRaisesRegex(ValueError,'multi'): loader()

if __name__=='__main__': unittest.main()
