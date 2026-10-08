"""Torch-first tests; run outside Qt process."""
import unittest
import numpy as np

class DbmTrainingTests(unittest.TestCase):
    def test_saved_dbm_is_used_not_adc_or_fit(self):
        import touch_dbm_training as m
        frame={'quality':{'press_label':3,'stable_adc':[100]*45,'stable_dbm':[-12.5]*45,'training_views':{'stable_optical_power_dbm':[-12.5]*45},'paxini_force':{'fx_n':1,'fy_n':2,'fz_n':3}}}
        sample=m.extract_dbm_sample(frame)
        self.assertIsNotNone(sample)
        values,label,force,valid=sample
        np.testing.assert_array_equal(values,np.full(45,-12.5,np.float32))
        self.assertEqual(label,3)
        self.assertTrue(valid)
        frame['quality'].pop('stable_dbm')
        frame['quality']['training_views'].pop('stable_optical_power_dbm')
        self.assertIsNone(m.extract_dbm_sample(frame))

    def test_random_split_is_fixed_stratified_and_disjoint(self):
        from touch_dbm_training import random_split
        labels=np.repeat(np.arange(26),100)
        split=random_split(labels,seed=20261008)
        self.assertEqual(split,random_split(labels,seed=20261008))
        sets=[set(split[k]) for k in ('train','validation','test')]
        self.assertEqual([len(s) for s in sets],[1820,390,390])
        self.assertFalse(sets[0]&sets[1] or sets[0]&sets[2] or sets[1]&sets[2])
        self.assertEqual(len(set.union(*sets)),2600)
        for key in ('train','validation','test'):
            self.assertEqual(len(np.unique(labels[split[key]])),26)
        self.assertNotEqual(split['train'],sorted(split['train']))

    def test_training_uses_validation_only_and_model_roundtrips(self):
        import tempfile
        from pathlib import Path
        import touch_dbm_training as m
        rng=np.random.default_rng(31)
        data={'dbm':rng.normal(-15,2,(100,45)).astype(np.float32),'labels':np.tile(np.arange(2),50),'forces':rng.normal(size=(100,3)).astype(np.float32),'valid':np.ones(100,bool)}
        split={'train':list(range(70)),'validation':list(range(70,85)),'test':list(range(85,100))}
        config={'id':'smoke','seed':31,'width':16,'heads':4,'layers':1,'ffn':32,'shared':32,'dropout':0.,'lr':.001,'weight_decay':.0001,'force_weight':.2,'batch_size':32,'max_epochs':2,'patience':2}
        with tempfile.TemporaryDirectory() as folder:
            result=m.train_trial(data,split,config,Path(folder),feedback_selector=2)
            self.assertNotIn('test',result.get('metrics',{}))
            __import__('torch').set_num_threads(3)
            bundle=m.DbmModelBundle(result['model_path'])
            self.assertEqual(__import__('torch').get_num_threads(),2)
            labels,forces,probs=bundle.predict_dbm(data['dbm'][:2])
            self.assertEqual(probs.shape,(2,26))
            from manual_touch_cnn import load_touch_model,prediction_payload
            universal=load_touch_model(result['model_path'])
            np.testing.assert_allclose(universal.predict_dbm(data['dbm'][:2])[2],probs,atol=1e-6)
            prediction=prediction_payload(universal,np.full(45,500).tolist())
            self.assertIn('position',prediction)
            self.assertEqual(forces.shape,(2,3))
            np.testing.assert_allclose(probs.sum(1),1,atol=1e-6)
            payload=__import__('torch').load(result['model_path'],weights_only=False)
            np.testing.assert_allclose(payload['feature_mean'],data['dbm'][:70].mean(0).reshape(9,5),atol=1e-5)
            # Direct saved dBm and online ADC conversion must use the same path.
            from temporary_optical_units import adc_code_to_dbm
            adc=np.full((1,45),500,np.float32)
            converted=adc_code_to_dbm(adc,2,floor_nonpositive=True)
            pred1=bundle.predict(adc)
            pred2=bundle.predict_dbm(converted)
            np.testing.assert_allclose(pred1[2],pred2[2],atol=1e-6)

    def test_prepare_preserves_sources_and_checks_conversion(self):
        import tempfile,json,hashlib
        from pathlib import Path
        from temporary_optical_units import adc_code_to_dbm
        from touch_dbm_training import prepare_data
        with tempfile.TemporaryDirectory(dir=r'C:\Users\Administrator\AppData\Local\hermes\cache\scratch') as folder:
            root=Path(folder); session=root/'session'; session.mkdir(); out=root/'out'; out.mkdir()
            adc=np.full(45,500,dtype=np.int32)
            frame={'quality':{'press_label':1,'stable_adc':adc.tolist(),'stable_dbm':adc_code_to_dbm(adc,2,floor_nonpositive=True).tolist(),'feedback_selector':2,'training_views':{'wavelengths_nm':list(range(45))},'paxini_force':{'fx_n':1,'fy_n':2,'fz_n':3}}}
            source=session/'frames.jsonl'; source.write_text(json.dumps(frame)+'\n',encoding='utf-8')
            before=hashlib.sha256(source.read_bytes()).hexdigest()
            data,audit=prepare_data([session],out)
            self.assertEqual(data['dbm'].shape,(1,45))
            self.assertEqual(audit['sources'][0]['sha256'],before)
            self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),before)
            self.assertLess(audit['maximum_adc_to_saved_dbm_error'],1e-4)
            self.assertEqual(audit['feedback_selector'],2)

if __name__=='__main__': unittest.main()
