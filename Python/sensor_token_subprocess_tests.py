"""Run directly, isolated from Qt-first imports."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
import torch
from manual_touch_sensor_token import SensorTokenModel, SparseCNN26, tokenize, extract_sample, multitask_loss, ModelBundle

class SensorTokenTests(unittest.TestCase):
    def test_tokens_preserve_nine_groups_of_five(self):
        x = np.arange(45, dtype=np.float32)
        result = tokenize(x, np.zeros(5), np.ones(5))
        self.assertEqual(result.shape, (1, 9, 5))
        np.testing.assert_array_equal(result[0, 8], np.arange(40, 45))

    def test_shared_models_have_26_class_and_three_force_outputs(self):
        for model, shape in [(SensorTokenModel(), (3, 9, 5)), (SparseCNN26(), (3, 6, 4, 3))]:
            logits, force = model(torch.zeros(shape))
            self.assertEqual(tuple(logits.shape), (3, 26))
            self.assertEqual(tuple(force.shape), (3, 3))
            (logits.sum() + force.sum()).backward()
            self.assertTrue(any(p.grad is not None for p in model.backbone.parameters()))

    def test_no_press_and_missing_force_are_retained_not_fabricated(self):
        sample = extract_sample({'quality': {'press_label': 0, 'stable_adc': list(range(45)), 'paxini_force': None}})
        self.assertEqual(sample[1], 0)
        self.assertFalse(sample[3])
        self.assertTrue(np.isnan(sample[2]).all())
        self.assertIsNone(extract_sample({'quality': {'press_label': 26, 'stable_adc': list(range(45))}}))

    def test_missing_force_contributes_no_regression_loss(self):
        logits = torch.zeros(2, 26, requires_grad=True)
        forces = torch.ones(2, 3, requires_grad=True)
        loss = multitask_loss(logits, forces, torch.tensor([0, 25]), torch.zeros(2, 3), torch.zeros(2, dtype=torch.bool), torch.ones(26))
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(float(forces.grad.abs().sum()), 0.)

    def test_block_split_is_disjoint_and_keeps_gaps(self):
        from manual_touch_sensor_token import blocked_split
        labels = np.repeat(np.arange(26), 100)
        split = blocked_split(labels)
        groups = [set(split[k]) for k in ['train', 'validation', 'test', 'purged']]
        self.assertEqual(sum(map(len, groups)), len(labels))
        for i in range(4):
            for j in range(i):
                self.assertFalse(groups[i] & groups[j])
        for label in range(26):
            indices = {k: [i for i in split[k] if labels[i]==label] for k in ['train', 'validation', 'test']}
            self.assertEqual(len(indices['train']), 60)
            self.assertEqual(len(indices['validation']), 15)
            self.assertEqual(len(indices['test']), 15)
            self.assertGreater(min(indices['validation'])-max(indices['train']), 5)
            self.assertGreater(min(indices['test'])-max(indices['validation']), 5)

    def test_universal_loader_accepts_new_token_and_cnn26(self):
        from manual_touch_cnn import load_touch_model, prediction_payload
        with tempfile.TemporaryDirectory() as folder:
            for kind in ['sensor_token','sparse_cnn26']:
                from manual_touch_sensor_token import build_model
                model=build_model(kind)
                path=Path(folder)/(kind+'.pt')
                torch.save({'schema':'fbg_touch26_v1','architecture':kind,'model_state':model.state_dict(),'feature_mean':np.zeros(5),'feature_std':np.ones(5),'force_mean':np.zeros(3),'force_std':np.ones(3)},path)
                payload=prediction_payload(load_touch_model(path),np.zeros(45))
                self.assertEqual(len(payload['probabilities']),26)
                self.assertTrue(0 <= payload['position'] <= 25)

    def test_saved_model_prediction_roundtrip(self):
        with tempfile.TemporaryDirectory() as folder:
            for kind, model in [('sensor_token', SensorTokenModel()), ('sparse_cnn26', SparseCNN26())]:
                model.eval()
                payload = {'schema': 'fbg_touch26_v1', 'architecture': kind, 'model_state': model.state_dict(), 'feature_mean': np.zeros(5), 'feature_std': np.ones(5), 'force_mean': np.zeros(3), 'force_std': np.ones(3)}
                path = Path(folder) / (kind + '.pt')
                torch.save(payload, path)
                bundle = ModelBundle(path)
                labels, force, probabilities = bundle.predict(np.zeros(45))
                self.assertEqual(probabilities.shape, (1, 26))
                self.assertTrue(0 <= int(labels[0]) <= 25)
                np.testing.assert_allclose(probabilities.sum(1), [1.], rtol=1e-5)
                self.assertEqual(force.shape, (1, 3))

if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()
