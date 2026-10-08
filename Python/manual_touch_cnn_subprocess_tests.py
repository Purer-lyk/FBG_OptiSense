"""PyTorch-isolated checks for manual_touch_cnn.

Run in a clean subprocess because on Windows Qt must not load before torch.
"""

import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from manual_touch_cnn import (
    acquisition_run_summary,
    GRATING_GRID,
    SENSOR_MASK,
    SparseTouchCNN,
    load_touch_model,
    prediction_payload,
    save_touch_model,
    spatialize_five_point_features,
    stratified_tail_split,
)


class SparseTouchCnnIsolatedTests(unittest.TestCase):
    def test_acquisition_run_summary_detects_labels_without_independent_rounds(self):
        summary = acquisition_run_summary(np.asarray([1, 1, 2, 2, 1, 1]))
        self.assertEqual(summary["runs_by_label"], {1: 2, 2: 1})
        self.assertEqual(summary["labels_with_multiple_runs"], [1])
        self.assertEqual(summary["labels_with_single_run"], [2])

    def test_spatial_tensor(self):
        values = np.asarray([[g * 10.0 + p for p in range(5)] for g in range(1, 10)]).reshape(1, 45)
        tensor = spatialize_five_point_features(values, feature_mean=np.zeros(5), feature_std=np.ones(5))
        self.assertEqual(tensor.shape, (1, 6, 4, 3))
        np.testing.assert_array_equal(tensor[0, 5], SENSOR_MASK)
        for row, grid_row in enumerate(GRATING_GRID):
            for column, grating in enumerate(grid_row):
                expected = np.zeros(5) if grating == 0 else [grating * 10.0 + p for p in range(5)]
                np.testing.assert_array_equal(tensor[0, :5, row, column], expected)

    def test_tail_split(self):
        labels = np.repeat(np.arange(1, 26), 10)
        train, validation = stratified_tail_split(labels, validation_fraction=0.2)
        self.assertEqual((len(train), len(validation)), (200, 50))
        for label in range(1, 26):
            self.assertEqual(int(np.sum(labels[validation] == label)), 2)

    def test_shapes_and_round_trip(self):
        model = SparseTouchCNN()
        logits, force = model(torch.zeros((4, 6, 4, 3), dtype=torch.float32))
        self.assertEqual(tuple(logits.shape), (4, 25))
        self.assertEqual(tuple(force.shape), (4, 3))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "touch_model.pt"
            save_touch_model(
                path, model=model, feature_mean=np.arange(5, dtype=float),
                feature_std=np.ones(5) * 2.0, force_mean=np.asarray([1.0, 2.0, 3.0]),
                force_std=np.asarray([0.5, 0.5, 1.0]), report={"position_accuracy": 0.75},
            )
            loaded = load_touch_model(path)
            labels, forces, probabilities = loaded.predict(np.ones((2, 45)) * 100.0)
        self.assertEqual(labels.shape, (2,))
        self.assertEqual(forces.shape, (2, 3))
        self.assertEqual(probabilities.shape, (2, 25))
        np.testing.assert_allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
        self.assertEqual(loaded.report["position_accuracy"], 0.75)

    def test_realtime_prediction_payload_contains_position_probability_and_force(self):
        model = SparseTouchCNN()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "touch_model.pt"
            save_touch_model(
                path,
                model=model,
                feature_mean=np.zeros(5, dtype=np.float32),
                feature_std=np.ones(5, dtype=np.float32),
                force_mean=np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
                force_std=np.ones(3, dtype=np.float32),
                report={"dataset_dir": "demo"},
            )
            result = prediction_payload(load_touch_model(path), np.arange(45))
        self.assertIn(result["position"], range(1, 26))
        self.assertEqual(len(result["probabilities"]), 25)
        self.assertEqual(len(result["force"]), 3)
        self.assertAlmostEqual(sum(result["probabilities"]), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
