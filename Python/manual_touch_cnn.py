"""Sparse 4x3 CNN for 25-point press localization and 3-D force regression.

The nine gratings are physically arranged as::

    [ -, 4, - ]
    [ 9, 5, 3 ]
    [ 8, 6, 2 ]
    [ 7, -, 1 ]

Each grating contributes five measured values.  The CNN input therefore has five
standardized signal channels plus one fixed sensor-presence mask: Bx6x4x3.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


MODEL_SCHEMA = "fbg_manual_touch_sparse_cnn_v1"
GRATING_GRID = (
    (0, 4, 0),
    (9, 5, 3),
    (8, 6, 2),
    (7, 0, 1),
)
SENSOR_MASK = np.asarray(
    [[1.0 if grating else 0.0 for grating in row] for row in GRATING_GRID],
    dtype=np.float32,
)


def feature_statistics(adc_values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(adc_values, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != 45:
        raise ValueError("ADC训练输入必须是N×45")
    grouped = values.reshape(-1, 9, 5)
    mean = grouped.mean(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std = grouped.std(axis=(0, 1), dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def spatialize_five_point_features(
    adc_values: np.ndarray,
    *,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    """Map N×45 values to standardized N×(5 signal + mask)×4×3."""

    values = np.asarray(adc_values, dtype=np.float32)
    one_sample = values.ndim == 1
    if one_sample:
        values = values[None, :]
    if values.ndim != 2 or values.shape[1] != 45:
        raise ValueError("ADC输入必须包含45个值（9个光栅×每个5点）")
    mean = np.asarray(feature_mean, dtype=np.float32).reshape(1, 1, 5)
    std = np.asarray(feature_std, dtype=np.float32).reshape(1, 1, 5)
    if mean.size != 5 or std.size != 5 or np.any(std <= 0):
        raise ValueError("五个特征通道的均值/标准差无效")
    grouped = (values.reshape(-1, 9, 5) - mean) / std
    result = np.zeros((len(values), 6, 4, 3), dtype=np.float32)
    for row, grid_row in enumerate(GRATING_GRID):
        for column, grating in enumerate(grid_row):
            if grating:
                result[:, :5, row, column] = grouped[:, grating - 1, :]
    result[:, 5, :, :] = SENSOR_MASK
    return result[0] if one_sample else result


def stratified_tail_split(
    labels: np.ndarray, *, validation_fraction: float = 0.2
) -> tuple[np.ndarray, np.ndarray]:
    """Hold out the chronologically latest fraction within every position."""

    y = np.asarray(labels, dtype=np.int64).reshape(-1)
    if not 0.0 < float(validation_fraction) < 0.5:
        raise ValueError("validation_fraction必须在0和0.5之间")
    train: list[int] = []
    validation: list[int] = []
    for label in sorted(np.unique(y).tolist()):
        indices = np.flatnonzero(y == label)
        if len(indices) < 3:
            raise ValueError(f"编号{label}样本不足3条，不能训练和验证")
        held = max(1, int(round(len(indices) * validation_fraction)))
        held = min(held, len(indices) - 2)
        train.extend(indices[:-held].tolist())
        validation.extend(indices[-held:].tolist())
    return np.asarray(sorted(train), dtype=np.int64), np.asarray(
        sorted(validation), dtype=np.int64
    )


def acquisition_run_summary(labels: np.ndarray) -> dict:
    """Count contiguous acquisition runs per label without inventing group IDs."""

    values = np.asarray(labels, dtype=np.int64).reshape(-1)
    runs: dict[int, int] = {}
    previous = None
    for value in values.tolist():
        label = int(value)
        if label != previous:
            runs[label] = runs.get(label, 0) + 1
            previous = label
    return {
        "runs_by_label": dict(sorted(runs.items())),
        "labels_with_multiple_runs": sorted(
            label for label, count in runs.items() if count >= 2
        ),
        "labels_with_single_run": sorted(
            label for label, count in runs.items() if count == 1
        ),
    }


class SparseTouchCNN(nn.Module):
    """Shared sparse-grid backbone with position and force heads."""

    def __init__(self) -> None:
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(6, 24, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(24, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(48, 48, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(48 * 4 * 3, 128),
            nn.ReLU(),
            nn.Dropout(0.12),
        )
        self.position_head = nn.Linear(128, 25)
        self.force_head = nn.Linear(128, 3)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shared = self.backbone(values)
        return self.position_head(shared), self.force_head(shared)


@dataclass
class TouchDataset:
    adc: np.ndarray
    labels: np.ndarray
    forces: np.ndarray
    sample_indices: np.ndarray
    source_path: Path


def _extract_frame(frame: dict) -> tuple[np.ndarray, int, np.ndarray, int] | None:
    quality = frame.get("quality") or {}
    label = quality.get("press_label")
    stable = quality.get("stable_adc")
    if stable is None:
        stable = (quality.get("training_views") or {}).get("stable_adc")
    if stable is None:
        stable = (frame.get("raw_adc_codes") or {}).get("1")
    force = quality.get("paxini_force")
    if label is None or stable is None or force is None:
        return None
    adc = np.asarray(stable, dtype=np.float32)
    force_values = np.asarray(
        [force.get("fx_n"), force.get("fy_n"), force.get("fz_n")],
        dtype=np.float32,
    )
    if adc.shape != (45,) or not np.isfinite(adc).all():
        return None
    if force_values.shape != (3,) or not np.isfinite(force_values).all():
        return None
    label = int(label)
    if not 1 <= label <= 25:
        return None
    sample_index = int(quality.get("sample_index", frame.get("sequence", 0)))
    return adc, label, force_values, sample_index


def load_touch_dataset(
    session_dir: str | Path,
    *,
    progress: Callable[[int, int], None] | None = None,
    use_cache: bool = True,
) -> TouchDataset:
    """Stream the large JSONL once, then reuse a compact validated NPZ cache."""

    root = Path(session_dir).resolve()
    frames_path = root / "frames.jsonl"
    if not frames_path.is_file():
        raise FileNotFoundError(f"未找到训练帧文件：{frames_path}")
    model_dir = root / "models"
    cache_path = model_dir / "manual_touch_training_cache.npz"
    stat = frames_path.stat()
    if use_cache and cache_path.is_file():
        cache = np.load(cache_path, allow_pickle=False)
        if (
            int(cache["source_size"]) == stat.st_size
            and int(cache["source_mtime_ns"]) == stat.st_mtime_ns
        ):
            return TouchDataset(
                adc=cache["adc"],
                labels=cache["labels"],
                forces=cache["forces"],
                sample_indices=cache["sample_indices"],
                source_path=frames_path,
            )

    adc_rows: list[np.ndarray] = []
    labels: list[int] = []
    forces: list[np.ndarray] = []
    sample_indices: list[int] = []
    total_hint = 0
    manifest_path = root / "manifest.json"
    if manifest_path.is_file():
        try:
            total_hint = int(
                (json.loads(manifest_path.read_text("utf-8")).get("counts") or {}).get(
                    "frames", 0
                )
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            total_hint = 0
    with frames_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = _extract_frame(json.loads(line))
            if record is not None:
                adc, label, force, sample_index = record
                adc_rows.append(adc)
                labels.append(label)
                forces.append(force)
                sample_indices.append(sample_index)
            if progress is not None and line_number % 25 == 0:
                progress(line_number, total_hint)
    if progress is not None:
        progress(line_number if "line_number" in locals() else 0, total_hint)
    if len(adc_rows) < 75:
        raise ValueError("有效训练样本不足75条")
    result = TouchDataset(
        adc=np.asarray(adc_rows, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int64),
        forces=np.asarray(forces, dtype=np.float32),
        sample_indices=np.asarray(sample_indices, dtype=np.int64),
        source_path=frames_path,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            adc=result.adc,
            labels=result.labels,
            forces=result.forces,
            sample_indices=result.sample_indices,
            source_size=np.asarray(stat.st_size, dtype=np.int64),
            source_mtime_ns=np.asarray(stat.st_mtime_ns, dtype=np.int64),
        )
    temporary.replace(cache_path)
    return result


def _metrics(
    labels: np.ndarray,
    predicted_labels: np.ndarray,
    forces: np.ndarray,
    predicted_forces: np.ndarray,
) -> dict:
    labels = np.asarray(labels, dtype=np.int64)
    predicted_labels = np.asarray(predicted_labels, dtype=np.int64)
    errors = np.asarray(predicted_forces, dtype=float) - np.asarray(forces, dtype=float)
    axis_mae = np.mean(np.abs(errors), axis=0)
    axis_rmse = np.sqrt(np.mean(errors ** 2, axis=0))
    confusion = np.zeros((25, 25), dtype=np.int64)
    for actual, predicted in zip(labels, predicted_labels):
        confusion[int(actual) - 1, int(predicted) - 1] += 1
    per_point = []
    for label in range(1, 26):
        selected = labels == label
        actual_force = forces[selected]
        predicted_force = predicted_forces[selected]
        per_point.append({
            "label": label,
            "support": int(np.sum(selected)),
            "position_accuracy": float(np.mean(predicted_labels[selected] == label)),
            "actual_force_mean": np.mean(actual_force, axis=0).astype(float).tolist(),
            "predicted_force_mean": np.mean(predicted_force, axis=0).astype(float).tolist(),
            "force_mae": np.mean(
                np.abs(predicted_force - actual_force), axis=0
            ).astype(float).tolist(),
            "force_trace": {
                "sample_order": list(range(1, int(np.sum(selected)) + 1)),
                "actual": actual_force.astype(float).tolist(),
                "predicted": predicted_force.astype(float).tolist(),
            },
        })
    return {
        "position_accuracy": float(np.mean(predicted_labels == labels)),
        "force_mae_n": axis_mae.astype(float).tolist(),
        "force_rmse_n": axis_rmse.astype(float).tolist(),
        "force_vector_mae_n": float(np.mean(np.linalg.norm(errors, axis=1))),
        "confusion_matrix": confusion.tolist(),
        "per_point": per_point,
    }


def save_touch_model(
    path: str | Path,
    *,
    model: SparseTouchCNN,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
    force_mean: np.ndarray,
    force_std: np.ndarray,
    report: dict,
) -> Path:
    target = Path(path).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": MODEL_SCHEMA,
        "grating_grid": GRATING_GRID,
        "sensor_mask": SENSOR_MASK,
        "feature_mean": np.asarray(feature_mean, dtype=np.float32),
        "feature_std": np.asarray(feature_std, dtype=np.float32),
        "force_mean": np.asarray(force_mean, dtype=np.float32),
        "force_std": np.asarray(force_std, dtype=np.float32),
        "model_state": model.state_dict(),
        "report": dict(report),
    }
    temporary = target.with_suffix(target.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(target)
    return target


class TouchModelBundle:
    def __init__(self, payload: dict, path: Path):
        if payload.get("schema") != MODEL_SCHEMA:
            raise ValueError("不支持的触觉CNN模型格式")
        if tuple(tuple(row) for row in payload.get("grating_grid", ())) != GRATING_GRID:
            raise ValueError("模型的9光栅空间布局与当前软件不一致")
        self.path = path
        self.feature_mean = np.asarray(payload["feature_mean"], dtype=np.float32)
        self.feature_std = np.asarray(payload["feature_std"], dtype=np.float32)
        self.force_mean = np.asarray(payload["force_mean"], dtype=np.float32)
        self.force_std = np.asarray(payload["force_std"], dtype=np.float32)
        self.report = dict(payload.get("report") or {})
        self.model = SparseTouchCNN()
        self.model.load_state_dict(payload["model_state"])
        self.model.eval()

    def predict(self, adc_values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        spatial = spatialize_five_point_features(
            adc_values,
            feature_mean=self.feature_mean,
            feature_std=self.feature_std,
        )
        if spatial.ndim == 3:
            spatial = spatial[None, :]
        with torch.no_grad():
            logits, standardized_force = self.model(
                torch.from_numpy(spatial.astype(np.float32))
            )
            probabilities = torch.softmax(logits, dim=1).cpu().numpy()
            force = standardized_force.cpu().numpy() * self.force_std + self.force_mean
        return np.argmax(probabilities, axis=1) + 1, force, probabilities


def load_touch_model(path: str | Path) -> TouchModelBundle:
    source = Path(path).resolve()
    payload = torch.load(source, map_location="cpu", weights_only=False)
    if payload.get("schema") == "fbg_touch_dbm26_v1":
        from touch_dbm_training import DbmModelBundle
        return DbmModelBundle(source)
    if payload.get("schema") == "fbg_touch26_v1":
        from manual_touch_sensor_token import ModelBundle
        torch.set_num_threads(2)
        return ModelBundle(source)
    return TouchModelBundle(payload, source)


def prediction_payload(bundle: TouchModelBundle, adc_values) -> dict:
    """Return one JSON-safe real-time prediction from exactly 45 stable ADC values."""

    adc = np.asarray(adc_values, dtype=np.float32).reshape(-1)
    if adc.shape != (45,) or not np.isfinite(adc).all():
        raise ValueError("实时推理必须提供同一帧的45个有效稳定ADC值")
    labels, forces, probabilities = bundle.predict(adc)
    probability_row = probabilities[0].astype(float)
    return {
        "position": int(labels[0]),
        "confidence": float(np.max(probability_row)),
        "probabilities": probability_row.tolist(),
        "force": forces[0].astype(float).tolist(),
    }


def serve_touch_model(path: str | Path) -> None:
    """Persistent JSON-lines inference service, intentionally isolated from Qt."""

    bundle = load_touch_model(path)
    print(json.dumps({
        "type": "ready",
        "model_path": str(bundle.path),
        "dataset_dir": bundle.report.get("dataset_dir"),
        "input_shape": bundle.report.get("input_shape", [9, 5] if getattr(bundle, "kind", "") == "sensor_token" else [6, 4, 3]),
    }, ensure_ascii=False), flush=True)
    for line in sys.stdin:
        try:
            request = json.loads(line)
            result = prediction_payload(bundle, request.get("adc"))
            result.update({
                "type": "prediction",
                "request_id": request.get("request_id"),
            })
        except Exception as exc:
            result = {
                "type": "error",
                "request_id": request.get("request_id") if "request" in locals() else None,
                "message": str(exc) or type(exc).__name__,
            }
        print(json.dumps(result, ensure_ascii=False), flush=True)


def train_touch_cnn(
    session_dir: str | Path,
    *,
    output_path: str | Path | None = None,
    epochs: int = 180,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    validation_fraction: float = 0.2,
    seed: int = 20260927,
    progress: Callable[[int, int, dict], None] | None = None,
    load_progress: Callable[[int, int], None] | None = None,
    cancel_event: threading.Event | None = None,
) -> dict:
    """Train and save the shared position/force CNN on a manual-touch session."""

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(max(1, min(6, torch.get_num_threads())))
    dataset = load_touch_dataset(session_dir, progress=load_progress)
    train_indices, validation_indices = stratified_tail_split(
        dataset.labels, validation_fraction=validation_fraction
    )
    feature_mean, feature_std = feature_statistics(dataset.adc[train_indices])
    force_mean = dataset.forces[train_indices].mean(axis=0).astype(np.float32)
    force_std = dataset.forces[train_indices].std(axis=0).astype(np.float32)
    force_std[force_std < 1e-5] = 1.0
    spatial = spatialize_five_point_features(
        dataset.adc, feature_mean=feature_mean, feature_std=feature_std
    )
    standardized_force = (dataset.forces - force_mean) / force_std
    x_tensor = torch.from_numpy(spatial)
    y_tensor = torch.from_numpy(dataset.labels.astype(np.int64) - 1)
    f_tensor = torch.from_numpy(standardized_force.astype(np.float32))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        TensorDataset(
            x_tensor[train_indices], y_tensor[train_indices], f_tensor[train_indices]
        ),
        batch_size=int(batch_size),
        shuffle=True,
        generator=generator,
    )
    model = SparseTouchCNN()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(learning_rate), weight_decay=1e-4
    )
    class_counts = np.bincount(dataset.labels[train_indices] - 1, minlength=25)
    class_weights = len(train_indices) / (25.0 * np.maximum(class_counts, 1))
    position_loss = nn.CrossEntropyLoss(
        weight=torch.from_numpy(class_weights.astype(np.float32))
    )
    force_loss = nn.SmoothL1Loss()
    best_state = copy.deepcopy(model.state_dict())
    best_score = -math.inf
    best_epoch = 0
    stale_epochs = 0
    patience = max(18, min(35, int(epochs) // 5))
    val_x = x_tensor[validation_indices]
    val_labels = dataset.labels[validation_indices]
    val_force = dataset.forces[validation_indices]
    history = []
    for epoch in range(1, int(epochs) + 1):
        if cancel_event is not None and cancel_event.is_set():
            raise RuntimeError("训练已取消")
        model.train()
        total_loss = 0.0
        batches = 0
        for batch_x, batch_y, batch_force in loader:
            optimizer.zero_grad(set_to_none=True)
            logits, force_prediction = model(batch_x)
            loss = position_loss(logits, batch_y) + force_loss(
                force_prediction, batch_force
            )
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach())
            batches += 1
        model.eval()
        with torch.no_grad():
            logits, standardized_prediction = model(val_x)
            predicted_labels = torch.argmax(logits, dim=1).cpu().numpy() + 1
            predicted_force = (
                standardized_prediction.cpu().numpy() * force_std + force_mean
            )
        accuracy = float(np.mean(predicted_labels == val_labels))
        vector_mae = float(
            np.mean(np.linalg.norm(predicted_force - val_force, axis=1))
        )
        normalized_force_error = float(
            np.mean(np.abs((predicted_force - val_force) / force_std))
        )
        score = accuracy - 0.05 * normalized_force_error
        epoch_metrics = {
            "loss": total_loss / max(1, batches),
            "position_accuracy": accuracy,
            "force_vector_mae_n": vector_mae,
        }
        history.append(epoch_metrics)
        if progress is not None:
            progress(epoch, int(epochs), epoch_metrics)
        if score > best_score + 1e-5:
            best_score = score
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits, standardized_prediction = model(val_x)
        probabilities = torch.softmax(logits, dim=1).cpu().numpy()
        predicted_labels = np.argmax(probabilities, axis=1) + 1
        predicted_force = standardized_prediction.cpu().numpy() * force_std + force_mean
    report = _metrics(val_labels, predicted_labels, val_force, predicted_force)
    run_summary = acquisition_run_summary(dataset.labels)
    single_run_labels = run_summary["labels_with_single_run"]
    if single_run_labels:
        validation_warning = (
            "当前数据不是25个编号的多轮独立采集：编号"
            + "、".join(str(value) for value in single_run_labels)
            + "只有一个连续采集段。当前时间尾部验证仍与训练数据来自同一次按压轮次，"
              "指标只能用于开发评估，不能代表跨轮次或实际部署泛化性能。"
        )
    else:
        validation_warning = "全部编号均检测到多轮采集；仍建议按明确的press_cycle_id分组验证。"
    report.update({
        "schema": MODEL_SCHEMA,
        "dataset_dir": str(Path(session_dir).resolve()),
        "source_frames": str(dataset.source_path),
        "sample_count": int(len(dataset.labels)),
        "train_count": int(len(train_indices)),
        "validation_count": int(len(validation_indices)),
        "best_epoch": int(best_epoch),
        "epochs_completed": int(len(history)),
        "validation_split": "chronological_tail_within_each_of_25_labels",
        "acquisition_run_summary": run_summary,
        "validation_warning": validation_warning,
        "deployment_ready": False,
        "input_shape": [6, 4, 3],
        "signal_channels": 5,
        "mask_channels": 1,
        "grating_grid": [list(row) for row in GRATING_GRID],
        "multi_task_design": {
            "shared_feature_extractor": "three_layer_sparse_2d_cnn",
            "position_head": "25_class_softmax_hotbit_classification",
            "position_loss": "cross_entropy_equivalent_to_one_hot_target",
            "force_head": "fx_fy_fz_regression",
            "force_loss": "smooth_l1_on_standardized_force",
        },
        "history": history,
    })
    target = (
        Path(output_path)
        if output_path is not None
        else Path(session_dir) / "models" / "manual_touch_sparse_cnn.pt"
    )
    save_touch_model(
        target,
        model=model,
        feature_mean=feature_mean,
        feature_std=feature_std,
        force_mean=force_mean,
        force_std=force_std,
        report=report,
    )
    report_path = Path(target).with_suffix(".report.json")
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    report["model_path"] = str(Path(target).resolve())
    report["report_path"] = str(report_path.resolve())
    return report


__all__ = [
    "acquisition_run_summary",
    "GRATING_GRID",
    "MODEL_SCHEMA",
    "SENSOR_MASK",
    "SparseTouchCNN",
    "TouchDataset",
    "TouchModelBundle",
    "feature_statistics",
    "load_touch_dataset",
    "load_touch_model",
    "prediction_payload",
    "save_touch_model",
    "spatialize_five_point_features",
    "stratified_tail_split",
    "train_touch_cnn",
    "serve_touch_model",
]


def _cli(argv=None):
    parser = argparse.ArgumentParser(description="训练稀疏4x3手动触觉CNN")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--train", help="手动触觉数据集目录")
    mode.add_argument("--serve", help="常驻实时推理的模型文件")
    parser.add_argument("--epochs", type=int, default=180)
    args = parser.parse_args(argv)

    if args.serve:
        serve_touch_model(args.serve)
        return

    def emit(payload):
        print(json.dumps(payload, ensure_ascii=False), flush=True)

    report = train_touch_cnn(
        args.train,
        epochs=args.epochs,
        load_progress=lambda current, total: emit({
            "type": "loading", "current": current, "total": total,
        }),
        progress=lambda epoch, total, metrics: emit({
            "type": "progress", "epoch": epoch, "total": total,
            "metrics": metrics,
        }),
    )
    emit({
        "type": "completed",
        "model_path": report["model_path"],
        "report_path": report["report_path"],
    })


if __name__ == "__main__":
    _cli()
