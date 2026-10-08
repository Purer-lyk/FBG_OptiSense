"""Fifth workspace page: sparse-CNN training and 25-point evaluation."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from manual_stress_capture import FingerSurfaceMap, ForceVectorWidget
from touch_model_catalog import find_latest_touch_model, model_description
class PerformanceFingerMap(FingerSurfaceMap):
    """Top-view finger whose 25 markers encode validation accuracy."""

    def __init__(self, *, on_selected=None, parent=None):
        super().__init__(on_selected=on_selected, parent=parent)
        self.performance_colors = [(0.23, 0.49, 0.86, 0.98)] * 25
        self.setMinimumSize(480, 550)

    @staticmethod
    def _accuracy_color(accuracy):
        value = max(0.0, min(1.0, float(accuracy)))
        if value < 0.5:
            ratio = value / 0.5
            return (0.90, 0.22 + 0.48 * ratio, 0.18, 0.98)
        ratio = (value - 0.5) / 0.5
        return (0.90 - 0.66 * ratio, 0.70 + 0.12 * ratio, 0.18, 0.98)

    def set_performance(self, per_point):
        by_label = {int(item["label"]): item for item in per_point}
        colors = []
        for label in range(1, 26):
            item = by_label.get(label) or {}
            accuracy = float(item.get("position_accuracy", 0.0))
            colors.append(self._accuracy_color(accuracy))
            button = self.group.button(label)
            if button is not None:
                button.setToolTip(
                    f"编号{label} · 验证准确率{accuracy:.1%} · "
                    f"样本{int(item.get('support', 0))}条"
                )
        self.performance_colors = colors
        self.marker_item.setData(
            pos=self.marker_positions,
            size=13.0,
            color=np.asarray(colors, dtype=float),
            pxMode=True,
        )

    def show_live_prediction(self, predicted_label, expected_label=None):
        colors = [(0.32, 0.40, 0.52, 0.65)] * 25
        sizes = [9.0] * 25
        predicted = int(predicted_label)
        expected = None if expected_label is None else int(expected_label)
        if expected is not None and 1 <= expected <= 25:
            colors[expected - 1] = (0.10, 0.72, 0.95, 1.0)
            sizes[expected - 1] = 17.0
        if 1 <= predicted <= 25:
            colors[predicted - 1] = (
                (0.15, 0.82, 0.42, 1.0)
                if expected is None or predicted == expected
                else (0.96, 0.22, 0.22, 1.0)
            )
            sizes[predicted - 1] = 30.0
        self.marker_item.setData(
            pos=self.marker_positions,
            size=np.asarray(sizes, dtype=float),
            color=np.asarray(colors, dtype=float),
            pxMode=True,
        )


class ForceCurveComparison(QtWidgets.QWidget):
    """Three plots where real/predicted overlap shows force accuracy."""

    AXIS_COLORS = {"Fx": "#E84C5B", "Fy": "#23B16F", "Fz": "#3A7CF6"}

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        heading = QtWidgets.QLabel(
            "三维力预测曲线 · 实线为真实值，橙色虚线为CNN预测值；越重合表示越准确"
        )
        heading.setObjectName("sectionCaption")
        heading.setWordWrap(True)
        self.heading = heading
        layout.addWidget(heading)
        plots = QtWidgets.QHBoxLayout()
        plots.setSpacing(8)
        self.actual_curves = {}
        self.predicted_curves = {}
        self.plots = {}
        for axis in ("Fx", "Fy", "Fz"):
            plot = pg.PlotWidget(background="w")
            plot.setMinimumHeight(190)
            plot.showGrid(x=True, y=True, alpha=0.18)
            plot.setTitle(f"{axis} 真实/预测", color="#EAF2FF", size="10pt")
            plot.setLabel("left", f"{axis} (N)")
            plot.setLabel("bottom", "验证样本序号")
            plot.addLegend(offset=(8, 6), labelTextColor="#EAF2FF")
            actual = plot.plot(
                [], [], name="真实", pen=pg.mkPen(self.AXIS_COLORS[axis], width=2.6)
            )
            predicted = plot.plot(
                [], [], name="预测",
                pen=pg.mkPen("#F59E0B", width=2.4, style=QtCore.Qt.DashLine),
            )
            self.plots[axis] = plot
            self.actual_curves[axis] = actual
            self.predicted_curves[axis] = predicted
            plots.addWidget(plot, 1)
        layout.addLayout(plots)

    def set_trace(self, trace):
        trace = trace or {}
        actual = np.asarray(trace.get("actual") or [], dtype=float)
        predicted = np.asarray(trace.get("predicted") or [], dtype=float)
        order = np.asarray(trace.get("sample_order") or [], dtype=float)
        valid = (
            actual.ndim == 2
            and predicted.ndim == 2
            and actual.shape == predicted.shape
            and actual.shape[1:] == (3,)
            and len(order) == len(actual)
        )
        for column, axis in enumerate(("Fx", "Fy", "Fz")):
            if valid:
                self.actual_curves[axis].setData(order, actual[:, column])
                self.predicted_curves[axis].setData(order, predicted[:, column])
                self.plots[axis].enableAutoRange()
            else:
                self.actual_curves[axis].setData([], [])
                self.predicted_curves[axis].setData([], [])


class TouchTrainingWorker(QtCore.QThread):
    loading = QtCore.pyqtSignal(int, int)
    progress = QtCore.pyqtSignal(int, int, object)
    completed = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, dataset_dir, *, epochs=180, parent=None):
        super().__init__(parent)
        self.dataset_dir = Path(dataset_dir)
        self.epochs = int(epochs)
        self.cancel_event = threading.Event()
        self.process = None

    def cancel(self):
        self.cancel_event.set()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()

    def run(self):
        try:
            creationflags = (
                subprocess.CREATE_NO_WINDOW
                if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0
            )
            self.process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "manual_touch_cnn",
                    "--train",
                    str(self.dataset_dir),
                    "--epochs",
                    str(self.epochs),
                ],
                cwd=str(Path(__file__).resolve().parent),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                creationflags=creationflags,
            )
            completed_payload = None
            for line in self.process.stdout:
                if self.cancel_event.is_set():
                    raise RuntimeError("训练已取消")
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if payload.get("type") == "loading":
                    self.loading.emit(int(payload["current"]), int(payload["total"]))
                elif payload.get("type") == "progress":
                    self.progress.emit(
                        int(payload["epoch"]), int(payload["total"]),
                        dict(payload["metrics"]),
                    )
                elif payload.get("type") == "completed":
                    completed_payload = payload
            return_code = self.process.wait()
            error_text = self.process.stderr.read().strip()
            if self.cancel_event.is_set():
                raise RuntimeError("训练已取消")
            if return_code != 0:
                raise RuntimeError(error_text or f"训练子进程退出码{return_code}")
            if completed_payload is None:
                raise RuntimeError("训练结束但没有生成评估报告")
            report = json.loads(
                Path(completed_payload["report_path"]).read_text("utf-8")
            )
            report["model_path"] = completed_payload["model_path"]
            report["report_path"] = completed_payload["report_path"]
        except Exception as exc:
            self.failed.emit(str(exc) or type(exc).__name__)
        else:
            self.completed.emit(report)
        finally:
            self.process = None


class TouchInferenceWorker(QtCore.QThread):
    """Keep torch in one child process and drop stale frames under load."""

    ready = QtCore.pyqtSignal(object)
    predicted = QtCore.pyqtSignal(object)
    failed = QtCore.pyqtSignal(str)

    def __init__(self, model_path, parent=None):
        super().__init__(parent)
        self.model_path = Path(model_path).resolve()
        self.process = None
        self._stop_event = threading.Event()
        self._requests = queue.Queue(maxsize=1)
        self._request_id = 0
        self._result_lock = threading.Lock()
        self._latest_result = None

    def publish_prediction(self, payload):
        # One-slot result mailbox; never enqueue one Qt event per inference.
        with self._result_lock:
            self._latest_result = payload

    def take_latest_prediction(self):
        with self._result_lock:
            result = self._latest_result
            self._latest_result = None
        return result

    def submit(self, adc_values):
        self._request_id += 1
        item = {
            "request_id": self._request_id,
            "adc": [float(value) for value in adc_values],
        }
        try:
            self._requests.put_nowait(item)
        except queue.Full:
            try:
                self._requests.get_nowait()
            except queue.Empty:
                pass
            self._requests.put_nowait(item)

    def cancel(self):
        self._stop_event.set()
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()

    def run(self):
        try:
            self.process = subprocess.Popen(
                [sys.executable, "-m", "manual_touch_cnn", "--serve", str(self.model_path)],
                cwd=str(Path(__file__).resolve().parent),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            ready_line = self.process.stdout.readline()
            if not ready_line:
                error = self.process.stderr.read().strip()
                raise RuntimeError(error or "实时推理子进程未就绪")
            ready = json.loads(ready_line)
            if ready.get("type") != "ready":
                raise RuntimeError(ready.get("message") or "实时推理模型载入失败")
            self.ready.emit(ready)
            while not self._stop_event.is_set():
                try:
                    request = self._requests.get(timeout=0.1)
                except queue.Empty:
                    continue
                if self.process.poll() is not None:
                    raise RuntimeError("实时推理子进程意外退出")
                self.process.stdin.write(json.dumps(request) + "\n")
                self.process.stdin.flush()
                line = self.process.stdout.readline()
                if not line:
                    error = self.process.stderr.read().strip()
                    raise RuntimeError(error or "实时推理没有返回结果")
                result = json.loads(line)
                if result.get("type") == "error":
                    raise RuntimeError(result.get("message") or "实时推理失败")
                self.publish_prediction(result)
        except Exception as exc:
            if not self._stop_event.is_set():
                self.failed.emit(str(exc) or type(exc).__name__)
        finally:
            process = self.process
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=3)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)
                for stream in (process.stdin, process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
            self.process = None


class ManualTouchTrainingPage(QtWidgets.QWidget):
    """Train and inspect one shared position/three-axis-force sparse CNN."""

    def __init__(
        self,
        *,
        default_dataset_dir: Path,
        model_search_root: Path | None = None,
        frame_source=None,
        force_provider=None,
        parent=None,
    ):
        super().__init__(parent)
        self.default_dataset_dir = self._resolve_default_dataset(default_dataset_dir)
        project_outputs = Path(__file__).resolve().parent / "outputs"
        self.model_search_root = (Path(model_search_root) if model_search_root is not None else project_outputs if Path(default_dataset_dir).resolve().is_relative_to(project_outputs) else None)
        self.frame_source = frame_source
        self.force_provider = force_provider
        self.worker: TouchTrainingWorker | None = None
        self.inference: TouchInferenceWorker | None = None
        self.model_path: Path | None = None
        self.report = None
        self.report_by_label = {}
        self._live_expected_label = None
        self._live_prediction_count = 0
        self._live_series_limit = 240
        self._live_samples = deque(maxlen=self._live_series_limit)
        self._live_predicted = deque(maxlen=self._live_series_limit)
        self._live_actual_force = deque(maxlen=self._live_series_limit)
        self._build_ui()
        self.live_render_timer = QtCore.QTimer(self)
        self.live_render_timer.setInterval(100)
        self.live_render_timer.timeout.connect(self._poll_live_prediction)
        self._set_dataset_path(self.default_dataset_dir)
        latest = find_latest_touch_model(self.model_search_root) if self.model_search_root is not None else None
        if latest is not None:
            self.load_model(latest)
        else:
            self._load_existing_report_if_present()
        self._attach_live_listener()

    @staticmethod
    def _resolve_default_dataset(path):
        candidate = Path(path).resolve()
        if (candidate / "frames.jsonl").is_file():
            return candidate
        if candidate.is_dir():
            sessions = sorted(
                (
                    item for item in candidate.iterdir()
                    if item.is_dir() and (item / "frames.jsonl").is_file()
                ),
                key=lambda item: item.stat().st_mtime_ns,
                reverse=True,
            )
            if sessions:
                return sessions[0]
        return candidate

    def _build_ui(self):
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(18, 16, 18, 18)
        outer.setSpacing(12)

        title = QtWidgets.QLabel("模型训练与评估 · 空载/25点按压位置 + 三维力")
        title.setObjectName("pageTitle")
        outer.addWidget(title)
        self.input_shape_label = QtWidgets.QLabel(
            "稀疏空间输入：9个光栅按4×3布局排列；每光栅5个实测值标准化后"
            "形成5个信号通道，再加入1个传感器存在性Mask通道，CNN输入为 B×6×4×3。"
            "网络共用卷积特征提取器：位置头执行25类Hot-bit/Softmax分类，"
            "三维力头执行Fx、Fy、Fz回归。"
        )
        self.input_shape_label.setWordWrap(True)
        self.input_shape_label.setObjectName("softHint")
        outer.addWidget(self.input_shape_label)

        controls_card = QtWidgets.QFrame()
        controls_card.setObjectName("contentCard")
        controls = QtWidgets.QGridLayout(controls_card)
        controls.setContentsMargins(16, 12, 16, 12)
        controls.addWidget(QtWidgets.QLabel("训练数据集"), 0, 0)
        self.dataset_edit = QtWidgets.QLineEdit()
        self.dataset_edit.setMinimumWidth(480)
        controls.addWidget(self.dataset_edit, 0, 1, 1, 4)
        self.browse_button = QtWidgets.QPushButton("选择目录")
        self.browse_button.clicked.connect(self.browse_dataset)
        controls.addWidget(self.browse_button, 0, 5)
        controls.addWidget(QtWidgets.QLabel("训练轮数"), 1, 0)
        self.epochs_spin = QtWidgets.QSpinBox()
        self.epochs_spin.setRange(10, 1000)
        self.epochs_spin.setValue(180)
        controls.addWidget(self.epochs_spin, 1, 1)
        self.train_button = QtWidgets.QPushButton("训练稀疏CNN")
        self.train_button.setProperty("role", "primary")
        self.train_button.clicked.connect(self.start_training)
        controls.addWidget(self.train_button, 1, 2)
        self.cancel_button = QtWidgets.QPushButton("取消训练")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_training)
        controls.addWidget(self.cancel_button, 1, 3)
        self.load_model_button = QtWidgets.QPushButton("载入已有模型")
        self.load_model_button.clicked.connect(self.browse_model)
        controls.addWidget(self.load_model_button, 1, 4)
        controls.setColumnStretch(1, 1)
        outer.addWidget(controls_card)

        live_card = QtWidgets.QFrame()
        live_card.setObjectName("contentCard")
        live = QtWidgets.QGridLayout(live_card)
        live.setContentsMargins(16, 12, 16, 12)
        live.addWidget(QtWidgets.QLabel("实时模型文件"), 0, 0)
        self.live_model_edit = QtWidgets.QLineEdit()
        self.live_model_edit.setReadOnly(True)
        self.live_model_edit.setMinimumWidth(420)
        live.addWidget(self.live_model_edit, 0, 1, 1, 4)
        self.start_live_button = QtWidgets.QPushButton("开始实时部署")
        self.start_live_button.setProperty("role", "primary")
        self.start_live_button.setEnabled(False)
        self.start_live_button.clicked.connect(self.start_live)
        live.addWidget(self.start_live_button, 0, 5)
        self.stop_live_button = QtWidgets.QPushButton("停止实时部署")
        self.stop_live_button.setEnabled(False)
        self.stop_live_button.clicked.connect(self.stop_live)
        live.addWidget(self.stop_live_button, 0, 6)
        live.addWidget(QtWidgets.QLabel("期望位置（可选）"), 1, 0)
        self.expected_position_combo = QtWidgets.QComboBox()
        self.expected_position_combo.addItem("不对比（仅观察预测）", None)
        self.expected_position_combo.addItem("无按压／空载（0号）", 0)
        for label in range(1, 26):
            self.expected_position_combo.addItem(f"编号{label}", label)
        self.expected_position_combo.currentIndexChanged.connect(
            self._expected_position_changed
        )
        live.addWidget(self.expected_position_combo, 1, 1)
        self.live_status_label = QtWidgets.QLabel(
            "实时部署未启动：载入模型后可订阅临时测试模式45点帧并在线推理"
        )
        self.live_status_label.setWordWrap(True)
        self.live_status_label.setObjectName("softHint")
        live.addWidget(self.live_status_label, 1, 2, 1, 5)
        live.setColumnStretch(4, 1)
        outer.addWidget(live_card)

        self.progress_bar = QtWidgets.QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        outer.addWidget(self.progress_bar)
        self.status_label = QtWidgets.QLabel("选择数据集后开始训练；验证集采用每个编号时间序列末尾20%。")
        self.status_label.setWordWrap(True)
        self.status_label.setObjectName("softHint")
        outer.addWidget(self.status_label)
        self.overall_metrics_label = QtWidgets.QLabel("尚未载入模型评估结果")
        self.overall_metrics_label.setObjectName("statusPill")
        self.overall_metrics_label.setWordWrap(True)
        outer.addWidget(self.overall_metrics_label)
        self.validation_warning_label = QtWidgets.QLabel(
            "验证可靠性：尚未读取数据集采集轮次信息"
        )
        self.validation_warning_label.setObjectName("statusPill")
        self.validation_warning_label.setProperty("statusKind", "warning")
        self.validation_warning_label.setWordWrap(True)
        outer.addWidget(self.validation_warning_label)

        results = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        results.setChildrenCollapsible(False)
        left_card = QtWidgets.QFrame()
        left_card.setObjectName("contentCard")
        left_layout = QtWidgets.QVBoxLayout(left_card)
        left_layout.addWidget(QtWidgets.QLabel("实时位置预测 · 5×5点阵"))
        self.live_position_label = QtWidgets.QLabel("等待实时位置预测")
        self.live_position_label.setWordWrap(True)
        self.live_position_label.setStyleSheet("font-size:28px; font-weight:700; color:#F97316; padding:12px; border:2px solid #F97316; border-radius:10px;")
        left_layout.addWidget(self.live_position_label)
        self.performance_map = PerformanceFingerMap(
            on_selected=self.select_result_point, parent=left_card
        )
        left_layout.addWidget(self.performance_map, 1)
        self.selected_point_label = QtWidgets.QLabel("点击手指上的编号查看该点表现")
        self.selected_point_label.setWordWrap(True)
        self.selected_point_label.setObjectName("softHint")
        left_layout.addWidget(self.selected_point_label)
        results.addWidget(left_card)

        right = QtWidgets.QWidget()
        right_layout = QtWidgets.QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        force_row = QtWidgets.QHBoxLayout()
        self.actual_force = ForceVectorWidget(
            parent=right,
            title="真实平均三维力",
            show_connection_status=False,
            subtitle="所选编号验证集三轴平均值",
        )
        self.predicted_force = ForceVectorWidget(
            parent=right,
            title="CNN预测三维力",
            show_connection_status=False,
            subtitle="所选编号CNN预测三轴平均值",
        )
        self.actual_force.zero_button.hide()
        self.predicted_force.zero_button.hide()
        self.actual_force.setMinimumSize(260, 320)
        self.predicted_force.setMinimumSize(260, 320)
        force_row.addWidget(self.actual_force)
        force_row.addWidget(self.predicted_force)
        right_layout.addLayout(force_row)

        self.force_curves = ForceCurveComparison(parent=right)
        right_layout.addWidget(self.force_curves)

        self.live_curves = ForceCurveComparison(parent=right)
        self.live_curves.heading.setText(
            "实时三维力曲线 · 实线为PX6D真实力，橙色虚线为模型预测；无真实力时仅显示预测"
        )
        self.live_curves.setVisible(True)
        for plot in self.live_curves.plots.values():
            plot.setMinimumHeight(400)
            plot.setLabel("bottom", "实时推理帧序号")
            plot.getAxis("left").enableAutoSIPrefix(False)
        right_layout.addWidget(self.live_curves)

        self.performance_table = QtWidgets.QTableWidget(25, 9)
        self.performance_table.setHorizontalHeaderLabels([
            "编号", "验证样本", "位置准确率", "真实Fx", "预测Fx",
            "真实Fy", "预测Fy", "真实Fz", "预测Fz",
        ])
        self.performance_table.verticalHeader().hide()
        self.performance_table.setSelectionBehavior(QtWidgets.QAbstractItemView.SelectRows)
        self.performance_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.performance_table.cellClicked.connect(
            lambda row, _column: self.select_result_point(row + 1)
        )
        header = self.performance_table.horizontalHeader()
        header.setSectionResizeMode(QtWidgets.QHeaderView.ResizeToContents)
        header.setStretchLastSection(True)
        for row in range(25):
            self.performance_table.setItem(row, 0, QtWidgets.QTableWidgetItem(str(row + 1)))
        right_layout.addWidget(self.performance_table, 1)
        results.addWidget(right)
        results.setStretchFactor(0, 4)
        results.setStretchFactor(1, 6)
        outer.addWidget(results, 1)
        self.actual_force.hide()
        self.predicted_force.hide()
        self.force_curves.hide()
        self.performance_table.hide()
        self.selected_point_label.hide()

    def _set_dataset_path(self, path):
        self.dataset_edit.setText(str(Path(path)))

    def browse_dataset(self):
        selected = QtWidgets.QFileDialog.getExistingDirectory(
            self, "选择手动触觉数据集", self.dataset_edit.text()
        )
        if selected:
            self._set_dataset_path(Path(selected))
            self._load_existing_report_if_present()

    def browse_model(self):
        selected, _filter = QtWidgets.QFileDialog.getOpenFileName(
            self, "载入触觉模型（注意力或CNN）", str(self.model_path.parent) if self.model_path else self.dataset_edit.text(), "PyTorch模型 (*.pt)"
        )
        if selected:
            self.load_model(selected)

    def _load_existing_report_if_present(self):
        model_path = Path(self.dataset_edit.text()) / "models" / "manual_touch_sparse_cnn.pt"
        if model_path.is_file():
            self.load_model(model_path)

    # ------------------------------------------------- live deployment ----

    def _extract_live_adc(self, frame):
        """45 stable second-sample codes, exactly the training contract."""

        records = list(frame.get("records") or [])
        if len(records) != 45:
            raise ValueError(f"临时帧不是45点，收到{len(records)}点")
        return [int(record["second_code"]) for record in records]

    def _attach_live_listener(self):
        listeners = getattr(self.frame_source, "frame_listeners", None)
        if listeners is not None and self._receive_live_frame not in listeners:
            listeners.append(self._receive_live_frame)

    def _detach_live_listener(self):
        listeners = getattr(self.frame_source, "frame_listeners", None)
        if listeners is not None and self._receive_live_frame in listeners:
            listeners.remove(self._receive_live_frame)

    def _expected_position_changed(self, _index):
        value = self.expected_position_combo.currentData()
        self._live_expected_label = None if value is None else int(value)

    def load_model(self, path):
        source = Path(path).resolve()
        report_path = source.with_suffix(".report.json")
        try:
            report = json.loads(report_path.read_text("utf-8"))
        except Exception as exc:
            self.status_label.setText(f"模型评估报告载入失败：{exc}")
            return False
        if not source.is_file():
            self.status_label.setText(f"模型文件不存在：{source}")
            return False
        if self.inference is not None and self.inference.isRunning():
            self.status_label.setText("请先停止实时部署，再切换模型")
            return False
        report["model_path"] = str(source)
        self.model_path = source
        self.live_model_edit.setText(str(source))
        self.start_live_button.setEnabled(True)
        self.apply_report(report)
        self.status_label.setText(f"已载入模型：{source}")
        return True

    def start_live(self):
        if self.model_path is None:
            self.live_status_label.setText("请先载入模型，再开始实时部署")
            return False
        if self.inference is not None and self.inference.isRunning():
            return True
        self._live_prediction_count = 0
        self._live_samples.clear()
        self._live_predicted.clear()
        self._live_actual_force.clear()
        self.live_curves.setVisible(True)
        self.inference = TouchInferenceWorker(self.model_path, parent=self)
        self.inference.ready.connect(self._live_ready)
        self.live_render_timer.start()
        self.inference.failed.connect(self._live_failed)
        self.inference.finished.connect(self._live_finished)
        self.inference.start()
        self.start_live_button.setEnabled(False)
        self.stop_live_button.setEnabled(True)
        self.live_status_label.setText("正在启动实时推理子进程并载入模型…")
        self._attach_live_listener()
        return True

    def stop_live(self):
        self.live_render_timer.stop()
        self._detach_live_listener()
        inference = self.inference
        if inference is None:
            self._detach_live_listener()
            return False
        inference.cancel()
        self.start_live_button.setEnabled(self.model_path is not None)
        self.stop_live_button.setEnabled(False)
        self.live_status_label.setText("正在停止实时部署…")
        return True

    def _live_ready(self, payload):
        dataset = payload.get("dataset_dir") or ""
        self.live_status_label.setText(
            f"实时部署运行中 · 模型 {Path(payload.get('model_path', '')).name}"
            + (f" · 训练数据 {Path(dataset).name}" if dataset else "")
        )

    def _live_failed(self, message):
        self.live_status_label.setText(f"实时部署失败：{message}")
        self.start_live_button.setEnabled(self.model_path is not None)
        self.stop_live_button.setEnabled(False)

    def _live_finished(self):
        self.live_render_timer.stop()
        if self.inference is not None:
            self.inference.deleteLater()
        self.inference = None
        self.start_live_button.setEnabled(self.model_path is not None)
        self.stop_live_button.setEnabled(False)

    def _receive_live_frame(self, frame):
        if self.inference is None or not self.inference.isRunning():
            return
        try:
            adc = self._extract_live_adc(frame)
        except (ValueError, KeyError, TypeError):
            return
        self.inference.submit(adc)

    def _poll_live_prediction(self):
        inference = self.inference
        if inference is None:
            return
        payload = inference.take_latest_prediction()
        if payload is not None:
            self._live_prediction(payload)

    def _live_prediction(self, payload):
        self._live_prediction_count += 1
        position = int(payload["position"])
        confidence = float(payload.get("confidence", 0.0))
        force = tuple(float(value) for value in payload.get("force", (0.0, 0.0, 0.0)))
        actual_force = None
        if self.force_provider is not None:
            try:
                candidate = self.force_provider()
            except Exception:
                candidate = None
            if candidate is not None:
                actual_force = tuple(float(value) for value in candidate)
        position_text = "无按压／空载（0号）" if position == 0 else f"预测位置  {position:02d} 号"
        expected_text = "" if self._live_expected_label is None else " · 期望 " + ("无按压／空载" if self._live_expected_label == 0 else f"{self._live_expected_label:02d} 号")
        self.live_position_label.setText(f"{position_text}\n置信度 {confidence:.1%}" + expected_text)
        self._live_samples.append(self._live_prediction_count)
        self._live_predicted.append(force)
        self._live_actual_force.append(
            actual_force if actual_force is not None else (np.nan, np.nan, np.nan)
        )
        self.performance_map.show_live_prediction(
            position, expected_label=self._live_expected_label
        )
        sample_count = len(self._live_samples)
        order = np.asarray(list(self._live_samples)[-120:], dtype=float)
        predicted_window = np.asarray(list(self._live_predicted)[-120:], dtype=float)
        actual_window = np.asarray(list(self._live_actual_force)[-120:], dtype=float)
        for column, axis in enumerate(("Fx", "Fy", "Fz")):
            self.live_curves.predicted_curves[axis].setData(order, predicted_window[:, column])
            actual_column = actual_window[:, column]
            if np.isfinite(actual_column).any():
                self.live_curves.actual_curves[axis].setData(order, actual_column)
            else:
                self.live_curves.actual_curves[axis].setData([], [])
                self.live_curves.actual_curves[axis].setPen(
                    pg.mkPen(ForceCurveComparison.AXIS_COLORS[axis], width=2.6)
                )
        self.live_status_label.setText(
            f"实时部署运行中 · 已推理{self._live_prediction_count}帧 · "
            f"位置{position} · 置信度{confidence:.1%} · "
            f"预测力 Fx{force[0]:+.2f} / Fy{force[1]:+.2f} / Fz{force[2]:+.2f} N"
            + ("" if actual_force is not None else " · 无真实力参考")
        )

    def start_training(self):
        if self.worker is not None and self.worker.isRunning():
            return False
        dataset_dir = Path(self.dataset_edit.text()).resolve()
        if not (dataset_dir / "frames.jsonl").is_file():
            self.status_label.setText("所选目录中没有frames.jsonl")
            return False
        self.worker = TouchTrainingWorker(
            dataset_dir, epochs=self.epochs_spin.value(), parent=self
        )
        self.worker.loading.connect(self._loading_progress)
        self.worker.progress.connect(self._training_progress)
        self.worker.completed.connect(self._training_completed)
        self.worker.failed.connect(self._training_failed)
        self.worker.finished.connect(self._training_finished)
        self.train_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("正在流式读取1.3GB标注文件并构建紧凑训练缓存…")
        self.worker.start()
        return True

    def cancel_training(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.cancel()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("正在停止训练…")
            return True
        return False

    def _loading_progress(self, current, total):
        if total > 0:
            self.progress_bar.setValue(min(15, int(15 * current / total)))
            self.status_label.setText(f"正在读取训练帧：{current}/{total}")
        else:
            self.status_label.setText(f"正在读取训练帧：{current}条")

    def _training_progress(self, epoch, total, metrics):
        self.progress_bar.setValue(15 + int(85 * epoch / max(1, total)))
        self.status_label.setText(
            f"训练 {epoch}/{total} · 验证位置准确率"
            f"{float(metrics['position_accuracy']):.2%} · "
            f"三维力向量MAE {float(metrics['force_vector_mae_n']):.3f} N"
        )

    def _training_completed(self, report):
        self.progress_bar.setValue(100)
        self.apply_report(report)
        self.status_label.setText(
            f"训练完成并已保存：{report.get('model_path', '')}"
        )

    def _training_failed(self, message):
        self.status_label.setText(f"训练失败：{message}")

    def _training_finished(self):
        self.train_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if self.worker is not None:
            self.worker.deleteLater()
        self.worker = None

    @staticmethod
    def _force_text(value):
        return f"{float(value):+.3f} N"

    def apply_report(self, report):
        self.report = dict(report)
        nested_metrics = self.report.get("metrics") or {}
        metric_label = "测试集" if "test" in nested_metrics else "验证集"
        if nested_metrics:
            self.report.update(nested_metrics.get("test") or nested_metrics.get("validation") or {})
        source = self.report.get("model_path")
        if source and Path(source).is_file():
            self.model_path = Path(source).resolve()
            self.live_model_edit.setText(str(self.model_path))
            self.start_live_button.setEnabled(True)
        self.input_shape_label.setText(model_description(self.report))
        if self.report.get("schema") == "fbg_touch26_v1":
            self.status_label.setText("最新26类模型已载入；可开始实时部署（训练按钮仍为旧25类CNN单数据集训练）")
            self.report.setdefault("validation_warning", "逐类连续分块开发测试，不是独立会话泛化；空载与按压来自不同会话，存在会话混淆。")
        per_point = list(self.report.get("per_point") or [])
        self.report_by_label = {int(item["label"]): item for item in per_point}
        accuracy = float(self.report.get("position_accuracy", 0.0))
        force_mae = list(self.report.get("force_mae_n") or [0.0, 0.0, 0.0])
        self.overall_metrics_label.setText(
            f"{metric_label}位置准确率 {accuracy:.2%} · "
            f"Fx/Fy/Fz MAE = {force_mae[0]:.3f} / {force_mae[1]:.3f} / "
            f"{force_mae[2]:.3f} N · 三维向量MAE "
            f"{float(self.report.get('force_vector_mae_n', 0.0)):.3f} N · "
            f"训练/验证 {int(self.report.get('train_count', 0))}/"
            f"{int(self.report.get('validation_count', 0))}条 · "
            f"最佳轮次 {int(self.report.get('best_epoch', 0))}"
        )
        warning = str(self.report.get("validation_warning") or "")
        self.validation_warning_label.setText(
            "验证可靠性提示：" + (warning or "报告未包含独立采集轮次说明")
        )
        self.validation_warning_label.setProperty(
            "statusKind", "warning" if warning else "info"
        )
        self.validation_warning_label.style().unpolish(self.validation_warning_label)
        self.validation_warning_label.style().polish(self.validation_warning_label)
        self.performance_map.set_performance(per_point)
        for label in range(1, 26):
            row = label - 1
            item = self.report_by_label.get(label) or {}
            actual = list(item.get("actual_force_mean") or [0.0, 0.0, 0.0])
            predicted = list(item.get("predicted_force_mean") or [0.0, 0.0, 0.0])
            values = [
                str(label),
                str(int(item.get("support", 0))),
                f"{float(item.get('position_accuracy', 0.0)):.2%}",
                self._force_text(actual[0]), self._force_text(predicted[0]),
                self._force_text(actual[1]), self._force_text(predicted[1]),
                self._force_text(actual[2]), self._force_text(predicted[2]),
            ]
            for column, text in enumerate(values):
                cell = QtWidgets.QTableWidgetItem(text)
                cell.setTextAlignment(QtCore.Qt.AlignCenter)
                self.performance_table.setItem(row, column, cell)
        if per_point:
            self.select_result_point(1)

    def select_result_point(self, label):
        label = int(label)
        item = self.report_by_label.get(label)
        if item is None:
            self.selected_point_label.setText(f"编号{label}暂无验证结果")
            return False
        actual = tuple(float(v) for v in item["actual_force_mean"])
        predicted = tuple(float(v) for v in item["predicted_force_mean"])
        self.actual_force.set_force(*actual, connected=True)
        self.predicted_force.set_force(*predicted, connected=True)
        self.force_curves.set_trace(item.get("force_trace"))
        self.selected_point_label.setText(
            f"编号{label} · 位置准确率{float(item['position_accuracy']):.2%} · "
            f"验证样本{int(item['support'])}条 · 三轴力误差 "
            + " / ".join(self._force_text(v) for v in item.get("force_mae", (0, 0, 0)))
        )
        self.performance_table.selectRow(label - 1)
        button = self.performance_map.group.button(label)
        if button is not None:
            button.setChecked(True)
        return True

    def shutdown(self):
        self.live_render_timer.stop()
        self._detach_live_listener()
        inference = self.inference
        if inference is not None:
            inference.cancel()
            inference.wait(5000)
        worker = self.worker
        if worker is not None and worker.isRunning():
            worker.cancel()
            worker.wait(10000)

    def closeEvent(self, event):
        self.shutdown()
        super().closeEvent(event)


__all__ = [
    "ManualTouchTrainingPage",
    "PerformanceFingerMap",
    "TouchTrainingWorker",
]
