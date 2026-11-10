"""Fifth workspace page: sparse-CNN training and 25-point evaluation."""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
import time
import hashlib
from collections import deque
from pathlib import Path

import numpy as np
from qt_runtime import configure_qt
configure_qt()
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph as pg

from manual_stress_capture import FingerSurfaceMap, ForceVectorWidget
from touch_model_catalog import model_description
from touch_worker_runtime import torch_worker_python
class PerformanceFingerMap(FingerSurfaceMap):
    """Top-view finger whose 25 markers encode validation accuracy."""

    def __init__(self, *, on_selected=None, parent=None):
        super().__init__(on_selected=on_selected, parent=parent)
        self.performance_colors = [(0.23, 0.49, 0.86, 0.98)] * 25
        self.setMinimumSize(220, 220)
        self.view.setMinimumSize(160, 180)
        self.marker_item.setGLOptions("translucent")
        from manual_stress_capture import _ContextScatterItem
        self.view.setBackgroundColor('#F3F5F7')
        neutral=np.tile(np.asarray((.76,.79,.82,.95)),(len(self.silicone_vertices),1))
        self.silicone_item.setMeshData(vertexes=self.silicone_vertices,faces=self.silicone_faces,vertexColors=neutral)
        self.prediction_halo_item=_ContextScatterItem(pos=np.empty((0,3)),size=24,color=(1.,.62,.16,.30),pxMode=True)
        self.prediction_halo_item.setGLOptions('translucent');self.view.addItem(self.prediction_halo_item)
        self.show_live_prediction([])

    def resizeEvent(self, event):
        super().resizeEvent(event)
        if not hasattr(self,'direction_vertices'):return
        # Fit distance only: top-down pose and canonical mesh/grid stay unchanged.
        # Use actual vertices, not fictitious box corners: depth can make box
        # projection grossly over-conservative and shrink the tactile surface.
        for _ in range(3):
            projected=[self._project_to_widget(v) for v in self.direction_vertices]
            if any(v is None for v in projected):return
            xy=np.array(projected);center=np.array([self.width()/2,self.height()/2])
            half=np.abs(xy-center).max(axis=0)
            factor=max(half[0]/max(1,self.width()*.46),half[1]/max(1,self.height()*.46))
            if factor>0:self.view.setCameraPosition(distance=self.view.opts['distance']*factor)
        self._position_buttons()

    def _position_buttons(self):
        projected=[self._project_to_widget(point) for point in self.marker_positions]
        if any(point is None for point in projected): return
        gaps=[np.linalg.norm(np.asarray(a)-b) for i,a in enumerate(projected) for b in projected[i+1:]]
        side=max(3,int(min(g for g in gaps if g>0)*.72))
        for button,(x,y) in zip(self.buttons,projected):
            button.setStyleSheet('QPushButton { background:transparent; border:none; min-width:0px; min-height:0px; padding:0px; }')
            button.setGeometry(round(x-side/2),round(y-side/2),side,side);button.show();button.raise_()

    def set_selected_points(self, labels):
        super().set_selected_points(labels)
        self._position_buttons()

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
        labels = predicted_label if isinstance(predicted_label,(list,tuple,set)) else ([] if predicted_label is None or predicted_label == 0 else [int(predicted_label)])
        if any(type(i) != int or not 1 <= i <= 25 for i in labels): raise ValueError('Invalid predicted position set')
        self.live_selected_labels = sorted(set(labels))
        colors = np.array([(0.32,0.40,0.52,0.65)]*25);sizes=np.full(25,7.)
        for label in self.live_selected_labels: colors[label-1]=(0.98,0.48,0.08,1);sizes[label-1]=12
        self.marker_item.setData(pos=self.marker_positions,size=sizes,color=colors,pxMode=True)
        self.set_selected_points(self.live_selected_labels)
        halo_positions=self.marker_positions[[i-1 for i in self.live_selected_labels]].copy() if labels else np.empty((0,3))
        if len(halo_positions):halo_positions[:,2]+=.015
        self.prediction_halo_item.setData(pos=halo_positions,size=24,color=(1.,.62,.16,.30),pxMode=True)
        if hasattr(self,'selection_marker_item'):
            self.selection_marker_item.setData(pos=self.marker_positions[[i-1 for i in self.live_selected_labels]] if labels else np.empty((0,3)),color=(.98,.48,.08,1),size=9,pxMode=True)


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
        plots = QtWidgets.QVBoxLayout()
        plots.setSpacing(4)
        self.actual_curves = {}
        self.predicted_curves = {}
        self.plots = {}
        for axis in ("Fx", "Fy", "Fz"):
            plot = pg.PlotWidget(background="#15191F")
            plot.setMinimumHeight(100)
            plot.showGrid(x=True, y=True, alpha=0.18)
            plot.setTitle(f"{axis} 真实/预测", color="#EAF2FF", size="10pt")
            plot.setLabel("left", f"{axis} (N)")
            plot.getAxis("left").enableAutoSIPrefix(False)
            for name in ("left","bottom"):
                plot.getAxis(name).setTextPen("#BCCBDC")
                plot.getAxis(name).setPen("#73839A")
            plot.setLabel("bottom", "光学源经过时间 (s)")
            plot.addLegend(offset=(8, 6), labelTextColor="#EAF2FF", colCount=2)
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
        self.plots["Fy"].setXLink(self.plots["Fx"])
        self.plots["Fz"].setXLink(self.plots["Fx"])
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

    def __init__(self, dataset_dir, *, epochs=180, architecture="sensor_token", force_history=False, parent=None):
        super().__init__(parent)
        self.dataset_dir = Path(dataset_dir)
        self.epochs = int(epochs)
        self.architecture = architecture
        self.force_history = bool(force_history)
        self.cancel_event = threading.Event()
        self.process = None

    def cancel(self):
        self.cancel_event.set()
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()

    def command(self):
        command = [torch_worker_python(), "-m", "touch_45_models", "--train", str(self.dataset_dir),
                   "--epochs", str(self.epochs), "--architecture", self.architecture]
        if self.force_history:
            command.append("--force-history")
        return command

    def run(self):
        try:
            creationflags = (
                subprocess.CREATE_NO_WINDOW
                if sys.platform == "win32" and hasattr(subprocess, "CREATE_NO_WINDOW")
                else 0
            )
            self.process = subprocess.Popen(
                self.command(),
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
                        int(payload["epoch"]), int(payload.get("total", self.epochs)),
                        dict(payload.get("metrics", payload)),
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

    def submit(self, adc_values, *, source_seq=None, timestamp_s=None, session_id=None, reset=False, feature_kind="adc", input_contract=None):
        self._request_id += 1
        item = {
            "request_id": self._request_id,
            (feature_kind if feature_kind in ("delta_nm","dbm") else "adc"): [float(value) for value in adc_values],
            "source_seq": self._request_id if source_seq is None else source_seq,
            "timestamp_s": timestamp_s,
            "session_id": session_id,
            "reset": bool(reset),
            "input_contract": input_contract,
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
                ([torch_worker_python(), "-m", self.service_module, str(self.model_path)] if hasattr(self,"service_module") else [torch_worker_python(), "-m", "manual_touch_cnn", "--serve", str(self.model_path)]),
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


from touch_deployment_ui import IndependentSlotsMixin

class ManualTouchTrainingPage(IndependentSlotsMixin, QtWidgets.QWidget):
    """Train and inspect one shared position/three-axis-force sparse CNN."""

    def __init__(
        self,
        *,
        default_dataset_dir: Path,
        model_search_root: Path | None = None,
        frame_source=None,
        force_provider=None,
        force_sensor=None,
        capture_window=None,
        parent=None,
    ):
        super().__init__(parent)
        self.default_dataset_dir = self._resolve_default_dataset(default_dataset_dir)
        project_outputs = Path(__file__).resolve().parent / "outputs"
        self.model_search_root = (Path(model_search_root) if model_search_root is not None else project_outputs if Path(default_dataset_dir).resolve().is_relative_to(project_outputs) else None)
        self.frame_source = frame_source
        self.capture_window = capture_window
        self._stream_reset_pending = True
        self._live_time_origin = None
        from paxini_px6d import PaxiniForceSensor
        self._owns_force_sensor = force_sensor is None
        self.force_sensor = force_sensor if force_sensor is not None else PaxiniForceSensor()
        self.force_provider = force_provider or self._acquired_force_reference
        self._use_shared_sensor_reference = force_sensor is not None or force_provider is None
        self._display_tare_count = self.force_sensor.tare_count
        self.force_notice = ""
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
        self._live_raw_actual_force = deque(maxlen=self._live_series_limit)
        self._build_ui()
        self.acquired_force_timer = QtCore.QTimer(self)
        self.acquired_force_timer.setInterval(100)
        self.acquired_force_timer.timeout.connect(self._refresh_acquired_force)
        self.acquired_force_timer.start()
        self.live_render_timer = QtCore.QTimer(self)
        self.live_render_timer.setInterval(100)
        self.live_render_timer.timeout.connect(self._poll_live_prediction)
        self._set_dataset_path(self.default_dataset_dir)
        self.live_status_label.setText('默认停用 · 等待45点9×5模型训练/选择；不会恢复9峰模型')
        self.status_label.setText('训练完成按可信同协议验证独立选优；自动加载但不自动开始')
        self._init_slots()
        self._attach_live_listener()

    @staticmethod
    def _resolve_default_dataset(path):
        return Path(path).resolve()
    def _build_ui(self):
        from touch_training_force_mini import TrainingForceMini
        outer = QtWidgets.QVBoxLayout(self)
        outer.setContentsMargins(4, 4, 4, 4); outer.setSpacing(4)
        self.input_shape_label = QtWidgets.QLabel('45点9×5 (dBm) · 真实second_code转换 · 25位多标签 / 整体三轴力 · 独立骨干')
        self.input_shape_label.setObjectName('softHint'); self.input_shape_label.hide()
        compact = QtWidgets.QWidget(); compact.setStyleSheet('QPushButton { min-height:22px; padding:2px 8px; } QLineEdit,QComboBox,QSpinBox { min-height:22px; padding:2px; }')
        controls = QtWidgets.QGridLayout(compact); controls.setContentsMargins(0,0,0,0); controls.setSpacing(4)
        self.dataset_edit=QtWidgets.QLineEdit(); self.browse_button=QtWidgets.QPushButton('数据目录'); self.browse_button.clicked.connect(self.browse_dataset)
        self.epochs_spin=QtWidgets.QSpinBox(); self.epochs_spin.setRange(1,1000); self.epochs_spin.setValue(180)
        self.architecture_combo=QtWidgets.QComboBox()
        for title,key in (('TCN','tcn'),('Token Transformer','sensor_token'),('稀疏CNN','sparse_cnn'),('Inception 1D','inception_1d')): self.architecture_combo.addItem(title,key)
        self.force_history_check=QtWidgets.QCheckBox('仅力历史');self.force_history_check.setToolTip('位置仅当前45点，力可用因果自预测历史；无真实力teacher forcing')
        self.train_button=QtWidgets.QPushButton('训练'); self.train_button.clicked.connect(self.start_training)
        self.cancel_button=QtWidgets.QPushButton('取消');self.cancel_button.setEnabled(False);self.cancel_button.clicked.connect(self.cancel_training)
        controls.addWidget(QtWidgets.QLabel('45点数据'),0,0);controls.addWidget(self.dataset_edit,0,1,1,5);controls.addWidget(self.browse_button,0,6)
        controls.addWidget(QtWidgets.QLabel('轮数'),1,0);controls.addWidget(self.epochs_spin,1,1);controls.addWidget(self.architecture_combo,1,2);controls.addWidget(self.force_history_check,1,3);controls.addWidget(self.train_button,1,4);controls.addWidget(self.cancel_button,1,5)
        self.details_button=QtWidgets.QToolButton();self.details_button.setText('评估详情 ▸');self.details_button.setCheckable(True);controls.addWidget(self.details_button,1,6)
        self.live_model_edit=QtWidgets.QLineEdit();self.live_model_edit.setReadOnly(True)
        self.load_model_button=QtWidgets.QPushButton('载入');self.load_model_button.clicked.connect(self.browse_model)
        self.start_live_button=QtWidgets.QPushButton('部署');self.start_live_button.setEnabled(False);self.start_live_button.clicked.connect(self.start_live)
        self.stop_live_button=QtWidgets.QPushButton('停止');self.stop_live_button.setEnabled(False);self.stop_live_button.clicked.connect(self.stop_live)
        self.live_model_edit.hide();self.load_model_button.hide()
        self.position_model_edit=QtWidgets.QLineEdit();self.position_model_edit.setReadOnly(True)
        self.force_model_edit=QtWidgets.QLineEdit();self.force_model_edit.setReadOnly(True)
        self.position_slot_status=QtWidgets.QLabel('位置未加载');self.force_slot_status=QtWidgets.QLabel('力未加载')
        for label in (self.position_slot_status,self.force_slot_status):label.setMinimumWidth(0);label.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Fixed)
        self.position_load_button=QtWidgets.QPushButton('选位置');self.position_load_button.clicked.connect(lambda:self._browse_slot('position'))
        self.force_load_button=QtWidgets.QPushButton('选力');self.force_load_button.clicked.connect(lambda:self._browse_slot('force'))
        self.reload_slots_button=QtWidgets.QPushButton('重新加载');self.reload_slots_button.clicked.connect(lambda:self.load_slot_pair(self.slot_pair,persist=False) if self.slot_pair else None)
        self.start_live_button.setText('开始推理')
        controls.addWidget(QtWidgets.QLabel('位置模型'),2,0);controls.addWidget(self.position_model_edit,2,1,1,2);controls.addWidget(self.position_slot_status,2,3,1,2);controls.addWidget(self.position_load_button,2,5);controls.addWidget(self.reload_slots_button,2,6)
        controls.addWidget(QtWidgets.QLabel('力模型'),3,0);controls.addWidget(self.force_model_edit,3,1,1,2);controls.addWidget(self.force_slot_status,3,3,1,2);controls.addWidget(self.force_load_button,3,5)
        for task,row in (('position',2),('force',3)):
            box=QtWidgets.QWidget();buttons=QtWidgets.QHBoxLayout(box);buttons.setContentsMargins(0,0,0,0);buttons.setSpacing(4)
            for kind,title in (('best','Best推荐'),('last','Last最新')):
                button=QtWidgets.QPushButton(title);setattr(self,task+'_'+kind+'_button',button)
                button.clicked.connect(lambda checked=False,t=task,k=kind:self._quick_load(t,k));buttons.addWidget(button)
            # Keep the existing file picker; place quick choices alongside it.
            picker=getattr(self,task+'_load_button');controls.removeWidget(picker);buttons.addWidget(picker)
            controls.addWidget(box,row,5,1,2)
        controls.removeWidget(self.reload_slots_button);controls.addWidget(self.reload_slots_button,3,7)
        self.slot_warning_label=QtWidgets.QLabel('双槽等待加载 · 不自动启动硬件');self.slot_warning_label.setStyleSheet('color:#B45309;font-weight:600;');self.slot_warning_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Fixed)
        controls.addWidget(self.slot_warning_label,4,0,1,5);controls.addWidget(self.start_live_button,4,5);controls.addWidget(self.stop_live_button,4,6)
        controls.setColumnStretch(2,1);controls.setColumnStretch(3,1);outer.addWidget(compact)
        status_row=QtWidgets.QHBoxLayout();self.progress_bar=QtWidgets.QProgressBar();self.progress_bar.setMaximumWidth(100);self.progress_bar.setMaximumHeight(18)
        self.status_label=QtWidgets.QLabel('训练只读取已保存45点dBm；候选不自动部署');self.status_label.setMinimumWidth(0);self.status_label.setMaximumHeight(24)
        status_row.addWidget(self.progress_bar);status_row.addWidget(self.status_label,1);outer.addLayout(status_row)
        self.live_status_label=QtWidgets.QLabel('默认停用；手动载入模型后部署');self.live_status_label.setMaximumHeight(24);status_row.addWidget(self.live_status_label,2)
        self.details_box=QtWidgets.QWidget();details=QtWidgets.QVBoxLayout(self.details_box);details.setContentsMargins(0,0,0,0)
        self.overall_metrics_label=QtWidgets.QLabel('尚无评估报告');self.overall_metrics_label.setWordWrap(True)
        self.validation_warning_label=QtWidgets.QLabel('验证选模、测试仅一次；尚无实测精度证据');self.validation_warning_label.setWordWrap(True)
        details.addWidget(self.overall_metrics_label);details.addWidget(self.validation_warning_label)
        self.details_box.hide();outer.addWidget(self.details_box);self.details_button.toggled.connect(self.details_box.setVisible)
        results=QtWidgets.QSplitter(QtCore.Qt.Horizontal);results.setChildrenCollapsible(False)
        left=QtWidgets.QWidget();left_layout=QtWidgets.QVBoxLayout(left);left_layout.setContentsMargins(0,0,0,0);left_layout.setSpacing(3)
        self.live_position_label=QtWidgets.QLabel('等待位置预测');self.live_position_label.setStyleSheet('font-size:17px;font-weight:700;color:#F97316;');self.live_position_label.setWordWrap(False);self.live_position_label.setFixedHeight(30);self.live_position_label.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Fixed);left_layout.addWidget(self.live_position_label)
        self.performance_map=PerformanceFingerMap(on_selected=self.select_result_point);self.performance_map.setMinimumHeight(180);self.performance_map.view.setMinimumHeight(160);left_layout.addWidget(self.performance_map,1)
        self.actual_force=TrainingForceMini(title='PX6D实测力');self.predicted_force=TrainingForceMini(title='预测整体三维力');self.predicted_force.hide();self.actual_force.zero_button.clicked.connect(self._request_force_zero);left_layout.addWidget(self.actual_force)
        sensor_row=QtWidgets.QHBoxLayout();self.force_connect_button=QtWidgets.QPushButton('连接PX6D');self.force_disconnect_button=QtWidgets.QPushButton('断开');self.force_connect_button.clicked.connect(self._connect_force_sensor);self.force_disconnect_button.clicked.connect(self.force_sensor.close)
        sensor_row.addWidget(self.force_connect_button);sensor_row.addWidget(self.force_disconnect_button);left_layout.addLayout(sensor_row)
        self.force_label=QtWidgets.QLabel('PX6D未连接');self.force_label.hide()
        self.selected_point_label=QtWidgets.QLabel('');self.selected_point_label.hide()
        self.expected_position_combo=QtWidgets.QComboBox();self.expected_position_combo.addItem('不对比',None);self.expected_position_combo.addItem('空载',0)
        for label in range(1,26):self.expected_position_combo.addItem(str(label),label)
        self.expected_position_combo.currentIndexChanged.connect(self._expected_position_changed);self.expected_position_combo.hide()
        results.addWidget(left)
        self.live_curves=ForceCurveComparison();self.live_curves.heading.setText('整体Fx/Fy/Fz · 实线PX6D，橙色虚线预测 · 近时采样非严格同步')
        results.addWidget(self.live_curves);results.setStretchFactor(0,45);results.setStretchFactor(1,55);results.setSizes([540,660]);self.results_splitter=results;outer.addWidget(results,1)
        self.force_curves=ForceCurveComparison();self.force_curves.hide()
        self.performance_table=QtWidgets.QTableWidget(25,9);self.performance_table.hide()
        for label in (self.input_shape_label,self.status_label,self.live_status_label):
            label.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Fixed)
            label.setMinimumWidth(0)
        self.live_curves.heading.setSizePolicy(QtWidgets.QSizePolicy.Ignored,QtWidgets.QSizePolicy.Fixed)
    def _acquired_force_reference(self):
        sample = self.force_sensor.measurement_snapshot()
        return tuple(sample['corrected'][:3]) if sample else None

    def _connect_force_sensor(self):
        try:
            self.force_sensor.start()
        except Exception as exc:
            self.force_notice = str(exc)
        self._refresh_acquired_force()

    def _request_force_zero(self, _checked=False):
        try:
            self.force_sensor.tare()
        except Exception as exc:
            self.force_notice = f"采集力归零失败：{exc}"
            success = False
        else:
            self.force_notice = "采集力归零成功（主机去皮，两页共享，不改变预测）"
            success = True
            # The refresh rebases retained raw measurements, never NN predictions.
        self._refresh_acquired_force()
        return success

    def _refresh_acquired_force(self):
        if self._display_tare_count != self.force_sensor.tare_count:
            self._display_tare_count = self.force_sensor.tare_count
            offset = self.force_sensor.tare_offset[:3]
            self._live_actual_force = deque(
                (tuple(v - o for v, o in zip(raw, offset)) if raw is not None else corrected
                 for raw, corrected in zip(self._live_raw_actual_force, self._live_actual_force)),
                maxlen=self._live_series_limit)
            self._redraw_live_force_curves()
        sample = self.force_sensor.measurement_snapshot()
        values = sample['corrected'] if sample else None
        self.actual_force.set_force(*(values[:3] if values else (None, None, None)),
                                    connected=values is not None,
                                    moments=values[3:] if values else None)
        if sample:
            fx, fy, fz = values[:3]
            status = (f"PX6D（{self.force_sensor.port or '协议回放'}）已连接 · "
                      f"Fx={fx:+.3f} Fy={fy:+.3f} Fz={fz:+.3f} N · "
                      f"主机去皮{sample['tare']['count']}次 · {sample['age_ms']:.0f} ms")
        else:
            status = 'PX6D未连接／数据过期；点击连接 · 实测参考为空'
            if self.force_sensor.last_error:
                status += ' · ' + self.force_sensor.last_error
        self.actual_force.setToolTip(status)
        self.force_label.setText(status + ' · 主机去皮，不发送设备标定命令' +
                                 (' · ' + self.force_notice if self.force_notice else ''))

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
        # Dataset selection is not explicit approval to load an old session model.
        return

    # ------------------------------------------------- live deployment ----

    def _extract_live_adc(self, frame):
        """45 stable second-sample codes, exactly the training contract."""

        records = list(frame.get("records") or [])
        if len(records) != 45 or frame.get('points_per_peak',5) != 5:
            raise ValueError(f"临时帧不是五点/45点，收到{len(records)}点")
        contract=(self.report or {}).get('input_contract') or {}
        if contract:
            if contract.get('input_shape') != [9,5] or frame.get('feedback_selector') != contract.get('feedback_selector'):
                raise ValueError('45点模型与实时跨阻档位/输入契约不匹配')
        return [int(record["second_code"]) for record in records]

    def _extract_live_dbm(self, frame):
        from touch_45_data import validate_input_contract
        from temporary_optical_units import adc_code_to_dbm
        contract=(self.report or {}).get('input_contract') or {}
        validate_input_contract(contract)
        channel=int(frame.get('adc_channel',1))
        selector=0 if channel>=2 else int(frame.get('feedback_selector',0))
        if channel!=contract['channel'] or selector!=contract['feedback_selector']:
            raise ValueError('45点模型与实时通道/有效跨阻档位不一致')
        records=frame.get('records') or []
        if len(records)!=45 or frame.get('points_per_peak',5)!=5:
            raise ValueError('必须45个真实second_code，不接受9峰补齐')
        values=np.asarray([r['second_code'] for r in records],float)
        if not np.isfinite(values).all():raise ValueError('45点包含非有限测量值')
        rows=(getattr(self.frame_source,'plan',None) or {}).get('rows',[])
        # Temperature shifts require reselecting real sampling points. Keep the
        # model's feature order (P1..P9, five ascending wavelengths each), but
        # do not bind live indices or absolute wavelengths to the training run.
        if len(rows)!=45:
            raise ValueError('实时路线必须为9峰×5点，共45点')
        waves=np.asarray([r.get('measured_nm') for r in rows],float)
        if not np.isfinite(waves).all() or not np.all(np.diff(waves)>0):
            raise ValueError('实时45点须按P1至P9及峰内波长递增排列')
        groups=[r.get('group') for r in rows]
        if any(g is not None for g in groups) and groups!=[i//5+1 for i in range(45)]:
            raise ValueError('实时路线必须保持P1至P9顺序，每峰5点')
        return adc_code_to_dbm(values,selector,floor_nonpositive=True).tolist()

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
        if report.get('schema') == 'fbg_touch_seven_temporal26_v1' or report.get('input_shape') in ([9,7], [63]):
            self.status_label.setText('当前只支持9×5/45点，七点/63点模型拒绝载入')
            return False
        if report.get('schema') == 'fbg_touch_delta9_multilabel_v1':
            self.status_label.setText('9峰模型与当前45点输入不兼容；请选择45点模型');return False
        if report.get('schema') == 'fbg_touch45_multilabel_v1':
            from touch_45_data import validate_input_contract
            try:validate_input_contract(report.get('input_contract'))
            except ValueError as exc:self.status_label.setText(str(exc));return False
            if report.get('input_shape') != [9,5] or report.get('checkpoint_sha256') != hashlib.sha256(source.read_bytes()).hexdigest() or Path(report.get('model_path','')).resolve() != source:
                self.status_label.setText('45点checkpoint哈希／报告绑定不匹配');return False
        if report.get('schema') == 'fbg_touch_temporal26_v1':
            if report.get('input_shape') != [9,5] or report.get('checkpoint_sha256') != hashlib.sha256(source.read_bytes()).hexdigest():
                self.status_label.setText('45点模型输入契约或checkpoint报告哈希不匹配')
                return False
        if self.inference is not None:
            self.status_label.setText("请先等待实时部署完全停止，再切换模型")
            return False
        report["model_path"] = str(source)
        self.model_path = source
        self.live_model_edit.setText(str(source))
        self.start_live_button.setEnabled(True)
        self.apply_report(report)
        self.status_label.setText(f"已载入模型：{source.name} · 未自动启动")
        self.status_label.setToolTip(str(source))
        return True

    def start_live(self):
        if getattr(self,'slot_pair',None) is not None:return self._start_slots()
        if self.model_path is None:
            self.live_status_label.setText("请先载入模型，再开始实时部署")
            return False
        if self.inference is not None:
            if self.inference.isRunning() and not self.inference._stop_event.is_set():
                return True
            self.live_status_label.setText("等待上一实时部署完全停止后再启动")
            return False
        self._live_time_origin = None
        self._stream_reset_pending = True
        self._live_source_seq = 0
        self._live_prediction_count = 0
        self._live_samples.clear()
        for curve in list(self.live_curves.predicted_curves.values())+list(self.live_curves.actual_curves.values()): curve.setData([],[])
        self._live_predicted.clear()
        self._live_actual_force.clear()
        self._live_raw_actual_force.clear()
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
        inference = self.inference
        if inference is None:
            return False
        inference.cancel()
        self.start_live_button.setEnabled(self.model_path is not None)
        self.stop_live_button.setEnabled(False)
        self.live_status_label.setText("正在停止实时部署…")
        return True

    def _live_ready(self, payload):
        if self._is_multilabel() and payload.get("input_contract") != (self.report or {}).get("input_contract"):
            self.live_status_label.setText("报告与checkpoint输入契约不匹配，拒绝部署")
            self.inference.cancel()
            return
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

    def _is_multilabel(self):
        return (self.report or {}).get('schema') == 'fbg_touch45_multilabel_v1'

    def _receive_live_frame(self, frame):
        if self.inference is None or not self.inference.isRunning(): return
        self._live_source_seq=getattr(self,'_live_source_seq',0)+1
        try:
            measured=(self.report or {}).get('schema')=='fbg_touch45_multilabel_v1'
            values=self._extract_live_dbm(frame) if measured else self._extract_live_adc(frame)
            timestamp=frame.get('timestamp_s')
            if timestamp is None and frame.get('cycle_start_us') is not None:timestamp=float(frame['cycle_start_us'])/1e6
            if timestamp is None and frame.get('monotonic_ns') is not None:timestamp=float(frame['monotonic_ns'])/1e9
            timestamp=None if timestamp is None else float(timestamp)
        except (ValueError,KeyError,TypeError) as exc:
            self._stream_reset_pending=True;self.live_status_label.setText('拒绝推理：'+str(exc));return
        self.inference.submit(values,source_seq=frame.get('source_seq',frame.get('sequence',self._live_source_seq)),timestamp_s=timestamp,session_id=frame.get('session_id'),reset=frame.get('reset',False) or self._stream_reset_pending,feature_kind='dbm' if measured else 'adc',input_contract=(self.report or {}).get('input_contract'))
        self._stream_reset_pending=False

    def _poll_live_prediction(self):
        inference = self.inference
        if inference is None:
            return
        payload = inference.take_latest_prediction()
        if payload is not None:
            self._live_prediction(payload)

    def _set_position_caption(self, text, state):
        color={'pressed':'#D97706','no_press':'#237A57','uncertain':'#64748B'}.get(state,'#64748B')
        self.live_position_label.setStyleSheet(f'font-size:17px;font-weight:700;color:{color};')
        self.live_position_label.setToolTip(text)
        self.live_position_label.setText(self.live_position_label.fontMetrics().elidedText(text.replace('\n',' · '),QtCore.Qt.ElideRight,max(30,self.live_position_label.width()-8)))

    def _live_prediction(self, payload):
        if getattr(self,'slot_pair',None) is not None:
            seq=payload.get('source_seq');last=getattr(self,'_last_dual_source',None)
            if seq is None or (last is not None and seq<=last):return
            if 'force' not in payload or len(payload['force'])!=3 or 'position_labels' not in payload:
                self.live_status_label.setText('双模型返回不完整，拒绝伪造零值');return
            self._last_dual_source=seq
        self._live_prediction_count += 1
        position = None if payload.get("position") is None else int(payload["position"])
        uncertain = payload.get("decision_status") == "uncertain" or position is None
        confidence = float(payload.get("confidence", 0.0))
        if "decision_status" not in payload and confidence < 0.65:
            uncertain = True
        force = tuple(float(value) for value in payload.get("force", (0.0, 0.0, 0.0)))
        self._refresh_acquired_force()
        actual_force = None
        raw_force = None
        if self._use_shared_sensor_reference:
            sample = self.force_sensor.measurement_snapshot()
            if sample:
                actual_force = tuple(sample['corrected'][:3])
                raw_force = tuple(sample['raw'][:3])
        elif self.force_provider is not None:
            try:
                candidate = self.force_provider()
            except Exception:
                candidate = None
            if candidate is not None:
                actual_force = tuple(float(value) for value in candidate)
        if uncertain:
            position = None
            position_text = "不确定／接触状态待确认"
        else:
            position_text = "无按压／空载（0号）" if position == 0 else f"预测位置  {position:02d} 号"
        expected_text = "" if self._live_expected_label is None else " · 期望 " + ("无按压／空载" if self._live_expected_label == 0 else f"{self._live_expected_label:02d} 号")
        contact_text = "" if payload.get("contact_probability") is None else f" · 接触概率 {float(payload['contact_probability']):.1%}"
        self.live_position_label.setText(f"{position_text}\n置信度 {confidence:.1%}" + contact_text + expected_text)
        timestamp = payload.get("timestamp_s")
        if timestamp is not None and np.isfinite(timestamp):
            if self._live_time_origin is None: self._live_time_origin = float(timestamp)
            elapsed = float(timestamp) - self._live_time_origin
        else:
            elapsed = np.nan  # Clock unavailable; never fabricate elapsed time from GUI index.
        if self._is_multilabel():
            position = list(payload.get("position_labels") or [])
            position_text = ("无按压／空载" if not position else "位置集合 " + ", ".join(map(str,position)))
            uncertain = payload.get('decision') == 'uncertain'
            if uncertain:position=None;position_text='不确定／接触状态待确认'
            self._set_position_caption(('按压 · '+position_text) if position else position_text,'uncertain' if uncertain else ('pressed' if position else 'no_press'))
        if not self._is_multilabel():self._set_position_caption(position_text+expected_text+f' · 置信度 {confidence:.1%}', 'uncertain' if uncertain else ('no_press' if position==0 else 'pressed'))
        self._live_samples.append(elapsed)
        self._live_predicted.append(force)
        self._live_actual_force.append(
            actual_force if actual_force is not None else (np.nan, np.nan, np.nan)
        )
        self._live_raw_actual_force.append(raw_force)
        self.performance_map.show_live_prediction(
            position, expected_label=self._live_expected_label
        )
        self._redraw_live_force_curves()
        self.live_status_label.setText(
            f"实时部署运行中 · 已推理{self._live_prediction_count}帧 · "
            + ("接触状态待确认 · " if uncertain else f"位置{position} · ")
            + f"置信度{confidence:.1%} · "
            f"预测力 Fx{force[0]:+.2f} / Fy{force[1]:+.2f} / Fz{force[2]:+.2f} N"
            + ("" if actual_force is not None else " · 无真实力参考")
        )

    def _redraw_live_force_curves(self):
        if not self._live_samples:
            return
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

    def start_training(self):
        if self.worker is not None:
            return False
        dataset_dir = Path(self.dataset_edit.text()).resolve()
        if not (dataset_dir / "frames.jsonl").is_file() and not any(dataset_dir.rglob("frames.jsonl")):
            self.status_label.setText("所选目录中没有frames.jsonl")
            return False
        self.worker = TouchTrainingWorker(
            dataset_dir, epochs=self.epochs_spin.value(), architecture=self.architecture_combo.currentData(),
            force_history=self.force_history_check.isChecked(), parent=self
        )
        self.worker.loading.connect(self._loading_progress)
        self.worker.progress.connect(self._training_progress)
        self.worker.completed.connect(self._training_completed)
        self.worker.failed.connect(self._training_failed)
        self.worker.finished.connect(self._training_finished)
        self.train_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.progress_bar.setValue(0)
        self.status_label.setText("读取已保存45点实测dBm；位置/力独立训练、固定split验证选模…")
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
            f"训练 {metrics.get('task', '')} {epoch}/{total} · "
            + (f"验证位置准确率 {float(metrics.get('validation_accuracy', metrics.get('position_accuracy', 0))):.2%}"
               if metrics.get('task') != 'force' else f"验证三轴力MAE {metrics.get('validation_force_mae_n', '暂无')}")
        )

    def _training_completed(self, report):
        self.progress_bar.setValue(100)
        self.last_training_report = dict(report)
        self._pending_training_report = dict(report)
        self.status_label.setText('训练完成；等待训练线程安全退出后自动验证选优')

    def _training_failed(self, message):
        self.status_label.setText(f"训练失败：{message}")

    def _training_finished(self):
        self.train_button.setEnabled(True)
        self.cancel_button.setEnabled(False)
        if self.worker is not None:
            self.worker.deleteLater()
        self.worker = None
        report=getattr(self,"_pending_training_report",None)
        self._pending_training_report=None
        if report is not None:self._auto_select_training(report)

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
        if self._is_multilabel():
            metrics = nested_metrics.get("test") or nested_metrics.get("validation") or {}
            self.input_shape_label.setText("45点9×5 (dBm) · 25位sigmoid多标签 · 独立位置/整体力骨干 · "+str(self.report.get("architecture")))
            self.overall_metrics_label.setText(f"{metric_label}集合精确匹配 {metrics.get('position_exact_match', '未提供')} · 三轴MAE {metrics.get('force_mae_n')} N · 最佳轮次 {self.report.get('best_epoch')} · 缺失训练位置 {self.report.get('missing_training_positions')}")
            self.validation_warning_label.setText(str(self.report.get("validation_warning", "")))
            self.status_label.setToolTip(self.overall_metrics_label.text()+"\n"+self.validation_warning_label.text())
            return
        self.input_shape_label.setText(model_description(self.report))
        if self.report.get("schema") == "fbg_touch26_v1":
            self.status_label.setText("最新26类模型已载入；可开始实时部署（训练按钮仍为旧25类CNN单数据集训练）")
            self.report.setdefault("validation_warning", "逐类连续分块开发测试，不是独立会话泛化；空载与按压来自不同会话，存在会话混淆。")
        per_point = list(self.report.get("per_point") or [])
        self.report_by_label = {int(item["label"]): item for item in per_point}
        accuracy = float(self.report.get("position_accuracy", 0.0))
        force_mae = list(self.report.get("force_mae_n") or [0.0, 0.0, 0.0])
        best_epoch = self.report.get('best_epoch', 0)
        best_text = (f"位置{best_epoch.get('position', 0)} / 力{best_epoch.get('force', 0)}"
                     if isinstance(best_epoch, dict) else str(best_epoch))
        self.overall_metrics_label.setText(
            f"{metric_label}位置准确率 {accuracy:.2%} · "
            f"Fx/Fy/Fz MAE = {force_mae[0]:.3f} / {force_mae[1]:.3f} / "
            f"{force_mae[2]:.3f} N · 三维向量MAE "
            f"{float(self.report.get('force_vector_mae_n', 0.0)):.3f} N · "
            f"训练/验证 {int(self.report.get('train_count', 0))}/"
            f"{int(self.report.get('validation_count', 0))}条 · "
            f"最佳轮次 {best_text}"
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
        predicted = tuple(float(v) for v in item["predicted_force_mean"])
        # Historical validation means are not live sensor readings.
        self._refresh_acquired_force()
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
        self._shutdown_slots()
        self.acquired_force_timer.stop()
        if self._owns_force_sensor:
            self.force_sensor.close()
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
