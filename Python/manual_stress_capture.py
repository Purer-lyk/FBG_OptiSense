"""Fixed 5×5 top-view manual press data collection fed by temporary test mode.

The 45-point source is the temporary-test page's live scan stream
(``TemporaryTestWindow.update_frame``), NOT the main spectrum page's native
USB ``frames_queue``. Every decoded temporary frame carries 45 records; this
collector defaults to saving all45 real post-settle second-code measurements,
their voltage/engineering-dBm values and Makima curves. The capture-only signed
wavelength heatmap retains its optional unloaded baseline;45 saving, training
and deployment do not depend on it. Explicit archived delta9 fixtures remain
isolated from the restored45 data roots.
"""

from __future__ import annotations

from pathlib import Path
import re
import threading
import time
import json
import os

import numpy as np
from qt_runtime import configure_qt
configure_qt()
from PyQt5 import QtCore, QtGui, QtWidgets
import pyqtgraph.opengl as gl

from finger_dataset import ContactEvent, FingerSessionWriter, SpectrumFrame
from mechanical_finger_3d import (
    BODY_BASE_RGBA,
    SILICONE_BASE_RGBA,
    VIEW_BACKGROUND_RGBA,
    _silicone_pad_mesh,
    load_stl_mesh,
    prepare_thumb_mesh,
    silicone_surface_height,
)
from paxini_px6d import PaxiniForceSensor
from temporary_peak_fit_makima import fit_temporary_frame
import temporary_optical_units as units
from signed_wavelength import SignedWavelengthState, PEAK_LABELS, signed_colors, spatial_field
from machine_profile import get_runtime_machine_id


GRID_ROWS = 5
GRID_COLUMNS = 5
GRID_POINT_COUNT = GRID_ROWS * GRID_COLUMNS
STRESS_POINT_COUNT = 45
LATEST_FRAME_MAX_AGE_MS = 2000.0

# Label, top-view mesh-normalized x/y. The 25 markers form a true 5×5
# rectangle over the same silicone sensing area visible in the original 3-D
# stress model's top view.
_FINGER_X_LEVELS = (0.30, 0.40, 0.50, 0.60, 0.70)
_FINGER_Y_LEVELS = (0.16, 0.255, 0.35, 0.445, 0.54)
FINGER_POINT_LAYOUT = tuple(
    (row * 5 + column + 1, x, y)
    for row, y in enumerate(_FINGER_Y_LEVELS)
    for column, x in enumerate(_FINGER_X_LEVELS)
)


# Display-only inset; never use these coordinates for persisted class metadata.
FINGER_DISPLAY_LAYOUT = tuple(
    (label, 0.5 + 0.8 * (x - 0.5), 0.35 + 0.8 * (y - 0.35))
    for label, x, y in FINGER_POINT_LAYOUT
)

FINGER_POINT_BY_LABEL = {
    label: (x, y) for label, x, y in FINGER_POINT_LAYOUT
}
FINGER_TOP_MESH_PATH = Path(__file__).resolve().parent / "assets" / "thumb_real.STL"


def _surface_region(y):
    if y < 0.18:
        return "指尖"
    if y < 0.47:
        return "上段指腹"
    if y < 0.64:
        return "中段指腹"
    if y < 0.79:
        return "下段指腹"
    return "指根"


class _ContextMeshItem(gl.GLMeshItem):
    """Keep pyqtgraph 0.14's mesh program out of its process-global cache.

    QOpenGLWidgets in different top-level windows do not necessarily share
    contexts (capture's central widget is reparented into the workspace).
    A program ID from the old mechanical page may be invalid here, or worse,
    collide with another valid but unrelated program. Clone the shader per
    context without changing pyqtgraph's registry or the legacy renderer.
    """

    def shader(self):
        from copy import copy

        context = QtGui.QOpenGLContext.currentContext()
        if getattr(self, "_shader_context", None) is not context:
            self._local_shader = copy(super().shader())
            self._local_shader.prog = None
            self._shader_context = context
        return self._local_shader


class _ContextScatterItem(gl.GLScatterPlotItem):
    """Isolate the scatter class's global program cache for this GL context."""

    def getShaderProgram(self):
        context = QtGui.QOpenGLContext.currentContext()
        if getattr(self, "_shader_context", None) is not context:
            previous = gl.GLScatterPlotItem._shaderProgram
            try:
                gl.GLScatterPlotItem._shaderProgram = None
                self._local_program = super().getShaderProgram()
                self._shader_context = context
            finally:
                gl.GLScatterPlotItem._shaderProgram = previous
        return self._local_program


class FingerSurfaceMap(QtWidgets.QWidget):
    """Original 3-D finger model with a clickable 5×5 surface annotation grid."""

    def __init__(self, *, on_selected=None, show_directions=False, parent=None):
        super().__init__(parent)
        self.on_selected = on_selected
        self.uses_real_3d_model = False
        self.setMinimumSize(520, 560)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding
        )

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self.view = gl.GLViewWidget(self)
        self.view.setBackgroundColor(VIEW_BACKGROUND_RGBA)
        self.view.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        layout.addWidget(self.view)

        vertices, faces = load_stl_mesh(FINGER_TOP_MESH_PATH)
        prepared, _normalized = prepare_thumb_mesh(vertices)
        lower = np.min(prepared, axis=0)
        upper = np.max(prepared, axis=0)
        span = upper - lower
        colors = np.tile(np.asarray(BODY_BASE_RGBA, dtype=float), (len(prepared), 1))
        self.body_item = _ContextMeshItem(
            vertexes=prepared,
            faces=faces,
            vertexColors=colors,
            smooth=True,
            drawEdges=False,
            shader="edgeHilight",
            glOptions="opaque",
        )
        self.view.addItem(self.body_item)

        silicone_vertices, silicone_faces, _silicone_norm = _silicone_pad_mesh(
            (lower[:2].copy(), upper[:2].copy()), float(span[2])
        )
        silicone_colors = np.tile(
            np.asarray(SILICONE_BASE_RGBA, dtype=float),
            (len(silicone_vertices), 1),
        )
        self.silicone_item = _ContextMeshItem(
            vertexes=silicone_vertices,
            faces=silicone_faces,
            vertexColors=silicone_colors,
            smooth=True,
            drawEdges=False,
            shader="edgeHilight",
            glOptions="translucent",
        )
        self.view.addItem(self.silicone_item)

        self.model_lower = lower
        self.model_upper = upper
        self.model_span = span
        self.silicone_vertices = silicone_vertices
        self.silicone_faces = silicone_faces
        self.marker_positions = np.asarray(
            [self._model_position(x, y) for _label, x, y in FINGER_DISPLAY_LAYOUT],
            dtype=float,
        )
        self.marker_item = _ContextScatterItem(
            pos=self.marker_positions,
            size=9.0,
            color=np.tile(np.asarray((0.05, 0.32, 0.62, 0.98)), (25, 1)),
            pxMode=True,
        )
        self.view.addItem(self.marker_item)


        self.text_items = []
        self.number_labels = []
        self.direction_labels = {}
        if show_directions:
            for text in ("指尖", "指腹", "指根"):
                tag = QtWidgets.QLabel(text, self)
                tag.setObjectName("sectionCaption")
                tag.setAlignment(QtCore.Qt.AlignCenter)
                tag.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
                self.direction_labels[text] = tag
        self.direction_vertices = np.vstack((prepared, silicone_vertices))
        # No number annotations: IDs remain in tooltips and selected-point text.

        center = (lower + upper) * 0.5
        self.view.opts["center"] = QtGui.QVector3D(
            float(center[0]), float(center[1]), float(center[2])
        )
        distance = max(float(np.max(span)) * 1.58, 72.0)
        self.view.setCameraPosition(elevation=90, azimuth=-90, distance=distance)
        self.uses_real_3d_model = True

        self.group = QtWidgets.QButtonGroup(self)
        self.group.setExclusive(True)
        self.buttons = []
        # Invisible full-surface hit targets keep point selection working while
        # the visible annotation stays minimal (small round markers only).
        for label, _x, _y in FINGER_POINT_LAYOUT:
            button = QtWidgets.QPushButton("", self)
            button.setCheckable(True)
            button.setCursor(QtCore.Qt.PointingHandCursor)
            button.setToolTip(f"手指表面位置编号 {label}")
            button.setStyleSheet(
                "QPushButton { background: transparent; border: none; }"
                "QPushButton:checked {"
                " background: rgba(255,159,67,70);"
                " border: 2px solid rgba(255,159,67,230);"
                " border-radius: 12px;"
                "}"
            )
            button.clicked.connect(
                lambda _checked=False, value=label: self._selected(value)
            )
            self.group.addButton(button, label)
            self.buttons.append(button)
        QtCore.QTimer.singleShot(0, self._position_buttons)


    def _model_position(self, x_fraction, y_fraction):
        x_norm = 2.0 * float(x_fraction) - 1.0
        y_norm = 1.0 - 2.0 * float(y_fraction)
        x = self.model_lower[0] + float(x_fraction) * self.model_span[0]
        y = self.model_upper[1] - float(y_fraction) * self.model_span[1]
        z = float(silicone_surface_height(x_norm, y_norm, self.model_span[2]))
        return (x, y, z + max(float(np.max(self.model_span)) * 0.010, 0.32))

    def set_signed_wavelengths(self, deltas, limit):
        """Recolor the existing tactile surface, retaining the actual STL body."""
        if not hasattr(self, 'sensor_item'):
            self.sensor_positions = self.marker_positions[np.array(PEAK_LABELS)-1].copy()
            self.sensor_item = _ContextScatterItem(pos=self.sensor_positions,
                size=13., color=signed_colors([None]*9), pxMode=True, glOptions='translucent')
            self.view.addItem(self.sensor_item)
        field = spatial_field(self.silicone_vertices[:,:2],
                              self.sensor_positions[:,:2], deltas)
        self.wavelength_vertex_colors = signed_colors(field, limit)
        # No lighting distortion of the signed color scale.
        self.silicone_item.setShader(None)
        self.silicone_item.setMeshData(vertexes=self.silicone_vertices,
            faces=self.silicone_faces, vertexColors=self.wavelength_vertex_colors)
        self.sensor_item.setData(pos=self.sensor_positions, color=signed_colors(deltas,limit))
        self.view.update()

    def _selected(self, label):
        if self.on_selected is not None:
            self.on_selected(int(label))

    def set_selected_points(self, labels):
        """Optional capture overlay; does not change training-map exclusivity or mesh."""
        labels = set(labels)
        exclusive = self.group.exclusive()
        self.group.setExclusive(False)
        colors = np.tile(np.asarray((0.05, 0.32, 0.62, 0.98)), (25, 1))
        sizes = np.full(25, 9.0)
        for label, button in enumerate(self.buttons, 1):
            button.setStyleSheet("QPushButton { background: transparent; border: none; }")
            button.setChecked(label in labels)
            if label in labels:
                colors[label - 1] = (1.0, 0.58, 0.08, 1.0)
                sizes[label - 1] = 12.0
        self.group.setExclusive(exclusive)
        self.marker_item.setData(color=colors, size=sizes)
        if not hasattr(self, "selection_item"):
            self.selection_item = _ContextScatterItem(size=12.0, pxMode=True, glOptions="translucent")
            self.selection_item.setDepthValue(20)
            self.view.addItem(self.selection_item)
        positions = self.marker_positions[np.asarray(sorted(labels), dtype=int) - 1].copy()
        positions[:, 2] += 0.2
        self.selection_item.setData(pos=positions, color=(1.0, 0.58, 0.08, 1.0))
        self.selected_marker_labels = labels
        self.view.update()

    def _project_to_widget(self, point):
        width = max(1, int(self.view.width()))
        height = max(1, int(self.view.height()))
        viewport = (0, 0, width, height)
        combined = (
            self.view.projectionMatrix(viewport, viewport) * self.view.viewMatrix()
        )
        clip = combined * QtGui.QVector4D(
            float(point[0]), float(point[1]), float(point[2]), 1.0
        )
        if abs(float(clip.w())) <= 1e-12:
            return None
        ndc_x = float(clip.x() / clip.w())
        ndc_y = float(clip.y() / clip.w())
        origin = self.view.mapTo(self, QtCore.QPoint(0, 0))
        return (
            origin.x() + (ndc_x + 1.0) * 0.5 * width,
            origin.y() + (1.0 - ndc_y) * 0.5 * height,
        )

    def _position_buttons(self):
        # Invisible hit targets: compact squares around each projected point.
        projected = [
            self._project_to_widget(point) for point in self.marker_positions
        ]
        if any(point is None for point in projected):
            for widget in self.buttons:
                widget.hide()
            return
        gaps = [
            projected[row * 5 + column + 1][0] - projected[row * 5 + column][0]
            for row in range(5)
            for column in range(4)
        ]
        vertical = [
            projected[(row + 1) * 5 + column][1] - projected[row * 5 + column][1]
            for row in range(4)
            for column in range(5)
        ]
        gap = max(22.0, min(gaps + vertical))
        marker_size = int(max(18, min(30, gap * 0.58)))
        for button, (x, y) in zip(self.buttons, projected):
            button.setGeometry(
                int(x - marker_size / 2),
                int(y - marker_size / 2),
                marker_size,
                marker_size,
            )
            button.show()
            button.raise_()
        self._position_directions()

    def _position_directions(self):
        """Read-only model projection; place tags in whitespace outside the mesh."""
        if not self.direction_labels:
            return
        points = np.asarray([self._project_to_widget(p) for p in self.direction_vertices])
        left, top = np.min(points, axis=0)
        right, bottom = np.max(points, axis=0)
        self.direction_model_rect = QtCore.QRectF(left, top, right-left, bottom-top)
        cx = (left + right) / 2
        tactile = [self._project_to_widget(p) for p in self.marker_positions]
        tactile_y = (min(p[1] for p in tactile) + max(p[1] for p in tactile)) / 2
        for text, tag in self.direction_labels.items():
            tag.adjustSize()
            w, h = tag.width(), tag.height()
            if text == "指尖":
                x, y = cx-w/2, top-h-10
            elif text == "指根":
                x, y = cx-w/2, bottom+10
            else:
                x, y = right+12, tactile_y-h/2
            tag.move(round(x), round(y))
            tag.raise_()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QtCore.QTimer.singleShot(0, self._position_buttons)

    def showEvent(self, event):
        super().showEvent(event)
        QtCore.QTimer.singleShot(0, self._position_buttons)



def extract_frame_vector(frame) -> list:
    """Return the 45-point second-sample CH1 vector of one temporary frame."""

    records = list(frame.get("records") or [])
    if len(records) != STRESS_POINT_COUNT:
        raise ValueError(f"临时帧不是45点，收到{len(records)}点")
    return [int(record["second_code"]) for record in records]


def _float_list(values):
    return [float(value) for value in values]


def build_training_views(frame, wavelengths) -> dict:
    """Build stable ADC, physical-unit and Makima training views.

    Only ``second_code`` is retained: it is the post-settle ADC reading used
    by the live peak fitter. dBm is an engineering estimate using the frame's
    feedback selector, not a traceable power-meter calibration.
    """

    x = [float(value) for value in wavelengths]
    if len(x) != STRESS_POINT_COUNT:
        raise ValueError(f"波长坐标不是45点，收到{len(x)}点")
    stable_adc = extract_frame_vector(frame)
    channel = int(frame.get("adc_channel", 1))
    selector = int(frame.get("feedback_selector", 0) or 0) if channel < 2 else 0
    stable_voltage = _float_list(units.adc_code_to_voltage(stable_adc))
    stable_dbm = _float_list(
        units.adc_code_to_dbm(stable_adc, selector, floor_nonpositive=True)
    )
    rows = [{"measured_nm": value} for value in x]
    fits = fit_temporary_frame(
        rows, stable_adc, warmed_up=int(frame.get("sequence", 0)) >= 16,
        feedback_selector=selector, adc_channel=channel
    )
    fit_records = []
    for peak_index, fit in enumerate(fits):
        start = peak_index * 5
        record = dict(fit)
        # Preserve the interpolated curve-global-dBm peak metadata.
        # Only the five real samples below are measured training inputs.
        record.update(
            peak_index=peak_index,
            input_wavelengths_nm=x[start:start + 5],
            input_adc=stable_adc[start:start + 5],
            input_voltage_v=stable_voltage[start:start + 5],
            input_dbm=stable_dbm[start:start + 5],
        )
        curve_adc = record.get("curve_adc")
        if curve_adc is not None:
            record["curve_voltage_v"] = _float_list(
                units.adc_code_to_voltage(curve_adc)
            )
            record["curve_dbm"] = _float_list(
                units.adc_code_to_dbm(
                    curve_adc, selector, floor_nonpositive=True
                )
            )
        fit_records.append(record)
    return {
        "wavelengths_nm": x,
        "stable_adc": stable_adc,
        "stable_voltage_v": stable_voltage,
        "stable_optical_power_dbm": stable_dbm,
        "conversion": {
            "adc_reference_v": float(units.ADC_REFERENCE_V),
            "adc_code_count": float(units.ADC_CODE_COUNT),
            "feedback_selector": selector,
            "transimpedance_ohm": float(units.transimpedance_ohm(selector)),
            "pd_responsivity_a_per_w": float(units.PD_RESPONSIVITY_A_PER_W),
            "dbm_is_engineering_estimate": True,
        },
        "makima_peak_fits": fit_records,
    }


class ForceVectorWidget(QtWidgets.QWidget):
    """Independent live-force display with colored vectors and value cards.

    This widget deliberately does not share the finger-model GL scene.  It
    combines the two visual elements requested by the operator: an intuitive
    RGB force-direction graphic and clearly readable Fx/Fy/Fz values.
    """

    AXIS_COLORS = {
        "Fx": (232, 72, 85),
        "Fy": (35, 177, 111),
        "Fz": (58, 124, 246),
    }
    AUTO_SCALE_N = 40.0

    def __init__(
        self,
        parent=None,
        *,
        title="实时力",
        show_connection_status=True,
        subtitle="空载并保持静止后执行归零",
        compact=False,
    ):
        super().__init__(parent)
        self.compact = compact
        self.title = str(title)
        self.show_connection_status = bool(show_connection_status)
        self.subtitle = str(subtitle)
        self.connected = False
        self.force_xyz = (0.0, 0.0, 0.0)
        self.moments = None
        self.setMinimumSize(200, 330) if compact else self.setMinimumSize(300, 330)
        self.setSizePolicy(
            QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Expanding
        )
        self.zero_button = QtWidgets.QPushButton("采集力归零", self)
        self.zero_button.setCursor(QtCore.Qt.PointingHandCursor)
        self.zero_button.setToolTip(
            "主机去皮：以当前新鲜六轴实测值为共享偏置；不发送设备标定命令。"
            "请先空载并保持静止。只影响实测力，不改变模型预测。"
        )
        self.zero_button.setStyleSheet(
            "QPushButton {"
            " background: #E8F0FF; color: #2F6FED;"
            " border: 1px solid #B9D1FF; border-radius: 7px;"
            " padding: 4px 10px; font-weight: 600;"
            "}"
            "QPushButton:hover { background: #DCE9FF; border-color: #8FB4FF; }"
            "QPushButton:pressed { background: #CFE0FF; }"
            "QPushButton:disabled { color: #9AA5B5; background: #EEF1F5;"
            " border-color: #DDE2EA; }"
        )
        self._position_zero_button()

    def _position_zero_button(self):
        if self.compact:
            self.zero_button.setGeometry(self.width()-122, self.height()-40, 110, 30)
        else:
            self.zero_button.setGeometry(max(176, self.width() - 124), 10, 110, 30)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._position_zero_button()

    def set_force(self, fx, fy, fz, *, connected=True, moments=None):
        if fx is None or fy is None or fz is None or not connected:
            self.connected = False
            self.force_xyz = (0.0, 0.0, 0.0)
            self.moments = None
        else:
            self.connected = True
            self.force_xyz = (float(fx), float(fy), float(fz))
            self.moments = None if moments is None else tuple(
                None if value is None else float(value) for value in moments
            )
        self.update()

    @staticmethod
    def _draw_arrow_head(painter, tip, direction, color, size=12.0):
        import math

        angle = math.atan2(direction.y(), direction.x())
        back = angle + math.pi
        spread = 0.48
        left = QtCore.QPointF(
            tip.x() + size * math.cos(back + spread),
            tip.y() + size * math.sin(back + spread),
        )
        right = QtCore.QPointF(
            tip.x() + size * math.cos(back - spread),
            tip.y() + size * math.sin(back - spread),
        )
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(color)
        painter.drawPolygon(QtGui.QPolygonF([tip, left, right]))

    def _compact_force_geometry(self):
        """Signed normalized projection; missing data has no active endpoints."""
        plot_height = min(150.0, self.height() - 242.0)
        origin = QtCore.QPointF(self.width() * .48, 180 + plot_height * .5)
        reach = min(self.width() * .27, plot_height * .34)
        scale = max((abs(v) for v in self.force_xyz), default=0.0)
        vectors = []
        for name, value, dx, dy in zip(('Fx','Fy','Fz'), self.force_xyz,
                                      (1.,-.72,0.), (.20,.52,-1.)):
            norm = (dx*dx + dy*dy)**.5
            direction = QtCore.QPointF(dx/norm, dy/norm)
            tip = origin + direction * (reach * value/scale) if self.connected and scale else None
            vectors.append((name, direction, tip))
        return origin, reach, vectors

    def _paint_compact(self):
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        painter.fillRect(self.rect(), QtGui.QColor(247,249,252))
        painter.setPen(QtGui.QColor(35,43,58))
        font = painter.font(); font.setPointSizeF(10); font.setBold(True); painter.setFont(font)
        painter.drawText(QtCore.QRect(12,10,self.width()-24,24), QtCore.Qt.AlignLeft, self.title)
        font.setPointSizeF(8); font.setBold(False); painter.setFont(font)
        painter.setPen(QtGui.QColor(126,135,149))
        painter.drawText(QtCore.QRect(12,35,self.width()-24,35), QtCore.Qt.TextWordWrap,
                         ("已连接 · 主机去皮" if self.connected else "未连接 · 缺失力存null")+"\n空载并静止后归零")
        for row, (name,value) in enumerate(zip(("Fx","Fy","Fz"),self.force_xyz)):
            y=70+row*36
            rect=QtCore.QRectF(10,y,self.width()-20,32)
            painter.setBrush(QtGui.QColor(255,255,255)); painter.setPen(QtGui.QColor(231,235,242)); painter.drawRoundedRect(rect,8,8)
            painter.setPen(QtGui.QColor(*self.AXIS_COLORS[name]))
            font.setPointSizeF(9);font.setBold(True);painter.setFont(font)
            painter.drawText(QtCore.QRectF(18,y,32,32),QtCore.Qt.AlignVCenter,name)
            font.setPointSizeF(13);painter.setFont(font);painter.setPen(QtGui.QColor(34,42,56))
            text=(f"{value:+.3f}" if self.connected else "—")+" N"
            painter.drawText(QtCore.QRectF(52,y,self.width()-70,32),QtCore.Qt.AlignVCenter|QtCore.Qt.AlignRight,text)
        # Reference axes remain visible even at zero/unavailable. Solid arrows
        # alone represent corrected force; scale is visual, never a force cap.
        origin, reach, vectors = self._compact_force_geometry()
        font.setPointSizeF(8); font.setBold(True); painter.setFont(font)
        for name, direction, tip in vectors:
            color = QtGui.QColor(*self.AXIS_COLORS[name])
            reference = origin + direction * reach
            painter.setPen(QtGui.QPen(color, 1.3, QtCore.Qt.DashLine))
            painter.drawLine(origin - direction * reach, reference)
            self._draw_arrow_head(painter, reference, direction, color, size=5)
            painter.setPen(color)
            painter.drawText(reference + direction * 8 + QtCore.QPointF(-8,4), '+' + name)
            if tip is not None and (tip-origin).manhattanLength() > 2:
                painter.setPen(QtGui.QPen(color, 3.5, QtCore.Qt.SolidLine, QtCore.Qt.RoundCap))
                painter.drawLine(origin, tip)
                self._draw_arrow_head(painter, tip, tip-origin, color, size=min(8,(tip-origin).manhattanLength()*.4))
        painter.setPen(QtCore.Qt.NoPen); painter.setBrush(QtGui.QColor(54,62,77)); painter.drawEllipse(origin,3,3)
        font.setPointSizeF(7);font.setBold(False);painter.setFont(font);painter.setPen(QtGui.QColor(126,135,149))
        painter.drawText(QtCore.QRect(8,self.height()-62,self.width()-16,20),QtCore.Qt.AlignCenter,"虚线坐标参照 · 实线力同比缩放")
        font.setPointSizeF(8);font.setBold(False);painter.setFont(font);painter.setPen(QtGui.QColor(126,135,149))
        painter.drawText(QtCore.QRect(12,self.height()-38,65,30),QtCore.Qt.AlignVCenter,"整体合力")
        painter.end()

    def paintEvent(self, event):
        if self.compact:
            self._paint_compact()
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing, True)
        width = float(self.width())
        height = float(self.height())
        # Soft, borderless panel integrated with the application's light UI.
        outer = QtCore.QRectF(1, 1, width - 2, height - 2)
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(247, 249, 252))
        painter.drawRoundedRect(outer, 14, 14)

        title_font = painter.font()
        title_font.setPointSizeF(11)
        title_font.setBold(True)
        painter.setFont(title_font)
        painter.setPen(QtGui.QColor(35, 43, 58))
        painter.drawText(QtCore.QPointF(18, 27), self.title)
        if self.show_connection_status:
            status = "已连接" if self.connected else "未连接"
            painter.setPen(
                QtGui.QColor(38, 166, 104)
                if self.connected
                else QtGui.QColor(143, 151, 164)
            )
            title_width = QtGui.QFontMetrics(title_font).horizontalAdvance(self.title)
            status_x = min(width - 62.0, 18.0 + title_width + 12.0)
            painter.drawText(QtCore.QPointF(status_x, 27), status)
        hint_font = painter.font()
        hint_font.setPointSizeF(8)
        hint_font.setBold(False)
        painter.setFont(hint_font)
        painter.setPen(QtGui.QColor(126, 135, 149))
        painter.drawText(QtCore.QPointF(18, 49), self.subtitle)

        # Top section: a clean pseudo-3D triad. Arrow length and direction
        # respond to the signed live force while retaining a readable stub.
        fx, fy, fz = self.force_xyz
        origin = QtCore.QPointF(width * 0.47, height * 0.34)
        reach = min(width * 0.29, height * 0.20)
        vectors = (
            ("Fx", fx, QtCore.QPointF(1.0, 0.20)),
            ("Fy", fy, QtCore.QPointF(-0.72, 0.52)),
            ("Fz", fz, QtCore.QPointF(0.0, -1.0)),
        )
        label_font = painter.font()
        label_font.setPointSizeF(9.5)
        label_font.setBold(True)
        painter.setFont(label_font)
        for name, value, base_dir in vectors:
            sign = 1.0 if value >= 0 else -1.0
            direction = QtCore.QPointF(base_dir.x() * sign, base_dir.y() * sign)
            norm = (direction.x() ** 2 + direction.y() ** 2) ** 0.5
            direction = QtCore.QPointF(direction.x() / norm, direction.y() / norm)
            ratio = min(1.0, abs(value) / self.AUTO_SCALE_N)
            length = reach * (0.36 + 0.64 * ratio)
            tip = QtCore.QPointF(
                origin.x() + direction.x() * length,
                origin.y() + direction.y() * length,
            )
            rgb = self.AXIS_COLORS[name]
            color = (
                QtGui.QColor(*rgb)
                if self.connected
                else QtGui.QColor(160, 167, 178)
            )
            painter.setPen(QtGui.QPen(color, 5.0, QtCore.Qt.SolidLine,
                                      QtCore.Qt.RoundCap))
            painter.drawLine(origin, tip)
            self._draw_arrow_head(painter, tip, direction, color, size=13.0)
            painter.setPen(color)
            label_pos = QtCore.QPointF(
                tip.x() + direction.x() * 9 - 8,
                tip.y() + direction.y() * 9 + 4,
            )
            painter.drawText(label_pos, name)

        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtGui.QColor(54, 62, 77))
        painter.drawEllipse(origin, 5, 5)

        # Bottom section: three independent numeric cards. The number is the
        # primary visual element; axis color, sign, decimals and N are explicit.
        card_left = 14.0
        card_width = width - 28.0
        card_height = max(48.0, min(58.0, (height * 0.43 - 16.0) / 3.0))
        start_y = height - (card_height * 3.0 + 22.0)
        for row, (name, value) in enumerate((("Fx", fx), ("Fy", fy), ("Fz", fz))):
            y = start_y + row * (card_height + 7.0)
            rect = QtCore.QRectF(card_left, y, card_width, card_height)
            painter.setBrush(QtGui.QColor(255, 255, 255))
            painter.setPen(QtGui.QPen(QtGui.QColor(231, 235, 242), 1.0))
            painter.drawRoundedRect(rect, 10, 10)
            rgb = self.AXIS_COLORS[name]
            color = QtGui.QColor(*rgb) if self.connected else QtGui.QColor(150, 158, 170)
            painter.setPen(QtCore.Qt.NoPen)
            painter.setBrush(color)
            painter.drawRoundedRect(QtCore.QRectF(card_left + 10, y + 11, 5, card_height - 22), 2, 2)

            axis_font = painter.font()
            axis_font.setPointSizeF(10)
            axis_font.setBold(True)
            painter.setFont(axis_font)
            painter.setPen(color)
            painter.drawText(QtCore.QRectF(card_left + 25, y, 42, card_height),
                             QtCore.Qt.AlignVCenter | QtCore.Qt.AlignLeft, name)

            value_font = painter.font()
            value_font.setPointSizeF(16)
            value_font.setBold(True)
            painter.setFont(value_font)
            state = "—" if not self.connected else f"{value:+.3f}"
            painter.setPen(QtGui.QColor(34, 42, 56))
            painter.drawText(QtCore.QRectF(card_left + 68, y, card_width - 112, card_height),
                             QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight, state)

            unit_font = painter.font()
            unit_font.setPointSizeF(9)
            unit_font.setBold(False)
            painter.setFont(unit_font)
            painter.setPen(QtGui.QColor(109, 118, 133))
            painter.drawText(QtCore.QRectF(card_left + card_width - 35, y, 25, card_height),
                             QtCore.Qt.AlignVCenter | QtCore.Qt.AlignRight, "N")


class ManualStressCaptureWindow(QtWidgets.QMainWindow):
    """Numbered 25-point grid that saves one 45-value vector per button press."""

    force_zero_finished = QtCore.pyqtSignal(bool, str)
    grid_shape = (GRID_ROWS, GRID_COLUMNS)

    def __init__(
        self,
        frame_source,
        *,
        wavelength_provider,
        output_root: Path,
        capture_contract="legacy45",
        force_sensor=None,
        parent=None,
    ):
        """``frame_source`` is the TemporaryTestWindow providing live frames."""

        super().__init__(parent)
        self.frame_source = frame_source
        self.wavelength_provider = wavelength_provider
        if capture_contract not in ("delta9", "legacy45"):
            raise ValueError("unsupported capture contract")
        self.capture_contract = capture_contract
        self.output_root = (Path(output_root).parent / "manual_touch_delta9"
                            if capture_contract == "delta9" and Path(output_root).name == "manual_touch_45"
                            else Path(output_root))
        # GUI-local only: never enabled at startup and never schedules offline work.
        self._empty_clock = time.monotonic
        self.timed_empty_interval_s = 300
        self.timed_empty_writer = None
        self.timed_empty_session_dir = None
        self.timed_empty_saved = 0
        self._empty_due = None
        self._empty_frame_valid = False
        self._empty_frame_context = None
        self._empty_context = None
        self._empty_last_sequence = None
        self._empty_last_result = "未开启"
        self.writer: FingerSessionWriter | None = None
        self.session_dir: Path | None = None
        self.selected_point: int | None = None
        self.annotation_mode = "single"
        self.selected_points = set()
        self.sample_index = 0
        self.sequence = 0
        self.capture_counts = {index: 0 for index in range(0, 26)}
        self.capture_history = []
        self.combo_counts = {}
        self.latest_frame = None
        self.latest_values = None
        self.latest_peak_fits = None
        self.wavelength_state = SignedWavelengthState()
        self._wavelength_display_key = None
        self.rejected_non45_frames = 0
        self.frames_seen = 0
        self.last_frame_monotonic = None
        # Paxini PX6D six-axis force sensor (RS485 over USB CDC, auto-push).
        # Reading is best-effort: when absent/busy, samples simply record a
        # null force plus the reason.
        self._owns_force_sensor = force_sensor is None
        self.force_sensor = force_sensor if force_sensor is not None else PaxiniForceSensor()
        self.force_sensor_enabled = True
        self.force_last_error: str | None = None
        self.force_notice = None
        self.force_notice_until = 0.0
        self.force_zero_finished.connect(self._force_zero_done)
        # Always idle until the operator explicitly connects, including standalone pages.
        self._force_sample_snapshot = None

        self.setWindowTitle("25点矩形点阵应力数据采集（临时测试模式数据源）")
        self._build_ui()

    # ------------------------------------------------------------------ link

    def attach(self):
        """Start receiving frames from the temporary test page."""

        listeners = getattr(self.frame_source, "frame_listeners", None)
        if listeners is None:
            raise TypeError("frame_source 必须是临时测试页（TemporaryTestWindow）")
        if self._on_temporary_frame not in listeners:
            listeners.append(self._on_temporary_frame)

    def detach(self):
        listeners = getattr(self.frame_source, "frame_listeners", None)
        if listeners is not None and self._on_temporary_frame in listeners:
            listeners.remove(self._on_temporary_frame)

    def _on_temporary_frame(self, frame):
        """Called on the GUI thread for every decoded temporary-mode frame."""

        self.frames_seen += 1
        self._empty_frame_valid = False
        try:
            values = extract_frame_vector(frame)
        except (ValueError, KeyError, TypeError, OverflowError):
            self.rejected_non45_frames += 1
            self.latest_peak_fits = None
            # The latest invalid frame must not reuse a recent prior peak snapshot.
            self.wavelength_state.update([], self._wavelength_context(),
                                         frame.get('sequence'), time.monotonic())
            return
        self._empty_frame_valid = True
        self._empty_frame_context = self._wavelength_context()
        self.latest_frame = frame
        self.latest_values = values
        self.last_frame_monotonic = time.monotonic()
        # Consume the source's same-frame Makima computation, never refit here.
        snapshot = getattr(self.frame_source, 'makima_frame_snapshot', None)
        self.latest_peak_fits = snapshot[1] if snapshot and snapshot[0] is frame else None
        context = self._wavelength_context()
        frame_channel = int(frame.get('adc_channel',1))
        frame_selector = 0 if frame_channel >= 2 else int(frame.get('feedback_selector',0))
        fits = self.latest_peak_fits or []
        if (frame_channel,frame_selector) != (context[2],context[3]):
            fits = []  # A queued old-channel frame is not a reference for new UI context.
        self.wavelength_state.update(fits, context,
                                     frame.get('sequence'), self.last_frame_monotonic)

    # ------------------------------------------------------------------- UI

    def _build_ui(self):
        from responsive_layout import ScrollContentWidget
        central = ScrollContentWidget(self)
        central.setObjectName("manualStressGridPage")
        self.setCentralWidget(central)
        outer = QtWidgets.QVBoxLayout(central)
        outer.setContentsMargins(4, 3, 4, 3)
        outer.setSpacing(1)

        title = QtWidgets.QLabel("手动应力采集 · 手指表面25点位置图")
        title.setObjectName("pageTitle")
        title.hide()
        note = QtWidgets.QLabel(
            "数据来自“临时测试模式”的实时45点扫描。操作：先在临时测试页开始"
            "连续采集 → 回到本页选择点阵编号 → 手动按住该位置 → 点击"
            "“采集当前45点”。每次点击只保留稳定后的第二次ADC值及其"
            "电压、dBm估算值，并保存9峰Makima五点拟合"
            "中心和完整曲线，编号作为按压位置标签。"
        )
        note.setWordWrap(True)
        note.setObjectName("softHint")
        note.hide()

        controls = QtWidgets.QHBoxLayout()
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("单点标注", "single")
        self.mode_combo.addItem("多点同时按压标注", "multi")
        self.mode_combo.setToolTip("只标注操作员确认的同时接触位置，不自动判断接触；切换前请保存并关闭数据集。多点数据不用于现有单标签训练。")
        self.mode_combo.currentIndexChanged.connect(self._change_annotation_mode)
        controls.addWidget(self.mode_combo)
        controls.addWidget(QtWidgets.QLabel("数据集："))
        self.dataset_name_edit = QtWidgets.QLineEdit(
            "press_grid_" + time.strftime("%Y%m%d_%H%M%S")
        )
        self.dataset_name_edit.setMinimumWidth(130)
        self.dataset_name_edit.setMaximumWidth(180)
        controls.addWidget(self.dataset_name_edit)
        self.capture_button = QtWidgets.QPushButton("采集9峰Δλ" if self.capture_contract == "delta9" else "采集当前45点")
        self.capture_button.setProperty("role", "primary")
        self.capture_button.setToolTip("保存45个真实second_code及其电压/dBm、Makima曲线；训练输入为45个实测dBm。采集不依赖波长基准；热图基准仅用于显示。" if self.capture_contract == "legacy45" else "保存9峰Δλ，需要显式空载基准；独立旧契约。")
        self.capture_button.clicked.connect(self.capture_current_press)
        controls.addWidget(self.capture_button)
        self.no_press_button = QtWidgets.QPushButton("空载（0）")
        self.no_press_button.clicked.connect(self.select_no_press)
        controls.addWidget(self.no_press_button)
        self.undo_button = QtWidgets.QPushButton("撤销上次")
        self.undo_button.setEnabled(False)
        self.undo_button.setToolTip("删除本数据集中刚刚采集的最后一条标注数据")
        self.undo_button.clicked.connect(self.undo_last_annotation)
        controls.addWidget(self.undo_button)
        self.close_session_button = QtWidgets.QPushButton("保存并关闭数据集")
        self.close_session_button.clicked.connect(self.close_session)
        controls.addWidget(self.close_session_button)
        controls.addStretch(1)
        outer.addLayout(controls)
        empty_row = QtWidgets.QHBoxLayout()
        self.timed_empty_button = QtWidgets.QPushButton("开启定时空载")
        self.timed_empty_button.setCheckable(True)
        self.timed_empty_button.setToolTip(
            "开启前请保持无接触；按压/手动采集时请暂停，系统不自动判空载。"
            "每5分钟保存1帧，首组在开启5分钟后；页面保持运行即可，不必一直前台。"
            "停止/退出后不离线补采，休眠或延误不追赶；不重设光谱基准。"
            "每次采集前自动进行帕西尼主机力归零（影响共享实际力参考）；无新鲜力或归零失败则跳过。"
            "定时数据独立保存，不影响手动数据集/撤销。")
        self.timed_empty_button.clicked.connect(self._toggle_timed_empty)
        empty_row.addWidget(self.timed_empty_button)
        self.timed_empty_status = QtWidgets.QLabel("每5分钟1组 / 未开启 / 已保存0")
        self.timed_empty_status.setMinimumWidth(0)
        self.timed_empty_status.setObjectName("softHint")
        empty_row.addWidget(self.timed_empty_status, 1)
        outer.addLayout(empty_row)
        self.timed_empty_timer = QtCore.QTimer(self)
        self.timed_empty_timer.setInterval(250)
        self.timed_empty_timer.timeout.connect(self._tick_timed_empty)

        surface_card = QtWidgets.QFrame()
        surface_card.setObjectName("contentCard")
        surface_outer = QtWidgets.QVBoxLayout(surface_card)
        surface_outer.setContentsMargins(4, 4, 4, 4)
        surface_outer.setSpacing(2)
        surface_outer.setSizeConstraint(QtWidgets.QLayout.SetMinimumSize)
        wavelength_controls = QtWidgets.QHBoxLayout()
        self.wavelength_baseline_button = QtWidgets.QPushButton('设置空载波长基准')
        self.wavelength_baseline_button.setToolTip('请先解除按压；显式取最近同帧9峰有效Makima波长。不等同于力归零。')
        self.wavelength_baseline_button.clicked.connect(self._set_wavelength_baseline)
        wavelength_controls.addWidget(self.wavelength_baseline_button)
        wavelength_controls.addWidget(QtWidgets.QLabel('固定对称色标 ±'))
        self.wavelength_range = QtWidgets.QDoubleSpinBox()
        self.wavelength_range.setRange(.001,1.)
        self.wavelength_range.setDecimals(3)
        self.wavelength_range.setSingleStep(.01)
        self.wavelength_range.setValue(.1)
        self.wavelength_range.setSuffix(' nm')
        wavelength_controls.addWidget(self.wavelength_range)
        self.wavelength_colorbar = QtWidgets.QLabel(' 蓝 −       0 中性       + 红 ')
        self.wavelength_colorbar.setStyleSheet('background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #144ddb,stop:0.5 #e0e3e6,stop:1 #db1f29);color:#18202a;padding:5px;')
        wavelength_controls.addWidget(self.wavelength_colorbar)
        wavelength_controls.addStretch(1)
        surface_outer.addLayout(wavelength_controls)
        self.wavelength_status = QtWidgets.QLabel('空载基准待设置；缺失值灰色')
        self.wavelength_status.setWordWrap(True)
        surface_outer.addWidget(self.wavelength_status)
        self.wavelength_context_label = QtWidgets.QLabel()
        self.wavelength_context_label.setWordWrap(True)
        self.wavelength_context_label.setMinimumWidth(0)
        self.wavelength_context_label.setObjectName("softHint")
        self.wavelength_context_label.hide()

        # The requested order is force display on the LEFT and finger model on
        # the RIGHT. They are layout siblings, so the force graphic can never
        # be painted over the model or its 25 location markers.
        body_row = QtWidgets.QHBoxLayout()
        self.force_model_layout = body_row
        body_row.setSpacing(8)
        self.force_vector = ForceVectorWidget(parent=surface_card, compact=True)
        self.force_vector.setMinimumWidth(200)
        self.force_vector.setMaximumHeight(390)
        self.zero_button = self.force_vector.zero_button
        self.zero_button.clicked.connect(self._request_force_zero)
        body_row.addWidget(self.force_vector, 2, QtCore.Qt.AlignTop)

        self.surface_map = FingerSurfaceMap(
            on_selected=self.select_grid_point,
            show_directions=True,
            parent=surface_card,
        )
        self.surface_map.setMinimumSize(400, 330)
        self.grid_group = self.surface_map.group
        self.grid_buttons = self.surface_map.buttons
        body_row.addWidget(self.surface_map, 3)
        surface_outer.addLayout(body_row, 1)
        self.wavelength_table = QtWidgets.QTableWidget(9,4)
        self.wavelength_table.setMinimumWidth(350)
        self.wavelength_table.setHorizontalHeaderLabels(['传感器→点号','当前λ / nm','空载λ₀ / nm','Δλ / nm'])
        self.wavelength_table.verticalHeader().hide()
        self.wavelength_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.wavelength_table.verticalHeader().setDefaultSectionSize(22)
        self.wavelength_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.wavelength_table.setMaximumHeight(228)
        self.wavelength_table.setMinimumHeight(180)
        # A compact table beneath the existing force cards; never overlaid on them.
        self.statistics_column = QtWidgets.QWidget(surface_card)
        self.statistics_layout = QtWidgets.QVBoxLayout(self.statistics_column)
        self.statistics_layout.setContentsMargins(0, 0, 0, 0)
        self.statistics_layout.setSpacing(3)
        self.statistics_layout.addWidget(self.wavelength_table)
        body_row.addWidget(self.statistics_column, 2)
        self.wavelength_explanation = explanation = QtWidgets.QLabel('Δλ=当前Makima峰位置−空载基准；表面色为9传感器空间可视化，非25点实测、非力/分类预测。超界仅颜色饱和，读数保留。')
        explanation.setToolTip(explanation.text())
        explanation.setText("Δλ = 当前9峰波长 − 显式空载基准；仅用于热图显示，45点保存/训练不依赖此基准。")
        explanation.setWordWrap(True)
        explanation.setMinimumWidth(0)
        explanation.setSizePolicy(QtWidgets.QSizePolicy.Ignored, QtWidgets.QSizePolicy.Minimum)
        explanation.setObjectName("softHint")
        surface_outer.addWidget(explanation)
        outer.addWidget(surface_card, 1)

        self.selected_label = QtWidgets.QLabel("当前标签：无按压／空载（0号）")
        self.selected_label.setWordWrap(True)
        self.selected_label.setMinimumWidth(0)
        self.selected_label.setObjectName("statusPill")
        self.selected_label.setProperty("statusKind", "warning")
        outer.addWidget(self.selected_label)

        self.capture_status_label = QtWidgets.QLabel("先解除按压设置空载波长基准，再选择位置采集9峰Δλ。" if self.capture_contract == "delta9" else "等待完整45点临时模式扫描；不选位置可采空载，按压时请选择1～25号")
        self.capture_status_label.setWordWrap(True)
        self.capture_status_label.setObjectName("softHint")
        outer.addWidget(self.capture_status_label)

        self.counts_label = QtWidgets.QLabel("已采集：0条")
        self.counts_label.setWordWrap(True)
        self.counts_label.setObjectName("softHint")
        self.counts_label.hide()
        # Scrollable session ledger below the existing STL/Δλ card: no space
        # taken from its horizontal layout, and no separate popup window.
        self.combo_counts_table = QtWidgets.QTableWidget(0, 2)
        self.combo_counts_table.setHorizontalHeaderLabels(['按压组合', '已采组数'])
        self.combo_counts_table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.combo_counts_table.verticalHeader().hide()
        self.combo_counts_table.setEditTriggers(QtWidgets.QAbstractItemView.NoEditTriggers)
        self.combo_counts_table.setFixedHeight(72)
        self.combo_counts_table.verticalHeader().setDefaultSectionSize(22)
        self.combo_counts_table.setToolTip('一组 = 成功保存的一帧45点样本，减去已撤销帧；不是独立按压轮次。空载单独统计。')
        self.combo_counts_table.hide()
        self.statistics_layout.addWidget(self.combo_counts_table)
        self.statistics_layout.addStretch(1)

        self.link_label = QtWidgets.QLabel()
        self.link_label.setWordWrap(True)
        self.link_label.setObjectName("softHint")
        self.link_label.hide()

        self.force_connect_button = QtWidgets.QPushButton("连接PX6D")
        self.force_connect_button.clicked.connect(self._connect_force_sensor)
        force_controls = QtWidgets.QHBoxLayout()
        force_controls.addWidget(self.force_connect_button)
        self.force_disconnect_button = QtWidgets.QPushButton("断开PX6D")
        self.force_disconnect_button.clicked.connect(self.force_sensor.close)
        force_controls.addWidget(self.force_disconnect_button)
        outer.insertLayout(1, force_controls)
        self.force_label = QtWidgets.QLabel("帕西尼力传感器：未连接；缺失力保存null")
        self.force_label.setWordWrap(True)
        self.force_label.setObjectName("softHint")
        force_controls.addWidget(self.force_label, 1)
        self.force_rate_label = QtWidgets.QLabel('有效接收：— Hz')
        self.force_rate_label.setObjectName('softHint')
        self.force_rate_label.setToolTip(
            'PX6D主机有效接收率：最近2秒内校验正确且成功解析的完整六轴帧，'
            '(帧数−1)/(末帧−首帧接收时间)，最多4096帧。'
            '不是串口波特率或设备ADC原生采样率；不代表严格硬件采样时刻。'
            '协议无序号，相同数值的独立接收帧仍各计一次；归零不重置。'
            '不足两帧估计中，超过500ms无新鲜数据显示—。')
        force_controls.addWidget(self.force_rate_label)
        # Keep the selected contact set visible near the mode controls even
        # when the retained full STL/force/table card needs vertical scrolling.
        outer.removeWidget(self.selected_label)
        outer.insertWidget(2, self.selected_label)
        central.setStyleSheet("QPushButton {padding:4px 8px; min-height:20px;} QComboBox {padding:3px 6px; min-height:20px;} QLineEdit {padding:3px 6px; min-height:20px;}")


        self.link_timer = QtCore.QTimer(self)
        self.link_timer.setInterval(500)
        self.link_timer.timeout.connect(self._update_link_label)
        self.link_timer.start()

        # Force arrows animate on their own ~20 fps timer, independent of the
        # 0.5 s status refresh.
        self.force_timer = QtCore.QTimer(self)
        self.force_timer.setInterval(50)
        self.force_timer.timeout.connect(self._refresh_force_arrows)
        self.force_timer.start()
        self.wavelength_timer = QtCore.QTimer(self)
        self.wavelength_timer.setInterval(100)
        self.wavelength_timer.timeout.connect(self._refresh_wavelength_display)
        self.wavelength_timer.start()
        self._refresh_wavelength_display(force=True)
        self._refresh_single_count_hint()

    def _toggle_timed_empty(self, checked):
        if not checked:
            self._stop_timed_empty()
            return
        if self.capture_contract != "legacy45" or self.selected_points or self.selected_point:
            self.timed_empty_button.setChecked(False)
            self._empty_last_result = "当前有按压标签或非45点契约，请先选择空载"
            self._refresh_timed_empty_status()
            return
        answer = QtWidgets.QMessageBox.question(self, "确认持续空载",
            "开启前请保持无接触；按压/手动采集时请暂停，系统不自动判空载。\n"
            "当前标签须为空载，每5分钟保存1帧到独立数据集，首组在5分钟后。\n"
            "每次采集前自动进行帕西尼主机力归零（影响共享实际力参考），请始终保持空载。\n"
            "无新鲜力或归零失败则跳过；不发送设备标定命令、不重设光谱基准。是否开启？",
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No, QtWidgets.QMessageBox.No)
        if answer != QtWidgets.QMessageBox.Yes:
            self.timed_empty_button.setChecked(False)
            return
        import uuid
        path = self.output_root / "manual_touch_multi45" / (
            "empty_interval_" + time.strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8])
        metadata = dict(dataset_kind="manual_touch_multi45", training_eligible=True,
            annotation_mode="multi", position_label_schema="multi_hot25_v1",
            input_contract="45_measured_second_code_9x5_with_operator_multi_hot25_v1",
            annotation_source="timer_empty_operator_confirmed", operator_confirmed=True,
            interval_s=300, first_sample_delay_s=300, offline_catchup=False,
            label_semantics="position_labels=operator_confirmed_simultaneous_contacts; empty=no_press",
            force_semantics="shared_PX6D_tare_corrected_whole_contact_resultant_not_per_position")
        try:
            self._empty_context = self._wavelength_context()
            metadata.update(channel=self._empty_context[2], feedback_selector=self._empty_context[3],
                route=list(self._empty_context[4]), acquisition_generation=self._empty_context[6])
            self.timed_empty_writer = FingerSessionWriter(path, queue_capacity=512, manifest_extra=metadata)
            self.timed_empty_writer.write_event(ContactEvent(kind="session_start", metadata=metadata))
            self.timed_empty_writer.flush()
        except Exception as exc:
            self._stop_timed_empty("开启失败：" + str(exc))
            return
        self.timed_empty_session_dir = path
        self.timed_empty_saved = 0
        self._empty_last_sequence = None
        self._empty_due = self._empty_clock() + self.timed_empty_interval_s
        self._empty_last_result = "等待首组"
        self.timed_empty_button.setText("停止定时空载")
        self.timed_empty_timer.start()
        self._refresh_timed_empty_status()

    def _refresh_timed_empty_status(self):
        import math
        remaining = max(0, math.ceil(self._empty_due - self._empty_clock())) if self._empty_due is not None else None
        countdown = f"距下次{remaining//60:02d}:{remaining%60:02d}" if remaining is not None else "已停止"
        self.timed_empty_status.setText(
            f"每5分钟1组 / {countdown} / 已保存{self.timed_empty_saved} / {self._empty_last_result}")
        self.timed_empty_status.setToolTip(
            "定时空载数据：" + str(self.timed_empty_session_dir or "尚未创建") + "\n"
            + self._empty_last_result + "\n" + self.timed_empty_button.toolTip())

    def _stop_timed_empty(self, result="已停止并保存"):
        self.timed_empty_timer.stop()
        self._empty_due = None
        self.timed_empty_button.setChecked(False)
        self.timed_empty_button.setText("开启定时空载")
        writer, self.timed_empty_writer = self.timed_empty_writer, None
        if writer is not None:
            try:
                writer.write_event(ContactEvent(kind="session_end", metadata={"total_samples": self.timed_empty_saved}))
            except Exception as exc:
                result += "；结束审计失败：" + str(exc)
            finally:
                try:
                    writer.close()
                except Exception as exc:
                    result += "；关闭失败：" + str(exc)
        self._empty_last_result = result
        self._refresh_timed_empty_status()

    def _persist_timed_empty_counts(self):
        """Small derived snapshot; frame count advances only after production flush."""
        path = self.timed_empty_session_dir / "combination_counts.json"
        payload = dict(schema="manual_touch_multi45_combination_counts_v1",
            session=self.timed_empty_session_dir.name, dataset_name=self.timed_empty_session_dir.name,
            dataset_kind="manual_touch_multi45", counting_semantics="successful45pointframes_minusundone",
            sample_unit="one_successfully_saved_45_point_frame_not_press_round",
            active_samples_count=self.timed_empty_saved,
            combinations=[dict(position_labels=[], count=self.timed_empty_saved)] if self.timed_empty_saved else [])
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

    def _timed_empty_skip_reason(self):
        """Software guards only; no force/NN threshold can certify physical empty."""
        import math
        if self.selected_points or self.selected_point:
            return "当前有按压标签"
        if bool(getattr(self.frame_source, "peak_trend_stopped", False)):
            return "数据源已停止"
        if not self._empty_frame_valid or self.latest_frame is None or self._frame_age_ms() > LATEST_FRAME_MAX_AGE_MS:
            return "无新鲜完整45点帧"
        context = self._wavelength_context()
        if context != self._empty_context or context != self._empty_frame_context:
            return "数据源上下文已改变，请停止后重新开启"
        frame = self.latest_frame
        try:
            channel = int(frame["adc_channel"])
            selector = 0 if channel >= 2 else int(frame["feedback_selector"])
            if (channel, selector) != (context[2], context[3]):
                return "帧通道/增益与当前上下文不符"
            if frame.get("kind") != "data" or frame.get("i2c_errors", 0) or frame.get("spi_errors", 0):
                return "帧通信状态异常"
            sequence = frame["sequence"]
            if type(sequence) is not int or sequence < 0:
                return "帧序号无效"
            if self._empty_last_sequence is not None and sequence <= self._empty_last_sequence:
                return "重复或回退的帧序号"
            codes = [row["second_code"] for row in frame["records"]]
            if len(codes) != 45 or any(isinstance(v, bool) or not math.isfinite(v) or int(v) != v for v in codes):
                return "非完整有限45点帧"
            waves = list(self.wavelength_provider() or ())
            route = context[4]
            if len(waves) != 45 or not all(math.isfinite(w) for w in waves):
                return "45点波长表无效"
            if len(route) != 45 or [row[1] for row in route] != waves or len({row[0] for row in route}) != 45:
                return "45点路由与波长表不符"
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            return "帧/上下文无效：" + str(exc)
        return None

    def _tick_timed_empty(self):
        if self._empty_due is None:
            return
        now = self._empty_clock()
        if now >= self._empty_due:
            # One attempt only; never backfill missed intervals after GUI delay/sleep.
            self._empty_due = now + self.timed_empty_interval_s
            try:
                reason = self._timed_empty_skip_reason()
            except Exception as exc:
                # Keep an unexpected source/provider error visible; never escape a Qt timer slot.
                reason = "读取数据源失败：" + (str(exc) or type(exc).__name__)
            if reason:
                self._empty_last_result = reason + "，已跳过"
            else:
                # All optical/label/context gates precede the shared host-only tare.
                # No serial start, hardware calibration, waiting or repeated GUI-tick tare.
                if not self._request_force_zero(confirm=False):
                    self._empty_last_result = "帕西尼力归零失败/无新鲜力数据，已跳过"
                    self._refresh_timed_empty_status()
                    return
                try:
                    index = self.timed_empty_saved + 1
                    record, event = self._build_45_capture(list(self.wavelength_provider() or ()), [], "multi", index, index, self._frame_age_ms())
                    if record.quality.get("paxini_force") is None:
                        self._empty_last_result = "归零后无新鲜力数据，已跳过（共享去皮已生效）"
                        self._refresh_timed_empty_status()
                        return
                    record.quality.update(annotation_source="timer_empty_operator_confirmed", interval_s=300)
                    event.metadata.update(annotation_source="timer_empty_operator_confirmed")
                    # Rejecting a frame enqueue must not leave a false press_sample event.
                    self.timed_empty_writer.write_frame(record)
                    self.timed_empty_writer.write_event(event)
                    self.timed_empty_writer.flush()
                    self.timed_empty_saved += 1
                    self._empty_last_sequence = int(self.latest_frame["sequence"])
                    try:
                        self._persist_timed_empty_counts()
                    except Exception as exc:
                        self._stop_timed_empty("样本已保存，统计保存失败，已暂停：" + str(exc))
                        return
                    self._empty_last_result = "归零后保存成功"
                except Exception as exc:
                    self._stop_timed_empty("写入失败，已暂停；共享力归零已生效，无法回滚：" + str(exc))
                    return
        self._refresh_timed_empty_status()

    def _wavelength_context(self):
        source = self.frame_source
        plan = getattr(source, 'plan', None) or {}
        channel_fn = getattr(source, 'selected_channel', None)
        channel = channel_fn() if callable(channel_fn) else (self.latest_frame or {}).get('adc_channel',1)
        selector_fn = getattr(source, 'effective_feedback_selector', None)
        selector = selector_fn(channel) if callable(selector_fn) else (self.latest_frame or {}).get('feedback_selector',0)
        transport = getattr(source, 'wavelength_transport_context', None)
        transport = transport() if callable(transport) else transport
        method = getattr(source, 'peak_fit_method', None)
        method = method.currentData() if method is not None else 'makima'
        return (get_runtime_machine_id(), id(source), channel, selector,
            tuple((r.get('index'),r.get('measured_nm'),r.get('peak_id')) for r in plan.get('rows',[])),
            tuple(PEAK_LABELS), getattr(source,'wavelength_session_generation',0), transport, method)

    def _set_wavelength_baseline(self, _checked=False):
        self.wavelength_state.reset_context(self._wavelength_context())
        stopped = bool(getattr(self.frame_source,'peak_trend_stopped',False))
        frame = self.latest_frame or {}
        transport_error = frame.get('i2c_errors',0) or frame.get('spi_errors',0)
        if stopped or transport_error or not self.wavelength_state.set_baseline(time.monotonic()):
            self.wavelength_status.setText('空载基准未设置：需正在采集、最近2秒内同帧完整9峰有限波长且无通信错误；请解除按压后重新采集。采集不会停止。')
            return
        import uuid
        self.wavelength_baseline_id = uuid.uuid4().hex
        self._refresh_wavelength_display(force=True)

    def _refresh_wavelength_display(self, *, force=False):
        table = self.wavelength_table
        table.setFixedHeight(sum(table.rowHeight(i) for i in range(9))
            + table.horizontalHeader().height() + 2*table.frameWidth() + 4)
        state = self.wavelength_state
        state.reset_context(self._wavelength_context())
        stale = state.stale(time.monotonic()) or bool(getattr(self.frame_source,'peak_trend_stopped',False))
        key = (state.context,state.sequence,state.baseline_time,tuple(state.deltas),self.wavelength_range.value(),stale)
        if not force and (key == self._wavelength_display_key or not self.surface_map.isVisible()):
            return
        self._wavelength_display_key = key
        delta = state.deltas
        self.surface_map.set_signed_wavelengths(delta,self.wavelength_range.value())
        baseline = state.baseline or [None]*9
        for i,label in enumerate(PEAK_LABELS):
            values = [f'P{i+1} → {label}',
                f'{state.current[i]:.4f}' if state.current[i] is not None else '—',
                f'{baseline[i]:.4f}' if baseline[i] is not None else '待基准',
                f'{delta[i]:+.4f}' if delta[i] is not None else '—']

            for col,text in enumerate(values):
                self.wavelength_table.setItem(i,col,QtWidgets.QTableWidgetItem(text))
        ctx = state.context
        status = f'空载基准 {state.baseline_time}' if state.baseline is not None else '空载基准待设置（请解除按压后点击）'
        self.wavelength_status.setText(status + (' · 已停采/过期，冻结显示' if stale else ' · 实时'))
        self.wavelength_context_label.setText(f'当前数据：{ctx[0]} · CH{ctx[2]} · 档位{ctx[3]} · {len(ctx[4])}点扫描；9传感器对应点号见右表')
        self.wavelength_context_label.setToolTip(f'完整上下文：{ctx!r}；P1…P9→25,15,5,3,13,23,21,11,1')
        self.wavelength_status.setToolTip(f'完整上下文：{ctx!r}')

    def _connect_force_sensor(self):
        try:
            self.force_sensor.start()
        except Exception as exc:
            self.force_last_error = str(exc)
        self._update_force_label()

    def _refresh_force_arrows(self):
        sensor = self.force_sensor
        sample = sensor.measurement_snapshot() if isinstance(sensor, PaxiniForceSensor) and self.force_sensor_enabled else None
        values = sample['corrected'] if sample else None
        rate = sample.get('sample_rate_hz') if sample else None
        self.force_rate_label.setText(
            f'有效接收：{rate:.1f} Hz' if rate is not None else
            '有效接收：估计中' if sample else '有效接收：— Hz（无新鲜数据）')
        self.force_vector.set_force(*(values[:3] if values else (None, None, None)),
                                    connected=values is not None,
                                    moments=values[3:] if values else None)

    def _frame_age_ms(self):
        if self.last_frame_monotonic is None:
            return float("inf")
        return (time.monotonic() - self.last_frame_monotonic) * 1000.0

    def _update_link_label(self):
        """Surface why no temporary 45-point frame is arriving, with the fix."""

        if self.frames_seen == 0:
            text = (
                "数据链路：尚未收到临时测试模式帧。请切到“临时测试模式”页，"
                "完成选点后点击“开始”连续采集（45点路线），再回本页采集。"
            )
            kind = "warning"
        elif self.latest_values is None:
            text = (
                f"数据链路：已收到{self.frames_seen}帧，但都不是45点；"
                "请确认临时测试页使用的是45点选点路线。"
            )
            kind = "warning"
        else:
            age = self._frame_age_ms()
            fresh = age <= LATEST_FRAME_MAX_AGE_MS
            text = (
                f"数据链路正常：最新45点帧 {age:.0f} ms 前"
                + ("" if fresh else "（已过期，扫描可能已停止）")
            )
            kind = "ok" if fresh else "warning"
        self.link_label.setText(text)
        self.link_label.setProperty("statusKind", kind)
        self.link_label.style().unpolish(self.link_label)
        self.link_label.style().polish(self.link_label)
        self._update_force_label()

    def _request_force_zero(self, _checked=False, *, confirm=True):
        """Explicit host tare; never sends cmd10 or alters predicted force."""
        try:
            if not self.force_sensor_enabled:
                raise RuntimeError('采集力已禁用，不能归零')
            self.force_sensor.tare()
        except Exception as exc:
            self.force_notice = f"采集力归零失败：{exc}"
            success = False
        else:
            self.force_notice = "采集力归零成功（主机去皮，不是设备标定；两页共享）"
            success = True
        self.force_notice_until = time.monotonic() + 4.0
        self._refresh_force_arrows()
        self._update_force_label()
        return success

    def _force_zero_done(self, success, error):
        self.force_notice = str(error) if not success else "设备命令结束（独立于主机去皮）"
        self.force_notice_until = time.monotonic() + 4.0
        self._update_force_label()

    def _update_force_label(self):
        sensor = self.force_sensor
        sample = sensor.measurement_snapshot() if isinstance(sensor, PaxiniForceSensor) and self.force_sensor_enabled else None
        self._refresh_force_arrows()
        if sample:
            fx, fy, fz = sample['corrected'][:3]
            text = (f"帕西尼PX6D（{sensor.port or '协议回放'}）：已连接 · "
                    f"Fx={fx:+.3f} N  Fy={fy:+.3f} N  Fz={fz:+.3f} N · "
                    f"主机去皮{sensor.tare_count}次 · 最新帧{sample['age_ms']:.0f} ms")
            kind = 'ok'
        else:
            reason = sensor.last_error or self.force_last_error or '未连接／数据过期；点击连接'
            text = f"帕西尼PX6D（{sensor.port or '自动识别'}）：无新鲜实测力（{reason}）；缺失保存null，不阻断45点采集"
            kind = 'warning'
        if self.force_notice and time.monotonic() < self.force_notice_until:
            text += ' · ' + self.force_notice
        self.force_label.setToolTip(text)
        text = ('PX6D：已连接 · 主机去皮' if sample else 'PX6D：无新鲜力，缺失存null') + (' · 45实测点9×5' if self.capture_contract=='legacy45' else ' · 源45实测点 → 9峰Makima Δλ')
        if self.force_notice and time.monotonic() < self.force_notice_until:
            text += ' · ' + self.force_notice
        self.force_label.setText(text)
        self.force_label.setProperty('statusKind', kind)
        self.force_label.style().unpolish(self.force_label)
        self.force_label.style().polish(self.force_label)

    @staticmethod
    def _safe_name(value):
        clean = re.sub(r'[^0-9A-Za-z_\-\u4e00-\u9fff]+', '_', str(value).strip())
        return clean.strip('_') or "press_grid"

    def select_no_press(self, _checked=False):
        self.selected_point = 0
        self.selected_points.clear()
        self.grid_group.setExclusive(False)
        for button in self.grid_group.buttons():
            button.setChecked(False)
        self.grid_group.setExclusive(self.annotation_mode == "single")
        self.surface_map.set_selected_points([])
        self.selected_label.setText(f"当前标签：无按压／空载（0号） · 已采{self.capture_counts[0]}条")
        self._refresh_single_count_hint()
        if self.annotation_mode == "multi":
            self._refresh_multi_counts()
        self.capture_status_label.setText("请保持空载，然后点击“采集9峰Δλ”（不自动重设基准）" if self.capture_contract == "delta9" else "请保持手指无按压，然后点击“采集当前45点”")
        return True

    def select_grid_point(self, label):
        label = int(label)
        if not 1 <= label <= GRID_POINT_COUNT:
            raise ValueError("grid point must be 1..25")
        if self.annotation_mode == "multi":
            if label in self.selected_points:
                self.selected_points.remove(label)
            else:
                self.selected_points.add(label)
            self.surface_map.set_selected_points(self.selected_points)
            self.selected_point = None
            self._refresh_multi_counts()
            self.capture_status_label.setText("请同时按住所选全部位置；空集合表示空载。共享PX6D仅记录整体合力，不分配每点力。")
            return True
        self.selected_point = label
        self.selected_points = {label}
        self.surface_map.set_selected_points([label])
        button = self.grid_group.button(label)
        if button is not None:
            button.setChecked(True)
        x, y = FINGER_POINT_BY_LABEL[label]
        region = _surface_region(y)
        count = self.capture_counts[label]
        self.selected_label.setText(
            f"当前标签：编号 {label} · {region} · 已采{count}条"
        )
        self._refresh_single_count_hint()
        self.selected_label.setProperty("statusKind", "ok")
        self.selected_label.style().unpolish(self.selected_label)
        self.selected_label.style().polish(self.selected_label)
        self.capture_status_label.setText(
            f"请手动按住编号{label}，点击“采集9峰Δλ”" if self.capture_contract == "delta9" else f"请手动按住编号{label}对应位置，然后点击“采集当前45点”"
        )
        return True

    # -------------------------------------------------------------- capture

    def _change_annotation_mode(self, _index):
        if self.writer is not None:
            self.mode_combo.blockSignals(True)
            self.mode_combo.setCurrentIndex(0 if self.annotation_mode == "single" else 1)
            self.mode_combo.blockSignals(False)
            self.capture_status_label.setText("请先保存并关闭当前数据集，再切换单点／多点模式；同一会话禁止混合。")
            return
        self.annotation_mode = self.mode_combo.currentData()
        self.selected_label.setStyleSheet('')
        self.combo_counts_table.setVisible(self.annotation_mode == "multi")
        self.select_no_press()


    def _ensure_session(self):
        if self.writer is not None:
            return
        if self.capture_contract == "delta9":
            output_root = self.output_root if self.output_root.name == "manual_touch_delta9" else self.output_root / "manual_touch_delta9"
        else:
            output_root = self.output_root / "manual_touch_multi45" if self.annotation_mode == "multi" else self.output_root
        output_root.mkdir(parents=True, exist_ok=True)
        stem = self._safe_name(self.dataset_name_edit.text())
        session_dir = output_root / stem
        suffix = 1
        while session_dir.exists():
            session_dir = output_root / f"{stem}_{suffix:02d}"
            suffix += 1
        metadata = {
            "dataset_kind": "fixed_25_point_press_grid",
            "training_eligible": True,
            "input_contract": "stable_adc_with_physical_and_makima_views",
            "frame_source": "temporary_test_mode_live_frames",
            "sample_kinds": [
                "stable_second_code",
                "stable_voltage_v",
                "stable_optical_power_dbm",
                "makima_curves_adc_voltage_dbm",
            ],
            "grid_shape": [GRID_ROWS, GRID_COLUMNS],
            "label_range": [0, GRID_POINT_COUNT],
            "no_press_label": 0,
            "label_semantics": "0=no_press; 1..25=pressed_position",
            "view": "finger_surface_front_25_points",
            "point_layout": [
                {"label": label, "x": x, "y": y}
                for label, x, y in FINGER_POINT_LAYOUT
            ],
            "intended_task": "capture_no_press_0_and_press_labels_1_to_25",
        }
        if self.annotation_mode == "multi":
            metadata.update(dataset_kind="manual_touch_multi45", training_eligible=True,
                annotation_mode="multi", position_label_schema="multi_hot25_v1",
                input_contract="45_measured_second_code_9x5_with_operator_multi_hot25_v1",
                label_semantics="position_labels=operator_confirmed_simultaneous_contacts; empty=no_press",
                intended_task="independent45_multilabel_position_and_whole_contact_force",
                force_semantics="shared_PX6D_tare_corrected_whole_contact_resultant_not_per_position")
        if self.capture_contract == "delta9":
            from delta9_capture import Delta9SessionWriter
            metadata.update(dataset_kind="manual_touch_delta9",training_eligible=False,
                input_contract="delta_wavelength9",input_shape=[9],feature_units="nm",
                annotation_mode=self.annotation_mode,peak_order=[f"P{i}" for i in range(1,10)],
                peak_to_position_labels=list(PEAK_LABELS),sample_kinds=["delta_wavelength_nm"],
                intended_task="delta9_capture_only_training_not_adapted",
                source_acquisition="45 measured points -> 9 same-frame Makima curve-global peaks",
                force_semantics="shared_PX6D_tare_corrected_whole_contact_resultant_not_per_position")
            self.writer = Delta9SessionWriter(session_dir,queue_capacity=512,manifest_extra=metadata)
        else:
            self.writer = FingerSessionWriter(session_dir,queue_capacity=512,manifest_extra=metadata)
        self.session_dir = session_dir
        self.capture_counts = {index: 0 for index in range(0, 26)}
        self.capture_history.clear()
        self.combo_counts.clear()
        self.sample_index = 0
        self.sequence = 0
        self.writer.write_event(ContactEvent(kind="session_start", metadata=metadata))
        self.dataset_name_edit.setEnabled(False)

    def _read_force_sample(self):
        """Nonblocking atomic raw/corrected snapshot; missing force stays None."""
        self._force_sample_snapshot = None
        if not self.force_sensor_enabled:
            self.force_last_error = 'force_sensor_disabled'
            return None
        self._force_sample_snapshot = self.force_sensor.measurement_snapshot()
        if self._force_sample_snapshot is None:
            self.force_last_error = self.force_sensor.last_error or 'force_disconnected_stale_or_invalid'
            return None
        self.force_last_error = None
        return tuple(self._force_sample_snapshot['corrected'])

    def _build_45_capture(self, wavelengths, labels, annotation_mode, sample_index, sequence, age_ms):
        """Shared manual/timer record contract, including the atomic corrected force snapshot."""
        force_error = None
        labels = list(labels)
        label = labels[0] if len(labels) == 1 else 0
        surface_x, surface_y = FINGER_POINT_BY_LABEL[label] if label else (None, None)
        region = _surface_region(surface_y) if label else "无按压／空载"
        # Retain the historical row/column fields for old readers, but the
        # authoritative annotation is now the hand-surface coordinate below.
        row = (label - 1) // GRID_COLUMNS + 1 if label else None
        column = (label - 1) % GRID_COLUMNS + 1 if label else None
        capture_ns = time.monotonic_ns()
        frame = self.latest_frame or {}
        training_views = build_training_views(frame, wavelengths)
        stable_adc = training_views["stable_adc"]
        selector = training_views["conversion"]["feedback_selector"]
        try:
            force = self._read_force_sample()
        except Exception as exc:
            force = None
            force_error = str(exc) or type(exc).__name__
        quality = {
            "dataset_kind": "fixed_25_point_press_grid",
            "press_label": label,
            "press_state": "pressed" if label else "no_press",
            "is_pressed": bool(label),
            "grid_row": row,
            "grid_column": column,
            "surface_x_normalized": float(surface_x) if surface_x is not None else None,
            "surface_y_normalized": float(surface_y) if surface_y is not None else None,
            "surface_region": region,
            "sample_index": sample_index,
            "source_frame_sequence": int(frame.get("sequence", -1)),
            "source_frame_age_ms": float(age_ms),
            "source_frame_quality": dict(frame.get("quality") or {}),
            "training_views": training_views,
            # Paxini PX6D six-axis force at the capture instant.  None means
            # the sensor was absent/busy; force_error says why.
            "paxini_force": None if force is None else {
                "fx_n": force[0],
                "fy_n": force[1],
                "fz_n": force[2],
                "mx_nm": force[3],
                "my_nm": force[4],
                "mz_nm": force[5],
                "monotonic_ns": (self._force_sample_snapshot or {}).get("sample_monotonic_ns"),
                "raw_six_axis": (self._force_sample_snapshot or {}).get("raw"),
                "axis_order": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
                "axis_units": ["N", "N", "N", "N*m", "N*m", "N*m"],
                "sample_age_ms": (self._force_sample_snapshot or {}).get("age_ms"),
                "port": (self._force_sample_snapshot or {}).get("port"),
                "protocol": (self._force_sample_snapshot or {}).get("protocol"),
                "tare": (self._force_sample_snapshot or {}).get("tare"),
                "official_zero_count": self.force_sensor.zero_count,
                "last_official_zero_ns": self.force_sensor.last_zero_ns,
            },
            "force_error": (force_error or self.force_last_error or "force_unavailable") if force is None else None,
            "points_per_peak": 5, "point_count": 45, "input_shape": [9, 5],
            "stable_records": [{k: v for k, v in r.items() if k != "first_code"} for r in frame.get("records", [])],
            "frame_provenance": frame.get("provenance", "temporary_test_live_frame"),
            "smoke_only": "synthetic" in str(frame.get("provenance", "")).lower(),
            "cycle_start_us": frame.get("cycle_start_us"),
            # Flat aliases keep old analysis scripts working while new models
            # consume the complete nested training_views schema.
            "feedback_selector": selector,
            "adc_channel": int(frame.get("adc_channel",1)),
            "route_indices": [r.get("index") for r in (getattr(self.frame_source,"plan",None) or {}).get("rows",[])],
            "transimpedance_ohm": training_views["conversion"]["transimpedance_ohm"],
            "stable_adc": stable_adc,
            "stable_voltage_v": training_views["stable_voltage_v"],
            "stable_dbm": training_views["stable_optical_power_dbm"],
            "peak_fits": training_views["makima_peak_fits"],
        }
        if annotation_mode == "multi":
            quality.update(dataset_kind="manual_touch_multi45", annotation_mode="multi",
                training_eligible=True, position_label_schema="multi_hot25_v1",
                input_contract="45_measured_second_code_9x5_with_operator_multi_hot25_v1",
                position_labels=labels,
                position_multihot=[int(i in labels) for i in range(1, 26)],
                position_coordinates=[{"label": i, "x": FINGER_POINT_BY_LABEL[i][0], "y": FINGER_POINT_BY_LABEL[i][1]} for i in labels],
                annotation_source="operator_confirmed_simultaneous_contacts",
                press_label=labels[0] if len(labels) == 1 else (None if labels else 0),
                press_state="pressed" if labels else "no_press", is_pressed=bool(labels),
                force_semantics="whole_contact_resultant_not_per_position")
            if len(labels) > 1:
                for key in ("grid_row", "grid_column", "surface_x_normalized", "surface_y_normalized", "surface_region"):
                    quality[key] = None
        event = ContactEvent(
            kind="press_sample",
            monotonic_ns=capture_ns,
            metadata={
                "press_label": label,
                "press_state": "pressed" if label else "no_press",
                "is_pressed": bool(label),
                "grid_row": row,
                "grid_column": column,
                "surface_x_normalized": float(surface_x) if surface_x is not None else None,
                "surface_y_normalized": float(surface_y) if surface_y is not None else None,
                "surface_region": region,
                "sample_index": sample_index,
                "source_frame_sequence": int(frame.get("sequence", -1)),
                **({key: quality[key] for key in ("annotation_mode", "position_labels", "position_multihot", "position_coordinates", "position_label_schema", "press_label", "press_state", "is_pressed", "grid_row", "grid_column", "surface_x_normalized", "surface_y_normalized", "surface_region")} if annotation_mode == "multi" else {}),
            },
        )
        record = SpectrumFrame(
            sequence=sequence,
            monotonic_ns=capture_ns,
            wavelengths_nm=wavelengths,
            raw_adc_codes={int(frame.get("adc_channel",1)): stable_adc},
            adc_value_kind="direct_adc",
            quality=quality,
            mode="stress",
            source="temporary_test_live_frame",
            transimpedance_ohm={int(frame.get("adc_channel",1)): units.transimpedance_ohm(selector)},
        )
        return record, event

    def capture_current_press(self):
        if self.writer is not None:
            from delta9_capture import Delta9SessionWriter
            if isinstance(self.writer,Delta9SessionWriter) != (self.capture_contract == "delta9"):
                self.capture_status_label.setText("已有不同存储契约的会话；请先保存关闭，禁止混写9峰Δλ与45点数据。")
                return False
        if self.capture_contract == "delta9":
            return self._capture_delta9()
        force = None
        force_error = None
        wavelengths = list(self.wavelength_provider() or ())
        if len(wavelengths) != STRESS_POINT_COUNT:
            self.capture_status_label.setText(
                "临时测试页尚未完成45点参考线，波长表不是45点，无法采集"
            )
            return False
        if self.latest_values is None:
            self.capture_status_label.setText(
                "尚未收到临时模式45点帧；请先在临时测试页开始连续采集"
            )
            return False
        age_ms = self._frame_age_ms()
        if age_ms > LATEST_FRAME_MAX_AGE_MS:
            self.capture_status_label.setText(
                f"最新45点帧已过期（{age_ms:.0f} ms），请在临时测试页保持连续采集"
            )
            return False

        previous_sample_index, previous_sequence = self.sample_index, self.sequence
        try:
            self._ensure_session()
            self.sample_index += 1
            self.sequence += 1
            labels = sorted(self.selected_points) if self.annotation_mode == "multi" else ([int(self.selected_point)] if self.selected_point else [])
            record, event = self._build_45_capture(wavelengths, labels, self.annotation_mode,
                self.sample_index, self.sequence, age_ms)
            label = labels[0] if len(labels) == 1 else 0
            region = record.quality["surface_region"]
            force = record.quality["paxini_force"]
            self.writer.write_event(event)
            self.writer.write_frame(record)
            self.writer.flush()
        except Exception as exc:
            self.sample_index, self.sequence = previous_sample_index, previous_sequence
            self.capture_status_label.setText(f"采集失败，未计入组合统计：{exc}")
            return False
        self.capture_history.append({
            "press_label": label,
            "position_labels": labels,
            "sample_index": self.sample_index,
            "sequence": self.sequence,
        })
        self.undo_button.setEnabled(True)
        if self.annotation_mode == "multi":
            combo = tuple(sorted(set(labels)))
            self.combo_counts[combo] = self.combo_counts.get(combo, 0) + 1
            for position in labels or [0]:
                self.capture_counts[position] += 1
            self._refresh_multi_counts()
            self.capture_status_label.setText(f"多点采集成功：{labels or '空载'} → 第{self.sample_index}组45点数据（整体合力，无每点力）")
            self._persist_combo_counts()

            return True
        self.capture_counts[label] += 1
        count = self.capture_counts[label]
        total = sum(self.capture_counts.values())
        button = self.grid_group.button(label)
        if button is not None:
            button.setToolTip(
                f"手指表面位置编号 {label} · {region} · 已采{count}条"
            )
        self.selected_label.setText(
            f"当前标签：编号 {label} · {region} · 已采{count}条"
        )
        self.counts_label.setText(
            f"已采集：{total}条 · 当前编号{label}累计{count}条 · "
            f"丢弃非45点帧{self.rejected_non45_frames}"
        )
        self.capture_status_label.setText(
            f"采集成功：编号{label} → 第{self.sample_index}条数据（45点CH1）"
            + (
                ""
                if force is not None
                else " · 力传感器未读到"
            )
        )
        self._refresh_single_count_hint()
        return True

    def _capture_delta9(self):
        """Atomically snapshot the already computed nine peak deltas; never refit/save raw45."""
        from delta9_capture import Delta9Frame
        state = self.wavelength_state
        context = self._wavelength_context()
        state.reset_context(context)
        frame = self.latest_frame
        snapshot = getattr(self.frame_source,"makima_frame_snapshot",None)
        fits = self.latest_peak_fits or []
        valid = (state.baseline is not None and frame is not None
            and snapshot is not None and snapshot[0] is frame and snapshot[1] is self.latest_peak_fits
            and state.sequence == frame.get("sequence") and context[-1] == "makima"
            and len(context[4]) == STRESS_POINT_COUNT and bool(getattr(self,"wavelength_baseline_id",None))
            and not state.stale(time.monotonic()) and self._frame_age_ms() <= LATEST_FRAME_MAX_AGE_MS
            and not bool(getattr(self.frame_source,"peak_trend_stopped",False))
            and not frame.get("i2c_errors",0) and not frame.get("spi_errors",0)
            and len(fits)==9 and all(f.get("peak_selection_method")=="makima_curve_global_max_dbm" for f in fits)
            and all(v is not None and np.isfinite(v) for v in state.current)
            and all(np.isfinite(v) for v in state.baseline))
        if not valid:
            self.capture_status_label.setText("未采集：先解除按压设置空载波长基准；需新鲜同帧完整9峰MakimaΔλ。")
            return False
        current, baseline = tuple(state.current), tuple(state.baseline)
        labels = sorted(self.selected_points) if self.annotation_mode == "multi" else ([int(self.selected_point)] if self.selected_point else [])
        label = labels[0] if len(labels)==1 else 0
        quality = dict(dataset_kind="manual_touch_delta9",input_contract="delta_wavelength9",input_shape=[9],
            feature_units="nm",training_eligible=False,annotation_mode=self.annotation_mode,
            press_label=(labels[0] if len(labels)==1 else (None if labels else 0)),
            press_state="pressed" if labels else "no_press",is_pressed=bool(labels),
            source_frame_sequence=frame.get("sequence"),cycle_start_us=frame.get("cycle_start_us"),
            source_frame_age_ms=float(self._frame_age_ms()),baseline_id=getattr(self,"wavelength_baseline_id",None),
            frame_provenance=frame.get("provenance","temporary_test_live_frame"),
            smoke_only="synthetic" in str(frame.get("provenance","")).lower(),
            peak_selection_method="makima_curve_global_max_dbm",force_semantics="whole_contact_resultant_not_per_position")
        if self.annotation_mode == "multi":
            quality.update(position_labels=labels,position_multihot=[int(i in labels) for i in range(1,26)],
                position_label_schema="multi_hot25_v1",annotation_source="operator_confirmed_simultaneous_contacts",
                position_coordinates=[dict(label=i,x=FINGER_POINT_BY_LABEL[i][0],y=FINGER_POINT_BY_LABEL[i][1]) for i in labels])
        x,y = FINGER_POINT_BY_LABEL[label] if label else (None,None)
        quality.update(grid_row=(label-1)//5+1 if label else None,grid_column=(label-1)%5+1 if label else None,
            surface_x_normalized=x,surface_y_normalized=y,surface_region=_surface_region(y) if label else (None if labels else "无按压／空载"))
        force_error = None
        try:
            force = self._read_force_sample()
        except Exception as exc:
            force = None; force_error = str(exc)
        sample = self._force_sample_snapshot or {}
        quality["paxini_force"] = None if force is None else dict(zip(("fx_n","fy_n","fz_n","mx_nm","my_nm","mz_nm"),force))
        if force is not None:
            quality["paxini_force"].update(raw_six_axis=sample.get("raw"),tare=sample.get("tare"),
                monotonic_ns=sample.get("sample_monotonic_ns"),sample_age_ms=sample.get("age_ms"),
                port=sample.get("port"),protocol=sample.get("protocol"),axis_order=["Fx","Fy","Fz","Mx","My","Mz"],
                axis_units=["N","N","N","N*m","N*m","N*m"])
        quality["force_error"] = (force_error or self.force_last_error or "force_unavailable") if force is None else None
        previous = self.sample_index,self.sequence
        try:
            self._ensure_session()
            self.sample_index += 1; self.sequence += 1
            quality["sample_index"] = self.sample_index
            captured = time.monotonic_ns()
            import hashlib
            context_record = dict(machine=context[0],source_identity=context[1],channel=context[2],feedback_selector=context[3],
                route_sha256=hashlib.sha256(json.dumps(context[4],sort_keys=True).encode()).hexdigest(),
                source_point_count=len(context[4]),peak_to_position_labels=list(context[5]),generation=context[6],
                transport=context[7],fit_method=context[8])
            self.writer.write_event(ContactEvent(kind="press_sample",monotonic_ns=captured,metadata={k:quality[k] for k in ("dataset_kind","input_contract","annotation_mode","press_label","press_state","sample_index","source_frame_sequence","baseline_id")} | ({"position_labels":labels} if self.annotation_mode == "multi" else {})))
            self.writer.write_frame(Delta9Frame(self.sequence,captured,current,baseline,context_record,state.baseline_time,quality))
            self.writer.flush()
        except Exception as exc:
            self.sample_index,self.sequence = previous
            self.capture_status_label.setText(f"采集失败，未计入组合统计：{exc}")
            return False
        self.capture_history.append(dict(press_label=label,position_labels=labels,sample_index=self.sample_index,sequence=self.sequence))
        self.undo_button.setEnabled(True)
        for position in labels or [0]:
            self.capture_counts[position] += 1
        snapshot_ok = True
        if self.annotation_mode == "multi":
            combo=tuple(labels);self.combo_counts[combo]=self.combo_counts.get(combo,0)+1
            self._refresh_multi_counts();snapshot_ok = self._persist_combo_counts()
        else:
            self.selected_label.setText(f"当前标签：{label or '空载'} · 已采{self.capture_counts[label]}组 | 总样本 {len(self.capture_history)} | 空载 {self.capture_counts[0]}")
        if snapshot_ok:
            self.capture_status_label.setText(f"已保存第{self.sample_index}组：9峰Δλ / nm · {labels or '空载'}"+(" · 力缺失存null" if force is None else ""))
        return True

    def _persist_combo_counts(self):
        """Atomic replace of a small derived snapshot; never rewrite raw audit files."""
        if self.annotation_mode != "multi" or self.session_dir is None:
            return True
        path = self.session_dir / 'combination_counts.json'
        payload = {
            'schema': 'manual_touch_delta9_combination_counts_v1' if self.capture_contract == 'delta9' else 'manual_touch_multi45_combination_counts_v1',
            'session': self.session_dir.name, 'dataset_name': self.session_dir.name,
            'dataset_kind': 'manual_touch_delta9' if self.capture_contract == 'delta9' else 'manual_touch_multi45',
            'counting_semantics': 'successful_delta9_frames_minusundone' if self.capture_contract == 'delta9' else 'successful45pointframes_minusundone',
            'sample_unit': 'one_successfully_saved_9delta_frame_not_press_round' if self.capture_contract == 'delta9' else 'one_successfully_saved_45_point_frame_not_press_round',
            'active_samples_count': len(self.capture_history),
            'combinations': [{'position_labels': list(k), 'count': self.combo_counts[k]}
                             for k in sorted(self.combo_counts, key=lambda k: (len(k), k))],
        }
        temporary = path.with_suffix('.json.tmp')
        try:
            with temporary.open('w', encoding='utf-8', newline='\n') as stream:
                json.dump(payload, stream, ensure_ascii=False, indent=2, allow_nan=False)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        except Exception as exc:
            self.capture_status_label.setText(f'样本状态已更新，但组合统计保存失败（可重试保存关闭）：{exc}')
            return False
        return True

    def _refresh_single_count_hint(self):
        label = int(self.selected_point or 0)
        self.selected_label.setText(f"当前位置 {label or '空载'} · 已采 {self.capture_counts[label]} 组 | 总样本 {len(self.capture_history)} | 空载 {self.capture_counts[0]}")

    def _refresh_multi_counts(self):
        combo = tuple(sorted(set(self.selected_points)))
        summary = (', '.join(map(str, combo[:8])) + (f', … 共{len(combo)}点' if len(combo)>8 else '')) if combo else '空载'
        self.selected_label.setStyleSheet('background-color:#e8f6ee;color:#166b48;padding:6px;')
        self.selected_label.setWordWrap(True)
        self.selected_label.setText(f'当前组合 {{{summary}}} · 当前组合已采集 {self.combo_counts.get(combo, 0)} 组 | 总样本 {len(self.capture_history)} | 空载 {self.capture_counts[0]}')
        keys = sorted(self.combo_counts, key=lambda k: (len(k), k))
        self.combo_counts_table.setRowCount(len(keys))
        for row, key in enumerate(keys):
            for col, text in enumerate(('{' + ', '.join(map(str,key)) + '}' if key else '空载', str(self.combo_counts[key]))):
                self.combo_counts_table.setItem(row, col, QtWidgets.QTableWidgetItem(text))
        total = len(self.capture_history)
        counts = " · ".join(f"{i}号:{self.capture_counts[i]}" for i in range(1, 26))
        self.counts_label.setToolTip(f"每位置接触样本数（非总样本数）：{counts}")
        self.counts_label.setText(f"总样本 {total} · 空载 {self.capture_counts[0]}")
        self.selected_label.setToolTip(self.counts_label.toolTip()+"；操作员确认接触，不自动判断")
        selected_row = keys.index(combo) if combo in keys else None
        if selected_row is not None:
            self.combo_counts_table.selectRow(selected_row)
            self.combo_counts_table.scrollToItem(self.combo_counts_table.item(selected_row,0))
        for i, button in enumerate(self.grid_buttons, 1):
            button.setToolTip(f"手指表面位置编号 {i} · 已采{self.capture_counts[i]}条")

    def undo_last_annotation(self, _checked=False, *, confirm=True):
        """Remove the most recently captured frame and restore the UI counters."""

        if self.writer is None or not self.capture_history:
            self.capture_status_label.setText("当前数据集中没有可回退的标注")
            self.undo_button.setEnabled(False)
            return False
        latest = self.capture_history[-1]
        label = int(latest["press_label"])
        labels = latest.get("position_labels", [])
        sample_index = int(latest["sample_index"])
        if confirm:
            answer = QtWidgets.QMessageBox.question(
                self,
                "回退上一次标注",
                f"将删除刚刚采集的位置{labels or '空载'}第{sample_index}条数据，是否继续？",
                QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
                QtWidgets.QMessageBox.No,
            )
            if answer != QtWidgets.QMessageBox.Yes:
                return False
        try:
            self.writer.undo_last_frame(expected_sample_index=sample_index)
        except Exception as exc:
            self.capture_status_label.setText(f"回退失败，数据未修改：{exc}")
            return False
        audit_error = None
        try:
            self.writer.write_event(ContactEvent(
                kind="press_sample_undo",
                metadata={
                    "press_label": label,
                    "sample_index": sample_index,
                    "reason": "operator_undo_last_annotation",
                    **({"position_labels": labels, "annotation_mode": "multi", "press_label": labels[0] if len(labels) == 1 else (None if labels else 0)} if self.annotation_mode == "multi" else {}),
                },
            ))
        except Exception as exc:
            audit_error = str(exc)

        self.capture_history.pop()
        for position in ((labels or [0]) if self.annotation_mode == "multi" else [label]):
            self.capture_counts[position] = max(0, self.capture_counts[position] - 1)
        if self.capture_history:
            self.sample_index = int(self.capture_history[-1]["sample_index"])
            self.sequence = int(self.capture_history[-1]["sequence"])
        else:
            self.sample_index = 0
            self.sequence = 0
        self.undo_button.setEnabled(bool(self.capture_history))
        if self.annotation_mode == "multi":
            combo = tuple(sorted(set(labels)))
            remaining = self.combo_counts.get(combo, 0) - 1
            if remaining > 0:
                self.combo_counts[combo] = remaining
            else:
                self.combo_counts.pop(combo, None)
            self._refresh_multi_counts()
            self.capture_status_label.setText(f"已回退：删除位置{labels or '空载'}的第{sample_index}条标注数据")
            self._persist_combo_counts()
            if audit_error:
                self.capture_status_label.setText(f"样本已撤销、组合计数已更新，但撤销审计写入失败：{audit_error}")
            return True
        total = sum(self.capture_counts.values())
        count = self.capture_counts[label]
        _x, y = FINGER_POINT_BY_LABEL[label] if label else (None, None)
        region = _surface_region(y) if label else "无按压／空载"
        button = self.grid_group.button(label)
        if button is not None:
            button.setToolTip(
                f"手指表面位置编号 {label} · {region} · 已采{count}条"
            )
        if int(self.selected_point or 0) == label:
            self.selected_label.setText(
                f"当前标签：编号 {label} · {region} · 已采{count}条"
            )
        self.counts_label.setText(
            f"已采集：{total}条 · 编号{label}累计{count}条 · "
            f"丢弃非45点帧{self.rejected_non45_frames}"
        )
        self.capture_status_label.setText(
            f"已回退：删除编号{label}的第{sample_index}条标注数据"
        )
        self._refresh_single_count_hint()
        return True

    def close_session(self):
        if self.writer is None:
            return False
        total = len(self.capture_history)
        if not self._persist_combo_counts():
            return False
        self.writer.write_event(ContactEvent(
            kind="session_end",
            metadata={
                "total_samples": total,
                "counts_by_label": dict(self.capture_counts),
                "rejected_non45_frames": self.rejected_non45_frames,
                **({"annotation_mode": "multi", "count_semantics": "counts_by_label_are_position_contact_counts_not_frame_total"} if self.annotation_mode == "multi" else {}),
            },
        ))
        path = self.session_dir
        self.writer.close()
        self.writer = None
        self.capture_history.clear()
        self.combo_counts.clear()
        if self.annotation_mode == "multi":
            self.capture_counts = {index: 0 for index in range(26)}
            self._refresh_multi_counts()
        self.undo_button.setEnabled(False)
        self.dataset_name_edit.setEnabled(True)
        self.capture_status_label.setToolTip(str(path))
        self.capture_status_label.setText(f"数据集已保存 · 共{total}组9峰Δλ（路径见提示）" if self.capture_contract == "delta9" else f"数据集已保存：{path} · 共{total}条45点数据")
        return True

    def closeEvent(self, event):
        try:
            self._stop_timed_empty()
            self.link_timer.stop()
            self.force_timer.stop()
            self.wavelength_timer.stop()
            self.detach()
            self.close_session()
        finally:
            if self._owns_force_sensor:
                self.force_sensor.close()
            event.accept()
