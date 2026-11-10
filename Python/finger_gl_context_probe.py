"""Desktop GL regression: independent Qt contexts must not share program IDs.
Run with Python/.venv/Scripts/python.exe -m unittest test_finger_gl_contexts -v.
Does not create serial devices or load checkpoints; screenshots are evidence.
"""
import os
import sys
import unittest
from pathlib import Path

plugins = Path(sys.prefix) / 'Lib/site-packages/PyQt5/Qt5/plugins'
os.environ['QT_QPA_PLATFORM'] = 'windows'
os.environ['QT_OPENGL'] = 'desktop'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = str(plugins / 'platforms')
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtTest import QTest
import pyqtgraph.opengl as gl
from OpenGL import GL
from manual_stress_capture import FingerSurfaceMap
from manual_touch_training_page import PerformanceFingerMap

EVIDENCE = Path(__file__).resolve().parents[1] / 'migration_evidence/finger_compact_no_numbers'

class FingerGLContextTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        QtCore.QCoreApplication.setLibraryPaths([str(plugins)])
        cls.app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
        QtGui.QFontDatabase.addApplicationFont('C:/Windows/Fonts/msyh.ttc')
        cls.app.setFont(QtGui.QFont('Microsoft YaHei', 9))
        cls.legacy = gl.GLViewWidget()
        cls.legacy.resize(260, 220)
        cls.legacy.addItem(gl.GLMeshItem(meshdata=gl.MeshData.sphere(rows=8, cols=8), shader='edgeHilight'))
        cls.legacy.show()
        QTest.qWait(200)

    @classmethod
    def tearDownClass(cls):
        cls.legacy.close()

    def test_display_is_compact_but_semantics_are_unchanged(self):
        from manual_stress_capture import FINGER_POINT_LAYOUT, FINGER_POINT_BY_LABEL, _surface_region
        canonical = tuple((r * 5 + c + 1, x, y)
            for r, y in enumerate((.16, .255, .35, .445, .54))
            for c, x in enumerate((.30, .40, .50, .60, .70)))
        self.assertEqual(FINGER_POINT_LAYOUT, canonical)
        surface = FingerSurfaceMap()
        try:
            for (label, x, y), position in zip(canonical, surface.marker_positions):
                import numpy as np
                np.testing.assert_allclose(position, surface._model_position(
                    .5 + .8 * (x - .5), .35 + .8 * (y - .35)))
                self.assertEqual(FINGER_POINT_BY_LABEL[label], (x, y))
                self.assertEqual(surface.group.button(label).toolTip(), f'手指表面位置编号 {label}')
            self.assertEqual([_surface_region(y) for y in (.16, .255, .35, .445, .54)],
                             ['指尖', '上段指腹', '上段指腹', '上段指腹', '中段指腹'])
        finally:
            surface.close()
            surface.deleteLater()
            QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)

    def test_maps_have_no_number_annotations(self):
        for cls in (FingerSurfaceMap, PerformanceFingerMap):
            surface = cls()
            surface.resize(750, 560)
            surface.show()
            QTest.qWait(200)
            try:
                self.assertEqual(surface.text_items, [])
                self.assertEqual(surface.number_labels, [])
                self.assertEqual(surface.findChildren(QtWidgets.QLabel), [])
                self.assertFalse(any(isinstance(item, gl.GLTextItem) for item in surface.view.items))
                self.assertEqual(len(surface.marker_positions), 25)
                self.assertEqual(surface.view.opts['elevation'], 90)
            finally:
                surface.close()
                surface.deleteLater()
                QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)

    def test_full_pages_projected_mouse_selection_and_no_press(self):
        from manual_stress_capture import ManualStressCaptureWindow
        from manual_touch_training_page import ManualTouchTrainingPage
        import tempfile

        class FakeSource:
            frame_listeners = []
        class IdleSensor:
            _latest = None
            _latest_ns = None
            frames_received = 0
            last_error = None
            port = None
            def start(self):
                raise AssertionError('no hardware permitted in visual probe')
            def close(self): pass

        scratch = Path.home() / 'AppData/Local/hermes/cache/scratch'
        scratch.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as folder:
            capture = ManualStressCaptureWindow(FakeSource(), wavelength_provider=lambda: [],
                output_root=Path(folder), force_sensor=IdleSensor())
            training = ManualTouchTrainingPage(default_dataset_dir=Path(folder))
            shell = QtWidgets.QMainWindow()
            stack = QtWidgets.QStackedWidget()
            shell.setCentralWidget(stack)
            stack.addWidget(capture.takeCentralWidget())
            stack.addWidget(training)
            shell.resize(1500, 1100)
            shell.show()
            try:
                for index, surface in enumerate((capture.surface_map, training.performance_map)):
                    stack.setCurrentIndex(index)
                    QTest.qWait(250)
                    for label, model_point in enumerate(surface.marker_positions, 1):
                        x, y = surface._project_to_widget(model_point)
                        projected = QtCore.QPoint(round(x), round(y))
                        target = surface.childAt(projected)
                        self.assertIs(target, surface.group.button(label), 'projected dot has wrong click hit target')
                        QTest.mouseClick(target, QtCore.Qt.LeftButton, pos=target.mapFrom(surface, projected))
                        self.assertEqual(surface.group.checkedId(), label)
                        if index == 0:
                            self.assertEqual(capture.selected_point, label)
                        else:
                            self.assertEqual(training.selected_point_label.text(), f'编号{label}暂无验证结果')
                    # GL errors are caught internally by pyqtgraph; check the
                    # actual framebuffer of each embedded production page too.
                    surface.view.makeCurrent()
                    surface.body_item.paint()
                    surface.silicone_item.paint()
                    surface.marker_item.paint()
                    self.assertEqual(GL.glGetError(), GL.GL_NO_ERROR)
                    surface.view.doneCurrent()
                    QTest.qWait(50)
                    shell.grab().save(str(EVIDENCE / ('capture_page.png' if index == 0 else 'training_page.png')))
                    shell.screen().grabWindow(int(shell.winId())).save(str(EVIDENCE / ('capture_page_desktop.png' if index == 0 else 'training_page_desktop.png')))
                stack.setCurrentIndex(0)
                QTest.mouseClick(capture.no_press_button, QtCore.Qt.LeftButton)
                self.assertEqual(capture.selected_point, 0)
                self.assertEqual(capture.grid_group.checkedId(), -1)
                training.expected_position_combo.setCurrentIndex(training.expected_position_combo.findData(0))
                self.assertEqual(training._live_expected_label, 0)
                self.assertEqual(len(training.performance_map.marker_positions), 25)
            finally:
                capture.close()
                training.close()
                shell.close()
                capture.deleteLater()
                shell.deleteLater()
                QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)

    def test_mesh_programs_are_valid_in_each_independent_context(self):
        maps = [FingerSurfaceMap(), PerformanceFingerMap()]
        try:
            for index, surface in enumerate(maps):
                surface.resize(750, 560)
                surface.show()
                QTest.qWait(200)
                self.assertTrue(surface.view.isValid())
                surface.view.makeCurrent()
                program = surface.body_item.shader().program()
                self.assertTrue(GL.glIsProgram(program), 'mesh shader ID belongs to a different GL context')
                surface.body_item.paint()
                surface.silicone_item.paint()
                surface.marker_item.paint()
                self.assertEqual(GL.glGetError(), GL.GL_NO_ERROR)
                image = surface.view.grabFramebuffer()
                image.save(str(EVIDENCE / ('capture_map.png' if index == 0 else 'deployment_map.png')))
                # Real framebuffer, not item counts: require a substantial mesh
                # footprint differing from the off-white viewport background.
                rgb = image.convertToFormat(QtGui.QImage.Format_RGBA8888)
                ptr = rgb.bits(); ptr.setsize(rgb.byteCount())
                import numpy as np
                pixels = np.frombuffer(ptr, dtype=np.uint8).reshape(rgb.height(), rgb.width(), 4)[:, :, :3]
                mesh = (pixels.max(axis=2).astype(int) - pixels.min(axis=2).astype(int)) > 25
                self.assertGreater(int(mesh.sum()), 15000, 'finger mesh not visible in framebuffer')
        finally:
            for surface in maps:
                surface.view.doneCurrent()
                surface.close()
                surface.deleteLater()
            QtCore.QCoreApplication.sendPostedEvents(None, QtCore.QEvent.DeferredDelete)

if __name__ == '__main__':
    unittest.main()
