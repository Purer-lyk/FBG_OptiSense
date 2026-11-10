"""Text-only layout contracts; independent Qt process, no hardware."""
import os, sys, unittest
from pathlib import Path
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH']=str(Path(sys.prefix)/'Lib/site-packages/PyQt5/Qt5/plugins/platforms')
from PyQt5 import QtCore, QtWidgets
from manual_stress_capture import ManualStressCaptureWindow
from test_manual_stress_capture import FakeTemporaryPage

class CaptureTextLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        QtCore.QCoreApplication.setLibraryPaths([str(Path(sys.prefix)/'Lib/site-packages/PyQt5/Qt5/plugins')])
        cls.app=QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def test_explanation_is_a_wrapping_row_below_complete_three_columns(self):
        source=FakeTemporaryPage()
        win=ManualStressCaptureWindow(source, wavelength_provider=lambda:source.reference_wavelengths,output_root=Path(os.environ['TMPDIR']))
        self.addCleanup(win.close)
        explanation=win.wavelength_explanation
        for width in (1280,1642):
            win.resize(width,1400);win.show();self.app.processEvents()
            origin=explanation.mapTo(win,QtCore.QPoint())
            model=win.surface_map.mapTo(win,QtCore.QPoint())
            self.assertGreaterEqual(origin.y(),model.y()+win.surface_map.height(), 'explanatory text overlaps the GL model')
            self.assertTrue(explanation.wordWrap())
            self.assertGreaterEqual(explanation.height(), explanation.heightForWidth(explanation.width()))
        self.assertTrue(win.selected_label.wordWrap(), 'multi-contact summary must wrap rather than widen the page')

    def test_page_reserves_width_aware_wrapping_height_and_readable_table(self):
        from responsive_layout import ScrollContentWidget
        source=FakeTemporaryPage()
        win=ManualStressCaptureWindow(source,wavelength_provider=lambda:source.reference_wavelengths,output_root=Path(os.environ['TMPDIR']))
        self.addCleanup(win.close)
        self.assertIsInstance(win.centralWidget(),ScrollContentWidget, 'plain QWidget loses height-for-width when reparented to unified scroll area')
        self.assertGreaterEqual(win.wavelength_table.minimumWidth(),350, 'compact windows must scroll rather than crush four columns')

    def test_core_layout_fits_compact_capture_without_page_scroll(self):
        source=FakeTemporaryPage()
        win=ManualStressCaptureWindow(source,wavelength_provider=lambda:source.reference_wavelengths,output_root=Path(os.environ['TMPDIR']))
        self.addCleanup(win.close)
        win.mode_combo.setCurrentIndex(1)
        win.centralWidget().resize(1070,570)
        self.app.processEvents()
        self.assertLessEqual(win.centralWidget().minimumSizeHint().height(),570, 'all capture controls/counts/model must fit the 1366 desktop viewport')
        self.assertLessEqual(win.surface_map.minimumHeight(),350)
        self.assertLessEqual(win.force_vector.minimumWidth(),220)
        self.assertLessEqual(win.combo_counts_table.maximumHeight(),100)

    def test_direction_tags_follow_projected_model_not_whole_panel(self):
        source=FakeTemporaryPage()
        win=ManualStressCaptureWindow(source,wavelength_provider=lambda:source.reference_wavelengths,output_root=Path(os.environ['TMPDIR']))
        self.addCleanup(win.close)
        win.resize(1642,1400);win.show();self.app.processEvents()
        surface=win.surface_map
        tags={w.text():w for w in surface.findChildren(QtWidgets.QLabel)}
        self.assertEqual(set(tags),{'指尖','指腹','指根'}, 'directions belong to the model column, with a tactile-side label')
        bounds=surface.direction_model_rect
        self.assertLess(tags['指尖'].geometry().bottom(),bounds.top())
        self.assertGreater(tags['指根'].geometry().top(),bounds.bottom())
        self.assertGreater(tags['指腹'].geometry().left(),bounds.right())
        for tag in tags.values():
            self.assertTrue(surface.rect().contains(tag.geometry()))
            self.assertTrue(tag.testAttribute(QtCore.Qt.WA_TransparentForMouseEvents))
        self.assertTrue(win.wavelength_context_label.wordWrap())
        self.assertNotIn('P1…P9',win.wavelength_status.text())

if __name__=='__main__': unittest.main()
