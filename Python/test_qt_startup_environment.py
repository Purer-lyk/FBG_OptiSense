import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path


APP_DIR = Path(__file__).resolve().parent


class QtStartupEnvironmentTests(unittest.TestCase):
    def test_stale_platform_plugin_path_is_replaced_before_qapplication(self):
        script = textwrap.dedent(
            """
            import importlib
            import os
            import sys
            from pathlib import Path

            importlib.import_module(sys.argv[1])
            expected = (
                Path(sys.prefix) / "Lib" / "site-packages" / "PyQt5" /
                "Qt5" / "plugins" / "platforms"
            ).resolve()
            actual = Path(os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"]).resolve()
            if actual != expected:
                raise RuntimeError(f"plugin path was not repaired: {actual} != {expected}")
            from PyQt5 import QtWidgets
            app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
            print("qt-startup-ok")
            """
        )
        for module_name in (
            "app_JDSU",
            "fbg_unified_app",
            "fullband_calibration_progress",
            "manual_touch_training_page",
            "manual_stress_capture",
        ):
            with self.subTest(module=module_name):
                environment = os.environ.copy()
                environment["QT_QPA_PLATFORM"] = "offscreen"
                environment["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(
                    APP_DIR / "definitely-not-a-qt-plugin-directory"
                )
                result = subprocess.run(
                    [sys.executable, "-c", script, module_name],
                    cwd=APP_DIR,
                    env=environment,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                    timeout=30,
                    check=False,
                )
                self.assertEqual(
                    result.returncode,
                    0,
                    f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}",
                )
                self.assertIn("qt-startup-ok", result.stdout)

    def test_shared_bootstrap_repairs_both_paths_and_preserves_render_modes(self):
        script = textwrap.dedent(
            """
            import os
            from qt_runtime import configure_qt
            root = configure_qt()
            assert root is not None and root.is_dir()
            assert os.environ['QT_PLUGIN_PATH'] == str(root)
            assert os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] == str(root / 'platforms')
            assert os.environ['QT_QPA_PLATFORM'] == 'offscreen'
            assert os.environ['QT_OPENGL'] == 'software'
            from PyQt5.QtCore import QCoreApplication
            assert QCoreApplication.libraryPaths() == [str(root)]
            assert configure_qt() == root
            from PyQt5.QtWidgets import QApplication
            app = QApplication([])
            assert app.platformName() == 'offscreen'
            print('bootstrap-ok')
            """
        )
        env = os.environ.copy()
        env.update(QT_PLUGIN_PATH='nonexistent-plugins',
                   QT_QPA_PLATFORM_PLUGIN_PATH='nonexistent-platforms',
                   QT_QPA_PLATFORM='offscreen', QT_OPENGL='software')
        result = subprocess.run([sys.executable, '-c', script], cwd=APP_DIR,
                                env=env, capture_output=True, text=True,
                                encoding='utf-8', errors='replace', timeout=30)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn('bootstrap-ok', result.stdout)


if __name__ == "__main__":
    unittest.main()
