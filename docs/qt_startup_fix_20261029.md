# Windows Qt平台插件初始化

本机的项目目录含中文。当前PyQt5的`QLibraryInfo.PluginsPath`自动路径会将中文变为问号，导致独立GUI脚本创建`QApplication`时找不到平台插件；继承无效的`QT_QPA_PLATFORM_PLUGIN_PATH`也会触发同一错误。

共用入口为`Python/qt_runtime.py`。它从当前解释器实际安装的PyQt5包定位插件，修正进程内的`QT_PLUGIN_PATH`、`QT_QPA_PLATFORM_PLUGIN_PATH`和Qt库搜索路径。不修改系统环境、Python安装、`QT_QPA_PLATFORM`或`QT_OPENGL`。

正式启动器、单独训练页与采集页已调用此入口。今后新增界面验证脚本必须先初始化，再创建应用：

```python
from qt_runtime import configure_qt
configure_qt()
from PyQt5 import QtWidgets
app = QtWidgets.QApplication([])
```

脚本不在`Python`目录时，先将项目`Python`目录加入模块路径。不要仅依赖Qt自动路径，也不要用`setdefault`保留无效插件目录。不要在初始化之前创建`QApplication`。

验证：`Python/.venv/Scripts/python.exe -B -m unittest test_qt_startup_environment -v`（工作目录为`Python`）。另以正式启动器、Windows平台、故意损坏的继承插件路径实测打开并关闭demo窗口。未修改或烧录固件；未重建客户安装包。
