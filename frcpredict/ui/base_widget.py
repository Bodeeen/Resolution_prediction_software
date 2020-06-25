import os

from PyQt5 import uic
from PyQt5.QtCore import QFile
from PyQt5.QtWidgets import QWidget


class BaseWidget(QWidget):
    """
    A base widget that loads the UI from a UI file with the same name as the Python script (with a
    .ui extension instead of .py). Child classes should pass __file__ as py_file_path.
    """

    # Methods
    def __init__(self, py_file_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Load UI from file
        ui_file_path = f"{os.path.splitext(py_file_path)[0]}.ui"
        ui_file = QFile(ui_file_path)
        ui_file.open(QFile.ReadOnly)
        uic.loadUi(ui_file, self)
        ui_file.close()
