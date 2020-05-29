import os

from PyQt5 import uic
from PyQt5.QtCore import QFile
from PyQt5.QtWidgets import QWidget

class BaseWidget(QWidget):
    """
    A base widget that loads the UI from a specific UI file.
    """

    # Functions
    def __init__(self, py_file_path: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        
        # Load UI file
        ui_file_path = f"{os.path.splitext(py_file_path)[0]}.ui"
        ui_file = QFile(ui_file_path)
        ui_file.open(QFile.ReadOnly)
        uic.loadUi(ui_file, self)
        ui_file.close()