import os

from PyQt5 import uic
from PyQt5.QtWidgets import QMainWindow
from PyQt5.QtCore import QFile

from frcpredict.model import RunInstance


class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self._loadUi()

    def _loadUi(self) -> None:
        path = os.path.join(os.path.dirname(__file__), "main_window.ui")
        ui_file = QFile(path)
        ui_file.open(QFile.ReadOnly)
        uic.loadUi(ui_file, self)
        ui_file.close()
