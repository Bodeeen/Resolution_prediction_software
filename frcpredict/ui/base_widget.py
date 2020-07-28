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
    def __init__(self, pyFilePath: str, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        # Load UI from file
        uiFilePath = f"{os.path.splitext(pyFilePath)[0]}.ui"
        uiFile = QFile(uiFilePath)
        uiFile.open(QFile.ReadOnly)
        uic.loadUi(uiFile, self)
        uiFile.close()

        self._uiLoaded = True

    def isUiLoaded(self) -> bool:
        try:
            return self._uiLoaded
        except AttributeError:
            return False  # UI loaded flag not yet set, which means the UI isn't loaded
