from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QDialog

import frcpredict
from frcpredict.ui import BaseWidget


class PreferencesDialog(QDialog, BaseWidget):
    """
    Dialog for modifying persistent program preferences. TODO: Add logic.
    """

    # Methods
    def __init__(self,  parent: Optional[QWidget] = None) -> None:
        super().__init__(__file__, parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint)

    @staticmethod
    def display(parent: Optional[QWidget] = None) -> None:
        """ Synchronously opens the preferences dialog. """
        dialog = PreferencesDialog(parent)
        dialog.exec_()
