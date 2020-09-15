from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QDialog

import frcpredict
from frcpredict.ui import BaseWidget


class AboutDialog(QDialog, BaseWidget):
    """
    About dialog for the program, showing version etc.
    """

    # Methods
    def __init__(self,  parent: Optional[QWidget] = None) -> None:
        super().__init__(__file__, parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint)

        # Prepare UI elements
        self.lblTitle.setText(frcpredict.__summary__)
        self.lblVersion.setText(f"Version {frcpredict.__version__}")

    @staticmethod
    def display(parent: Optional[QWidget] = None) -> None:
        """ Synchronously opens the about dialog. """
        dialog = AboutDialog(parent)
        dialog.exec_()
