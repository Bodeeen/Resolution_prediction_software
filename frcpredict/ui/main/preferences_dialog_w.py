from typing import Optional

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QWidget, QDialog

from frcpredict.ui import BaseWidget, Preferences
from .preferences_dialog_p import PreferencesPresenter


class PreferencesDialog(QDialog, BaseWidget):
    """
    Dialog for modifying persistent program preferences.
    """

    # Signals
    precacheFrcCurvesChanged = pyqtSignal(int)
    precacheExpectedImagesChanged = pyqtSignal(int)
    cacheKernels2DChanged = pyqtSignal(int)

    # Methods
    def __init__(self,  parent: Optional[QWidget] = None) -> None:
        super().__init__(__file__, parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint)

        # Connect forwarded signals
        self.chkPrecacheFrcCurves.stateChanged.connect(self.precacheFrcCurvesChanged)
        self.chkPrecacheExpectedImages.stateChanged.connect(self.precacheExpectedImagesChanged)
        self.chkCacheKernels2D.stateChanged.connect(self.cacheKernels2DChanged)

        # Initialize presenter
        self._presenter = PreferencesPresenter(self)

    def value(self) -> Preferences:
        return self._presenter.model

    def setValue(self, model: Preferences) -> None:
        self._presenter.model = model

    def updateBasicFields(self, model: Preferences) -> None:
        self.chkPrecacheFrcCurves.setChecked(model.precacheFrcCurves)
        self.chkPrecacheExpectedImages.setChecked(model.precacheExpectedImages)
        self.chkCacheKernels2D.setChecked(model.cacheKernels2D)

    @staticmethod
    def display(parent: Optional[QWidget] = None) -> None:
        """ Synchronously opens the preferences dialog. """
        dialog = PreferencesDialog(parent)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            Preferences.save(dialog.value())
