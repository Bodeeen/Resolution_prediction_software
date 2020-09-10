from typing import Optional, Tuple

import numpy as np
from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtWidgets import QWidget, QDialog, QDialogButtonBox

from frcpredict.model import SampleImage, Array2DPatternData
from frcpredict.ui import BaseWidget


class SampleFromFileDialog(QDialog, BaseWidget):
    """
    A dialog for fine-tuning sample images that have been loaded from files.
    """

    # Methods
    def __init__(self, parent: Optional[QWidget] = None,
                 rawFileValues: np.ndarray = np.zeros(())) -> None:
        self._rawFileValues = rawFileValues
        super().__init__(__file__, parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint)

        # Prepare UI elements
        self._updateOKButton()
        self.rdoValuesRaw.setChecked(True)
        self.editMaxValue.setValue(rawFileValues.max())

        # Connect signals
        self.rdoValuesRaw.toggled.connect(self._onRawValuesToggle)
        self.rdoValuesCustom.toggled.connect(self._onCustomValuesToggle)
        self.editAreaSide.valueChanged.connect(self._updateOKButton)

    @staticmethod
    def getSampleImage(parent: Optional[QWidget] = None,
                       rawFileValues: np.ndarray = np.zeros(())) -> Tuple[Optional[SampleImage], bool]:
        """
        Synchronously opens a dialog for fine-tuning sample image parameters. The second value in
        the returned tuple refers to whether the "OK" button was pressed when the dialog closed. If
        it's true, the first value will contain the sample image.
        """

        dialog = SampleFromFileDialog(parent, rawFileValues)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            value = SampleImage(
                area_side_um=dialog.editAreaSide.value(),
                image=Array2DPatternData(
                    value=(rawFileValues if not dialog.rdoValuesCustom.isChecked()
                           else rawFileValues / rawFileValues.max() * dialog.editMaxValue.value())
                )
            )
        else:
            value = None

        dialog.deleteLater()  # Prevent memory leak
        return value, value is not None

    # Internal methods
    def _updateOKButton(self) -> None:
        """ Enables or disables the "OK" button depending on whether the entered data is valid. """
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(
            self.editAreaSide.isValid() and self.editAreaSide.value() > 0
        )

    # Event handling
    @pyqtSlot(bool)
    def _onRawValuesToggle(self, enabled: bool) -> None:
        if enabled:
            self.editMaxValue.setEnabled(False)
            self.editMaxValue.setValue(self._rawFileValues.max())

    @pyqtSlot(bool)
    def _onCustomValuesToggle(self, enabled: bool) -> None:
        if enabled:
            self.editMaxValue.setEnabled(True)
