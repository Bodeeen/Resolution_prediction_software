import numpy as np
from typing import Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QDialog, QDialogButtonBox

from frcpredict.model import Pulse
from frcpredict.util import patterns
from frcpredict.ui import BaseWidget


class AddPulseDialog(QDialog, BaseWidget):
    """
    A dialog for adding a pulse.
    """

    # Methods
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(__file__, parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint)
        self._updateOKButton()
        self.editProperties.setChangeOrderVisible(False)
        self.editProperties.editWavelength.selectAll()

        # Connect signals
        self.editProperties.editWavelength.valueChanged.connect(self._updateOKButton)
        self.editProperties.editDuration.valueChanged.connect(self._updateOKButton)
        self.editProperties.editMaxIntensity.valueChanged.connect(self._updateOKButton)
        self.editProperties.listPatterns.currentRowChanged.connect(self._updateOKButton)

    @staticmethod
    def getPulse(parent: Optional[QWidget] = None) -> Tuple[Optional[Pulse], bool]:
        """
        Synchronously opens a dialog for entering pulse properties. The second value in the
        returned tuple refers to whether the "OK" button was pressed when the dialog closed. If
        it's true, the first value will contain the pulse.
        """

        dialog = AddPulseDialog(parent)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            pulse = Pulse(
                wavelength=dialog.editProperties.editWavelength.value(),
                duration=dialog.editProperties.editDuration.value(),
                max_intensity=dialog.editProperties.editMaxIntensity.value(),
                illumination_pattern=patterns[
                    dialog.editProperties.listPatterns.currentItem().text()
                ]
            )
        else:
            pulse = None

        dialog.deleteLater()  # Prevent memory leak
        return pulse, result == QDialog.Accepted

    # Internal methods
    def _updateOKButton(self) -> None:
        """ Enables or disables the "OK" button depending on whether the entered data is valid. """
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(
            self.editProperties.editWavelength.value() > 0 and
            self.editProperties.editDuration.isValid() and
            self.editProperties.editDuration.value() > 0 and
            self.editProperties.editMaxIntensity.isValid() and
            self.editProperties.listPatterns.currentRow() > -1
        )
