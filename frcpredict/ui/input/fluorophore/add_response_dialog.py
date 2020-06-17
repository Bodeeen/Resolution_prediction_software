from typing import Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QDialog, QDialogButtonBox

from frcpredict.model import IlluminationResponse
from frcpredict.ui import BaseWidget
from frcpredict.util import with_cleared_signals


class AddResponseDialog(QDialog, BaseWidget):
    """
    A dialog for adding an illumination response.
    """

    # Methods
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(__file__, parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint)
        self._updateOKButton()
        self.editProperties.editWavelength.selectAll()

        # Connect signals
        self.editProperties.editWavelength.valueChanged.connect(self._updateOKButton)
        self.editProperties.editOffToOn.valueChanged.connect(self._updateOKButton)
        self.editProperties.editOnToOff.valueChanged.connect(self._updateOKButton)
        self.editProperties.editEmission.valueChanged.connect(self._updateOKButton)

    @staticmethod
    def getResponse(parent: Optional[QWidget] = None) -> Tuple[Optional[IlluminationResponse], bool]:
        """
        Synchronously opens a dialog for entering illumination response properties. The second
        value in the returned tuple refers to whether the "OK" button was pressed when the dialog
        closed. If it's true, the first value will contain the illumination response.
        """

        dialog = AddResponseDialog(parent)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            response = with_cleared_signals(dialog.editProperties.value())
        else:
            response = None

        dialog.deleteLater()  # Prevent memory leak
        return response, result == QDialog.Accepted

    # Internal methods
    def _updateOKButton(self) -> None:
        """ Enables or disables the "OK" button depending on whether the entered data is valid. """
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(
            self.editProperties.editWavelength.value() > 0 and
            self.editProperties.editOffToOn.isValid() and
            self.editProperties.editOnToOff.isValid() and
            self.editProperties.editEmission.isValid()
        )
