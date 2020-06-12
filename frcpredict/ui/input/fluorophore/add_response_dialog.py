from typing import Optional, Tuple

from PyQt5.QtWidgets import QDialog, QDialogButtonBox

from frcpredict.model import IlluminationResponse
from frcpredict.util import patterns
from frcpredict.ui import BaseDialog


class AddResponseDialog(BaseDialog):
    """
    A dialog for adding an illumination response.
    """

    # Methods
    def __init__(self, parent=None, *args, **kwargs) -> None:
        super().__init__(__file__, parent, *args, **kwargs)
        self._updateOKButton()
        self.editProperties.editWavelength.selectAll()

        # Connect signals
        self.editProperties.editWavelength.valueChanged.connect(self._updateOKButton)
        self.editProperties.editOffToOn.valueChanged.connect(self._updateOKButton)
        self.editProperties.editOnToOff.valueChanged.connect(self._updateOKButton)
        self.editProperties.editEmission.valueChanged.connect(self._updateOKButton)

    @staticmethod
    def getResponse(parent=None) -> Tuple[Optional[IlluminationResponse], bool]:
        """
        Synchronously opens a dialog for entering illumination response properties. The second
        value in the returned tuple refers to whether the "OK" button was pressed when the dialog
        closed.
        """

        dialog = AddResponseDialog(parent)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            response = IlluminationResponse(
                wavelength_start=dialog.editProperties.editWavelength.value(),
                wavelength_end=dialog.editProperties.editWavelength.value(),
                cross_section_off_to_on=dialog.editProperties.editOffToOn.value(),
                cross_section_on_to_off=dialog.editProperties.editOnToOff.value(),
                cross_section_emission=dialog.editProperties.editEmission.value()
            )
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
