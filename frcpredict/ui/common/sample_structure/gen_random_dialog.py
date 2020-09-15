from typing import Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QDialog, QDialogButtonBox

from frcpredict.model import SampleStructure
from frcpredict.ui import BaseWidget
from frcpredict.util import make_random_positions_structure


class RandomPointsStructureDialog(QDialog, BaseWidget):
    """
    A dialog for generating a random points sample structure.
    """

    # Methods
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(__file__, parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint)

        # Prepare UI elements
        self._updateOKButton()
        self.editAreaSide.selectAll()

        # Connect signals
        self.editAreaSide.valueChanged.connect(self._updateOKButton)
        self.editFluorophores.valueChanged.connect(self._updateOKButton)

    @staticmethod
    def getStructure(parent: Optional[QWidget] = None) -> Tuple[Optional[SampleStructure], bool]:
        """
        Synchronously opens a dialog for generating the structure. The second value in the returned
        tuple refers to whether the "OK" button was pressed when the dialog closed. If it's true,
        the first value will contain the structure.
        """

        dialog = RandomPointsStructureDialog(parent)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            areaSide = dialog.editAreaSide.value()

            relativeArrayX, relativeArrayY = make_random_positions_structure(
                area_side_um=areaSide,
                f_per_um2=dialog.editFluorophores.value()
            )

            value = SampleStructure(
                area_side_um=areaSide,
                relative_array_x=relativeArrayX,
                relative_array_y=relativeArrayY,
                structure_type="random"
            )
        else:
            value = None

        dialog.deleteLater()  # Prevent memory leak
        return value, value is not None

    # Internal methods
    def _updateOKButton(self) -> None:
        """ Enables or disables the "OK" button depending on whether the entered data is valid. """
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(
            (self.editAreaSide.isValid() and self.editAreaSide.value() > 0 and
             self.editFluorophores.isValid() and self.editFluorophores.value() >= 0)
        )
