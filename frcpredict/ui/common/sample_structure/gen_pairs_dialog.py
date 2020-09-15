from typing import Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QDialog, QDialogButtonBox

from frcpredict.model import SampleStructure
from frcpredict.ui import BaseWidget
from frcpredict.util import make_pool_pairs_structure


class PairsStructureDialog(QDialog, BaseWidget):
    """
    A dialog for generating a fluorophore pool pairs sample structure.
    """

    # Methods
    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(__file__, parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint)

        # Prepare UI elements
        self._updateOKButton()
        self.editAreaSide.selectAll()

        # Connect signals
        self.editAreaSide.valueChanged.connect(self._updateOKButton)
        self.editNumPairs.valueChanged.connect(self._updateOKButton)
        self.editDistance.valueChanged.connect(self._updateOKButton)
        self.editFluorophores.valueChanged.connect(self._updateOKButton)

    @staticmethod
    def getStructure(parent: Optional[QWidget] = None) -> Tuple[Optional[SampleStructure], bool]:
        """
        Synchronously opens a dialog for generating the structure. The second value in the returned
        tuple refers to whether the "OK" button was pressed when the dialog closed. If it's true,
        the first value will contain the structure.
        """

        dialog = PairsStructureDialog(parent)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            areaSide = dialog.editAreaSide.value()

            relativeArrayX, relativeArrayY = make_pool_pairs_structure(
                area_side_um=areaSide,
                num_pairs=dialog.editNumPairs.value(),
                d=dialog.editDistance.value(),
                f_per_pool=dialog.editFluorophores.value(),
                poisson_labelling=dialog.chkPoisson.isChecked()
            )

            value = SampleStructure(
                area_side_um=areaSide,
                relative_array_x=relativeArrayX,
                relative_array_y=relativeArrayY,
                structure_type="pairs"
            )
        else:
            value = None

        dialog.deleteLater()  # Prevent memory leak
        return value, value is not None

    # Internal methods
    def _updateOKButton(self) -> None:
        """ Enables or disables the "OK" button depending on whether the entered data is valid. """
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(
            self.editAreaSide.isValid() and self.editAreaSide.value() > 0 and
            self.editDistance.isValid() and self.editDistance.value() > 0
        )
