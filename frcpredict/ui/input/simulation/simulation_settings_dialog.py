from copy import deepcopy
from typing import Optional, Tuple

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QDialog, QDialogButtonBox

from frcpredict.model import SimulationSettings
from frcpredict.ui import BaseWidget
from frcpredict.util import clear_signals


class SimulationSettingsDialog(QDialog, BaseWidget):
    """
    A dialog for adding an illumination response.
    """

    # Methods
    def __init__(self, parent: Optional[QWidget] = None,
                 initialValue: Optional[SimulationSettings] = None) -> None:
        super().__init__(__file__, parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint)
        self._updateOKButton()

        if initialValue is not None:
            self.editSettings.setValue(clear_signals(deepcopy(initialValue)))

        # Connect signals
        self.editSettings.editCanvasSideLength.valueChanged.connect(self._updateOKButton)
        self.editSettings.editNumDetectionIterations.valueChanged.connect(self._updateOKButton)

    @staticmethod
    def getSettings(parent: Optional[QWidget] = None,
                    initialValue: Optional[SimulationSettings] = None) -> Tuple[Optional[SimulationSettings], bool]:
        """
        Synchronously opens a dialog for entering simulation settings. The second value in the
        returned tuple refers to whether the "OK" button was pressed when the dialog closed. If it's
        true, the first value will contain the simulation settings.
        """

        dialog = SimulationSettingsDialog(parent, initialValue)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            response = clear_signals(dialog.editSettings.value())
        else:
            response = None

        dialog.deleteLater()  # Prevent memory leak
        return response, result == QDialog.Accepted

    # Internal methods
    def _updateOKButton(self) -> None:
        """ Enables or disables the "OK" button depending on whether the entered data is valid. """
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(
            self.editSettings.editCanvasSideLength.isValid() and
            self.editSettings.editCanvasSideLength.value() > 0 and
            self.editSettings.editNumDetectionIterations.value() > 0
        )
