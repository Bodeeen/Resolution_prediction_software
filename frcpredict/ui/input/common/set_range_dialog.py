from typing import Optional, Union, Tuple

from PyQt5.QtCore import pyqtSlot, Qt
from PyQt5.QtWidgets import QWidget, QDialog, QDialogButtonBox

from frcpredict.model import RangeType, ValueRange
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import getEnumEntryName


class SetRangeDialog(QDialog, BaseWidget):
    """
    A dialog for setting a value range.
    """

    # Methods
    def __init__(self, parent: Optional[QWidget] = None,
                 initialValue: Union[float, ValueRange[float]] = 0.0) -> None:
        self._noRange = False
        super().__init__(__file__, parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint)

        # Prepare UI elements
        noRangeButton = self.buttonBox.addButton("Remove Range", QDialogButtonBox.ResetRole)

        for rangeType in RangeType:
            self.editRangeType.addItem(getEnumEntryName(rangeType), rangeType)

        if isinstance(initialValue, ValueRange):
            self.editMinimum.setValue(initialValue.start)
            self.editMaximum.setValue(initialValue.end)
            self.editNumEvaluations.setValue(initialValue.num_evaluations)
            self.editRangeType.setCurrentText(getEnumEntryName(initialValue.range_type))

            noRangeButton.setEnabled(True)
            noRangeButton.clicked.connect(self._onClickNoRange)
        else:
            noRangeButton.setEnabled(False)

        self._updateOKButton()
        self.editMinimum.selectAll()

        # Connect signals
        self.editMinimum.valueChanged.connect(self._updateOKButton)
        self.editMaximum.valueChanged.connect(self._updateOKButton)

    @staticmethod
    def getRange(parent: Optional[QWidget] = None,
                 initialValue: Union[float, ValueRange[float]] = 0.0) -> Tuple[Optional[ValueRange], bool]:
        """
        Synchronously opens a dialog for setting a value range. The second value in the returned
        tuple refers to whether the "OK" button was pressed when the dialog closed. If it's true,
        the first value will contain the value range.
        """

        dialog = SetRangeDialog(parent, initialValue)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            if not dialog._noRange:
                value = ValueRange(
                    start=dialog.editMinimum.value(),
                    end=dialog.editMaximum.value(),
                    num_evaluations=dialog.editNumEvaluations.value(),
                    range_type=dialog.editRangeType.itemData(dialog.editRangeType.currentIndex()),
                )
            elif isinstance(initialValue, ValueRange):
                value = 0.0
            else:
                value = None
        else:
            value = None

        dialog.deleteLater()  # Prevent memory leak
        return value, value is not None

    # Internal methods
    def _updateOKButton(self) -> None:
        """ Enables or disables the "OK" button depending on whether the entered data is valid. """
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(
            self.editMinimum.isValid() and
            self.editMaximum.isValid() and
            self.editMinimum.value() < self.editMaximum.value()
        )

    # Event handling
    @pyqtSlot()
    def _onClickNoRange(self) -> None:
        self._noRange = True
        self.accept()
