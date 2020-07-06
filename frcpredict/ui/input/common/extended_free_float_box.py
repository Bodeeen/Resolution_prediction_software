from typing import Union

from PyQt5.QtCore import pyqtSignal, pyqtSlot, pyqtProperty, Qt
from PyQt5.QtWidgets import QMessageBox

from frcpredict.model import ValueRange
from frcpredict.ui import BaseWidget
from .set_range_dialog import SetRangeDialog
import frcpredict.ui.resources


class ExtendedFreeFloatBox(BaseWidget):
    """
    A wrapper around FreeFloatBox that contains additional features:

    * Allows for setting ranged values
    * If the infoText property is set, adds a button that shows information about the field when
      clicked.
    """

    # Properties
    @pyqtProperty(str)
    def infoText(self) -> str:
        return self._infoText

    @infoText.setter
    def infoText(self, value: str) -> None:
        self._infoText = value
        self.btnInfo.setVisible(bool(value))

    # Signals
    valueChanged = pyqtSignal([float], [ValueRange])
    valueChangedByUser = pyqtSignal([float], [ValueRange])

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        self._rangeValue = None
        self._infoText = ""
        super().__init__(__file__, *args, **kwargs)

        # Prepare UI elements
        self.btnInfo.setVisible(False)

        # Connect own signal slots
        self.btnSetRange.clicked.connect(self._onClickSetRange)
        self.btnInfo.clicked.connect(self._onClickInfo)

        # Connect forwarded signals
        self.editValue.valueChanged[float].connect(self.valueChanged)
        self.editValue.valueChangedByUser[float].connect(self.valueChangedByUser)

    def isValid(self) -> bool:
        """ Returns whether the field contains a valid float or range value. """
        return self.editValue.isValid() or isinstance(self.value(), ValueRange)

    def value(self) -> Union[float, ValueRange[float]]:
        """ Returns the value of the field. """
        return self._rangeValue if self._rangeValue is not None else self.editValue.value()

    def setValue(self, value: Union[float, ValueRange[float]]) -> None:
        """ Sets the value of the field. """
        if type(value) is not type(self.value()) or value != self.value():
            valueIsRange = isinstance(value, ValueRange)

            self.editValue._shouldHighlightIfInvalid = not valueIsRange
            self.editValue.setEnabled(not valueIsRange)

            if valueIsRange:
                self._rangeValue = value
                self.editValue.blockSignals(True)  # Ensure that change event isn't triggered automatically
                self.editValue.setText(str(value))
                self.editValue.blockSignals(False)
                self.valueChanged[ValueRange].emit(value)
            else:
                self._rangeValue = None
                self.editValue._valid = False
                self.editValue.setValue(value)

    # Event handling
    @pyqtSlot()
    def _onClickSetRange(self) -> None:
        """
        Opens a dialog for the user to configure a value range, and then sets the value to that
        range.
        """

        valueRange, okClicked = SetRangeDialog.getRange(self, self.value())
        if okClicked:
            self.setValue(valueRange)

    @pyqtSlot()
    def _onClickInfo(self) -> None:
        """ Shows information about the field (e.g. an explanation of it). """

        infoBox = QMessageBox(QMessageBox.NoIcon, "Field Information", self._infoText, parent=self)
        infoBox.setAttribute(Qt.WA_DeleteOnClose)  # Prevent memory leak
        infoBox.exec_()
