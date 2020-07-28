from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QLineEdit


class FreeFloatBox(QLineEdit):
    """
    A box where the user can freely type float values. The values entered are validated. E notation
    is allowed. Both dots and commas are allowed as decimal separators.
    """

    # Signals
    valueChanged = pyqtSignal(float)
    valueChangedByUser = pyqtSignal(float)

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        self._value = 0.0
        self._valid = True
        self._shouldHighlightIfInvalid = True

        super().__init__(*args, **kwargs)

        # Attempt to ensure that the box doesn't become smaller when we set a stylesheet
        self.setMinimumHeight(20)

        # Update the text
        self._onFinishEditing()

        # Connect signals
        self.textChanged.connect(self._onChange)
        self.editingFinished.connect(self._onFinishEditing)

    def isValid(self) -> bool:
        """ Returns whether the field contains a valid float value. """
        return self._valid

    def value(self) -> float:
        """ Returns the value of the field. """
        return self._value

    def setValue(self, value: float) -> None:
        """ Sets the value of the field. """
        if type(value) is type(self._value) and value == self._value and self._valid:
            return

        self._value = value

        self.blockSignals(True)  # Ensure that change event isn't triggered automatically
        self.setText(str(value))
        self.blockSignals(False)
        self._onChange(str(value), changedByUser=False)  # Trigger change event manually

    # Event handling
    @pyqtSlot(str)
    def _onChange(self, strValue: str, changedByUser: bool = True) -> None:
        if len(strValue) < 1:
            strValue = "0"  # Assume empty string == 0

        try:
            self._value = float(strValue.replace(",", "."))
            self._valid = True

            self.valueChanged.emit(self._value)
            if changedByUser:
                self.valueChangedByUser.emit(self._value)

            self.setStyleSheet("")  # Remove red border if it was there
        except ValueError:
            self._valid = False
            if self._shouldHighlightIfInvalid and strValue != "-":
                try:
                    float(strValue.replace(",", ".").rstrip(".e+-"))
                except ValueError:
                    self.setStyleSheet("border: 1px solid red")  # Red border if invalid value

    @pyqtSlot()
    def _onFinishEditing(self) -> None:
        if self.isReadOnly():
            return

        # Auto-format e.g. "0.00001" to "1e-05" when the user leaves the box
        self.setText(str(self._value))
