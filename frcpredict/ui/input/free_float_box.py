from PyQt5.QtCore import pyqtSignal, pyqtSlot
from PyQt5.QtWidgets import QLineEdit


class FreeFloatBox(QLineEdit):
    """
    A box where the user can freely type float values. The values entered are validated. E notation
    is allowed.
    """

    # Signals
    valueChanged = pyqtSignal(float)

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._value = 0.0
        self._valid = True
        self._onFinishEditing()  # Update the text

        # Connect signals
        self.textChanged.connect(self._onChange)
        self.editingFinished.connect(self._onFinishEditing)

    def value(self) -> float:
        """ Returns the value of the field. """
        return self._value

    def setValue(self, value: float) -> None:
        """ Sets the value of the field. """
        if value != self._value:
            self._value = value
            self.setText(str(value))

    def isValid(self) -> bool:
        """ Returns whether the field contains a valid float. """
        return self._valid

    # Event handling
    @pyqtSlot(str)
    def _onChange(self, str_value: str) -> None:
        if len(str_value) < 1:
            str_value = "0"  # Assume empty string == 0

        try:
            self._value = float(str_value.replace(",", ".").rstrip("-.e"))
            self._valid = True
            self.valueChanged.emit(self._value)
            self.setStyleSheet("")  # Remove red border if it was there
        except ValueError:
            self._valid = False
            self.setStyleSheet("border: 1px solid red")

    @pyqtSlot()
    def _onFinishEditing(self) -> None:
        # Auto-format e.g. "0.00001" to "1e-05" when the user leaves the box
        self.setText(str(self._value))
