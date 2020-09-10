from typing import Optional

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
        self._staticText = None

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

    def setStaticText(self, text: Optional[str]) -> None:
        """
        Sets the displayed value to a static text string. In most use cases, this should be combined
        with setEnabled(False). Note that this doesn't affect the returned value from value(). Pass
        None as the argument to restore the previous value.
        """
        if text == self._staticText:
            return

        self._staticText = text
        if text is not None:
            self._setRedBorderVisible(False)
            self.setText(text)
        else:
            self.setValue(self._value, forceUpdate=True)

    def value(self) -> float:
        """ Returns the value of the field. """
        return self._value

    def setValue(self, value: Optional[float], forceUpdate: bool = False) -> None:
        """ Sets the value of the field. """
        if value is None or (not forceUpdate
                             and type(value) is type(self._value)
                             and value == self._value and self._valid):
            return

        self._value = value

        if self._staticText is not None:
            return

        self.blockSignals(True)  # Ensure that change event isn't triggered automatically
        self.setText(str(value))
        self.blockSignals(False)
        self._onChange(str(value), changedByUser=False)  # Trigger change event manually

    # Internal methods
    def _setRedBorderVisible(self, visible: bool) -> None:
        self.setStyleSheet("border: 1px solid red" if visible else "")

    # Event handling
    @pyqtSlot(str)
    def _onChange(self, strValue: str, changedByUser: bool = True) -> None:
        if self._staticText is not None:
            return

        if len(strValue) < 1:
            strValue = "0"  # Assume empty string == 0

        try:
            self._value = float(strValue.replace(",", "."))
            self._valid = True

            self.valueChanged.emit(self._value)
            if changedByUser:
                self.valueChangedByUser.emit(self._value)

            self._setRedBorderVisible(False)  # Remove red border if it was there
        except ValueError:
            self._valid = False
            if strValue != "-":
                try:
                    float(strValue.replace(",", ".").rstrip(".e+-"))
                except ValueError:
                    self._setRedBorderVisible(True)  # Red border if invalid value

    @pyqtSlot()
    def _onFinishEditing(self) -> None:
        if self.isReadOnly():
            return

        # Auto-format e.g. "0.00001" to "1e-05" when the user leaves the box
        self.setText(str(self._value))
