from typing import Union

from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtWidgets import QSpinBox

from frcpredict.model import Multivalue
from .base_extended_value_box import BaseExtendedValueBox


class ExtendedSpinBox(BaseExtendedValueBox[int]):
    """
    A partial wrapper around QSpinBox with the additional features of BaseExtendedValueBox.
    """

    # Signals
    valueChanged = pyqtSignal([int], [Multivalue])
    valueChangedByUser = pyqtSignal([int], [Multivalue])
    returnPressed = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        spinBox = QSpinBox(minimum=0, maximum=1000000)
        super().__init__(spinBox, *args,
                         widgetValueGetter=spinBox.value,
                         widgetValueSetter=spinBox.setValue,
                         widgetTextSetter=self._setText,
                         **kwargs)

        self._value = spinBox.value()

        # Connect own signal slots
        self.editValue.valueChanged[int].connect(self._onChange)

        # Connect forwarded signals
        self.editValue.valueChanged[int].connect(self.valueChanged)

    def selectAll(self) -> None:
        self.editValue.selectAll()

    def setValue(self, value: Union[int, Multivalue[int]]) -> None:
        """ Sets the value of the field. """

        if type(value) is type(self._value) and value == self._value:
            return

        self._value = value

        if isinstance(value, int):
            self.blockSignals(True)  # Ensure that change event isn't triggered automatically
            self.editValue.setSpecialValueText("")
            self.editValue.setValue(value)
            self.blockSignals(False)
            self.valueChanged[int].emit(value)  # Trigger change event manually

        super().setValue(value)

    # Internal methods
    def _setText(self, text: str) -> None:
        self.blockSignals(True)  # Ensure that change event isn't triggered
        self.editValue.setSpecialValueText(text)
        self.editValue.setValue(self.editValue.minimum())
        self.blockSignals(False)

    # Event handling
    def keyPressEvent(self, event):
        """ Overriden QSpinBox function; emit returnPressed on pressing return/enter. """
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.returnPressed.emit()

        super().keyPressEvent(event)

    @pyqtSlot(int)
    def _onChange(self, value: int, changedByUser: bool = True) -> None:
        if type(value) is type(self._value) and value == self._value:
            return

        self._value = value
        self.valueChanged[int].emit(value)
        if changedByUser:
            self.valueChangedByUser[int].emit(value)
