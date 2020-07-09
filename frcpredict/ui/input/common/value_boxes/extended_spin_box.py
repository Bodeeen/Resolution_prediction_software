from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QSpinBox

from frcpredict.model import Multivalue
from .base_extended_value_box import BaseExtendedValueBox


class ExtendedSpinBox(BaseExtendedValueBox[int]):
    """
    A partial wrapper around QSpinBox with the additional features of BaseExtendedValueBox.
    """

    # Signals
    valueChanged = pyqtSignal([int], [Multivalue])
    returnPressed = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        spinbox = QSpinBox(minimum=0, maximum=1000000)
        super().__init__(spinbox, *args,
                         widgetValueGetter=spinbox.value,
                         widgetValueSetter=spinbox.setValue,
                         widgetTextSetter=spinbox.setSpecialValueText,
                         **kwargs)

        # Connect forwarded signals
        self.editValue.valueChanged[int].connect(self.valueChanged)

    def selectAll(self) -> None:
        self.editValue.selectAll()

    # Event handling
    def keyPressEvent(self, event):
        """ Overriden QSpinBox function; emit returnPressed on pressing return/enter. """
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.returnPressed.emit()

        super().keyPressEvent(event)
