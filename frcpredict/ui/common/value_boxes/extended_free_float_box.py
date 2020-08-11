from typing import Union

from PyQt5.QtCore import pyqtSignal

from frcpredict.model import Multivalue
from .base_extended_value_box import BaseExtendedValueBox
from .free_float_box import FreeFloatBox


class ExtendedFreeFloatBox(BaseExtendedValueBox[float]):
    """
    A partial wrapper around FreeFloatBox with the additional features of BaseExtendedValueBox.
    """

    # Signals
    valueChanged = pyqtSignal([float], [Multivalue])
    valueChangedByUser = pyqtSignal([float], [Multivalue])
    returnPressed = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        freeFloatBox = FreeFloatBox()
        super().__init__(freeFloatBox, *args,
                         widgetValueGetter=freeFloatBox.value,
                         widgetValueSetter=freeFloatBox.setValue,
                         widgetTextSetter=freeFloatBox.setText,
                         **kwargs)

        # Connect forwarded signals
        self.editValue.valueChanged[float].connect(self.valueChanged)
        self.editValue.valueChangedByUser[float].connect(self.valueChangedByUser)
        self.editValue.returnPressed.connect(self.returnPressed)

    def selectAll(self) -> None:
        self.editValue.selectAll()

    def isValid(self) -> bool:
        """ Returns whether the box contains a valid float or multivalue. """
        return self.editValue.isValid() or isinstance(self.value(), Multivalue)

    def text(self) -> str:
        return self.editValue.text()

    def setValue(self, value: Union[float, Multivalue[float]]) -> None:
        if type(value) is not type(self.value()) or value != self.value():
            if isinstance(value, Multivalue):
                self.editValue._shouldHighlightIfInvalid = False
            else:
                self.editValue._shouldHighlightIfInvalid = True
                self.editValue._valid = False

        super().setValue(value)
