from abc import abstractmethod
from typing import Union, TypeVar, Generic, Callable

from PyQt5.QtCore import pyqtBoundSignal, pyqtSlot, pyqtProperty, Qt
from PyQt5.QtWidgets import QWidget, QMessageBox, QMenu

from frcpredict.model import Multivalue
from frcpredict.ui import BaseWidget
from .set_value_list_dialog_w import SetValueListDialog
from .set_value_range_dialog import SetValueRangeDialog
import frcpredict.ui.resources

T = TypeVar("T")


class BaseExtendedValueBox(BaseWidget, Generic[T]):
    """
    A wrapper around an value input box (which type of box is decided by the deriving class) with
    additional features:

    * Allows for setting a value list or range
    * If the infoText property is set, adds a button that shows information about the field when
      clicked.
    * returnPressed signal
    """

    # Properties
    @pyqtProperty(str)
    def infoText(self) -> str:
        return self._infoText

    @infoText.setter
    def infoText(self, value: str) -> None:
        self._infoText = value
        if self.isUiLoaded():
            self.btnInfo.setVisible(bool(value))

    @pyqtProperty(bool)
    def allowSetList(self) -> bool:
        return self._allowSetList

    @allowSetList.setter
    def allowSetList(self, value: bool) -> None:
        self._allowSetList = value
        if self.isUiLoaded():
            self._updateMultivalueButtons()

    @pyqtProperty(bool)
    def allowSetRange(self) -> bool:
        return self._allowSetRange

    @allowSetRange.setter
    def allowSetRange(self, value: bool) -> None:
        self._allowSetRange = value
        if self.isUiLoaded():
            self._updateMultivalueButtons()

    @pyqtProperty(float)
    def defaultScalarValue(self) -> T:
        return self._defaultScalarValue

    @defaultScalarValue.setter
    def defaultScalarValue(self, value: T) -> None:
        self._defaultScalarValue = float(value) if self._containsFloatValue else int(value)

    @property
    @abstractmethod
    def valueChanged(self) -> pyqtBoundSignal:
        pass

    @property
    @abstractmethod
    def valueChangedByUser(self) -> pyqtBoundSignal:
        pass

    @property
    @abstractmethod
    def returnPressed(self) -> pyqtBoundSignal:
        pass

    # Methods
    def __init__(self, valueWidgetToWrap: QWidget, *args, widgetValueGetter: Callable[[], T],
                 widgetValueSetter: Callable[[T], None], widgetTextSetter: Callable[[str], None],
                 **kwargs) -> None:
        self._containsFloatValue = isinstance(widgetValueGetter(), float)
        self._multivalue = None
        self._allowSetList = True
        self._allowSetRange = self._containsFloatValue
        self._defaultScalarValue = 0.0 if self._containsFloatValue else 0

        self._innerBoxValueGetter = widgetValueGetter
        self._innerBoxValueSetter = widgetValueSetter
        self._innerBoxTextSetter = widgetTextSetter

        self._innerBoxValueSetter(self._defaultScalarValue)

        super().__init__(__file__, *args, **kwargs)
        self.infoText = ""
        self._updateMultivalueButtons()

        # Prepare UI elements
        self.editValue.parent().layout().replaceWidget(self.editValue, valueWidgetToWrap)
        self.editValue = valueWidgetToWrap

        # Connect own signal slots
        self.btnSetList.clicked.connect(self._onClickSetList)
        self.btnSetRange.clicked.connect(self._onClickSetRange)
        self.btnInfo.clicked.connect(self._onClickInfo)

    def value(self) -> Union[T, Multivalue[T]]:
        """ Returns the value of the field. """
        return self._multivalue if self._multivalue is not None else self._innerBoxValueGetter()

    def setValue(self, value: Union[T, Multivalue[T]]) -> None:
        """ Sets the value of the field. """

        if type(value) is type(self.value()) and value == self.value():
            return

        valueIsMulti = isinstance(value, Multivalue)

        self._multivalue = value if valueIsMulti else None
        if hasattr(self.editValue, "setReadOnly") and callable(self.editValue.setReadOnly):
            self.editValue.setReadOnly(valueIsMulti)
        else:
            self.editValue.setEnabled(not valueIsMulti)

        if valueIsMulti:
            self.editValue.blockSignals(True)  # Ensure that change event isn't triggered automatically
            self._innerBoxTextSetter(str(value))
            self.editValue.blockSignals(False)
            self.valueChanged[Multivalue].emit(value)
        else:
            self._innerBoxValueSetter(value)

        self._updateMultivalueButtons()

    # Internal methods
    def _updateMultivalueButtons(self) -> None:
        """ Updates which multivalue-related actions are available to the user. """

        self.btnSetList.setVisible(self._allowSetList and not self._allowSetRange)
        self.btnSetRange.setVisible(self._allowSetRange)

        if self._allowSetList or self._allowSetRange:
            actionMenu = QMenu()

            if self._allowSetRange:
                actionMenu.addAction("Set value range…", self._onClickSetRange)

            if self._allowSetList:
                actionMenu.addAction("Set value list…", self._onClickSetList)

            resetAction = actionMenu.addAction("Reset to scalar value", self._onClickResetToScalar)
            resetAction.setEnabled(self._multivalue is not None)

            buttonToAddMenuTo = self.btnSetRange if self._allowSetRange else self.btnSetList
            buttonToAddMenuTo.setMenu(actionMenu)

    # Event handling
    @pyqtSlot()
    def _onClickSetList(self) -> None:
        """
        Opens a dialog for the user to configure a value list, and then sets the value to that value
        list.
        """

        valueList, okClicked = SetValueListDialog.getValueList(
            self, containsFloatValues=self._containsFloatValue, initialValue=self.value()
        )

        if okClicked:
            self.setValue(valueList)
            self.valueChangedByUser[Multivalue].emit(valueList)

    @pyqtSlot()
    def _onClickSetRange(self) -> None:
        """
        Opens a dialog for the user to configure a value range, and then sets the value to that
        value range.
        """

        valueRange, okClicked = SetValueRangeDialog.getValueRange(self, initialValue=self.value())
        if okClicked:
            self.setValue(valueRange)
            self.valueChangedByUser[Multivalue].emit(valueRange)

    @pyqtSlot()
    def _onClickResetToScalar(self) -> None:
        """ Resets the value to a scalar value. """
        valueToSet = self._defaultScalarValue
        self.setValue(valueToSet)
        self.valueChangedByUser[float if self._containsFloatValue else int].emit(valueToSet)

    @pyqtSlot()
    def _onClickInfo(self) -> None:
        """ Shows information about the field (e.g. an explanation of it). """
        infoBox = QMessageBox(QMessageBox.NoIcon, "Field Information", self._infoText, parent=self)
        infoBox.setAttribute(Qt.WA_DeleteOnClose)  # Prevent memory leak
        infoBox.exec_()
