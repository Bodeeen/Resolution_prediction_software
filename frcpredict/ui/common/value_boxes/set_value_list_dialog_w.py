from copy import deepcopy
from typing import Optional, Union, TypeVar, Generic, Tuple

from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtWidgets import QWidget, QDialog, QDialogButtonBox, QListWidgetItem

from frcpredict.model import ValueList
from frcpredict.ui import BaseWidget
from ..list_item_with_value import ListItemWithValue
from .set_value_list_dialog_p import SetValueListPresenter

T = TypeVar("T")


class SetValueListDialog(QDialog, BaseWidget, Generic[T]):
    """
    A dialog for setting a value list.
    """

    # Signals
    valueSelectionChanged = pyqtSignal(QListWidgetItem, QListWidgetItem)
    addValueClicked = pyqtSignal()
    removeValueClicked = pyqtSignal()

    # Methods
    def __init__(self, parent: Optional[QWidget] = None, containsFloatValues: bool = False,
                 initialValue: Union[T, ValueList[T]] = 0) -> None:
        self._containsFloatValues = containsFloatValues
        self._hasHandledInitialRowChange = not(
            isinstance(initialValue, ValueList) and len(initialValue.values) > 0
        )

        super().__init__(__file__, parent, Qt.WindowSystemMenuHint | Qt.WindowTitleHint)

        # Prepare UI elements
        if containsFloatValues:
            from .extended_free_float_box import ExtendedFreeFloatBox
            editValueToAdd = ExtendedFreeFloatBox(allowSetList=False, allowSetRange=False)
        else:
            from .extended_spin_box import ExtendedSpinBox
            editValueToAdd = ExtendedSpinBox(allowSetList=False, allowSetRange=False)

        self.editValueToAdd.parent().layout().replaceWidget(self.editValueToAdd, editValueToAdd)
        self.editValueToAdd = editValueToAdd
        self.editValueToAdd.selectAll()

        self._updateOKButton()

        # Connect own signal slots
        self.listValues.currentItemChanged.connect(self._onValueSelectionChange)

        # Connect forwarded signals
        self.listValues.currentItemChanged.connect(self.valueSelectionChanged)
        self.editValueToAdd.returnPressed.connect(self.addValueClicked)
        self.btnAddValue.clicked.connect(self.addValueClicked)
        self.btnRemoveValue.clicked.connect(self.removeValueClicked)

        # Initialize presenter
        self._presenter = SetValueListPresenter(self)
        if isinstance(initialValue, ValueList):
            self._presenter.model = deepcopy(initialValue)

    def addValueToList(self, value: T) -> None:
        """ Adds the specified value to the list. """
        item = ListItemWithValue(str(value), value)
        self.listValues.addItem(item)
        self._updateOKButton()

    def removeValueFromList(self, value: T) -> None:
        """ Removes the specified value from the list and deselects it. """
        matchingRows = self.listValues.findItems(str(value), Qt.MatchExactly)

        for matchingRow in matchingRows:
            self.listValues.takeItem(self.listValues.row(matchingRow))

        self.listValues.setCurrentRow(-1)
        self._updateOKButton()

    def clearList(self) -> None:
        """ Removes all values from the list. """
        self.listValues.clear()
        self._updateOKButton()

    def containsFloatValues(self) -> bool:
        """
        Returns whether the value list contains float values (if not, then it contains integers).
        """
        return self._containsFloatValues

    def valueToAdd(self) -> Optional[T]:
        if self._containsFloatValues and not self.editValueToAdd.text():
            return None

        return self.editValueToAdd.value()

    @staticmethod
    def getValueList(parent: Optional[QWidget] = None, containsFloatValues: bool = False,
                     initialValue: Union[T, ValueList[T]] = 0) -> Tuple[Optional[ValueList[T]], bool]:
        """
        Synchronously opens a dialog for setting a value list. The second value in the returned
        tuple refers to whether the "OK" button was pressed when the dialog closed. If it's true,
        the first value will contain the value list.
        """

        dialog = SetValueListDialog(parent, containsFloatValues, initialValue)
        result = dialog.exec_()

        if result == QDialog.Accepted:
            value = dialog._presenter.model
        else:
            value = None

        dialog.deleteLater()  # Prevent memory leak
        return value, value is not None

    # Internal methods
    def _updateOKButton(self) -> None:
        """ Enables or disables the "OK" button depending on whether the entered data is valid. """
        self.buttonBox.button(QDialogButtonBox.Ok).setEnabled(
            self.listValues.count() > 0
        )

    # Event handling
    def keyPressEvent(self, event):
        """ Overriden QDialog function; don't close dialog on pressing return/enter. """
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            return

        super().keyPressEvent(event)

    @pyqtSlot(QListWidgetItem, QListWidgetItem)
    def _onValueSelectionChange(self, selectedItem: Optional[ListItemWithValue], _=None) -> None:
        """ Updates remove button enabled state based on whether an item is selected or not. """

        if not self._hasHandledInitialRowChange and selectedItem is not None:
            # We do this to make sure no row is selected when the dialog opens
            self.listValues.setCurrentRow(-1)
            self._hasHandledInitialRowChange = True
            selectedItem = None

        self.btnRemoveValue.setEnabled(selectedItem is not None)
