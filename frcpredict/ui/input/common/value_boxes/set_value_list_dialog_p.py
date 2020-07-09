from typing import TypeVar, Generic, Optional

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QListWidgetItem

from frcpredict.model import ValueList
from frcpredict.ui import BasePresenter
from ..list_item_with_value import ListItemWithValue

T = TypeVar("T")


class SetValueListPresenter(BasePresenter[ValueList], Generic[T]):
    """
    Presenter for the generate pattern dialog.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: ValueList) -> None:
        # Disconnect old model event handling
        try:
            self._model.value_added.disconnect(self._onValueAdded)
            self._model.value_removed.disconnect(self._onValueRemoved)
        except AttributeError:
            pass

        # Set model
        self._model = model

        # Trigger model change event handlers
        self.widget.clearList()
        for value in model.values:
            self.widget.addValueToList(value)

        self._uiValueSelectionChange(None)

        # Prepare model events
        model.value_added.connect(self._onValueAdded)
        model.value_removed.connect(self._onValueRemoved)

    # Methods
    def __init__(self, widget) -> None:
        # Initialize model
        model = ValueList(values=[])

        super().__init__(model, widget)
        self._selectedValue = None

        # Prepare UI events
        widget.valueSelectionChanged.connect(self._uiValueSelectionChange)
        widget.addValueClicked.connect(self._uiClickAddValue)
        widget.removeValueClicked.connect(self._uiClickRemoveValue)

    # Model event handling
    def _onValueAdded(self, value: T) -> None:
        self.widget.addValueToList(value)

    def _onValueRemoved(self, value: T) -> None:
        self.widget.removeValueFromList(value)

    # UI event handling
    @pyqtSlot(QListWidgetItem, QListWidgetItem)
    def _uiValueSelectionChange(self, selectedItem: Optional[ListItemWithValue], _=None) -> None:
        """ Updates state based on the current selection. """

        if selectedItem is None:
            self._selectedValue = None
        else:
            self._selectedValue = selectedItem.value()

    @pyqtSlot()
    def _uiClickAddValue(self) -> None:
        """ Adds the value that the user wishes to add to the value list in the model. """
        valueToAdd = self.widget.valueToAdd()
        if valueToAdd is not None:
            self.model.add(valueToAdd)

    @pyqtSlot()
    def _uiClickRemoveValue(self) -> None:
        """ Removes the value that the user wishes to remove from the value list in the model. """
        self.model.remove(self._selectedValue)
