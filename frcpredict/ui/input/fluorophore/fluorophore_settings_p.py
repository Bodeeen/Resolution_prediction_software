from typing import Optional

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMessageBox

from frcpredict.model import FluorophoreSettings, IlluminationResponse
from frcpredict.ui import BasePresenter
from .add_response_dialog import AddResponseDialog


class FluorophoreSettingsPresenter(BasePresenter[FluorophoreSettings]):
    """
    Presenter for the fluorophore settings widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: FluorophoreSettings) -> None:
        self._model = model

        # Trigger model change event handlers
        self.widget.clearResponseList()
        for response in model.responses:
            self.widget.addResponseToList(response)

        self._uiResponseSelectionChange(-1)

        # Prepare model events
        model.response_added.connect(self._onResponseAdded)
        model.response_removed.connect(self._onResponseRemoved)

    # Methods
    def __init__(self, widget) -> None:
        # Initialize model
        model = FluorophoreSettings(responses=[])
        
        super().__init__(model, widget)
        self._selectedResponse = None

        # Prepare UI events
        widget.responseSelectionChanged.connect(self._uiResponseSelectionChange)
        widget.addResponseClicked.connect(self._uiClickAddResponse)
        widget.removeResponseClicked.connect(self._uiClickRemoveResponse)

    # Model event handling
    def _onResponseAdded(self, response: IlluminationResponse) -> None:
        self.widget.addResponseToList(response)

    def _onResponseRemoved(self, response: IlluminationResponse) -> None:
        self.widget.removeResponseFromList(response)

    # UI event handling
    @pyqtSlot(int)
    def _uiResponseSelectionChange(self, selectedIndex: int) -> None:
        """ Updates state and response properties widget based on the current selection. """

        if selectedIndex < 0:
            self._selectedResponse = None
        else:
            # TODO: Move item key logic so that we can get rid of this bad and potentially buggy
            #       way of getting the selected item
            self._selectedResponse = sorted(self.model.responses,
                                            key=lambda response: response.wavelength_start)[selectedIndex]

        self.widget.setSelectedResponse(self._selectedResponse)

    @pyqtSlot()
    def _uiClickAddResponse(self) -> None:
        """
        Adds a response. A dialog will open for the user to enter the properties first.
        """
        
        response, ok_pressed = AddResponseDialog.getResponse(self.widget)
        if ok_pressed:
            self.model.add_response(response)

    @pyqtSlot()
    def _uiClickRemoveResponse(self) -> None:
        """
        Removes the currently selected response. A dialog will open for the user to confirm first.
        """

        confirmation_result = QMessageBox.question(
            self.widget, "Remove Response", f"Remove the selected response \"{self._selectedResponse}\"?")

        if confirmation_result == QMessageBox.Yes:
            self.model.remove_response(self._selectedResponse.wavelength_start)
