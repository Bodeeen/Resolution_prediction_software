from typing import Optional

from PyQt5.QtCore import pyqtSlot, QObject
from PyQt5.QtWidgets import QMessageBox, QInputDialog

from frcpredict.model import FluorophoreSettings, IlluminationResponse


class FluorophoreSettingsPresenter(QObject):
    """
    Presenter for the fluorophore settings widget.
    """

    # Properties
    @property
    def model(self) -> FluorophoreSettings:
        return self._model

    @model.setter
    def model(self, model: FluorophoreSettings) -> None:
        self._model = model

        # Update response list in widget
        self._widget.clearResponseList()
        for response in self.model.responses:
            self._widget.addResponseToList(response.wavelength_start, response.wavelength_end)

        self._onResponseSelectionChange(-1)

        # Prepare model events
        model.response_added.connect(self._onResponseAdded)
        model.response_removed.connect(self._onResponseRemoved)

    # Functions
    def __init__(self, widget, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._widget = widget
        self._selectedResponse = None

        # Prepare UI events
        self._widget.listResponses.currentRowChanged.connect(self._onResponseSelectionChange)
        self._widget.btnAddResponse.clicked.connect(self._onClickAddResponse)
        self._widget.btnRemoveResponse.clicked.connect(self._onClickRemoveResponse)

        # Init model
        self.model = FluorophoreSettings(responses={})

    # Model event handling
    def _onResponseAdded(self, response: IlluminationResponse) -> None:
        self._widget.addResponseToList(response)

    def _onResponseRemoved(self, response: IlluminationResponse) -> None:
        self._widget.removeResponseFromList(response)

    # UI event handling
    @pyqtSlot(int)
    def _onResponseSelectionChange(self, selectedIndex: int) -> None:
        """ Updates state and response properties widget based on the current selection. """

        if selectedIndex < 0:
            self._selectedResponse = None
        else:
            # TODO: Move item key logic so that we can get rid of this bad and buggy way of getting
            #       the selected item
            self._selectedResponse = sorted(self.model.responses,
                                            key=lambda response: response.wavelength_start)[selectedIndex]

        self._widget.setSelectedResponse(self._selectedResponse)

    @pyqtSlot()
    def _onClickAddResponse(self) -> None:
        """
        Adds a new response. A dialog will open for the user to enter the desired wavelength first.

        TODO: Custom dialog where you can choose to enter a range, and maybe even set all
              parameters (could just reuse response_properties with a couple of additional fields)
        """

        wavelength, ok_pressed = QInputDialog.getInt(
            self._widget, "Add Response", "Enter wavelength in nanometres:")

        if ok_pressed:
            self.model.add_response(IlluminationResponse(
                wavelength_start=wavelength, wavelength_end=wavelength,
                cross_section_off_to_on=0.5, cross_section_on_to_off=0.25, cross_section_emission=0.25
            ))

    @pyqtSlot()
    def _onClickRemoveResponse(self) -> None:
        """
        Removes the currently selected response. A popup will open for the user to confirm first.
        """

        confirmation_result = QMessageBox.question(
            self._widget, "Remove Response", "Remove the selected response?")

        if confirmation_result == QMessageBox.Yes:
            self.model.remove_response(self._selectedResponse.wavelength_start)
