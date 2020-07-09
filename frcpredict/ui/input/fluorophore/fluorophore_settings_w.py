from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtWidgets import QListWidgetItem

from frcpredict.model import FluorophoreSettings, IlluminationResponse
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import UserFileDirs
from .fluorophore_settings_p import FluorophoreSettingsPresenter
from .response_list_item import ResponseListItem


class FluorophoreSettingsWidget(BaseWidget):
    """
    A widget where the user may add or remove fluorophore responses.
    """

    # Signals
    valueChanged = pyqtSignal(FluorophoreSettings)
    responseSelectionChanged = pyqtSignal(QListWidgetItem, QListWidgetItem)
    addResponseClicked = pyqtSignal()
    removeResponseClicked = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        # Prepare UI elements
        self.presetPicker.setModelType(FluorophoreSettings)
        self.presetPicker.setStartDirectory(UserFileDirs.FluorophoreSettings)
        self.presetPicker.setValueGetter(self.value)
        self.presetPicker.setValueSetter(self.setValue)

        self.editProperties.setWavelengthVisible(False)

        # Connect own signal slots
        self.presetPicker.dataLoaded.connect(self._onLoadPreset)
        
        # Connect forwarded signals
        self.listResponses.currentItemChanged.connect(self.responseSelectionChanged)
        self.btnAddResponse.clicked.connect(self.addResponseClicked)
        self.btnRemoveResponse.clicked.connect(self.removeResponseClicked)

        # Initialize presenter
        self._presenter = FluorophoreSettingsPresenter(self)

    def addResponseToList(self, response: IlluminationResponse) -> None:
        """ Adds the specified response to the response list and selects it. """
        item = ResponseListItem(response)
        self.listResponses.addItem(item)
        self.listResponses.setCurrentItem(item)

    def removeResponseFromList(self, response: IlluminationResponse) -> None:
        """ Removes the specified response from the response list and deselects it. """
        matchingRows = self.listResponses.findItems(str(response), Qt.MatchExactly)

        for matchingRow in matchingRows:
            self.listResponses.takeItem(self.listResponses.row(matchingRow))

        self.deselectSelectedRow()

    def clearResponseList(self) -> None:
        """ Removes all responses from the response list. """
        self.listResponses.clear()

    def setSelectedResponse(self, response: IlluminationResponse) -> None:
        """ Updates controls and response properties widget based on the current selection. """

        if response is not None:
            self.groupProperties.setTitle(f"Selected Response: {response}")
            self.editProperties.setValue(response)
            self.editProperties.setEnabled(True)
            self.btnRemoveResponse.setEnabled(True)
        else:
            # Clear properties
            self.groupProperties.setTitle("Selected Response")
            self.editProperties.setValue(  # Clear properties
                IlluminationResponse(
                    wavelength_start=0, wavelength_end=0,
                    cross_section_off_to_on=0.0,
                    cross_section_on_to_off=0.0,
                    cross_section_emission=0.0
                )
            )
            self.editProperties.setEnabled(False)
            self.btnRemoveResponse.setEnabled(False)

    def deselectSelectedRow(self) -> None:
        """ Deselects the currently selected row in the response list, if any row is selected. """
        self.listResponses.setCurrentRow(-1)

    def value(self) -> FluorophoreSettings:
        return self._presenter.model

    def setValue(self, model: FluorophoreSettings, emitSignal: bool = True) -> None:
        self.presetPicker.setLoadedPath(None)
        self._presenter.model = model
        self.deselectSelectedRow()

        if emitSignal:
            self.valueChanged.emit(model)
    
    # Event handling
    @pyqtSlot()
    def _onLoadPreset(self) -> None:
        self.deselectSelectedRow()
