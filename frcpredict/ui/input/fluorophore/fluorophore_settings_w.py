from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt
from PyQt5.QtWidgets import QListWidgetItem

from frcpredict.model import FluorophoreSettings, IlluminationResponse
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import PresetFileDirs, UserFileDirs
from .fluorophore_settings_p import FluorophoreSettingsPresenter
from .response_list_item import ResponseListItem


class FluorophoreSettingsWidget(BaseWidget):
    """
    A widget where the user may add or remove fluorophore responses.
    """

    # Signals
    valueChanged = pyqtSignal(FluorophoreSettings)
    modifiedFlagSet = pyqtSignal()

    responseSelectionChanged = pyqtSignal(QListWidgetItem, QListWidgetItem)
    addResponseClicked = pyqtSignal()
    removeResponseClicked = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        # Prepare UI elements
        self.configPanel.setModelType(FluorophoreSettings)
        self.configPanel.setPresetsDirectory(PresetFileDirs.FluorophoreSettings)
        self.configPanel.setStartDirectory(UserFileDirs.FluorophoreSettings)
        self.configPanel.setValueGetter(self.value)
        self.configPanel.setValueSetter(self.setValue)

        self.editProperties.setWavelengthVisible(False)

        # Connect own signal slots
        self.modifiedFlagSet.connect(self._onModifiedFlagSet)
        self.editProperties.modifiedFlagSet.connect(self._onModifiedFlagSet)
        self.configPanel.dataLoaded.connect(self._onLoadConfig)
        
        # Connect forwarded signals
        self.listResponses.currentItemChanged.connect(self.responseSelectionChanged)
        self.btnAddResponse.clicked.connect(self.addResponseClicked)
        self.btnRemoveResponse.clicked.connect(self.removeResponseClicked)

        # Initialize presenter
        self._presenter = FluorophoreSettingsPresenter(self)

    def addResponseToList(self, response: IlluminationResponse, select: bool = True) -> None:
        """ Adds the specified response to the response list and selects it. """
        item = ResponseListItem(response)
        self.listResponses.addItem(item)

        if select:
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
            self.editProperties.setValue(IlluminationResponse())
            self.editProperties.setEnabled(False)
            self.btnRemoveResponse.setEnabled(False)

    def deselectSelectedRow(self) -> None:
        """ Deselects the currently selected row in the response list, if any row is selected. """
        self.listResponses.setCurrentRow(-1)

    def value(self) -> FluorophoreSettings:
        return self._presenter.model

    def setValue(self, model: FluorophoreSettings, emitSignal: bool = True) -> None:
        self.configPanel.setLoadedPath(None)
        self._presenter.model = model
        self.deselectSelectedRow()

        if emitSignal:
            self.valueChanged.emit(model)

    # Event handling
    @pyqtSlot()
    def _onModifiedFlagSet(self, *_args, **_kwargs) -> None:
        self.configPanel.setModifiedFlag()

    @pyqtSlot()
    def _onLoadConfig(self) -> None:
        self.deselectSelectedRow()
