from PyQt5.QtCore import pyqtSignal, Qt

from frcpredict.model import FluorophoreSettings, IlluminationResponse
from frcpredict.ui import BaseWidget
from .fluorophore_settings_p import FluorophoreSettingsPresenter
from .response_list_item import ResponseListItem


class FluorophoreSettingsWidget(BaseWidget):
    """
    A widget where the user may add or remove fluorophore responses.
    """

    # Signals
    responseSelectionChanged = pyqtSignal(int)
    addResponseClicked = pyqtSignal()
    removeResponseClicked = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)
        self.editProperties.setWavelengthVisible(False)
        
        # Connect forwarded signals
        self.listResponses.currentRowChanged.connect(self.responseSelectionChanged)
        self.btnAddResponse.clicked.connect(self.addResponseClicked)
        self.btnRemoveResponse.clicked.connect(self.removeResponseClicked)

        # Initialize presenter
        self._presenter = FluorophoreSettingsPresenter(self)

    def setModel(self, model: FluorophoreSettings) -> None:
        self._presenter.model = model

    def addResponseToList(self, response: IlluminationResponse) -> None:
        """ Adds the specified response to the response list and selects it. """
        item = ResponseListItem(response)
        self.listResponses.addItem(item)
        self.listResponses.setCurrentItem(item)

    def removeResponseFromList(self, response: IlluminationResponse) -> None:
        """ Removes the specified response from the response list and deselects it. """
        self.listResponses.takeItem(
            self.listResponses.row(
                self.listResponses.findItems(str(response), Qt.MatchExactly)[0]
            )
        )

        self.listResponses.setCurrentRow(-1)

    def clearResponseList(self) -> None:
        """ Removes all responses from the response list. """
        self.listResponses.clear()

    def setSelectedResponse(self, response: IlluminationResponse) -> None:
        """ Updates controls and response properties widget based on the current selection. """

        if response is not None:
            self.groupProperties.setTitle(f"Selected Response: {response}")
            self.editProperties.setModel(response)
            self.editProperties.setEnabled(True)
            self.btnRemoveResponse.setEnabled(True)
        else:
            # Clear properties
            self.groupProperties.setTitle("Selected Response")
            self.editProperties.setModel(  # Clear properties
                IlluminationResponse(
                    wavelength_start=0.0, wavelength_end=0.0,
                    cross_section_off_to_on=0.0, cross_section_on_to_off=0.0, cross_section_emission=0.0
                )
            )
            self.editProperties.setEnabled(False)
            self.btnRemoveResponse.setEnabled(False)
