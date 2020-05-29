from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QListWidgetItem

from frcpredict.model import FluorophoreSettings, IlluminationResponse
from frcpredict.ui import BaseWidget
from .fluorophore_settings_p import FluorophoreSettingsPresenter


class FluorophoreSettingsWidget(BaseWidget):
    """
    A widget where the user may add or remove fluorophore responses.
    """

    # Functions
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)
        self._presenter = FluorophoreSettingsPresenter(self)

    def setModel(self, model: FluorophoreSettings) -> None:
        self._presenter.model = model

    def addResponseToList(self, response: IlluminationResponse) -> None:
        """ Adds the specified response to the response list and selects it. """
        item = ResponseItem(response)
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
            self.groupProperties.setTitle(f"Properties: {response}")
            self.editProperties.setModel(response)
            self.editProperties.setEnabled(True)
            self.btnRemoveResponse.setEnabled(True)
        else:
            self.groupProperties.setTitle("Properties")
            self.editProperties.setEnabled(False)
            self.btnRemoveResponse.setEnabled(False)


class ResponseItem(QListWidgetItem):
    """
    Custom QListWidgetItem that can be initialized/updated with and sorted by wavelength.
    """

    _wavelengthStart: int
    _wavelengthEnd: int

    # Functions
    def __init__(self, response: IlluminationResponse, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._wavelengthStart = response.wavelength_start
        self._wavelengthEnd = response.wavelength_end
        self.setText(str(response))

    def __lt__(self, other: QListWidgetItem) -> bool:
        if self._wavelengthStart != other._wavelengthStart:
            return self._wavelengthStart < other._wavelengthStart
        else:
            return self._wavelengthEnd < other._wavelengthEnd

    def __gt__(self, other: QListWidgetItem) -> bool:
        if self._wavelengthStart != other._wavelengthStart:
            return self._wavelengthStart > other._wavelengthStart
        else:
            return self._wavelengthEnd > other._wavelengthEnd
