from PyQt5.QtCore import pyqtSignal

from frcpredict.model import IlluminationResponse
from frcpredict.ui import BaseWidget
from .response_properties_p import ResponsePropertiesPresenter


class ResponsePropertiesWidget(BaseWidget):
    """
    A widget where the user may set propreties of a specific fluorophore response.
    """

    # Signals
    offToOnEdited = pyqtSignal(float)
    onToOffEdited = pyqtSignal(float)
    emissionEdited = pyqtSignal(float)

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        # Connect forwarded signals
        self.editOffToOn.valueEdited.connect(self.offToOnEdited)
        self.editOnToOff.valueEdited.connect(self.onToOffEdited)
        self.editEmission.valueEdited.connect(self.emissionEdited)

        # Initialize presenter
        self._presenter = ResponsePropertiesPresenter(self)

    def setModel(self, model: IlluminationResponse) -> None:
        self._presenter.model = model

    def setWavelengthVisible(self, value: bool) -> None:
        self.lblWavelength.setVisible(value)
        self.editWavelength.setVisible(value)

    def updateBasicFields(self, model: IlluminationResponse) -> None:
        self.editOffToOn.setValue(model.cross_section_off_to_on)
        self.editOnToOff.setValue(model.cross_section_on_to_off)
        self.editEmission.setValue(model.cross_section_emission)
