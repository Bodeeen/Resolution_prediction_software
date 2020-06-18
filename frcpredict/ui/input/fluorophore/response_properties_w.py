from PyQt5.QtCore import pyqtSignal

from frcpredict.model import IlluminationResponse
from frcpredict.ui import BaseWidget
from .response_properties_p import ResponsePropertiesPresenter


class ResponsePropertiesWidget(BaseWidget):
    """
    A widget where the user may set propreties of a specific fluorophore response.
    """

    # Signals
    offToOnChanged = pyqtSignal(float)
    onToOffChanged = pyqtSignal(float)
    emissionChanged = pyqtSignal(float)

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        # Connect forwarded signals
        self.editOffToOn.valueChanged.connect(self.offToOnChanged)
        self.editOnToOff.valueChanged.connect(self.onToOffChanged)
        self.editEmission.valueChanged.connect(self.emissionChanged)

        # Initialize presenter
        self._presenter = ResponsePropertiesPresenter(self)

    def value(self) -> IlluminationResponse:
        return self._presenter.model

    def setValue(self, model: IlluminationResponse) -> None:
        self._presenter.model = model

    def setWavelengthVisible(self, visible: bool) -> None:
        """ Sets whether the field for editing the wavelength is visible. """
        self.lblWavelength.setVisible(visible)
        self.editWavelength.setVisible(visible)

    def updateBasicFields(self, model: IlluminationResponse) -> None:
        self.editOffToOn.setValue(model.cross_section_off_to_on)
        self.editOnToOff.setValue(model.cross_section_on_to_off)
        self.editEmission.setValue(model.cross_section_emission)
