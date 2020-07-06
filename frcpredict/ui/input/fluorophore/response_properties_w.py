from PyQt5.QtCore import pyqtSignal

from frcpredict.model import IlluminationResponse, ValueRange
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import connectMulti
from .response_properties_p import ResponsePropertiesPresenter


class ResponsePropertiesWidget(BaseWidget):
    """
    A widget where the user may set properties of a specific fluorophore response.
    """

    # Signals
    wavelengthChanged = pyqtSignal(int)
    offToOnChanged = pyqtSignal([float], [ValueRange])
    onToOffChanged = pyqtSignal([float], [ValueRange])
    emissionChanged = pyqtSignal([float], [ValueRange])

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        # Connect forwarded signals
        self.editWavelength.valueChanged.connect(self.wavelengthChanged)
        connectMulti(self.editOffToOn.valueChanged, [float, ValueRange], self.offToOnChanged)
        connectMulti(self.editOnToOff.valueChanged, [float, ValueRange], self.onToOffChanged)
        connectMulti(self.editEmission.valueChanged, [float, ValueRange], self.emissionChanged)

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
        self.editWavelength.setValue(model.wavelength_start)
        self.editOffToOn.setValue(model.cross_section_off_to_on)
        self.editOnToOff.setValue(model.cross_section_on_to_off)
        self.editEmission.setValue(model.cross_section_emission)
