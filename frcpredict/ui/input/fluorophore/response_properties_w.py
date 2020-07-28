from PyQt5.QtCore import pyqtSignal

from frcpredict.model import IlluminationResponse, Multivalue
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import setFormLayoutRowVisibility, connectMulti
from .response_properties_p import ResponsePropertiesPresenter


class ResponsePropertiesWidget(BaseWidget):
    """
    A widget where the user may set properties of a specific fluorophore response.
    """

    # Signals
    wavelengthChanged = pyqtSignal(float)
    offToOnChanged = pyqtSignal([float], [Multivalue])
    onToOffChanged = pyqtSignal([float], [Multivalue])
    emissionChanged = pyqtSignal([float], [Multivalue])

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        # Connect forwarded signals
        self.editWavelength.valueChanged.connect(self.wavelengthChanged)
        connectMulti(self.editOffToOn.valueChanged, [float, Multivalue], self.offToOnChanged)
        connectMulti(self.editOnToOff.valueChanged, [float, Multivalue], self.onToOffChanged)
        connectMulti(self.editEmission.valueChanged, [float, Multivalue], self.emissionChanged)

        # Initialize presenter
        self._presenter = ResponsePropertiesPresenter(self)

    def value(self) -> IlluminationResponse:
        return self._presenter.model

    def setValue(self, model: IlluminationResponse) -> None:
        self._presenter.model = model

    def setWavelengthVisible(self, visible: bool) -> None:
        """ Sets whether the field for editing the wavelength is visible. """
        setFormLayoutRowVisibility(self.layout(), 0, self.lblWavelength, self.editWavelength,
                                   visible=visible)

    def updateBasicFields(self, model: IlluminationResponse) -> None:
        self.editWavelength.setValue(model.wavelength)
        self.editOffToOn.setValue(model.cross_section_off_to_on)
        self.editOnToOff.setValue(model.cross_section_on_to_off)
        self.editEmission.setValue(model.cross_section_emission)
