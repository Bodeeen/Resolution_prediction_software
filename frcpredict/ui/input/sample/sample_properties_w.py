from PyQt5.QtCore import pyqtSignal

from frcpredict.model import SampleProperties
from frcpredict.ui import BaseWidget
from .sample_properties_p import SamplePropertiesPresenter


class SamplePropertiesWidget(BaseWidget):
    """
    A widget where the user may set sample properties.
    """

    # Signals
    spectralPowerChanged = pyqtSignal(float)
    labellingDensityChanged = pyqtSignal(float)

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)
        
        # Connect forwarded signals
        self.editSpectralPower.valueChanged.connect(self.spectralPowerChanged)
        self.editLabellingDensity.valueChanged.connect(self.labellingDensityChanged)

        # Initialize presenter
        self._presenter = SamplePropertiesPresenter(self)

    def setValue(self, model: SampleProperties) -> None:
        self._presenter.model = model

    def updateBasicFields(self, model: SampleProperties) -> None:
        self.editSpectralPower.setValue(model.spectral_power)
        self.editLabellingDensity.setValue(model.labelling_density)
