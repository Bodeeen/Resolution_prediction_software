from PyQt5.QtCore import pyqtSignal

from frcpredict.model import CameraProperties
from frcpredict.ui import BaseWidget
from .camera_properties_p import CameraPropertiesPresenter


class CameraPropertiesWidget(BaseWidget):
    """
    A widget where the user may set camera properties.
    """

    # Signals
    readoutNoiseChanged = pyqtSignal(float)
    quantumEfficiencyChanged = pyqtSignal(float)

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)
        
        # Connect forwarded signals
        self.editReadoutNoise.valueChanged.connect(self.readoutNoiseChanged)
        self.editQuantumEfficiency.valueChanged.connect(self.quantumEfficiencyChanged)

        # Initialize presenter
        self._presenter = CameraPropertiesPresenter(self)

    def setModel(self, model: CameraProperties) -> None:
        self._presenter.model = model

    def updateBasicFields(self, model: CameraProperties) -> None:
        self.editReadoutNoise.setValue(model.readout_noise)
        self.editQuantumEfficiency.setValue(model.quantum_efficiency * 100)  # Convert to percent
