from PyQt5.QtCore import pyqtSignal

from frcpredict.model import CameraProperties
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import UserFileDirs
from .camera_properties_p import CameraPropertiesPresenter


class CameraPropertiesWidget(BaseWidget):
    """
    A widget where the user may set camera properties.
    """

    # Signals
    valueChanged = pyqtSignal(CameraProperties)
    readoutNoiseChanged = pyqtSignal(float)
    quantumEfficiencyChanged = pyqtSignal(float)

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        # Prepare UI elements
        self.presetPicker.setModelType(CameraProperties)
        self.presetPicker.setStartDirectory(UserFileDirs.CameraProperties)
        self.presetPicker.setValueGetter(self.value)
        self.presetPicker.setValueSetter(self.setValue)
        
        # Connect forwarded signals
        self.editReadoutNoise.valueChanged.connect(self.readoutNoiseChanged)
        self.editQuantumEfficiency.valueChanged.connect(self.quantumEfficiencyChanged)

        # Initialize presenter
        self._presenter = CameraPropertiesPresenter(self)

    def value(self) -> CameraProperties:
        return self._presenter.model

    def setValue(self, model: CameraProperties, emitSignal: bool = True) -> None:
        self.presetPicker.setLoadedPath(None)
        self._presenter.model = model

        if emitSignal:
            self.valueChanged.emit(model)

    def updateBasicFields(self, model: CameraProperties) -> None:
        self.editReadoutNoise.setValue(model.readout_noise)
        self.editQuantumEfficiency.setValue(model.quantum_efficiency * 100)  # Convert to percent
