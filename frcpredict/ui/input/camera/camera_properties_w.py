from PyQt5.QtCore import pyqtSignal, pyqtSlot

from frcpredict.model import CameraProperties, Multivalue
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import connectMulti, PresetFileDirs, UserFileDirs
from .camera_properties_p import CameraPropertiesPresenter


class CameraPropertiesWidget(BaseWidget):
    """
    A widget where the user may set camera properties.
    """

    # Signals
    valueChanged = pyqtSignal(CameraProperties)
    modifiedFlagSet = pyqtSignal()

    readoutNoiseChanged = pyqtSignal([float], [Multivalue])
    quantumEfficiencyChanged = pyqtSignal([float], [Multivalue])

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        # Prepare UI elements
        self.configPanel.setModelType(CameraProperties)
        self.configPanel.setPresetsDirectory(PresetFileDirs.CameraProperties)
        self.configPanel.setStartDirectory(UserFileDirs.CameraProperties)
        self.configPanel.setValueGetter(self.value)
        self.configPanel.setValueSetter(self.setValue)

        # Connect own signal slots
        self.modifiedFlagSet.connect(self._onModifiedFlagSet)
        
        # Connect forwarded signals
        connectMulti(self.editReadoutNoise.valueChanged, [float, Multivalue],
                     self.readoutNoiseChanged)
        connectMulti(self.editQuantumEfficiency.valueChanged, [float, Multivalue],
                     self.quantumEfficiencyChanged)

        # Initialize presenter
        self._presenter = CameraPropertiesPresenter(self)

    def value(self) -> CameraProperties:
        return self._presenter.model

    def setValue(self, model: CameraProperties, emitSignal: bool = True) -> None:
        self.configPanel.setLoadedPath(None)
        self._presenter.model = model

        if emitSignal:
            self.valueChanged.emit(model)

    def updateBasicFields(self, model: CameraProperties) -> None:
        self.editReadoutNoise.setValue(model.readout_noise)
        self.editQuantumEfficiency.setValue(model.quantum_efficiency)

    # Event handling
    @pyqtSlot()
    def _onModifiedFlagSet(self, *_args, **_kwargs) -> None:
        self.configPanel.setModifiedFlag()
