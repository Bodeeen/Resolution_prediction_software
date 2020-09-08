from PyQt5.QtCore import pyqtSignal, pyqtSlot

from frcpredict.model import DetectorProperties, Multivalue
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import (
    connectMulti, setTabOrderForChildren, setFormLayoutRowVisibility, PresetFileDirs, UserFileDirs
)
from .detector_properties_p import DetectorPropertiesPresenter


class DetectorPropertiesWidget(BaseWidget):
    """
    A widget where the user may set detector properties.
    """

    # Signals
    valueChanged = pyqtSignal(DetectorProperties)
    modifiedFlagSet = pyqtSignal()

    typePointDetectorToggled = pyqtSignal(bool)
    typeCameraToggled = pyqtSignal(bool)

    readoutNoiseChanged = pyqtSignal([float], [Multivalue])
    quantumEfficiencyChanged = pyqtSignal([float], [Multivalue])
    cameraPixelSizeChanged = pyqtSignal([float], [Multivalue])

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        # Prepare UI elements
        self.configPanel.setModelType(DetectorProperties)
        self.configPanel.setPresetsDirectory(PresetFileDirs.DetectorProperties)
        self.configPanel.setStartDirectory(UserFileDirs.DetectorProperties)
        self.configPanel.setValueGetter(self.value)
        self.configPanel.setValueSetter(self.setValue)

        setTabOrderForChildren(self, [self.configPanel, self.rdoPointDetector, self.rdoCamera,
                                      self.editReadoutNoise, self.editCameraPixelSize,
                                      self.editQuantumEfficiency])

        # Connect own signal slots
        self.modifiedFlagSet.connect(self._onModifiedFlagSet)
        
        # Connect forwarded signals
        self.rdoPointDetector.toggled.connect(self.typePointDetectorToggled)
        self.rdoCamera.toggled.connect(self.typeCameraToggled)
        connectMulti(self.editReadoutNoise.valueChanged, [float, Multivalue],
                     self.readoutNoiseChanged)
        connectMulti(self.editQuantumEfficiency.valueChanged, [float, Multivalue],
                     self.quantumEfficiencyChanged)
        connectMulti(self.editCameraPixelSize.valueChanged, [float, Multivalue],
                     self.cameraPixelSizeChanged)
        self.configPanel.dataLoaded.connect(self.modifiedFlagSet)

        # Initialize presenter
        self._presenter = DetectorPropertiesPresenter(self)

    def value(self) -> DetectorProperties:
        return self._presenter.model

    def setValue(self, model: DetectorProperties, emitSignal: bool = True) -> None:
        self.configPanel.setLoadedPath(None)
        self._presenter.model = model

        if emitSignal:
            self.valueChanged.emit(model)

    def updateBasicFields(self, model: DetectorProperties) -> None:
        cameraMode = model.camera_pixel_size is not None

        setFormLayoutRowVisibility(
            self.formLayout, 2, self.lblCameraPixelSize, self.editCameraPixelSize,
            visible=cameraMode
        )

        if cameraMode:
            self.rdoCamera.setChecked(True)
            self.editCameraPixelSize.setValue(model.camera_pixel_size)
        else:
            self.rdoPointDetector.setChecked(True)

        self.editQuantumEfficiency.setValue(model.quantum_efficiency)

    # Event handling
    @pyqtSlot()
    def _onModifiedFlagSet(self, *_args, **_kwargs) -> None:
        self.configPanel.setModifiedFlag()
