import numpy as np

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import QMainWindow

from frcpredict.model import (
    FluorophoreSettings, ImagingSystemSettings, PulseScheme, SampleProperties, CameraProperties,
    RunInstance
)
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import UserFileDirs
from .main_window_p import MainWindowPresenter


class MainWindow(QMainWindow, BaseWidget):
    """
    The main window of the program.
    """

    # Signals
    fluorophoreSettingsModelSet = pyqtSignal(FluorophoreSettings)
    imagingSystemSettingsModelSet = pyqtSignal(ImagingSystemSettings)
    pulseSchemeModelSet = pyqtSignal(PulseScheme)
    samplePropertiesModelSet = pyqtSignal(SampleProperties)
    cameraPropertiesModelSet = pyqtSignal(CameraProperties)
    
    simulateFrcClicked = pyqtSignal()

    # Methods
    def __init__(self) -> None:
        super().__init__(__file__)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, on=False)

        # Prepare UI elements
        self.presetPicker.setFieldName("Global config")
        self.presetPicker.setModelType(RunInstance)
        self.presetPicker.setStartDirectory(UserFileDirs.RunInstance)
        self.presetPicker.setValueGetter(self.value)
        self.presetPicker.setValueSetter(self.setValue)

        # Connect forwarded signals
        self.fluorophoreSettings.valueChanged.connect(self.fluorophoreSettingsModelSet)
        self.imagingSystemSettings.valueChanged.connect(self.imagingSystemSettingsModelSet)
        self.pulseScheme.valueChanged.connect(self.pulseSchemeModelSet)
        self.sampleProperties.valueChanged.connect(self.samplePropertiesModelSet)
        self.cameraProperties.valueChanged.connect(self.cameraPropertiesModelSet)
        
        self.btnSimulateFrc.clicked.connect(self.simulateFrcClicked)

        # Initialize presenter
        self._presenter = MainWindowPresenter(self)

    def setFrcResults(self, x: np.ndarray, y: np.ndarray) -> None:
        self.frcResults.setCurve(x, y)

    def setSimulating(self, simulating: bool) -> None:
        self.btnSimulateFrc.setEnabled(not simulating)
        if simulating:
            self.btnSimulateFrc.setText("SIMULATING…")
        else:
            self.btnSimulateFrc.setText("SIMULATE")

    def value(self) -> RunInstance:
        return self._presenter.model

    def setValue(self, model: RunInstance) -> None:
        self._presenter.model = model

    def updateFluorophoreSettings(self, fluorophoreSettings: FluorophoreSettings) -> None:
        self.fluorophoreSettings.setValue(fluorophoreSettings, emitSignal=False)

    def updateImagingSystemSettings(self, imagingSystemSettings: ImagingSystemSettings) -> None:
        self.imagingSystemSettings.setValue(imagingSystemSettings, emitSignal=False)

    def updatePulseScheme(self, pulseScheme: PulseScheme) -> None:
        self.pulseScheme.setValue(pulseScheme, emitSignal=False)

    def updateSampleProperties(self, sampleProperties: SampleProperties) -> None:
        self.sampleProperties.setValue(sampleProperties, emitSignal=False)

    def updateCameraProperties(self, cameraProperties: CameraProperties) -> None:
        self.cameraProperties.setValue(cameraProperties, emitSignal=False)
