import numpy as np

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QMainWindow

from frcpredict.model import FluorophoreSettings, ImagingSystemSettings, PulseScheme, SampleProperties, CameraProperties, RunInstance
from frcpredict.ui import BaseWidget
from .main_window_p import MainWindowPresenter


class MainWindow(QMainWindow, BaseWidget):
    """
    The main window of the program.
    """

    # Signals
    simulateFrcClicked = pyqtSignal()

    # Methods
    def __init__(self) -> None:
        super().__init__(__file__)

        # Connect forwarded signals
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

    def setValue(self, model: RunInstance) -> None:
        self._presenter.model = model

    def updateFluorophoreSettings(self, fluorophore_settings: FluorophoreSettings) -> None:
        self.fluorophoreSettings.setValue(fluorophore_settings)

    def updateImagingSystemSettings(self, imaging_system_settings: ImagingSystemSettings) -> None:
        self.imagingSystemSettings.setValue(imaging_system_settings)

    def updatePulseScheme(self, pulse_scheme: PulseScheme) -> None:
        self.pulseScheme.setValue(pulse_scheme)

    def updateSampleProperties(self, sample_properties: SampleProperties) -> None:
        self.sampleProperties.setValue(sample_properties)

    def updateCameraProperties(self, camera_properties: CameraProperties) -> None:
        self.cameraProperties.setValue(camera_properties)