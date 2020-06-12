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

    def setModel(self, model: RunInstance) -> None:
        self._presenter.model = model

    def setFrcResults(self, x: np.ndarray, y: np.ndarray) -> None:
        self.frcResults.setCurve(x, y)

    def updateFluorophoreSettings(self, fluorophore_settings: FluorophoreSettings) -> None:
        self.fluorophoreSettings.setModel(fluorophore_settings)

    def updateImagingSystemSettings(self, imaging_system_settings: ImagingSystemSettings) -> None:
        self.imagingSystemSettings.setModel(imaging_system_settings)

    def updatePulseScheme(self, pulse_scheme: PulseScheme) -> None:
        self.pulseScheme.setModel(pulse_scheme)

    def updateSampleProperties(self, sample_properties: SampleProperties) -> None:
        self.sampleProperties.setModel(sample_properties)

    def updateCameraProperties(self, camera_properties: CameraProperties) -> None:
        self.cameraProperties.setModel(camera_properties)