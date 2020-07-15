from typing import Optional

from PyQt5.QtCore import pyqtSignal, Qt, QRect
from PyQt5.QtGui import QFontMetrics
from PyQt5.QtWidgets import QMainWindow

from frcpredict.model import (
    FluorophoreSettings, ImagingSystemSettings, PulseScheme, SampleProperties, CameraProperties,
    RunInstance, FrcSimulationResults
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
    abortClicked = pyqtSignal()

    # Methods
    def __init__(self, screenGeometry: Optional[QRect] = None) -> None:
        super().__init__(__file__)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, on=False)

        # Try to resize the window to fit everything without being too small or too large. The text
        # size-based calculations are there to approximately account for differences in text size on
        # different systems, as this has an effect how large the window should be.
        textSizeInfo = QFontMetrics(self.font()).boundingRect(
            "The quick brown fox jumps over the lazy dog"
        )
        self.resize(
            min(screenGeometry.width() - 96, 1280 * (textSizeInfo.width() / 214) ** 0.5),
            min(screenGeometry.height() - 96, 740 * (textSizeInfo.height() / 13) ** 0.5)
        )

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
        self.btnAbort.clicked.connect(self.abortClicked)

        # Initialize presenter
        self._presenter = MainWindowPresenter(self)

    def setFrcSimulationResults(self, frcSimulationResults: FrcSimulationResults) -> None:
        """ Sets FRC simulation results. """
        self.frcResults.setValue(frcSimulationResults)

    def setSimulating(self, simulating: bool) -> None:
        """ Sets whether a simulation is currently in progress. """

        self.btnSimulateFrc.setEnabled(not simulating)
        self.btnAbort.setEnabled(simulating)

        if simulating:
            self.btnSimulateFrc.setText("SIMULATING…")
        else:
            self.btnSimulateFrc.setText("SIMULATE")

    def setAborting(self, aborting: bool) -> None:
        """ Sets whether the current simulation is being aborted. """

        if aborting:
            self.btnAbort.setEnabled(False)
            self.btnAbort.setText("Aborting…")
        else:
            self.btnAbort.setText("Abort")

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

    def updateSimulationProgress(self, progress: float) -> None:
        if 0 < progress < 1:
            self.pbProgress.setValue(progress * 100)
        else:
            self.pbProgress.setValue(0)
