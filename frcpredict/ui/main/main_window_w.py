from typing import Optional

from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QRect, QSettings
from PyQt5.QtGui import QFontMetrics, QCloseEvent
from PyQt5.QtWidgets import QMessageBox, QMainWindow

import frcpredict
from frcpredict.model import (
    FluorophoreSettings, ImagingSystemSettings, PulseScheme, SampleProperties, DetectorProperties,
    RunInstance, SimulationResults, KernelSimulationResult
)
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import centerWindow, PresetFileDirs, UserFileDirs
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
    detectorPropertiesModelSet = pyqtSignal(DetectorProperties)

    simulateFrcClicked = pyqtSignal()
    abortClicked = pyqtSignal()

    loadSimulationClicked = pyqtSignal()
    exportJsonClicked = pyqtSignal()
    exportBinaryClicked = pyqtSignal()
    preferencesClicked = pyqtSignal()
    aboutClicked = pyqtSignal()

    # Methods
    def __init__(self, screenGeometry: Optional[QRect] = None) -> None:
        super().__init__(__file__)
        self.setWindowFlag(Qt.WindowMaximizeButtonHint, on=False)

        # Try to resize the window to fit everything without being too small or too large. The text
        # size-based calculations are there to approximately account for differences in text size on
        # different computers, as this has an effect how large the window should be.
        textSizeInfo = QFontMetrics(self.font()).boundingRect(
            "The quick brown fox jumps over the lazy dog"
        )
        self.resize(
            min(screenGeometry.width() - 96, 1300 * (textSizeInfo.width() / 214) ** 0.5),
            min(screenGeometry.height() - 96, 800 * (textSizeInfo.height() / 13) ** 0.5)
        )

        # Prepare UI elements
        self.setWindowTitle(frcpredict.__summary__)

        self.tabifyDockWidget(self.dckInput, self.dckVirtualImagingResults)
        self.takeCentralWidget()
        self.dckInput.raise_()
        self.resizeDocks([self.dckInput], [int(self.width() * 0.7)], Qt.Horizontal)

        self.frcResults.setOutputDirector(self.outputDirector)
        self.virtualImagingResults.setOutputDirector(self.outputDirector)

        self.configPanel.setFieldName("Full configuration")
        self.configPanel.setModelType(RunInstance)
        self.configPanel.setPresetsDirectory(PresetFileDirs.RunInstance)
        self.configPanel.setStartDirectory(UserFileDirs.RunInstance)
        self.configPanel.setValueGetter(self.value)
        self.configPanel.setValueSetter(self.setValue)

        self.setSimulating(False)
        self.setAborting(False)

        self._defaultGeometry = self.saveGeometry()
        self._defaultState = self.saveState()
        self._loadWindowSettings()
        self.dckInput.raise_()  # Always show input dock on start

        # Connect own signal slots
        self.fluorophoreSettings.modifiedFlagSet.connect(self._onModifiedFlagSet)
        self.imagingSystemSettings.modifiedFlagSet.connect(self._onModifiedFlagSet)
        self.pulseScheme.modifiedFlagSet.connect(self._onModifiedFlagSet)
        self.sampleProperties.modifiedFlagSet.connect(self._onModifiedFlagSet)
        self.detectorProperties.modifiedFlagSet.connect(self._onModifiedFlagSet)

        self.outputDirector.kernelResultChanged.connect(self._onKernelResultChange)
        self.actionExit.triggered.connect(self.close)
        self.actionResetInterface.triggered.connect(self._resetWindowSettings)

        # Connect forwarded signals
        self.fluorophoreSettings.valueChanged.connect(self.fluorophoreSettingsModelSet)
        self.imagingSystemSettings.valueChanged.connect(self.imagingSystemSettingsModelSet)
        self.pulseScheme.valueChanged.connect(self.pulseSchemeModelSet)
        self.sampleProperties.valueChanged.connect(self.samplePropertiesModelSet)
        self.detectorProperties.valueChanged.connect(self.detectorPropertiesModelSet)

        self.btnSimulateFrc.clicked.connect(self.simulateFrcClicked)
        self.btnAbort.clicked.connect(self.abortClicked)

        self.actionLoadSimulation.triggered.connect(self.loadSimulationClicked)
        self.actionExportJson.triggered.connect(self.exportJsonClicked)
        self.actionExportBinary.triggered.connect(self.exportBinaryClicked)
        self.actionPreferences.triggered.connect(self.preferencesClicked)
        self.actionAbout.triggered.connect(self.aboutClicked)

        # Initialize presenter
        self._presenter = MainWindowPresenter(self)

    def precacheAllSimulationResults(self) -> None:
        """ Pre-caches all results from the currently loaded simulation. """
        self.outputDirector.precacheAllResults()

    def isModified(self) -> bool:
        """ Returns whether the input parameters have been modified since they were loaded. """
        return self.configPanel.isModified()

    def setFullSimulation(self, model: SimulationResults) -> None:
        """
        Sets the input parameters and the output data at the same time, from a simulation results
        object.
        """
        self.setValue(model.run_instance)
        self.configPanel.clearModifiedFlag()
        self.setSimulationResults(model)

    def simulationResults(self) -> SimulationResults:
        return self.outputDirector.simulationResults()

    def setSimulationResults(self, simulationResults: SimulationResults) -> None:
        """ Sets FRC simulation results. """
        self.outputDirector.setSimulationResults(simulationResults)

    def setSimulating(self, simulating: bool) -> None:
        """ Sets whether a simulation is currently in progress. """

        self.btnSimulateFrc.setEnabled(not simulating)
        self.btnAbort.setEnabled(simulating)

        if simulating:
            self.btnSimulateFrc.setText("SIMULATING…")
        else:
            self.btnSimulateFrc.setText("SIMULATE")
            self.pbProgress.setVisible(False)

    def setAborting(self, aborting: bool) -> None:
        """ Sets whether the current simulation is being aborted. """

        if aborting:
            self.btnAbort.setEnabled(False)
            self.btnAbort.setText("Aborting…")
        else:
            self.btnAbort.setText("Abort")

    def setProgressBarVisible(self, visible: bool) -> None:
        self.pbProgress.setVisible(visible)

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

    def updateDetectorProperties(self, detectorProperties: DetectorProperties) -> None:
        self.detectorProperties.setValue(detectorProperties, emitSignal=False)

    def updateSimulationProgress(self, progress: float) -> None:
        self.pbProgress.setValue(progress * 100 if 0 < progress < 1 else 0)

    # Internal methods
    def _loadWindowSettings(self) -> None:
        settings = self._getWindowSettingsObject()
        geometry = settings.value(_geometrySettingName)
        state = settings.value(_stateSettingName)

        if geometry is not None:
            self.restoreGeometry(geometry)

        if state is not None:
            self.restoreState(state)

    def _saveWindowSettings(self) -> None:
        settings = self._getWindowSettingsObject()
        settings.setValue(_geometrySettingName, self.saveGeometry())
        settings.setValue(_stateSettingName, self.saveState())

    def _resetWindowSettings(self) -> None:
        settings = self._getWindowSettingsObject()
        settings.remove(_geometrySettingName)
        settings.remove(_stateSettingName)

        self.restoreGeometry(self._defaultGeometry)
        self.restoreState(self._defaultState)
        centerWindow(self)

    def _getWindowSettingsObject(self) -> QSettings:
        return QSettings(frcpredict.__author__, frcpredict.__title__)

    # Event handling
    @pyqtSlot()
    def _onModifiedFlagSet(self, *_args, **_kwargs) -> None:
        self.configPanel.setModifiedFlag()

    @pyqtSlot(object, object, bool)
    def _onKernelResultChange(self, _, kernelResult: Optional[KernelSimulationResult], __) -> None:
        self.menuExportSimulation.setEnabled(kernelResult is not None)
        self.actionExportJson.setEnabled(kernelResult is not None)
        self.actionExportBinary.setEnabled(kernelResult is not None)

    def closeEvent(self, event: QCloseEvent) -> None:
        confirmation_result = QMessageBox.question(
            self, "Exit?",
            f"Exit the program? All unsaved input parameters and results will be lost.",
            defaultButton=QMessageBox.No
        )

        if confirmation_result != QMessageBox.Yes:
            event.ignore()
            return

        self._saveWindowSettings()


_geometrySettingName: str = "geometry"
_stateSettingName: str = "state"
