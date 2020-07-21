from traceback import format_exc

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, QThreadPool, QRunnable, QMutex
from PyQt5.QtWidgets import QMessageBox

from frcpredict.model import (
    Pattern, Array2DPatternData,
    FluorophoreSettings, ImagingSystemSettings, PulseScheme, SampleProperties, CameraProperties,
    RunInstance, FrcSimulationResults
)
from frcpredict.ui import BasePresenter


class MainWindowPresenter(BasePresenter[RunInstance]):
    """
    Presenter for the main window.
    """

    @BasePresenter.model.setter
    def model(self, model: RunInstance) -> None:
        # Disconnect old model event handling
        try:
            self._model.fluorophore_settings_loaded.disconnect(self._onFluorophoreSettingsLoad)
            self._model.imaging_system_settings_loaded.disconnect(self._onImagingSystemSettingsLoad)
            self._model.pulse_scheme_loaded.disconnect(self._onPulseSchemeLoad)
            self._model.sample_properties_loaded.disconnect(self._onSamplePropertiesLoad)
            self._model.camera_properties_loaded.disconnect(self._onCameraPropertiesLoad)
        except AttributeError:
            pass

        # Set model
        self._model = model

        # Trigger model change event handlers
        self._onFluorophoreSettingsLoad(model.fluorophore_settings)
        self._onImagingSystemSettingsLoad(model.imaging_system_settings)
        self._onPulseSchemeLoad(model.pulse_scheme)
        self._onSamplePropertiesLoad(model.sample_properties)
        self._onCameraPropertiesLoad(model.camera_properties)

        # Prepare model events
        model.fluorophore_settings_loaded.connect(self._onFluorophoreSettingsLoad)
        model.imaging_system_settings_loaded.connect(self._onImagingSystemSettingsLoad)
        model.pulse_scheme_loaded.connect(self._onPulseSchemeLoad)
        model.sample_properties_loaded.connect(self._onSamplePropertiesLoad)
        model.camera_properties_loaded.connect(self._onCameraPropertiesLoad)

    # Methods
    def __init__(self, widget) -> None:
        super().__init__(RunInstance(), widget)
        self._threadPool = QThreadPool.globalInstance()
        self._actionLock = QMutex()
        self._currentWorker = None

        # Connect UI events
        widget.fluorophoreSettingsModelSet.connect(self._uiSetFluorophoreSettingsModel)
        widget.imagingSystemSettingsModelSet.connect(self._uiSetImagingSystemSettingsModel)
        widget.pulseSchemeModelSet.connect(self._uiSetPulseSchemeModel)
        widget.samplePropertiesModelSet.connect(self._uiSetSamplePropertiesModel)
        widget.cameraPropertiesModelSet.connect(self._uiSetCameraPropertiesModel)

        widget.simulateFrcClicked.connect(self._uiClickSimulateFrc)
        widget.abortClicked.connect(self._uiClickAbort)

    # Internal methods
    def _doPostSimulationReset(self) -> None:
        self._actionLock.unlock()
        self.widget.setSimulating(False)
        self.widget.setAborting(False)
        self.widget.setProgressBarVisible(False)

    # Model event handling
    def _onFluorophoreSettingsLoad(self, fluorophoreSettings: FluorophoreSettings) -> None:
        self.widget.updateFluorophoreSettings(fluorophoreSettings)

    def _onImagingSystemSettingsLoad(self, imagingSystemSettings: ImagingSystemSettings) -> None:
        self.widget.updateImagingSystemSettings(imagingSystemSettings)

    def _onPulseSchemeLoad(self, pulseScheme: PulseScheme) -> None:
        self.widget.updatePulseScheme(pulseScheme)

    def _onSamplePropertiesLoad(self, sampleProperties: SampleProperties) -> None:
        self.widget.updateSampleProperties(sampleProperties)

    def _onCameraPropertiesLoad(self, cameraProperties: CameraProperties) -> None:
        self.widget.updateCameraProperties(cameraProperties)

    # UI event handling
    @pyqtSlot(FluorophoreSettings)
    def _uiSetFluorophoreSettingsModel(self, fluorophoreSettings: FluorophoreSettings) -> None:
        self.model.fluorophore_settings = fluorophoreSettings

    @pyqtSlot(ImagingSystemSettings)
    def _uiSetImagingSystemSettingsModel(self,
                                         imagingSystemSettings: ImagingSystemSettings) -> None:
        self.model.imaging_system_settings = imagingSystemSettings

    @pyqtSlot(PulseScheme)
    def _uiSetPulseSchemeModel(self, pulseScheme: PulseScheme) -> None:
        self.model.pulse_scheme = pulseScheme

    @pyqtSlot(SampleProperties)
    def _uiSetSamplePropertiesModel(self, sampleProperties: SampleProperties) -> None:
        self.model.sample_properties = sampleProperties

    @pyqtSlot(CameraProperties)
    def _uiSetCameraPropertiesModel(self, cameraProperties: CameraProperties) -> None:
        self.model.camera_properties = cameraProperties

    @pyqtSlot()
    def _uiClickSimulateFrc(self) -> None:
        if self._actionLock.tryLock():
            self.widget.setSimulating(True)

            self._currentWorker = self.FRCWorker(self.model)
            self._currentWorker.signals.done.connect(self._onWorkerDone)
            self._currentWorker.signals.aborted.connect(self._onWorkerAbort)
            self._currentWorker.signals.error.connect(self._onWorkerError)
            self._currentWorker.signals.preprocessingFinished.connect(self._onWorkerPreprocessed)
            self._currentWorker.signals.progressUpdated.connect(self._onWorkerProgressUpdate)
            self._threadPool.start(self._currentWorker)

    @pyqtSlot()
    def _uiClickAbort(self) -> None:
        if self._currentWorker is None or self._currentWorker.hasFinished():
            return

        confirmation_result = QMessageBox.question(
            self.widget, "Abort Simulation", "Abort the current simulation?")

        if confirmation_result == QMessageBox.Yes:
            self.widget.setAborting(True)
            self._currentWorker.abort()

    # Worker stuff
    @pyqtSlot(FrcSimulationResults)
    def _onWorkerDone(self, frcSimulationResults: FrcSimulationResults) -> None:
        try:
            self.widget.setFrcSimulationResults(frcSimulationResults)
        finally:
            self._doPostSimulationReset()

    @pyqtSlot()
    def _onWorkerAbort(self) -> None:
        self._doPostSimulationReset()
        self.widget.updateSimulationProgress(0)

    @pyqtSlot(str)
    def _onWorkerError(self, message: str) -> None:
        try:
            QMessageBox.critical(self.widget, "FRC simulation error", message)
        finally:
            self._doPostSimulationReset()
            self.widget.updateSimulationProgress(0)

    @pyqtSlot(int)
    def _onWorkerPreprocessed(self, numEvaluations: int) -> None:
        self.widget.setProgressBarVisible(numEvaluations > 1)

    @pyqtSlot(float)
    def _onWorkerProgressUpdate(self, progress: float) -> None:
        self.widget.updateSimulationProgress(progress)

    class FRCWorker(QRunnable):
        def __init__(self, runInstance: RunInstance) -> None:
            super().__init__()

            self.signals = self.Signals()
            self._runInstance = runInstance
            self._hasFinished = False

        def run(self) -> None:
            try:
                results = self._runInstance.simulate_frc(
                    self.signals.preprocessingFinished, self.signals.progressUpdated
                )

                if results is not None:
                    self.signals.done.emit(results)
                else:
                    self.signals.aborted.emit()
            except Exception as e:
                print(format_exc())
                self.signals.error.emit(str(e))
            finally:
                self._hasFinished = True

        def abort(self) -> None:
            self._runInstance.abort_running_simulations()

        def hasFinished(self) -> bool:
            return self._hasFinished

        class Signals(QObject):
            done = pyqtSignal(FrcSimulationResults)
            aborted = pyqtSignal()
            error = pyqtSignal(str)

            preprocessingFinished = pyqtSignal(int)
            progressUpdated = pyqtSignal(float)
