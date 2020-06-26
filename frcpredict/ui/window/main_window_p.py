from copy import deepcopy
import numpy as np
from traceback import format_exc

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, QThreadPool, QRunnable, QMutex
from PyQt5.QtWidgets import QMessageBox

from frcpredict.model import (
    Pattern, Array2DPatternData,
    FluorophoreSettings, ImagingSystemSettings, PulseScheme, SampleProperties, CameraProperties,
    RunInstance
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
        # Initialize model
        model = RunInstance(
            fluorophore_settings=FluorophoreSettings(responses=[]),
            imaging_system_settings=ImagingSystemSettings(
                optical_psf=Pattern(pattern_data=Array2DPatternData()),
                pinhole_function=Pattern(pattern_data=Array2DPatternData()),
                scanning_step_size=20.0
            ),
            pulse_scheme=PulseScheme(pulses=[]),
            sample_properties=SampleProperties(
                spectral_power=1.0,
                labelling_density=1.0,
                K_origin=1.0
            ),
            camera_properties=CameraProperties(
                readout_noise=0.0,
                quantum_efficiency=0.75
            )
        )

        super().__init__(model, widget)
        self._threadPool = QThreadPool()
        self._actionLock = QMutex()

        # Connect UI events
        self.widget.fluorophoreSettingsModelSet.connect(self._uiSetFluorophoreSettingsModel)
        self.widget.imagingSystemSettingsModelSet.connect(self._uiSetImagingSystemSettingsModel)
        self.widget.pulseSchemeModelSet.connect(self._uiSetPulseSchemeModel)
        self.widget.samplePropertiesModelSet.connect(self._uiSetSamplePropertiesModel)
        self.widget.cameraPropertiesModelSet.connect(self._uiSetCameraPropertiesModel)
        
        self.widget.simulateFrcClicked.connect(self._uiClickSimulateFrc)

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
    def _uiSetImagingSystemSettingsModel(self, imagingSystemSettings: ImagingSystemSettings) -> None:
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

            worker = self.FRCWorker(deepcopy(self.model))
            worker.signals.done.connect(self._onWorkerDone)
            worker.signals.error.connect(self._onWorkerError)
            self._threadPool.start(worker)

    # Worker stuff
    @pyqtSlot(np.ndarray, np.ndarray)
    def _onWorkerDone(self, x: np.ndarray, y: np.ndarray) -> None:
        try:
            self.widget.setFrcResults(x, y)
        finally:
            self._actionLock.unlock()
            self.widget.setSimulating(False)

    @pyqtSlot(str)
    def _onWorkerError(self, message: str) -> None:
        try:
            QMessageBox.critical(self.widget, "FRC calculation error", message)
        finally:
            self._actionLock.unlock()
            self.widget.setSimulating(False)

    class FRCWorker(QRunnable):
        def __init__(self, run_instance: RunInstance) -> None:
            super().__init__()
            self.run_instance = run_instance
            self.signals = self.Signals()

        def run(self) -> None:
            try:
                x, y = self.run_instance.frc()
                self.signals.done.emit(x, y)
            except Exception as e:
                self.signals.error.emit(str(e))
                print(format_exc())

        class Signals(QObject):
            done = pyqtSignal(np.ndarray, np.ndarray)
            error = pyqtSignal(str)
