from copy import deepcopy
import numpy as np

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, QThreadPool, QRunnable, QMutex

from frcpredict.model import (
    FluorophoreSettings, IlluminationResponse,
    Pattern, Array2DPatternData, GaussianPatternData, DoughnutPatternData, AiryPatternData, DigitalPinholePatternData,
    ImagingSystemSettings,
    PulseScheme, Pulse, PulseType,
    SampleProperties,
    CameraProperties,
    RunInstance,
    JsonContainer
)
from frcpredict.ui import BasePresenter


class MainWindowPresenter(BasePresenter[RunInstance]):
    """
    Presenter for the main window.
    """

    @BasePresenter.model.setter
    def model(self, model: RunInstance) -> None:
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
        self.widget.simulateFrcClicked.connect(self._uiOnClickSimulateFrc)
        
        # TODO: These are predefined values; remove them when we have proper preset support
        model.fluorophore_settings.add_response(IlluminationResponse(wavelength_start=405, wavelength_end=405, cross_section_off_to_on=6.6e-07, cross_section_on_to_off=0.0, cross_section_emission=5e-08))
        model.fluorophore_settings.add_response(IlluminationResponse(wavelength_start=488, wavelength_end=488, cross_section_off_to_on=3.3e-08, cross_section_on_to_off=7.3e-07, cross_section_emission=7e-06))
        model.imaging_system_settings.optical_psf = Pattern(pattern_data=GaussianPatternData(fwhm=250))
        model.imaging_system_settings.pinhole_function = Pattern(pattern_data=DigitalPinholePatternData(fwhm=220))
        model.pulse_scheme.add_pulse(Pulse(pulse_type=PulseType.on, wavelength=405, duration=0.1, max_intensity=10.0, illumination_pattern=Pattern(pattern_data=AiryPatternData(fwhm=200))))
        model.pulse_scheme.add_pulse(Pulse(pulse_type=PulseType.off, wavelength=488, duration=4.0, max_intensity=5.1, illumination_pattern=Pattern(pattern_data=DoughnutPatternData(periodicity=510))))
        model.pulse_scheme.add_pulse(Pulse(pulse_type=PulseType.readout, wavelength=488, duration=0.4, max_intensity=13.0, illumination_pattern=Pattern(pattern_data=AiryPatternData(fwhm=230))))
        model.sample_properties=SampleProperties(spectral_power=6.1, labelling_density=5.0, K_origin=3.19)
        model.camera_properties=CameraProperties(readout_noise=0.0, quantum_efficiency=0.82)
        self.model = model
        self.widget.fluorophoreSettings.listResponses.setCurrentRow(0)

    # Model event handling
    def _onFluorophoreSettingsLoad(self, fluorophore_settings: FluorophoreSettings) -> None:
        self._widget.updateFluorophoreSettings(fluorophore_settings)

    def _onImagingSystemSettingsLoad(self, imaging_system_settings: ImagingSystemSettings) -> None:
        self._widget.updateImagingSystemSettings(imaging_system_settings)
        
    def _onPulseSchemeLoad(self, pulse_scheme: PulseScheme) -> None:
        self._widget.updatePulseScheme(pulse_scheme)
        
    def _onSamplePropertiesLoad(self, sample_properties: SampleProperties) -> None:
        self._widget.updateSampleProperties(sample_properties)
        
    def _onCameraPropertiesLoad(self, camera_properties: CameraProperties) -> None:
        self._widget.updateCameraProperties(camera_properties)

    # UI event handling
    @pyqtSlot()
    def _uiOnClickSimulateFrc(self) -> None:
        if self._actionLock.tryLock():
            self.widget.setSimulating(True)
            
            worker = self.FRCWorker(deepcopy(self.model))
            worker.signals.done.connect(self._onWorkerDone)
            self._threadPool.start(worker)

    # Worker stuff
    @pyqtSlot(np.ndarray, np.ndarray)
    def _onWorkerDone(self, x: np.ndarray, y: np.ndarray) -> None:
        try:
            self._widget.setFrcResults(x, y)
        finally:
            self._actionLock.unlock()
            self.widget.setSimulating(False)

    class FRCWorker(QRunnable):
        def __init__(self, run_instance: RunInstance) -> None:
            super().__init__()
            self.run_instance = run_instance
            self.signals = self.Signals()

        def run(self) -> None:
            x, y = self.run_instance.frc()
            self.signals.done.emit(x, y)

        class Signals(QObject):
            done = pyqtSignal(np.ndarray, np.ndarray)
