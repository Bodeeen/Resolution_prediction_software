from copy import deepcopy
import numpy as np

from PyQt5.QtCore import pyqtSignal, pyqtSlot, QObject, QThreadPool, QRunnable

from frcpredict.model import FluorophoreSettings, IlluminationResponse, Pulse, ImagingSystemSettings, PulseScheme, SampleProperties, CameraProperties, RunInstance
from frcpredict.ui import BasePresenter
from frcpredict.util import patterns


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
                optical_psf=np.zeros((80, 80)),
                pinhole_function=np.zeros((80, 80)),
                scanning_step_size=1.0
            ),
            pulse_scheme=PulseScheme(pulses=[]),
            sample_properties=SampleProperties(spectral_power=1.0, labelling_density=1.0),
            camera_properties=CameraProperties(readout_noise=0.0, quantum_efficiency=0.75)
        )

        super().__init__(model, widget)

        # Connect UI events
        self.widget.simulateFrcClicked.connect(self._uiOnClickSimulateFrc)

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
        pass  # TODO