from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from PySignal import Signal
from dataclasses_json import dataclass_json

from Old_scripts.spectral_analysis_new_temp import simulate  # TODO: Temp!
from frcpredict.util import dataclass_internal_attrs, observable_property
from .camera import CameraProperties
from .fluorophore import FluorophoreSettings
from .imaging import ImagingSystemSettings
from .pulse import PulseScheme
from .sample import SampleProperties


@dataclass_json
@dataclass_internal_attrs(
    fluorophore_settings_loaded=Signal,
    imaging_system_settings_loaded=Signal,
    pulse_scheme_loaded=Signal,
    sample_properties_loaded=Signal,
    camera_properties_loaded=Signal,

    _abort_signal=Signal
)
@dataclass
class RunInstance:
    """
    A description of all parameters part of a simulation that can be run.
    """

    fluorophore_settings: FluorophoreSettings = observable_property(
        "_fluorophore_settings", default=None,
        signal_name="fluorophore_settings_loaded", emit_arg_name="fluorophore_settings"
    )
    imaging_system_settings: ImagingSystemSettings = observable_property(
        "_imaging_system_settings", default=None,
        signal_name="imaging_system_settings_loaded", emit_arg_name="imaging_system_settings"
    )
    pulse_scheme: PulseScheme = observable_property(
        "_pulse_scheme", default=None,
        signal_name="pulse_scheme_loaded", emit_arg_name="pulse_scheme"
    )
    sample_properties: SampleProperties = observable_property(
        "_sample_properties", default=None,
        signal_name="sample_properties_loaded", emit_arg_name="sample_properties"
    )
    camera_properties: CameraProperties = observable_property(
        "_camera_properties", default=None,
        signal_name="camera_properties_loaded", emit_arg_name="camera_properties"
    )

    # Methods
    def simulate_frc(self, preprocessing_finished_callback: Optional[Signal] = None,
                     progress_updated_callback: Optional[Signal] = None):
        """ Simulates FRC curves. """
        return simulate(
            deepcopy(self),
            self._abort_signal,
            preprocessing_finished_callback,
            progress_updated_callback
        )

    def abort_running_simulations(self) -> None:
        """
        Emits a signal to abort any running simulations. Note that the simulations may not terminate
        immediately after this method returns.
        """
        self._abort_signal.emit()

