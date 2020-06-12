from dataclasses import dataclass, field
from dataclasses_json import config as json_config, Exclude
import numpy as np
from PySignal import Signal
from typing import Tuple

from frcpredict.util import observable_property, hidden_field
from .fluorophore import FluorophoreSettings
from .imaging import ImagingSystemSettings
from .pulse import PulseScheme
from .sample import SampleProperties
from .camera import CameraProperties


@dataclass
class RunInstance:
    fluorophore_settings: FluorophoreSettings = observable_property(
        "_fluorophore_settings", default=None,
        signal_name="fluorophore_settings_changed", emit_arg_name="fluorophore_settings"
    )
    imaging_system_settings: ImagingSystemSettings = observable_property(
        "_imaging_system_settings", default=None,
        signal_name="imaging_system_settings_changed", emit_arg_name="imaging_system_settings"
    )
    pulse_scheme: PulseScheme = observable_property(
        "_pulse_scheme", default=None,
        signal_name="pulse_scheme_changed", emit_arg_name="pulse_scheme"
    )
    sample_properties: SampleProperties = observable_property(
        "_sample_properties", default=None,
        signal_name="sample_properties_changed", emit_arg_name="sample_properties"
    )
    camera_properties: CameraProperties = observable_property(
        "_camera_properties", default=None,
        signal_name="camera_properties_changed", emit_arg_name="camera_properties"
    )

    # Signals
    fluorophore_settings_loaded: Signal = hidden_field(Signal)
    imaging_system_settings_loaded: Signal = hidden_field(Signal)
    pulse_scheme_loaded: Signal = hidden_field(Signal)
    sample_properties_loaded: Signal = hidden_field(Signal)
    camera_properties_loaded: Signal = hidden_field(Signal)
    
    # Methods
    def frc(self) -> Tuple[np.ndarray, np.ndarray]:
        """ TODO. Returns x and y. """
        pass
