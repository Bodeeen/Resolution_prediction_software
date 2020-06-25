from dataclasses import dataclass, field
from dataclasses_json import dataclass_json
import numpy as np
from PySignal import Signal
from typing import Tuple

from frcpredict.util import dataclass_internal_attrs, observable_property
from .fluorophore import FluorophoreSettings
from .imaging import ImagingSystemSettings
from .pulse import PulseScheme
from .sample import SampleProperties
from .camera import CameraProperties

from Old_scripts.spectral_analysis_new_temp import simulate  # TODO: Temp!


@dataclass_json
@dataclass_internal_attrs(
    fluorophore_settings_loaded=Signal,
    imaging_system_settings_loaded=Signal,
    pulse_scheme_loaded=Signal,
    sample_properties_loaded=Signal,
    camera_properties_loaded=Signal
)
@dataclass
class RunInstance:
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
    def frc(self) -> Tuple[np.ndarray, np.ndarray]:
        """ TODO. Returns x and y. """
        frc_spectra, df = simulate(self)
        x = (np.arange(0, len(frc_spectra)) * df)
        
        return x, frc_spectra
