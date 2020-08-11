from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from PySignal import Signal
from dataclasses_json import dataclass_json

from frcpredict.simulation import simulate
from frcpredict.util import (
    dataclass_internal_attrs, dataclass_with_properties, observable_property, extended_field
)
from .camera import CameraProperties
from .fluorophore import FluorophoreSettings
from .imaging import ImagingSystemSettings
from .pulse import PulseScheme
from .sample import SampleProperties


@dataclass_json
@dataclass_with_properties
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

    fluorophore_settings: FluorophoreSettings = extended_field(
        observable_property(
            "_fluorophore_settings", default=FluorophoreSettings,
            signal_name="fluorophore_settings_loaded", emit_arg_name="fluorophore_settings"
        ),
        description="fluorophore settings"
    )

    imaging_system_settings: ImagingSystemSettings = extended_field(
        observable_property(
            "_imaging_system_settings", default=ImagingSystemSettings,
            signal_name="imaging_system_settings_loaded", emit_arg_name="imaging_system_settings"
        ),
        description="imaging system settings"
    )

    pulse_scheme: PulseScheme = extended_field(
        observable_property(
            "_pulse_scheme", default=PulseScheme,
            signal_name="pulse_scheme_loaded", emit_arg_name="pulse_scheme"
        ),
        description="pulse scheme"
    )

    sample_properties: SampleProperties = extended_field(
        observable_property(
            "_sample_properties", default=SampleProperties,
            signal_name="sample_properties_loaded", emit_arg_name="sample_properties"
        ),
        description="sample properties"
    )

    camera_properties: CameraProperties = extended_field(
        observable_property(
            "_camera_properties", default=CameraProperties,
            signal_name="camera_properties_loaded", emit_arg_name="camera_properties"
        ),
        description="camera properties"
    )

    # Methods
    def simulate(self, preprocessing_finished_callback: Optional[Signal] = None,
                 progress_updated_callback: Optional[Signal] = None):
        """ Runs the simulation and returns the results. """
        return simulate(deepcopy(self),
                        abort_signal=self._abort_signal,
                        preprocessing_finished_callback=preprocessing_finished_callback,
                        progress_updated_callback=progress_updated_callback)

    def abort_running_simulations(self) -> None:
        """
        Emits a signal to abort any running simulations. Note that the simulations may not terminate
        immediately after this method returns.
        """
        self._abort_signal.emit()
