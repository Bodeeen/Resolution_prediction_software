from dataclasses import dataclass
from typing import Union, Optional

from PySignal import Signal
from dataclasses_json import dataclass_json

from frcpredict.util import (
    dataclass_internal_attrs, dataclass_with_properties, observable_property, extended_field
)
from .multivalue import Multivalue
from .pattern import Pattern


@dataclass_json
@dataclass_with_properties
@dataclass_internal_attrs(basic_field_changed=Signal)
@dataclass
class DetectorProperties:
    """
    A description of a detector.
    """

    readout_noise: Union[float, Multivalue[float]] = extended_field(  # rms
        observable_property("_readout_noise", default=0.0, signal_name="basic_field_changed"),
        description="readout noise [e⁻]", accept_multivalues=True
    )

    quantum_efficiency: Union[float, Multivalue[float]] = extended_field(
        observable_property("_quantum_efficiency", default=0.75, signal_name="basic_field_changed"),
        description="quantum efficiency", accept_multivalues=True
    )

    camera_pixel_size: Optional[Union[float, Multivalue[float]]] = extended_field(
        # nanometres (if point detector: None)
        observable_property("_camera_pixel_size", default=None, signal_name="basic_field_changed"),
        description="camera pixel size", accept_multivalues=True
    )

    def get_total_readout_noise_var(self, pinhole_function: Pattern) -> float:
        """ Returns the total readout noise of the detector. """

        if self.camera_pixel_size is not None:
            pinhole_sum = (pinhole_function.get_numpy_array(self.camera_pixel_size) ** 2).sum()
            return pinhole_sum * self.readout_noise ** 2
        elif self.readout_noise is not None:
            return self.readout_noise ** 2
        else:
            raise Exception("readout_noise or camera_pixel_size must not be None!")
