from dataclasses import dataclass
from typing import Union

from PySignal import Signal
from dataclasses_json import dataclass_json

from frcpredict.util import (
    dataclass_internal_attrs, dataclass_with_properties, observable_property, extended_field
)
from .multivalue import Multivalue


@dataclass_json
@dataclass_with_properties
@dataclass_internal_attrs(basic_field_changed=Signal)
@dataclass
class CameraProperties:
    """
    A description of a camera.
    """

    readout_noise: Union[float, Multivalue[float]] = extended_field(
        observable_property("_readout_noise", default=0.0, signal_name="basic_field_changed"),
        description="readout noise [e⁻]", accept_multivalues=True
    )

    quantum_efficiency: Union[float, Multivalue[float]] = extended_field(
        observable_property("_quantum_efficiency", default=0.75, signal_name="basic_field_changed"),
        description="quantum efficiency", accept_multivalues=True
    )
