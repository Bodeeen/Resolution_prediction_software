from dataclasses import dataclass
from typing import Union

from PySignal import Signal
from dataclasses_json import dataclass_json

from frcpredict.util import (
    dataclass_internal_attrs, dataclass_with_observables, observable_field, multi_accepting_field
)
from .multivalue import Multivalue


@dataclass_json
@dataclass_with_observables
@dataclass_internal_attrs(basic_field_changed=Signal)
@dataclass
class CameraProperties:
    """
    A description of a camera.
    """

    readout_noise: Union[float, Multivalue[float]] = multi_accepting_field(  # electrons
        observable_field("_readout_noise", default=0.0, signal_name="basic_field_changed")
    )

    quantum_efficiency: Union[float, Multivalue[float]] = multi_accepting_field(  # [0, 1]
        observable_field("_quantum_efficiency", default=0.75, signal_name="basic_field_changed")
    )
