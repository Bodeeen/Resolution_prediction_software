from dataclasses import dataclass
from typing import Union

from PySignal import Signal
from dataclasses_json import dataclass_json

from frcpredict.util import (
    dataclass_internal_attrs, observable_property, multi_accepting_field
)
from .multivalue import Multivalue


@dataclass_json
@dataclass_internal_attrs(basic_field_changed=Signal)
@dataclass
class CameraProperties:
    readout_noise: Union[float, Multivalue[float]] = multi_accepting_field(  # electrons
        observable_property("_readout_noise", default=0.0, signal_name="basic_field_changed")
    )

    quantum_efficiency: Union[float, Multivalue[float]] = multi_accepting_field(  # [0, 1]
        observable_property("_quantum_efficiency", default=0.0, signal_name="basic_field_changed")
    )
