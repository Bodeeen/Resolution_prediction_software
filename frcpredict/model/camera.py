from dataclasses import dataclass
from dataclasses_json import dataclass_json
from PySignal import Signal

from frcpredict.util import dataclass_internal_attrs, observable_property


@dataclass_json
@dataclass_internal_attrs(basic_field_changed=Signal)
@dataclass
class CameraProperties:
    readout_noise: float = observable_property("_readout_noise", default=0.0,
                                               signal_name="basic_field_changed")

    quantum_efficiency: float = observable_property("_quantum_efficiency", default=0.0,
                                                    signal_name="basic_field_changed")
