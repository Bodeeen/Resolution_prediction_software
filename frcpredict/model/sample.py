from dataclasses import dataclass
from dataclasses_json import dataclass_json
from PySignal import Signal

from frcpredict.util import dataclass_internal_attrs, observable_property


@dataclass_json
@dataclass_internal_attrs(basic_field_changed=Signal)
@dataclass
class SampleProperties:
    spectral_power: float = observable_property("_spectral_power", default=0.0,
                                                signal_name="basic_field_changed")

    labelling_density: float = observable_property("_labelling_density", default=0.0,
                                                   signal_name="basic_field_changed")

    K_origin: float = observable_property("_K_origin", default=0.0,
                                          signal_name="basic_field_changed")
