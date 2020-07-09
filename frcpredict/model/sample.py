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
class SampleProperties:
    spectral_power: Union[float, Multivalue[float]] = multi_accepting_field(
        observable_property("_spectral_power", default=0.0, signal_name="basic_field_changed")
    )

    labelling_density: Union[float, Multivalue[float]] = multi_accepting_field(  # fluorophores per mm^2
        observable_property("_labelling_density", default=0.0, signal_name="basic_field_changed")
    )

    K_origin: Union[float, Multivalue[float]] = multi_accepting_field(
        observable_property("_K_origin", default=0.0, signal_name="basic_field_changed")
    )
