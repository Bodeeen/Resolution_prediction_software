from dataclasses import dataclass
from typing import Union

from PySignal import Signal
from dataclasses_json import dataclass_json

from frcpredict.util import (
    dataclass_internal_attrs, observable_property, rangeable_field
)
from .value_range import ValueRange


@dataclass_json
@dataclass_internal_attrs(basic_field_changed=Signal)
@dataclass
class SampleProperties:
    spectral_power: Union[float, ValueRange[float]] = rangeable_field(
        observable_property("_spectral_power", default=0.0, signal_name="basic_field_changed")
    )

    labelling_density: Union[float, ValueRange[float]] = rangeable_field(  # fluorophores per mm^2
        observable_property("_labelling_density", default=0.0, signal_name="basic_field_changed")
    )

    K_origin: Union[float, ValueRange[float]] = rangeable_field(
        observable_property("_K_origin", default=0.0, signal_name="basic_field_changed")
    )
