from dataclasses import dataclass
from typing import Union

from PySignal import Signal
from dataclasses_json import dataclass_json

from frcpredict.util import (
    dataclass_internal_attrs, observable_property, rangeable_field
)
from .pattern import Pattern
from .value_range import ValueRange


@dataclass_json
@dataclass_internal_attrs(basic_field_changed=Signal)
@dataclass
class ImagingSystemSettings:
    optical_psf: Pattern

    pinhole_function: Pattern

    scanning_step_size: Union[float, ValueRange[float]] = rangeable_field(
        observable_property("_scanning_step_size", default=0.0, signal_name="basic_field_changed")
    )
