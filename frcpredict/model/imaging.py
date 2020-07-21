from dataclasses import dataclass
from typing import Union

from PySignal import Signal
from dataclasses_json import dataclass_json

from frcpredict.util import (
    dataclass_internal_attrs, dataclass_with_observables, observable_field, multi_accepting_field
)
from .pattern import Pattern
from .multivalue import Multivalue


@dataclass_json
@dataclass_with_observables
@dataclass_internal_attrs(basic_field_changed=Signal)
@dataclass
class ImagingSystemSettings:
    """
    A description of an imaging system.
    """

    optical_psf: Pattern = Pattern()

    pinhole_function: Pattern = Pattern()

    scanning_step_size: Union[float, Multivalue[float]] = multi_accepting_field(
        observable_field("_scanning_step_size", default=20.0, signal_name="basic_field_changed")
    )
