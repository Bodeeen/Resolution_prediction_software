from dataclasses import dataclass
from typing import Union

from PySignal import Signal
from dataclasses_json import dataclass_json

from frcpredict.util import (
    dataclass_internal_attrs, dataclass_with_properties, observable_property, extended_field
)
from .pattern import Pattern
from .multivalue import Multivalue


@dataclass_json
@dataclass_with_properties
@dataclass_internal_attrs(basic_field_changed=Signal)
@dataclass
class ImagingSystemSettings:
    """
    A description of an imaging system.
    """

    optical_psf: Pattern = extended_field(default_factory=Pattern,
                                          description="optical PSF")

    pinhole_function: Pattern = extended_field(default_factory=Pattern,
                                               description="pinhole function")

    scanning_step_size: Union[float, Multivalue[float]] = extended_field(
        observable_property("_scanning_step_size", default=20.0, signal_name="basic_field_changed"),
        description="scanning step size [nm]", accept_multivalues=True
    )
