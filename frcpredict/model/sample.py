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
class SampleProperties:
    """
    A description of sample-related properties of an environment.
    """

    spectral_power: Union[float, Multivalue[float]] = multi_accepting_field(
        observable_field("_spectral_power", default=1.0, signal_name="basic_field_changed")
    )

    labelling_density: Union[float, Multivalue[float]] = multi_accepting_field(  # fluorophores per mm^2
        observable_field("_labelling_density", default=1.0, signal_name="basic_field_changed")
    )

    K_origin: Union[float, Multivalue[float]] = multi_accepting_field(
        observable_field("_K_origin", default=1.0, signal_name="basic_field_changed")
    )
