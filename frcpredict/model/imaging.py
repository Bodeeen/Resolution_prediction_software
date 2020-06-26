from dataclasses import dataclass
from dataclasses_json import dataclass_json
from PySignal import Signal

from frcpredict.util import dataclass_internal_attrs, observable_property
from .pattern import Pattern


@dataclass_json
@dataclass_internal_attrs(basic_field_changed=Signal)
@dataclass
class ImagingSystemSettings:
    optical_psf: Pattern

    pinhole_function: Pattern

    scanning_step_size: float = observable_property(
        "_scanning_step_size", default=0.0,
        signal_name="basic_field_changed"
    )
