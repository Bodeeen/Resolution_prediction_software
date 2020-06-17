from dataclasses import dataclass, field
from dataclasses_json import config as json_config, Exclude
import numpy as np
from osgeo import gdal_array
from PySignal import Signal

from frcpredict.util import observable_property, hidden_field
from .pattern import Pattern


@dataclass
class ImagingSystemSettings:
    optical_psf: Pattern

    pinhole_function: Pattern

    scanning_step_size: float = observable_property(
        "_scanning_step_size", default=0.0,
        signal_name="basic_field_changed"
    )

    # Signals
    basic_field_changed: Signal = hidden_field(Signal)
