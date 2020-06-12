from dataclasses import dataclass
from PySignal import Signal

from frcpredict.util import observable_property, hidden_field


@dataclass
class SampleProperties:
    spectral_power: float = observable_property("_spectral_power", default=0.0,
                                                signal_name="basic_field_changed")

    labelling_density: float = observable_property("_labelling_density", default=0.0,
                                                   signal_name="basic_field_changed")

    # Signals
    basic_field_changed: Signal = hidden_field(Signal)
