from dataclasses import dataclass
from PySignal import Signal

from frcpredict.util import observable_property, hidden_field


@dataclass
class CameraProperties:
    readout_noise: float = observable_property("_readout_noise", default=0.0,
                                               signal_name="basic_field_changed")

    quantum_efficiency: float = observable_property("_quantum_efficiency", default=0.0,
                                                    signal_name="basic_field_changed")

    # Signals
    basic_field_changed: Signal = hidden_field(Signal)
