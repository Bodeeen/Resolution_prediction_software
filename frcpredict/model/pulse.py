from dataclasses import dataclass
from typing import List


@dataclass
class Pulse:
    wavelength: float
    duration: float
    intensity_pattern: str


@dataclass
class PulseScheme:
    pulses: List[Pulse]
