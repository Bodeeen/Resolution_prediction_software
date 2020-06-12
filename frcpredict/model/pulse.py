from dataclasses import dataclass, field
from dataclasses_json import config as json_config, Exclude
from collections import OrderedDict
from PySignal import Signal
import numpy as np
from typing import List, Dict
import uuid

from frcpredict.util import observable_property, hidden_field


@dataclass
class Pulse:
    wavelength: int = observable_property(
        "_wavelength", default=0.0,
        signal_name="basic_field_changed"
    )

    duration: float = observable_property(
        "_duration", default=0.0,
        signal_name="basic_field_changed"
    )

    max_intensity: float = observable_property(
        "_max_intensity", default=0.0,
        signal_name="basic_field_changed"
    )

    illumination_pattern: np.ndarray = observable_property(
        "_illumination_pattern", default=np.zeros((80, 80)),
        signal_name="illumination_pattern_changed", emit_arg_name="illumination_pattern")

    # Signals
    basic_field_changed: Signal = hidden_field(Signal)
    illumination_pattern_changed: Signal = hidden_field(Signal)


@dataclass
class PulseScheme:
    pulses: List[Pulse]

    # Internal fields
    _pulses: OrderedDict = hidden_field(OrderedDict)  # [key, value]: [str, Pulse]
    
    # Signals
    pulse_added: Signal = hidden_field(Signal)
    pulse_moved: Signal = hidden_field(Signal)
    pulse_removed: Signal = hidden_field(Signal)

    # Properties
    @property
    def pulses(self) -> List[Pulse]:
        return self._pulses.values()

    @pulses.setter
    def pulses(self, pulses: List[Pulse]) -> None:
        self._pulses = {}

        self.clear_pulses()
        for pulse in pulses:
            self.add_pulse(pulse)
    
    # Methods
    def add_pulse(self, pulse: Pulse) -> None:
        """ Adds a pulse to the pulse scheme. """
        key = str(uuid.uuid4())  # Random key
        self._pulses[key] = pulse
        self.pulse_added.emit(key, pulse)

    def remove_pulse(self, key: str) -> None:
        """ Removes the pulse with the specified key from the pulse scheme. """
        removed_pulse = self._pulses.pop(key)
        self.pulse_removed.emit(key, removed_pulse)

    def clear_pulses(self) -> None:
        """ Removes all pulses from the pulse scheme. """
        for key in self._pulses.keys():
            self.remove_pulse(key)

    def move_pulse_left(self, key: str) -> None:
        """ Moves the pulse with the specified key one step to the left in the order. """
        existing_keys = list(self._pulses.keys()).copy()
        
        i = len(existing_keys) - 1
        while i >= 0:
            has_moved_requested = False
            if key == existing_keys[i] and len(existing_keys) > i - 1:
                self._pulses.move_to_end(existing_keys[i - 1], last=False)
                has_moved_requested = True
            
            self._pulses.move_to_end(existing_keys[i], last=False)
            if has_moved_requested:
                i -= 2
            else:
                i -= 1

        self.pulse_moved.emit(key, self._pulses[key])

    def move_pulse_right(self, key: str) -> None:
        """ Moves the pulse with the specified key one step to the right in the order. """
        existing_keys = list(self._pulses.keys()).copy()
        
        i = 0
        while i < len(existing_keys):
            has_moved_requested = False
            if key == existing_keys[i] and len(existing_keys) > i + 1:
                self._pulses.move_to_end(existing_keys[i + 1], last=True)
                has_moved_requested = True
            
            self._pulses.move_to_end(existing_keys[i], last=True)
            if has_moved_requested:
                i += 2
            else:
                i += 1

        self.pulse_moved.emit(key, self._pulses[key])

