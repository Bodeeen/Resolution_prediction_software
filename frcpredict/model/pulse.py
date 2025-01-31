import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Union, List

from PySignal import Signal
from dataclasses_json import dataclass_json

from frcpredict.util import (
    dataclass_internal_attrs, dataclass_with_properties,
    dataclass_property, observable_property, extended_field
)
from .pattern import Pattern
from .multivalue import Multivalue


@dataclass_json
@dataclass_with_properties
@dataclass_internal_attrs(basic_field_changed=Signal)
@dataclass
class Pulse:
    """
    A description of a laser pulse.
    """

    wavelength: Union[float, Multivalue[float]] = extended_field(
        observable_property("_wavelength", default=0.0, signal_name="basic_field_changed"),
        description="wavelength [nm]", accept_multivalues=True
    )

    duration: Union[float, Multivalue[float]] = extended_field(
        observable_property("_duration", default=0.0, signal_name="basic_field_changed"),
        description="duration [ms]", accept_multivalues=True
    )

    max_intensity: Union[float, Multivalue[float]] = extended_field(
        observable_property("_max_intensity", default=0.0, signal_name="basic_field_changed"),
        description="max intensity [kW/cm²]", accept_multivalues=True
    )

    illumination_pattern: Pattern = extended_field(default_factory=Pattern,
                                                   description="illumination pattern")


@dataclass_json
@dataclass_with_properties
@dataclass_internal_attrs(
    _pulses=OrderedDict, pulse_added=Signal, pulse_moved=Signal, pulse_removed=Signal
)
@dataclass
class PulseScheme:
    """
    A description of a laser pulse scheme.
    """

    pulses: List[Pulse] = extended_field(
        dataclass_property(
            fget=lambda self: self._get_pulses(),
            fset=lambda self, pulses: self._set_pulses(pulses),
            default=list
        ),
        description=lambda _self, index: f"Pulse #{index + 1}"
    )

    # Methods
    def add_pulse(self, pulse: Pulse) -> None:
        """ Adds a pulse to the pulse scheme. """
        key = str(uuid.uuid4())  # Generate random key
        self._pulses[key] = pulse
        self.pulse_added.emit(key, pulse)

    def remove_pulse(self, key: str) -> None:
        """ Removes the pulse with the specified key from the pulse scheme. """
        removed_pulse = self._pulses.pop(key)
        self.pulse_removed.emit(key, removed_pulse)

    def clear_pulses(self) -> None:
        """ Removes all pulses from the pulse scheme. """
        keys = [*self._pulses.keys()]
        for key in keys:
            self.remove_pulse(key)

    def move_pulse_left(self, key: str) -> None:
        """ Moves the pulse with the specified key one step to the left in the order. """
        existing_keys = [*self._pulses.keys()].copy()

        i = len(existing_keys) - 1
        while i >= 0:
            has_moved_requested = False
            if key == existing_keys[i] and 0 < i:
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
        existing_keys = [*self._pulses.keys()].copy()

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

    def get_pulses_with_keys(self):
        return self._pulses.items()

    # Internal methods
    def _get_pulses(self) -> List[Pulse]:
        return [*self._pulses.values()]

    def _set_pulses(self, pulses: List[Pulse]) -> None:
        if not isinstance(pulses, list):
            return

        self.clear_pulses()
        for pulse in pulses:
            self.add_pulse(pulse)
