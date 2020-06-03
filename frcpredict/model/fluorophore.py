from dataclasses import dataclass, field
from dataclasses_json import config as json_config, Exclude
from PySignal import Signal
from typing import Any, List, Dict

from frcpredict.util import observable_property


@dataclass
class IlluminationResponse:
    wavelength_start: int
    wavelength_end: int

    cross_section_off_to_on: float = observable_property("_cross_section_off_to_on", default=0.0,
                                                      signal_name="basic_field_changed")

    cross_section_on_to_off: float = observable_property("_cross_section_on_to_off", default=0.0,
                                                      signal_name="basic_field_changed")

    cross_section_emission: float = observable_property("_cross_section_emission", default=0.0,
                                                     signal_name="basic_field_changed")

    # Signals
    basic_field_changed: Signal = field(
        init=False, repr=False, default_factory=Signal, metadata=json_config(exclude=Exclude.ALWAYS))

    # Methods
    def __str__(self) -> str:
        if self.wavelength_start == self.wavelength_end:
            return f"{self.wavelength_start} nm"
        else:
            return f"{self.wavelength_start}–{self.wavelength_end} nm"


@dataclass
class FluorophoreSettings:
    responses: List[IlluminationResponse]

    # Internal fields
    _responses: Dict[int, IlluminationResponse] = field(
        init=False, repr=False, default_factory=dict, metadata=json_config(exclude=Exclude.ALWAYS))

    # Signals
    response_added: Signal = field(
        init=False, repr=False, default_factory=Signal, metadata=json_config(exclude=Exclude.ALWAYS))
    response_removed: Signal = field(
        init=False, repr=False, default_factory=Signal, metadata=json_config(exclude=Exclude.ALWAYS))

    # Properties
    @property
    def responses(self) -> List[IlluminationResponse]:
        return self._responses.values()

    @responses.setter
    def responses(self, responses: List[IlluminationResponse]) -> None:
        self._responses = {}

        self.clear_responses()
        for response in responses:
            self.add_response(response)

    # Methods
    def add_response(self, response: IlluminationResponse) -> bool:
        """
        Adds a response. Returns true if successful, or false if there was a wavelength collision.
        """

        for existing_response in self._responses.values():
            if (response.wavelength_start >= existing_response.wavelength_start and
                    response.wavelength_end <= existing_response.wavelength_end):
                return False

        self._responses[response.wavelength_start] = response
        self.response_added.emit(response)
        return True

    def remove_response(self, wavelength_start: int) -> None:
        """ Removes the response with the specified wavelength attributes. """
        removed_response = self._responses.pop(wavelength_start)
        self.response_removed.emit(removed_response)

    def clear_responses(self) -> None:
        """ Removes all responses. """
        for wavelength_start in self._responses.keys():
            self.remove_response(wavelength_start)
