from dataclasses import dataclass, field
from PySignal import Signal
from typing import Any, List, Dict


@dataclass
class IlluminationResponse:
    wavelength_start: int
    wavelength_end: int
    cross_section_off_to_on: float
    cross_section_on_to_off: float
    cross_section_emission: float

    # Internal fields
    _cross_section_off_to_on: float = field(init=False, repr=False, default=0.0)
    _cross_section_on_to_off: float = field(init=False, repr=False, default=0.0)
    _cross_section_emission: float = field(init=False, repr=False, default=0.0)
    _initialized: bool = field(init=False, repr=False, default=False)  # TODO: Fix this ugly stuff

    # Properties
    @property
    def cross_section_off_to_on(self) -> float:
        return self._cross_section_off_to_on

    @cross_section_off_to_on.setter
    def cross_section_off_to_on(self, cross_section_off_to_on: float) -> None:
        self._cross_section_off_to_on = cross_section_off_to_on
        if self._initialized:
            self.basic_field_changed.emit(self)

    @property
    def cross_section_on_to_off(self) -> float:
        return self._cross_section_on_to_off

    @cross_section_on_to_off.setter
    def cross_section_on_to_off(self, cross_section_on_to_off: float) -> None:
        self._cross_section_on_to_off = cross_section_on_to_off
        if self._initialized:
            self.basic_field_changed.emit(self)

    @property
    def cross_section_emission(self) -> float:
        return self._cross_section_emission

    @cross_section_emission.setter
    def cross_section_emission(self, cross_section_emission: float) -> None:
        self._cross_section_emission = cross_section_emission
        if self._initialized:
            self.basic_field_changed.emit(self)

    # Functions
    def __post_init__(self):  # TODO: Fix this ugly stuff
        self.basic_field_changed = Signal()
        self._initialized = True

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
        init=False, repr=False, default_factory=dict)
    _initialized: bool = field(init=False, repr=False, default=False)  # TODO: Fix this ugly stuff

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

    # Functions
    def __post_init__(self):  # TODO: Fix this ugly stuff
        self.response_added = Signal()
        self.response_removed = Signal()
        self._initialized = True

    def add_response(self, response: IlluminationResponse) -> bool:
        """
        Adds a response. Returns true if successful, or false if there was a wavelength collision.
        """

        for existing_response in self._responses.values():
            if (response.wavelength_start >= existing_response.wavelength_start and
                    response.wavelength_end <= existing_response.wavelength_end):
                return False

        self._responses[response.wavelength_start] = response
        if self._initialized:
            self.response_added.emit(response)

        return True

    def remove_response(self, wavelength_start) -> None:
        """ Removes the response with the specified wavelength attributes. """
        removed_response = self._responses.pop(wavelength_start)
        if self._initialized:
            self.response_removed.emit(removed_response)

    def clear_responses(self) -> None:
        """ Removes all responses. """
        for wavelength_start in self._responses.keys():
            self.remove_response(wavelength_start)
