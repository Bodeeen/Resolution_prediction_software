from dataclasses import dataclass
from typing import Optional, Union, List

import numpy as np
from dataclasses_json import dataclass_json
from PySignal import Signal
from scipy.interpolate import interp1d

from frcpredict.util import (
    dataclass_internal_attrs, dataclass_with_properties,
    dataclass_property, observable_property, extended_field
)
from .multivalue import Multivalue


@dataclass_json
@dataclass_with_properties
@dataclass_internal_attrs(basic_field_changed=Signal)
@dataclass
class IlluminationResponse:
    """
    A description of the illumination response of a fluorophore at a certain wavelength.
    """

    wavelength: float = extended_field(
        observable_property("_wavelength", default=0.0, signal_name="basic_field_changed"),
        description="wavelength [nm]"
    )

    cross_section_off_to_on: Union[float, Multivalue[float]] = extended_field(
        observable_property("_cross_section_off_to_on", default=0.0,
                            signal_name="basic_field_changed"),
        description="ON cross section", accept_multivalues=True
    )

    cross_section_on_to_off: Union[float, Multivalue[float]] = extended_field(
        observable_property("_cross_section_on_to_off", default=0.0,
                            signal_name="basic_field_changed"),
        description="OFF cross section", accept_multivalues=True
    )

    cross_section_emission: Union[float, Multivalue[float]] = extended_field(
        observable_property("_cross_section_emission", default=0.0,
                            signal_name="basic_field_changed"),
        description="emission cross section", accept_multivalues=True
    )

    # Methods
    def __str__(self) -> str:
        return f"{self.wavelength} nm"


@dataclass_json
@dataclass_with_properties
@dataclass_internal_attrs(_responses=dict, response_added=Signal, response_removed=Signal)
@dataclass
class FluorophoreSettings:
    """
    A description of a fluorophore.
    """

    responses: List[IlluminationResponse] = extended_field(
        dataclass_property(
            fget=lambda self: self._get_responses(),
            fset=lambda self, responses: self._set_responses(responses),
            default=list
        ),
        description=lambda self, index: str(self.responses[index])
    )

    # Methods
    def add_response(self, response: IlluminationResponse) -> bool:
        """
        Adds a response. Returns true if successful, or false if the wavelength was invalid or
        there is an existing response that includes the wavelength.
        """

        for existing_response in self._responses.values():
            if existing_response.wavelength == response.wavelength:
                return False

        self._responses[response.wavelength] = response
        self.response_added.emit(response)
        return True

    def remove_response(self, wavelength: float) -> None:
        """ Removes the response with the specified wavelength attributes. """
        removed_response = self._responses.pop(wavelength)
        self.response_removed.emit(removed_response)

    def clear_responses(self) -> None:
        """ Removes all responses. """
        for wavelength in [*self._responses.keys()]:
            self.remove_response(wavelength)

    def get_response(self, wavelength: float) -> Optional[IlluminationResponse]:
        """
        Returns the response with the specified wavelength, if it is set. If not, a response
        obtained from interpolation is returned if there is more than one response, the only
        response if there is exactly one, or None if no responses are set.
        """

        # Look for exact match
        for response in self._responses.values():
            if wavelength == response.wavelength:
                return response

        # Interpolate if there's more than one response
        if len(self._responses.values()) > 1:
            interp_array_x = np.zeros(len(self._responses))
            interp_array_y = np.zeros((len(interp_array_x), 3))

            for index, response in enumerate(self._responses.values()):
                interp_array_x[index] = response.wavelength
                interp_array_y[index] = [response.cross_section_off_to_on,
                                         response.cross_section_on_to_off,
                                         response.cross_section_emission]

            interp_function = interp1d(interp_array_x, interp_array_y,
                                       axis=0,
                                       kind=max(1, min(len(interp_array_x) - 1, 3)),
                                       fill_value="extrapolate")

            return IlluminationResponse(
                wavelength,
                *interp_function(wavelength).clip(0)
            )

        # Return only response if one exists
        if len(self._responses.values()) == 1:
            return list(self._responses.values())[0]

        # No responses, return None
        return None

    # Internal methods
    def _get_responses(self) -> List[IlluminationResponse]:
        return [*self._responses.values()]

    def _set_responses(self, responses: List[IlluminationResponse]) -> None:
        if not isinstance(responses, list):
            return

        self.clear_responses()
        for response in responses:
            self.add_response(response)
