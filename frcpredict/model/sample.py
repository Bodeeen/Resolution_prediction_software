from dataclasses import dataclass
from typing import Optional, Union

from PySignal import Signal
from dataclasses_json import dataclass_json

from frcpredict.util import (
    dataclass_internal_attrs, dataclass_with_properties, observable_property, extended_field
)
from .multivalue import Multivalue
from .sample_structure import SampleStructure


@dataclass_json
@dataclass_with_properties
@dataclass_internal_attrs(basic_field_changed=Signal, loaded_structure_id_changed=Signal)
@dataclass
class SampleProperties:
    """
    A description of sample-related properties of an environment.
    """

    spectral_power: Union[float, Multivalue[float]] = extended_field(
        observable_property("_spectral_power", default=1.0, signal_name="basic_field_changed"),
        description="spectral power", accept_multivalues=True
    )

    labelling_density: Union[float, Multivalue[float]] = extended_field(
        observable_property("_labelling_density", default=1.0, signal_name="basic_field_changed"),
        description="labelling density [fl./mm²]", accept_multivalues=True
    )

    K_origin: Union[float, Multivalue[float]] = extended_field(
        observable_property("_K_origin", default=1.0, signal_name="basic_field_changed"),
        description="K(0, 0)", accept_multivalues=True
    )

    loaded_structure_id: Optional[str] = observable_property(
        "_loaded_structure_id", default=None, signal_name="loaded_structure_id_changed", emit_arg_name="loaded_structure_id"
    )

    def load_structure(self, structure: SampleStructure) -> None:
        """ Loads properties from the given sample structure. """
        self.loaded_structure_id = structure.image.id
        self.spectral_power = structure.properties.spectral_power
        self.K_origin = structure.properties.K_origin
