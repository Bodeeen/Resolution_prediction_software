import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Union, Dict

import numpy as np
from PySignal import Signal
from dataclasses_json import dataclass_json, config as json_config, Exclude

from frcpredict.util import (
    dataclass_internal_attrs, dataclass_with_properties,
    observable_property, extended_field, ndarray_field, positions2im
)
from .multivalue import Multivalue
from .pattern_data import Array2DPatternData


@dataclass_json
@dataclass_with_properties
@dataclass_internal_attrs(basic_field_changed=Signal)
@dataclass
class ExplicitSampleProperties:
    """
    An explicit description of sample-related properties in an environment.
    """

    input_power: Union[float, Multivalue[float]] = extended_field(
        observable_property("_input_power", default=1.0, signal_name="basic_field_changed"),
        description="input power", accept_multivalues=True
    )

    D_origin: Union[float, Multivalue[float]] = extended_field(
        observable_property("_D_origin", default=1.0, signal_name="basic_field_changed"),
        description="D(0, 0)", accept_multivalues=True
    )


@dataclass_json
@dataclass
class DisplayableSample(ABC):
    """
    A description of what a sample looks like. Note that due to the way this class handles caching,
    fields that are related to the generation of the sample image array MUST NOT be modified after
    initialization.
    """

    _cached_image_arrs: Dict[float, np.ndarray] = field(  # mapping from pixel size in nanometres
        default_factory=dict, metadata=json_config(exclude=Exclude.ALWAYS)
    )

    # Methods
    def get_explicit_properties(self, px_size_nm: float) -> "ExplicitSamplePropertiesData":
        """
        Returns the estimated properties of the sample based on what it looks like. This assumes
        that the sample is spectrally flat; if it's not, then the values won't be accurate.
        """
        image_arr = self.get_image_arr(px_size_nm)
        return ExplicitSampleProperties(input_power=image_arr.var(), D_origin=image_arr.mean())

    def get_image_arr(self, px_size_nm: float) -> np.ndarray:
        """
        Returns an image representation of the sample, as a numpy array. Generated images are
        automatically cached.
        """
        if px_size_nm in self._cached_image_arrs:
            return self._cached_image_arrs[px_size_nm]

        image_arr = self._get_image_arr(px_size_nm)
        self._cached_image_arrs[px_size_nm] = image_arr
        return image_arr

    @abstractmethod
    def is_loaded(self) -> bool:
        """ Returns whether this DisplayableSample instance contains a loaded sample. """
        pass

    @abstractmethod
    def get_id(self) -> str:
        """ Returns a unique, static ID of the sample, for caching purposes. """
        pass

    @abstractmethod
    def get_area_side_um(self) -> float:
        """ Returns the sample area side length, in microns. """
        pass

    # Internal methods
    @abstractmethod
    def _get_image_arr(self, px_size_nm: float) -> np.ndarray:
        """
        Internal method that must be overriden by derived classes; this is the method that the
        get_image_arr function calls to actually generate the image.
        """
        pass


@dataclass_json
@dataclass
class SampleStructure(DisplayableSample):
    """
    A description of a sample structure.
    """

    area_side_um: float = 5.0

    relative_array_x: np.ndarray = ndarray_field(default_factory=lambda: np.zeros(0))

    relative_array_y: np.ndarray = ndarray_field(default_factory=lambda: np.zeros(0))

    structure_type: Optional[str] = None

    structure_id: str = field(default_factory=uuid.uuid4)

    # Methods
    def is_loaded(self) -> bool:
        return len(self.relative_array_x) > 0 and len(self.relative_array_y) > 0

    def get_area_side_um(self) -> float:
        return self.area_side_um

    def get_id(self) -> str:
        return self.structure_id

    # Internal methods
    def _get_image_arr(self, px_size_nm: float) -> np.ndarray:
        px_size_um = px_size_nm / 1000
        return positions2im(self.area_side_um, px_size_um,
                            self.relative_array_x, self.relative_array_y)


@dataclass_json
@dataclass
class SampleImage(DisplayableSample):
    """
    A description of a sample image.
    """

    area_side_um: float = 5.0

    image: Array2DPatternData = field(default_factory=lambda: Array2DPatternData)

    id: str = field(default_factory=uuid.uuid4)

    # Methods
    def is_loaded(self) -> bool:
        return not self.image.is_empty()

    def get_id(self) -> str:
        return self.id

    def get_area_side_um(self) -> float:
        return self.area_side_um

    # Internal methods
    def _get_image_arr(self, px_size_nm: float) -> np.ndarray:
        return self.image.get_numpy_array(self.area_side_um * 500, px_size_nm)


@dataclass_json
@dataclass_with_properties
@dataclass_internal_attrs(data_loaded=Signal)
@dataclass
class SampleProperties:
    """
    A description of sample-related properties in an environment.
    """

    basic_properties: Optional[ExplicitSampleProperties] = extended_field(
        observable_property("_basic_properties", default=ExplicitSampleProperties(),
                            signal_name="data_loaded"),
        description="sample"
    )

    structure: Optional[SampleStructure] = extended_field(
        observable_property("_structure", default=None,
                            signal_name="data_loaded"),
        description="structure"
    )

    def get_combined_properties(self, px_size_nm: float) -> ExplicitSampleProperties:
        """
        Returns explicit sample properties. Properties derived from any loaded sample structure will
        take precedence over properties defined in basic_properties.
        """
        if self.structure is not None:
            return self.structure.get_explicit_properties(px_size_nm)
        elif self.basic_properties is not None:
            return self.basic_properties
        else:
            raise Exception("basic_properties or structure must be set!")
