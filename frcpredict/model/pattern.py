from dataclasses import dataclass
from dataclasses_json import dataclass_json
from enum import Enum
import numpy as np
from PySignal import Signal
from typing import Optional, Union

from frcpredict.util import dataclass_internal_attrs, extended_field
from .pattern_data import (
    PatternData, Array2DPatternData,
    GaussianPatternData, DoughnutPatternData, AiryFWHMPatternData, AiryNAPatternData,
    DigitalPinholePatternData, PhysicalPinholePatternData
)


class PatternType(Enum):
    """
    All supported pattern types.
    """
    array2D = "array2D"
    gaussian = "gaussian"
    doughnut = "doughnut"
    airy_from_FWHM = "airy_from_FWHM"
    airy_from_NA = "airy_from_NA"
    digital_pinhole = "digital_pinhole"
    physical_pinhole = "physical_pinhole"


@dataclass_json
@dataclass_internal_attrs(data_loaded=Signal)
@dataclass
class Pattern:
    """
    A description of a pattern.
    """

    pattern_type: PatternType = PatternType.array2D
    pattern_data: Union[dict, PatternData] = extended_field(default_factory=Array2DPatternData,
                                                            description="pattern data")

    # Properties
    @property
    def pattern_data(self) -> PatternData:
        # The way we do this with allowing and converting from dicts is a bit stupid, but it seems
        # to be necessary in order to get dataclasses-json to encode/decode our data as wanted
        if isinstance(self._pattern_data, dict):
            self._pattern_data = self._get_data_type_from_pattern_type(self.pattern_type).from_dict(
                self._pattern_data
            )

        return self._pattern_data

    @pattern_data.setter
    def pattern_data(self, pattern_data: Union[dict, PatternData]) -> None:
        self._pattern_data = pattern_data
        self.data_loaded.emit(self)

    # Methods
    def __init__(self, pattern_type: Optional[PatternType] = None,
                 pattern_data: Optional[Union[dict, PatternData]] = None):
        if pattern_type is None and pattern_data is None:
            self.pattern_type = PatternType.array2D
            self.pattern_data = Array2DPatternData()
        elif pattern_data is not None:
            if isinstance(pattern_data, dict):
                if pattern_type is None:
                    raise ValueError(
                        "Pattern type must be specified when loading pattern data from dict"
                    )

                self.pattern_type = pattern_type
                self.pattern_data = pattern_data
            else:
                self.load_from_data(pattern_data)
        else:
            self.load_from_type(pattern_type)
    
    def load_from_type(self, pattern_type: PatternType) -> None:
        """
        Loads the given pattern type into pattern_type and a default pattern of that type into
        pattern_data.
        """
        self.pattern_type = pattern_type
        self.pattern_data = self._get_data_type_from_pattern_type(pattern_type)()

    def load_from_data(self, pattern_data: PatternData) -> None:
        """
        Loads the given pattern data into pattern_data and the type of that data into pattern_type.
        """
        self.pattern_type = self._get_pattern_type_from_data(pattern_data)
        self.pattern_data = pattern_data

    def get_radial_profile(self, canvas_inner_radius_nm: float, px_size_nm: float) -> np.ndarray:
        """ Returns a numpy array representation of the pattern data as a radial profile. """
        return self.pattern_data.get_radial_profile(canvas_inner_radius_nm, px_size_nm)

    def get_numpy_array(self, canvas_inner_radius_nm: float, px_size_nm: float,
                        extend_sides_to_diagonal: bool = False) -> np.ndarray:
        """ Returns a numpy array representation of the pattern data. """
        return self.pattern_data.get_numpy_array(canvas_inner_radius_nm, px_size_nm,
                                                 extend_sides_to_diagonal)

    def __str__(self) -> str:
        return str(self.pattern_data)

    # Internal methods
    def _get_data_type_from_pattern_type(self, pattern_type: PatternType) -> type:
        if pattern_type == PatternType.array2D:
            return Array2DPatternData
        elif pattern_type == PatternType.gaussian:
            return GaussianPatternData
        elif pattern_type == PatternType.doughnut:
            return DoughnutPatternData
        elif pattern_type == PatternType.airy_from_FWHM:
            return AiryFWHMPatternData
        elif pattern_type == PatternType.airy_from_NA:
            return AiryNAPatternData
        elif pattern_type == PatternType.digital_pinhole:
            return DigitalPinholePatternData
        elif pattern_type == PatternType.physical_pinhole:
            return PhysicalPinholePatternData
        else:
            raise ValueError(f"Invalid pattern type \"{pattern_type}\"")

    def _get_pattern_type_from_data(self, pattern_data: PatternData) -> PatternType:
        if type(pattern_data) is Array2DPatternData:
            return PatternType.array2D
        elif type(pattern_data) is GaussianPatternData:
            return PatternType.gaussian
        elif type(pattern_data) is DoughnutPatternData:
            return PatternType.doughnut
        elif type(pattern_data) is AiryFWHMPatternData:
            return PatternType.airy_from_FWHM
        elif type(pattern_data) is AiryNAPatternData:
            return PatternType.airy_from_NA
        elif type(pattern_data) is DigitalPinholePatternData:
            return PatternType.digital_pinhole
        elif type(pattern_data) is PhysicalPinholePatternData:
            return PatternType.physical_pinhole
        else:
            raise TypeError(f"Invalid pattern data type \"{type(pattern_data).__name__}\"")
