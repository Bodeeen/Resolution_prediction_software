from dataclasses import dataclass
from dataclasses_json import dataclass_json
from enum import Enum
import numpy as np
from PySignal import Signal
from typing import Optional, Union, List, Dict, Type

from frcpredict.util import dataclass_internal_attrs
from .pattern_data import (
    PatternData, Array2DPatternData,
    GaussianPatternData, DoughnutPatternData, AiryPatternData,
    DigitalPinholePatternData
)


class PatternType(Enum):
    array2d = "array2d"
    gaussian = "gaussian"
    doughnut = "doughnut"
    airy = "airy"
    digital_pinhole = "digital_pinhole"


@dataclass_json
@dataclass_internal_attrs(data_loaded=Signal)
@dataclass
class Pattern:
    pattern_type: PatternType = PatternType.array2d
    pattern_data: Union[dict, PatternData] = Array2DPatternData()

    # Properties
    @property
    def pattern_data(self) -> PatternData:
        # The way we do this with allowing and converting from dicts is a bit stupid, but it seems
        # to be neccessary in order to get dataclasses_json to encode/decode our data as wanted
        if isinstance(self._pattern_data, dict):
            self._pattern_data = self._get_data_type_from_pattern_type(self.pattern_type).from_dict(self._pattern_data)

        return self._pattern_data

    @pattern_data.setter
    def pattern_data(self, pattern_data: Union[dict, PatternData]) -> None:
        self._pattern_data = pattern_data
        self.data_loaded.emit(self)

    # Methods
    def __init__(self, pattern_type: Optional[PatternType] = None, pattern_data: Optional[Union[dict, PatternData]] = None):
        if pattern_type is None and pattern_data is None:
            raise ValueError("Either pattern type or pattern data must be specified")

        if pattern_data is not None:
            if isinstance(pattern_data, dict):
                if pattern_type is None:
                    raise ValueError("Pattern type must be specified when loading pattern data from dict")

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

    def get_numpy_array(self, pixels_per_nm: float) -> np.ndarray:
        """ Returns a numpy array representation of the pattern data. """
        return self.pattern_data.get_numpy_array(pixels_per_nm)

    def __str__(self) -> str:
        return str(self.pattern_data)

    # Internal methods
    def _get_data_type_from_pattern_type(self, pattern_type: PatternType) -> type:
        if pattern_type == PatternType.array2d:
            return Array2DPatternData
        elif pattern_type == PatternType.gaussian:
            return GaussianPatternData
        elif pattern_type == PatternType.doughnut:
            return DoughnutPatternData
        elif pattern_type == PatternType.airy:
            return AiryPatternData
        elif pattern_type == PatternType.digital_pinhole:
            return DigitalPinholePatternData
        else:
            raise ValueError(f"Invalid pattern type \"{pattern_type}\"")

    def _get_pattern_type_from_data(self, pattern_data: PatternData) -> PatternType:
        if type(pattern_data) == Array2DPatternData:
            return PatternType.array2d
        elif type(pattern_data) == GaussianPatternData:
            return PatternType.gaussian
        elif type(pattern_data) == DoughnutPatternData:
            return PatternType.doughnut
        elif type(pattern_data) == AiryPatternData:
            return PatternType.airy
        elif type(pattern_data) == DigitalPinholePatternData:
            return PatternType.digital_pinhole
        else:
            raise TypeError(f"Invalid pattern data type \"{type(pattern_data).__name__}\"")
