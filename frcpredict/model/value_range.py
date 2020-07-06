from dataclasses import dataclass
from enum import Enum
from typing import TypeVar, Generic

import numpy as np
from dataclasses_json import dataclass_json

T = TypeVar("T")


class RangeType(Enum):
    linear = "linear"
    logarithmic = "logarithmic"
    inverse_logarithmic = "inverse_logarithmic"


@dataclass_json
@dataclass
class ValueRange(Generic[T]):
    start: T

    end: T

    num_evaluations: int = 21

    range_type: RangeType = RangeType.linear

    # Methods
    def as_array(self) -> np.ndarray:
        if self.range_type == RangeType.linear:
            return np.linspace(self.start, self.end, self.num_evaluations)
        elif self.range_type == RangeType.logarithmic:
            return np.geomspace(self.start, self.end, self.num_evaluations)
        elif self.range_type == RangeType.inverse_logarithmic:
            return (self.start + self.end - np.geomspace(self.start, self.end,
                                                         self.num_evaluations))[::-1]
        else:
            raise ValueError(f"Invalid range type \"{self.range_type}\"")

    def avg_value(self) -> T:
        if self.range_type == RangeType.linear:
            return (self.start + self.end) / 2
        if self.range_type == RangeType.logarithmic:
            return np.geomspace(self.start, self.end, 3)[1]
        if self.range_type == RangeType.inverse_logarithmic:
            return (self.start + self.end - np.geomspace(self.start, self.end, 3))[1]
        else:
            raise ValueError(f"Invalid range type \"{self.range_type}\"")

    def __str__(self) -> str:
        return f"[{self.start}, {self.end}]"
