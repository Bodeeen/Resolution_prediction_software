from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import TypeVar, Generic, List

import numpy as np
from PySignal import Signal
from dataclasses_json import dataclass_json

from frcpredict.util import dataclass_internal_attrs

T = TypeVar("T")


class Multivalue(ABC, Generic[T]):
    """
    Abstract class that describes a value that could be many different scalar values.
    """

    @abstractmethod
    def as_array(self) -> np.ndarray:
        """ Returns an array representation of all possible states that the value can be in. """
        pass

    @abstractmethod
    def avg_value(self) -> float:
        """ Returns the average of all possible states that the value can be in. """
        pass

    @abstractmethod
    def num_values(self) -> int:
        """ Returns the number of possible states that the value can be in. """
        pass

    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass_json
@dataclass_internal_attrs(value_added=Signal, value_removed=Signal)
@dataclass
class ValueList(Multivalue[T]):
    """
    A multivalue described by a list of values.
    """

    values: List[T]

    # Methods
    def add(self, value: T) -> None:
        if value in self.values:
            return

        self.values.append(value)
        self.value_added.emit(value)

    def remove(self, value: T) -> None:
        self.values.remove(value)
        self.value_removed.emit(value)

    def clear(self) -> None:
        for value in [*self.values]:
            self.remove(value)

    def as_array(self) -> np.ndarray:
        return np.array(self.values)

    def avg_value(self) -> float:
        return sum(self.values) / len(self.values)

    def num_values(self) -> int:
        return len(self.values)

    def __str__(self) -> str:
        valuesStringList = list(map(str, self.values))
        return f"{{{', '.join(valuesStringList)}}}"


class RangeType(Enum):
    """
    All supported range types for ValueRange.
    """

    linear = "linear"
    logarithmic = "logarithmic"
    inverse_logarithmic = "inverse_logarithmic"


@dataclass_json
@dataclass
class ValueRange(Multivalue[float]):
    """
    A multivalue described by a range.
    """

    start: float

    end: float

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

    def avg_value(self) -> float:
        if self.range_type == RangeType.linear:
            return (self.start + self.end) / 2
        if self.range_type == RangeType.logarithmic:
            return np.geomspace(self.start, self.end, 3)[1]
        if self.range_type == RangeType.inverse_logarithmic:
            return (self.start + self.end - np.geomspace(self.start, self.end, 3))[1]
        else:
            raise ValueError(f"Invalid range type \"{self.range_type}\"")

    def num_values(self) -> int:
        return self.num_evaluations

    def __str__(self) -> str:
        return f"[{self.start}, {self.end}]"
