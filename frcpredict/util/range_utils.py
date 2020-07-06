from copy import deepcopy
from dataclasses import field, fields, Field
from typing import Any, Union, Tuple, List

from dataclasses_json import config as json_config
import numpy as np

import frcpredict.model
from .dataclass_extras import clear_signals


FieldPath = List[Union[int, str]]


def rangeable_field(default: Any) -> Field:
    """
    A dataclass field that could be either a ValueRange or a scalar (single) value. Rangeable fields
    must be initialized with this in order to be deserialized from JSON properly.
    """

    return field(
        default=default,
        metadata=json_config(
            {"rangeable": True},
            decoder=lambda x: frcpredict.model.ValueRange.from_dict(x) if isinstance(x, dict) else x
        )
    )


def get_paths_of_ranges(dataclass_instance: object) -> Tuple[List[FieldPath], int]:
    """
    Walks through a dataclass instance, and returns a tuple where the first element describes the
    paths to all ranged values (ValueRange objects) in the dataclass instance, and the second
    element is the number of possible combinations resulting from these ranged values.

    For example, the path to a ValueRange located in dataclass_instance.pulses[0].max_intensity
    would look like this: ["pulses", 0, "max_intensity"]
    """

    paths = []
    num_combinations = 1

    def recurse(name: List[Union[int, str]], value: Any) -> None:
        nonlocal num_combinations

        sub_paths, sub_num_combinations = get_paths_of_ranges(value)
        for sub_path in sub_paths:
            paths.append(name + sub_path)

        num_combinations *= sub_num_combinations

    for dataclass_field in fields(dataclass_instance):
        field_name = dataclass_field.name
        field_value = getattr(dataclass_instance, dataclass_field.name)

        if isinstance(field_value, frcpredict.model.ValueRange):
            paths.append([field_name])
            num_combinations *= field_value.num_evaluations
        elif isinstance(field_value, list):
            for list_item_index, list_item_value in enumerate(field_value):
                recurse([field_name, list_item_index], list_item_value)
        elif hasattr(field_value, "__dataclass_fields__"):
            recurse([field_name], field_value)

    return paths, num_combinations


def expand_ranges(dataclass_instance: object, range_paths: List[FieldPath]) -> np.ndarray:
    """
    Takes a dataclass instance that contains ranged values (ValueRange objects) and a list of range
    paths, and returns an n-dimensional (n = max(1, number of range paths)) array that contains
    copies of this dataclass instance, except all ranged values are replaced by scalar (single)
    values. All possible combinations of these values are available in this array.

    Each element in the returned array is a tuple, where the first element is a list that contains
    a combination of scalar values (referring to each range path respectively), and the second
    element is the copy of the dataclass instance with these scalar values set.
    """

    if len(range_paths) < 1:
        results = np.ndarray((1,), dtype=np.object)
        results[0] = ([], dataclass_instance)
        return results

    def build_results(results: np.ndarray, obj: object,
                      path_index: int = 0, prev_values_for_result: List[object] = []) -> None:
        range_values = get_value_from_path(obj, range_paths[path_index]).as_array()

        for range_value_index, range_value in enumerate(range_values):
            result = deepcopy(obj)

            setattr(
                get_value_from_path(result, range_paths[path_index][:-1]),
                range_paths[path_index][-1],
                range_value
            )

            values_for_result = prev_values_for_result.copy()
            values_for_result.append(range_value)

            if path_index + 1 < len(range_paths):
                build_results(results[range_value_index], result, path_index + 1, values_for_result)
            else:
                results[range_value_index] = (values_for_result, result)

    results = np.ndarray(
        tuple(get_value_from_path(dataclass_instance, range_path).num_evaluations for range_path in
              range_paths),
        dtype=np.object
    )

    build_results(results, clear_signals(dataclass_instance))
    return results


def avg_value_if_range(range_or_scalar) -> float:
    """
    If the argument is a ValueRange, this returns its average value. Otherwise, the argument is
    assumed to be a scalar (single) value returned.
    """
    return (range_or_scalar.avg_value() if isinstance(range_or_scalar, frcpredict.model.ValueRange)
            else range_or_scalar)


def get_value_from_path(dataclass_instance: object, range_path: FieldPath) -> Any:
    """
    Returns the value of a field of a dataclass_instance from its path. For example, the path
    ["pulses", 0, "max_intensity"] would make this function return
    dataclass_instance.pulses[0].max_intensity.
    """

    if range_path:
        if isinstance(range_path[0], str):
            return get_value_from_path(getattr(dataclass_instance, range_path[0]), range_path[1:])
        else:
            return get_value_from_path(dataclass_instance[range_path[0]], range_path[1:])  # Handle list
    else:
        return dataclass_instance
