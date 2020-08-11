from copy import deepcopy
from dataclasses import field, fields, Field, MISSING
from typing import Any, Optional, Union, Tuple, List, Dict, TypeVar

from dataclasses_json import config as json_config
import numpy as np

import frcpredict.model
from .dataclass_extras import clear_signals


T = TypeVar("T")

FieldPath = List[Union[int, str]]


def multi_accepting_field(default: Any = MISSING, *, default_factory: Any = MISSING,
                          metadata: Optional[Dict[str, Any]] = None) -> Field:
    """
    A dataclass field that could be either a multivalue or a scalar (single) value. Multi-accepting
    fields must be initialized with this in order to be deserialized from JSON properly.
    """

    def decoder(value):
        if isinstance(value, dict):
            if "values" in value:
                return frcpredict.model.ValueList.from_dict(value)
            else:
                return frcpredict.model.ValueRange.from_dict(value)
        else:
            return value

    return field(
        default=default,
        default_factory=default_factory,
        metadata=json_config(
            metadata,
            decoder=decoder
        )
    )


def get_paths_of_multivalues(dataclass_instance: object) -> Tuple[List[FieldPath], int]:
    """
    Walks through a dataclass instance, and returns a tuple where the first element describes the
    paths to all multivalues in the dataclass instance, and the second element is the number of
    possible combinations resulting from these values.

    For example, the path to a multivalue located in dataclass_instance.pulses[0].max_intensity
    would look like this: ["pulses", 0, "max_intensity"]
    """

    paths = []
    num_combinations = 1

    def recurse(name: List[Union[int, str]], value: Any) -> None:
        nonlocal num_combinations

        sub_paths, sub_num_combinations = get_paths_of_multivalues(value)
        for sub_path in sub_paths:
            paths.append(name + sub_path)

        num_combinations *= sub_num_combinations

    for dataclass_field in fields(dataclass_instance):
        field_name = dataclass_field.name
        field_value = getattr(dataclass_instance, dataclass_field.name)

        if isinstance(field_value, frcpredict.model.Multivalue):
            paths.append([field_name])
            num_combinations *= field_value.num_values()
        elif isinstance(field_value, list):
            for list_item_index, list_item_value in enumerate(field_value):
                recurse([field_name, list_item_index], list_item_value)
        elif hasattr(field_value, "__dataclass_fields__"):
            recurse([field_name], field_value)

    return paths, num_combinations


def expand_multivalues(dataclass_instance: object, multivalue_paths: List[FieldPath]) -> np.ndarray:
    """
    Takes a dataclass instance that contains multivalues and a list of multivalue paths, and returns
    an n-dimensional (n = max(1, number of multivalue paths)) array that contains copies of this
    dataclass instance, except all multivalues are replaced by scalar (single) values. All possible
    combinations of these values are available in this array.

    Each element in the returned array is a tuple, where the first element is a list that contains
    a combination of scalar values (referring to each multivalue path respectively), and the second
    element is the copy of the dataclass instance with these scalar values set.
    """

    if len(multivalue_paths) < 1:
        results = np.ndarray((1,), dtype=np.object)
        results[0] = ([], clear_signals(deepcopy(dataclass_instance)))
        return results

    def build_results(results: np.ndarray, obj: object,
                      path_index: int = 0, prev_values_for_result: List[object] = []) -> None:
        multivalues = get_value_from_path(obj, multivalue_paths[path_index]).as_array()

        for multivalue_index, multivalue in enumerate(multivalues):
            result = deepcopy(obj)

            setattr(
                get_value_from_path(result, multivalue_paths[path_index][:-1]),
                multivalue_paths[path_index][-1],
                multivalue
            )

            values_for_result = prev_values_for_result.copy()
            values_for_result.append(multivalue)

            if path_index + 1 < len(multivalue_paths):
                build_results(results[multivalue_index], result, path_index + 1, values_for_result)
            else:
                results[multivalue_index] = (values_for_result, result)

    results = np.ndarray(
        tuple(get_value_from_path(dataclass_instance, multivalue_path).num_values()
              for multivalue_path in multivalue_paths),
        dtype=np.object
    )

    build_results(results, clear_signals(deepcopy(dataclass_instance)))
    return results


def expand_with_multivalues(dataclass_instance: T, multivalue_paths: List[FieldPath],
                            multivalue_values: List[Union[int, float]]) -> T:
    """
    Given a dataclass instance containing multivalues, a list of multivalue paths, and a list of
    corresponding multivalue values, this function creates a copy of the dataclass instance with
    cleared signals and the multivalue values set.
    """

    if len(multivalue_paths) != len(multivalue_values):
        raise ValueError("multivalue_paths and multivalue_values must be of the same length")

    result = clear_signals(deepcopy(dataclass_instance))

    for path_index in range(len(multivalue_paths)):
        setattr(
            get_value_from_path(result, multivalue_paths[path_index][:-1]),
            multivalue_paths[path_index][-1],
            multivalue_values[path_index]
        )

    return result


def avg_value_if_multivalue(multi_or_scalar_value) -> float:
    """
    If the argument is a multivalue, this returns its average value. Otherwise, the argument is
    assumed to be a scalar (single) value returned.
    """
    if isinstance(multi_or_scalar_value, frcpredict.model.Multivalue):
        return multi_or_scalar_value.avg_value()
    else:
        return multi_or_scalar_value


def get_value_from_path(dataclass_instance: object, multivalue_path: FieldPath) -> Any:
    """
    Returns the value of a field of a dataclass_instance from its path. For example, the path
    ["pulses", 0, "max_intensity"] would make this function return
    dataclass_instance.pulses[0].max_intensity.
    """

    if multivalue_path:
        if isinstance(multivalue_path[0], str):
            return get_value_from_path(
                getattr(dataclass_instance, multivalue_path[0]), multivalue_path[1:]
            )
        else:
            # Handle list
            return get_value_from_path(
                dataclass_instance[multivalue_path[0]], multivalue_path[1:]
            )
    else:
        return dataclass_instance
