import dataclasses
from dataclasses import is_dataclass, field, fields, Field, MISSING
from typing import Any, Optional, Union, Tuple, List, Callable, TypeVar

import dataclasses_json
import numpy as np
from PySignal import Signal

import frcpredict.util


T = TypeVar("T")


class dataclass_property(property):
    def __init__(self, fget, fset, default) -> None:
        _override_dataclasses_json_decode_dataclass()
        self.default = default
        super().__init__(fget, fset)


class observable_property(dataclass_property):
    """
    A dataclass property that emits a specific PySignal signal that is also a member of its owner
    dataclass.
    """

    def __init__(self, internal_name: str, default: Any,
                 signal_name: str, emit_arg_name: Optional[str] = None) -> None:
        """
        Arguments:
        internal_name -- the observable property will use this instance variable name internally to
                         store the value of the property
        default       -- default value of the property, or a factory for the default value
        signal_name   -- the name of an attribute in the same class instance, that is the signal that
                         should be emitted
        emit_arg_name -- the name of an attribute in the same class instance, the value of which will be
                         passed as an argument when the signal is emitted; if none is specified, the
                         entire class instance will be passed
        """

        def getter(self_):
            return getattr(self_, internal_name)

        def setter(self_, value: Any):
            try:
                should_emit = not hasattr(self_, internal_name) or getter(self_) != value
            except ValueError:
                should_emit = True

            setattr(self_, internal_name, value)

            signal = getattr(self_, signal_name)
            if signal and should_emit:
                if emit_arg_name is not None:
                    emit_arg = getattr(self_, emit_arg_name)
                    signal.emit(emit_arg)
                else:
                    signal.emit(self_)

        super().__init__(getter, setter, default)


def extended_field(default: Any = MISSING, *, default_factory: Any = MISSING,
                   description: Optional[Union[str, Callable]] = None,
                   accept_multivalues: bool = False) -> Field:
    """
    A dataclass field with extended functionality:

    * It is possible to set a description with later can be retrieved using the
      get_dataclass_field_description function. The description can be either a string or a
      function. If it's a function, the dataclass instance will be passed as the first argument when
      querying the description, and if the field is a list or a dict, an index will be passed as the
      second argument.
    * One can easily set whether the field accepts multivalues
    """

    field_type = frcpredict.util.multivalue_utils.multi_accepting_field if accept_multivalues else field

    metadata = {
        "description": description
    } if description is not None else None

    return field_type(
        default=default,
        default_factory=default_factory,
        metadata=metadata
    )


def dataclass_internal_attrs(cls=None, **internal_attr_factories):
    """
    Decorator for adding internal attributes into the dataclass. This is useful for declaring
    attributes that should be hidden when the dataclass is serialized to a string or JSON.
    """

    def wrap(cls):
        overridden_newfunc = cls.__new__

        def newfunc(new_cls, *_new_args, **_new_kwargs):
            instance = overridden_newfunc(new_cls)
            for key, value in internal_attr_factories.items():
                setattr(instance, key, value())

            return instance

        cls.__new__ = newfunc
        return cls

    if cls is None:
        return wrap

    return wrap(cls)


def dataclass_with_properties(cls=None):
    """
    Decorator for automatically initializing fields that are dataclass properties in a dataclass.
    """

    def wrap(cls):
        overridden_initfunc = cls.__init__

        def initfunc(self, *_init_args, **_init_kwargs):
            overridden_initfunc(self, *_init_args, **_init_kwargs)

            for dataclass_field in fields(cls):
                if (isinstance(getattr(self, dataclass_field.name), dataclass_property) and
                        isinstance(dataclass_field.default, dataclass_property)):
                    default = dataclass_field.default.default
                    if callable(default):
                        default = default()

                    setattr(self, dataclass_field.name, default)

        cls.__init__ = initfunc
        return cls

    if cls is None:
        return wrap

    return wrap(cls)


def get_dataclass_field_description(dataclass_instance: Any,
                                    dataclass_field: Field,
                                    list_or_dict_index: int = -1) -> Tuple[str, bool]:
    """
    Returns the set description of a given dataclass field if one exists. Otherwise, it generates a
    description dynamically. The second returned value represents whether the returned description
    was generated dynamically.
    """

    dataclass_field_value = getattr(dataclass_instance, dataclass_field.name)

    if "description" in dataclass_field.metadata:
        # Get description from field metadata
        description = dataclass_field.metadata["description"]
        if callable(description):
            if isinstance(dataclass_field_value, (list, dict)):
                description = description(dataclass_instance, list_or_dict_index)
            else:
                description = description(dataclass_instance)

        return description, False
    else:
        # Generate description
        description = dataclass_field.name.replace("_", " ")
        if isinstance(dataclass_field_value, list):
            description += f" item {list_or_dict_index + 1}"
        elif isinstance(dataclass_field_value, dict):
            description += f" item \"{list_or_dict_index + 1}\""

        return description, True


def is_dataclass_instance(obj: object) -> bool:
    return hasattr(obj, "__dataclass_fields__")


def recursive_field_iter(dataclass_instance: object, include_root: bool = True) -> List[object]:
    """
    Returns an iterator that contains all dataclass instances that the given dataclass instance
    contains, recursively.
    """

    result = [dataclass_instance] if include_root else []
    for attr_name in dir(dataclass_instance):
        if attr_name.startswith("__"):
            continue

        attr_value = getattr(dataclass_instance, attr_name)

        if isinstance(attr_value, list):
            for list_item in attr_value:
                result += recursive_field_iter(list_item)
        elif isinstance(attr_value, dict):
            for dict_value in attr_value.values():
                result += recursive_field_iter(dict_value)
        elif isinstance(attr_value, np.ndarray) and attr_value.dtype == np.object:
            for array_element in np.nditer(attr_value, flags=["refs_ok"]):
                result += recursive_field_iter(array_element.item())
        elif is_dataclass_instance(attr_value):
            result += recursive_field_iter(attr_value)
        else:
            result.append(attr_value)

    return result


def clear_signals(dataclass_instance: T) -> T:
    """
    Clears all signals in the given dataclass instance and returns it. This is done recursively.
    """

    for field_value in recursive_field_iter(dataclass_instance):
        if isinstance(field_value, Signal):
            field_value.clear()

    return dataclass_instance


def rebuild_dataclass(dataclass_instance: T) -> T:
    """
    Effectively creates a copy of the given dataclass instance where only the formally defined
    fields are included. This is done recursively.
    """

    if not is_dataclass(dataclass_instance):
        return dataclass_instance

    kwargs = {}

    for field_info in fields(dataclass_instance):
        field_name = field_info.name

        if not hasattr(dataclass_instance, field_name):
            continue

        field_value = getattr(dataclass_instance, field_name)
        if isinstance(field_value, list):
            kwargs[field_name] = [None] * len(field_value)
            for i in range(len(field_value)):
                kwargs[field_name][i] = rebuild_dataclass(field_value[i])
        if isinstance(field_value, dict):
            kwargs[field_name] = {}
            for (dict_key, dict_value) in field_value.items():
                kwargs[field_name][dict_key] = rebuild_dataclass(dict_value)
        elif isinstance(field_value, np.ndarray) and field_value.dtype == np.object:
            kwargs[field_name] = np.frompyfunc(rebuild_dataclass, 1, 1)(field_value)
        elif is_dataclass(field_value):
            kwargs[field_name] = rebuild_dataclass(field_value)
        else:
            kwargs[field_name] = field_value

    return type(dataclass_instance)(**kwargs)


def _override_dataclasses_json_decode_dataclass() -> None:
    """
    Overrides the is_dataclass and _is_dataclass_instance functions in dataclasses-json to be
    compatible with the way we use properties in dataclasses.
    """

    global _has_overridden_dataclasses_json_decode_dataclass
    if _has_overridden_dataclasses_json_decode_dataclass:
        return

    old_decode_dataclass = dataclasses_json.core._decode_dataclass

    def _decode_dataclass(cls, kvs, infer_missing):
        if isinstance(kvs, dataclass_property):
            return kvs

        return old_decode_dataclass(cls, kvs, infer_missing)

    dataclasses_json.core._decode_dataclass = _decode_dataclass
    _has_overridden_dataclasses_json_decode_dataclass = True


_has_overridden_dataclasses_json_decode_dataclass = False
