from dataclasses import field, fields, Field, MISSING
from typing import Any, Optional, Union, Tuple, Callable

from PySignal import Signal

import frcpredict.util


class dataclass_property(property):
    def __init__(self, fget, fset, default) -> None:
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
            setattr(self_, internal_name, value)

            signal = getattr(self_, signal_name)
            if signal:
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


def clear_signals(dataclass_instance):
    """ Clears all signals in the given dataclass instance and returns it. """

    for attr_name in dir(dataclass_instance):
        if attr_name.startswith("__"):
            continue

        attr_value = getattr(dataclass_instance, attr_name)

        if isinstance(attr_value, Signal):
            attr_value.clear()
        elif isinstance(attr_value, list):
            for list_item in attr_value:
                clear_signals(list_item)
        elif isinstance(attr_value, dict):
            for dict_value in attr_value.values():
                clear_signals(dict_value)
        elif hasattr(attr_value, "__dataclass_fields__"):
            clear_signals(attr_value)

    return dataclass_instance
