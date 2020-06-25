from base64 import b64encode, b64decode
from dataclasses import field, fields
from dataclasses_json import config as json_config, Exclude
import numpy as np
from PySignal import Signal
from typing import Any, Optional, List


def dataclass_internal_attrs(cls=None, super_cls: type = object, **internal_attr_factories):
    """
    Decorator for adding internal attributes into the dataclass. This is useful for declaring
    attributes that you want to be hidden when the dataclass is serialized to a string or JSON.
    """

    def wrap(cls):
        def newfunc(new_cls, *new_args, **new_kwargs):
            instance = super_cls().__new__(new_cls)
            for key, value in internal_attr_factories.items():
                setattr(instance, key, value())

            return instance

        cls.__new__ = newfunc
        return cls

    if cls is None:
        return wrap

    return wrap(cls)


def observable_property(internal_name: str, default: Any,
                        signal_name: str, emit_arg_name: Optional[str] = None):
    """
    A property that emits a specific PySignal signal that is also a member of its owner class.

    Arguments:
    internal_name -- the observable property will use this instance variable name internally to
                     store the value of the property
    default       -- default value of the property
    signal_name   -- the name of a property in the same class instance, that is the signal that
                     should be emitted
    emit_arg_name -- the name of a property in the same class instance, the value of which will be
                     passed as an argument when the signal is emitted; if none is specified, the
                     entire class instance will be passed
    """

    def getter(self):
        return getattr(self, internal_name, default)

    def setter(self, value: Any):
        setattr(self, internal_name, value)

        signal = getattr(self, signal_name, None)
        if signal:
            if emit_arg_name is not None:
                emit_arg = getattr(self, emit_arg_name, None)
                signal.emit(emit_arg)
            else:
                signal.emit(self)

    return property(getter, setter)


def with_cleared_signals(dataclass_instance):
    """ Clears all signals in the given dataclass instance and returns it. """

    for dataclass_field in fields(dataclass_instance):
        if issubclass(dataclass_field.type, Signal):
            getattr(dataclass_instance, dataclass_field.name).clear()

    return dataclass_instance
