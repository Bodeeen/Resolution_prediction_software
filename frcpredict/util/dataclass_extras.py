from dataclasses import fields
from typing import Any, Optional

from PySignal import Signal


class observable_property(property):
    def __init__(self, fget, fset, default) -> None:
        self.default = default
        super().__init__(fget, fset)


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


def dataclass_with_observables(cls=None):
    """
    Decorator for automatically initializing observable fields in a dataclass.
    """

    def wrap(cls):
        overridden_initfunc = cls.__init__

        def initfunc(self, *_init_args, **_init_kwargs):
            overridden_initfunc(self, *_init_args, **_init_kwargs)

            for dataclass_field in fields(cls):
                if (isinstance(getattr(self, dataclass_field.name), observable_property) and
                        isinstance(dataclass_field.default, observable_property)):
                    default = dataclass_field.default.default
                    if callable(default):
                        default = default()

                    setattr(self, dataclass_field.name, default)

        cls.__init__ = initfunc
        return cls

    if cls is None:
        return wrap

    return wrap(cls)


def observable_field(internal_name: str, default: Any,
                     signal_name: str, emit_arg_name: Optional[str] = None) -> observable_property:
    """
    A field that emits a specific PySignal signal that is also a member of its owner class.

    Arguments:
    internal_name -- the observable field will use this instance variable name internally to
                     store the value of the field
    default       -- default value of the field, or a factory for the default value
    signal_name   -- the name of an attribute in the same class instance, that is the signal that
                     should be emitted
    emit_arg_name -- the name of an attribute in the same class instance, the value of which will be
                     passed as an argument when the signal is emitted; if none is specified, the
                     entire class instance will be passed
    """

    def getter(self):
        return getattr(self, internal_name)

    def setter(self, value: Any):
        setattr(self, internal_name, value)

        signal = getattr(self, signal_name)
        if signal:
            if emit_arg_name is not None:
                emit_arg = getattr(self, emit_arg_name)
                signal.emit(emit_arg)
            else:
                signal.emit(self)

    return observable_property(getter, setter, default)


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
