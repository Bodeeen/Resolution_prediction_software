from typing import Any, Optional

from PySignal import Signal


def dataclass_internal_attrs(cls=None, super_cls: type = object, **internal_attr_factories):
    """
    Decorator for adding internal attributes into the dataclass. This is useful for declaring
    attributes that should be hidden when the dataclass is serialized to a string or JSON.
    """

    def wrap(cls):
        def newfunc(new_cls, *_new_args, **_new_kwargs):
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
                        signal_name: str, emit_arg_name: Optional[str] = None) -> property:
    """
    A property that emits a specific PySignal signal that is also a member of its owner class.

    Arguments:
    internal_name -- the observable property will use this instance variable name internally to
                     store the value of the property
    default       -- default value of the property
    signal_name   -- the name of an attribute in the same class instance, that is the signal that
                     should be emitted
    emit_arg_name -- the name of an attribute in the same class instance, the value of which will be
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
