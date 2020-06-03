from PySignal import Signal
from typing import Any, Optional, List


def observable_property(internal_name: str, default: Any, signal_name: str, emit_arg_name: Optional[str] = None):
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
