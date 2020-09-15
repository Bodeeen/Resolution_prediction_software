from typing import Callable, Any

from PyQt5.QtCore import pyqtBoundSignal
from PySignal import Signal

from frcpredict.util import recursive_field_iter


def connectModelToSignal(model: Any, combinedSignal: pyqtBoundSignal) -> Callable:
    """
    Recursively connects all signals in the given model to the given combined signal. Since the
    signal types may not be compatible, the signals are not connected directly, but a lambda
    function is created for emission of the combined signal. This lambda function may be passed to
    disconnectModelFromModifiedFlagSlot later for disconnection.
    """

    connectedFunction = lambda *_args, **_kwargs: combinedSignal.emit()

    for fieldValue in recursive_field_iter(model):
        if isinstance(fieldValue, Signal):
            fieldValue.connect(connectedFunction)

    return connectedFunction


def disconnectModelFromSignal(model: Any, connectedFunction: Callable) -> None:
    """
    Recursively disconnects all signals in the given model from the given function (which represents
    a combined signal).
    """

    for fieldValue in recursive_field_iter(model):
        if isinstance(fieldValue, Signal):
            fieldValue.disconnect(connectedFunction)
