from typing import Union, List, Callable

import numpy as np
from PyQt5.QtCore import pyqtSignal, pyqtBoundSignal
from PyQt5.QtGui import QImage, QPixmap, QGuiApplication, QCursor
from PyQt5.QtWidgets import QFormLayout, QWidget


def getArrayPixmap(arr: np.ndarray, normalize: bool = False) -> QPixmap:
    """
    Converts a numpy array to a QPixmap. The array is expected to be of the shape
    (width, height) and contain floats.
    """

    if normalize and ((arr < 0) | (arr > 1)).any():
        ptp = np.ptp(arr)
        arr = np.divide(arr - np.min(arr), ptp, out=np.zeros_like(arr), where=ptp != 0)

    uint8_arr = (arr.clip(0, 1) * 255).astype("uint8")  # Convert from floats to [0, 255] ints
    width = len(arr[0])
    height = len(arr)

    return QPixmap(
        QImage(uint8_arr, width, height, width, QImage.Format_Grayscale8)
    )


def setFormLayoutRowVisibility(formLayout: QFormLayout, rowNumber: int, labelWidget: QWidget,
                               valueWidget: QWidget, visible: bool) -> None:
    """ Sets the visibility of a QFormLayout row, without leaving empty space when hiding it. """

    if visible:
        if valueWidget.isHidden():
            formLayout.insertRow(rowNumber, labelWidget, valueWidget)
            labelWidget.show()
            valueWidget.show()
    else:
        if not valueWidget.isHidden():
            labelWidget.hide()
            valueWidget.hide()
            formLayout.removeWidget(labelWidget)
            formLayout.removeWidget(valueWidget)


def setTabOrderForChildren(parentWidget: QWidget, childWidgetOrder: List[QWidget]) -> None:
    """
    Sets the given parent widget's tab order to follow the order of the widgets in the given child
    widget list.
    """

    for i in range(0, len(childWidgetOrder) - 1):
        parentWidget.setTabOrder(childWidgetOrder[i], childWidgetOrder[i + 1])


def connectMulti(signal: pyqtSignal, arg_types: List[type],
                 handler: Union[Callable, pyqtBoundSignal]) -> None:
    """
    Connects a signal that has multiple possible argument types with a handler that accepts multiple
    argument types.
    """

    for arg_type in arg_types:
        try:
            signal[arg_type].connect(
                handler[arg_type] if isinstance(handler, pyqtBoundSignal) else handler
            )
        except KeyError:
            signal[arg_type].connect(handler)


def centerWindow(window: QWidget) -> None:
    """ Centers the given window in the screen. """
    screen = QGuiApplication.screenAt(QCursor().pos())
    frameGeometry = window.frameGeometry()
    frameGeometry.moveCenter(screen.availableGeometry().center())
    window.move(frameGeometry.topLeft())
