import numpy as np
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QFormLayout, QWidget


def getArrayPixmap(arr: np.ndarray, normalise: bool = False) -> QPixmap:
    """
    Converts a numpy array to a QPixmap. The array is expected to be of the shape
    (width, height).
    """

    if normalise:
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
        if valueWidget.isVisible():
            labelWidget.hide()
            valueWidget.hide()
            formLayout.removeWidget(labelWidget)
            formLayout.removeWidget(valueWidget)
