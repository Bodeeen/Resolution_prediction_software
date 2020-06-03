import numpy as np
from PyQt5.QtGui import QImage, QPixmap

def getArrayPixmap(arr: np.ndarray) -> QPixmap:
    """
    Converts a numpy array to a QPixmap. The array is expected to be of the shape
    (width, height).
    """

    uint8_arr = (arr * 255).astype("uint8")  # Convert from [0, 1] floats to [0, 255] ints
    width = len(arr[0])
    height = len(arr)

    return QPixmap(
        QImage(uint8_arr, width, height, width, QImage.Format_Grayscale8)
    )
