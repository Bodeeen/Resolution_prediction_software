import numpy as np
from typing import Tuple

from pyqtgraph import mkPen, PlotCurveItem

from frcpredict.util import wavelength_to_rgb


class PulseCurveItem(PlotCurveItem):
    # Properties
    @property
    def startTime(self):
        return self._startTime

    @property
    def duration(self):
        return self._duration

    # Methods
    def __init__(self, pulseKey: str, wavelength: int, startTime: float, duration: float, *args, **kwargs) -> None:
        x, y = self._getValues(startTime, duration)
        super().__init__(x, y, clickable=True, *args, **kwargs)

        self.pulseKey = pulseKey
        self._startTime = startTime
        self._duration = duration
        self._colour = wavelength_to_rgb(wavelength)

        self.unhighlight()

    def setTimeData(self, startTime: float = None, duration: float = None) -> None:
        """ Updates start time and duration in plot. """

        if startTime is not None:
            self._startTime = startTime

        if duration is not None:
            self._duration = duration

        x, y = self._getValues(startTime, duration)
        self.setData(x, y)

    def highlight(self) -> None:
        """ Highlights the item by making the curve thicker and moving it to the front. """
        self.setPen(self._colour, width=4)
        self.setZValue(1000000)

    def unhighlight(self) -> None:
        """ Unhighlights the item. """
        self.setPen(self._colour, width=2)
        self.setZValue(self.startTime)

    # Internal methods
    def _getValues(self, startTime: int, duration: int) -> Tuple[np.ndarray, np.ndarray]:
        x = np.arange(0, 20, 0.001)
        y = np.zeros(len(x))
        for i in range(0, len(y)):
            if x[i] >= startTime and x[i] < startTime + duration:
                y[i] = 1

        return x, y
