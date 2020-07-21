import numpy as np
from typing import Tuple

from pyqtgraph import PlotCurveItem

from frcpredict.util import wavelength_to_rgb


class PulseCurveItem(PlotCurveItem):
    # Properties
    @property
    def pulseKey(self) -> str:
        return self._pulseKey

    # Methods
    def __init__(self, pulseKey: str, wavelength: float, startTime: float, duration: float,
                 plotEndTime: float = 0, *args, **kwargs) -> None:
        self._pulseKey = pulseKey
        self._startTime = startTime
        self._duration = duration
        self._plotEndTime = plotEndTime
        self._colour = wavelength_to_rgb(wavelength)

        x, y = self._getValues(startTime, duration)
        super().__init__(x, y, clickable=True, *args, **kwargs)

        self.unhighlight()

    def highlight(self) -> None:
        """ Highlights the item by making the curve thicker and moving it to the front. """
        self.setPen(self._colour, width=4)
        self.setZValue(1000000)

    def unhighlight(self) -> None:
        """ Un-highlights the item. """
        self.setPen(self._colour, width=2)
        self.setZValue(self._startTime)

    # Internal methods
    def _getValues(self, startTime: float, duration: float) -> Tuple[np.ndarray, np.ndarray]:
        x = np.linspace(0, self._plotEndTime + 1, 10000)
        y = np.zeros(len(x))

        for i in range(0, len(y)):
            if startTime <= x[i] < startTime + duration:
                y[i] = 1

        return x, y
