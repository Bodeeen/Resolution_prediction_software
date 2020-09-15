import numpy as np
from typing import Tuple

from pyqtgraph import PlotCurveItem

from frcpredict.util import wavelength_to_rgb


class PulseCurveItem(PlotCurveItem):
    # Properties
    @property
    def pulseKey(self) -> str:
        return self._pulseKey

    @property
    def startTime(self) -> float:
        return self._startTime

    @property
    def duration(self) -> float:
        return self._duration

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
        linspaceEnd = self._plotEndTime + 1
        linspaceSteps = 10000

        x = np.linspace(0, linspaceEnd, linspaceSteps)
        y = np.zeros(len(x))

        for i in range(0, len(y)):
            if startTime <= x[i] < startTime + max(duration, (linspaceEnd / linspaceSteps)):
                y[i] = 1

        return x, y
