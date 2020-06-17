import numpy as np

from PyQt5.QtCore import Qt
import pyqtgraph as pg

from frcpredict.ui import BaseWidget


class FRCResultsWidget(BaseWidget):
    """
    A widget that displays a generated FRC graph.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

    def setCurve(self, x: np.ndarray, y: np.ndarray) -> None:
        """ TODO. """
        threshold = 0.15
        self.plotFrc.clear()
        self.plotFrc.plot(x, y, clickable=True)
        self.plotFrc.plot([x[0], x[-1]], [threshold, threshold], pen=pg.mkPen("r", style=Qt.DashLine))

        self.plotFrc.getAxis("bottom").setTicks(
            [[(1/value, str(value)) for value in [300, 100, 60, 40, 30]]]
        )
