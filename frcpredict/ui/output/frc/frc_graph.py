import numpy as np

from PyQt5.QtCore import Qt
import pyqtgraph as pg

from frcpredict.ui import BaseWidget


class FRCResultsWidget(BaseWidget):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)
        
        # Temp
        x = np.arange(0, 10, 0.01)
        y = -1 / (1 + np.exp(5 - x))
        threshold = np.repeat(-0.8, 10 + 1)
        self.plotFrc.plot(x, y)
        self.plotFrc.plot(threshold, pen=pg.mkPen("r", style=Qt.DashLine))
