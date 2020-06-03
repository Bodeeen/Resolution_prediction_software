from PyQt5.QtCore import pyqtSignal
import numpy as np

from frcpredict.model import PulseScheme, Pulse
from frcpredict.ui import BaseWidget
from .pulse_scheme_p import PulseSchemePresenter
from .pulse_curve_item import PulseCurveItem


class PulseSchemeWidget(BaseWidget):
    """
    A widget where the user may set the pulse scheme.
    """

    # Signals
    pulseClicked: pyqtSignal = pyqtSignal(PulseCurveItem)

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)
        self._presenter = PulseSchemePresenter(self)

        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setMenuEnabled(False)
        self.plot.getAxis("left").setTicks([[(0, "OFF"), (1, "ON")]])

    def setSelectedPulse(self, pulse: Pulse) -> None:
        """ Updates controls and pulse properties widget based on the current selection. """

        if pulse is not None:
            self.editProperties.setModel(pulse)
            self.editProperties.setEnabled(True)
            self.btnRemovePulse.setEnabled(True)
        else:
            # Clear properties
            self.editProperties.setModel(
                Pulse(wavelength=0.0, duration=0.0,
                      max_intensity=0.0, illumination_pattern=np.zeros((80, 80)))
            )
            self.editProperties.setEnabled(False)
            self.btnRemovePulse.setEnabled(False)

    def highlightPulse(self, pulseKey: str) -> None:
        """ Highlights a specific pulse in the pulse scheme plot (and unhighlights all others). """
        self.clearPlotHighlighting()
        for child in self.plot.listDataItems():
            if isinstance(child, PulseCurveItem) and child.pulseKey == pulseKey:
                child.highlight()

    def clearPlotHighlighting(self) -> None:
        """ Unhighlights all the pulses in the pulse scheme plot. """
        for child in self.plot.listDataItems():
            if isinstance(child, PulseCurveItem):
                child.unhighlight()

    def updatePlot(self, model: PulseScheme) -> None:
        """ Redraws the pulse scheme plot based on the passed model. """
        self.plot.clear()

        nextStartTime = 1
        for (key, pulse) in model._pulses.items():
            curve = PulseCurveItem(key, pulse.wavelength, nextStartTime, pulse.duration)
            curve.sigClicked.connect(self._onCurveClicked)
            self.plot.addItem(curve)
            nextStartTime += curve.duration + 1

    # Internal methods
    def _onCurveClicked(self, curve) -> None:
        self.pulseClicked.emit(curve)
