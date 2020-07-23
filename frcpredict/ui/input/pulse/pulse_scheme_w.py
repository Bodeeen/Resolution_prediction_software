from functools import reduce
from typing import Optional

from PyQt5.QtCore import pyqtSignal, pyqtSlot

from frcpredict.model import PulseScheme, Pulse, Multivalue
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import connectMulti, UserFileDirs
from frcpredict.util import avg_value_if_multivalue
from .pulse_scheme_p import PulseSchemePresenter
from .pulse_curve_item import PulseCurveItem


class PulseSchemeWidget(BaseWidget):
    """
    A widget where the user may set the pulse scheme by adding, removing, and editing individual
    pulses. It contains a dynamically updated graph that shows these pulses and allows the user to
    select existing pulses.
    """

    # Signals
    valueChanged = pyqtSignal(PulseScheme)
    pulseClicked = pyqtSignal(PulseCurveItem)
    plotClicked = pyqtSignal(object)
    addPulseClicked = pyqtSignal()
    removePulseClicked = pyqtSignal()

    pulseWavelengthChangedByUser = pyqtSignal()
    pulseDurationChangedByUser = pyqtSignal()
    pulseMoveLeftClicked = pyqtSignal()
    pulseMoveRightClicked = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        # Prepare UI elements 
        self.presetPicker.setModelType(PulseScheme)
        self.presetPicker.setStartDirectory(UserFileDirs.PulseScheme)
        self.presetPicker.setValueGetter(self.value)
        self.presetPicker.setValueSetter(self.setValue)

        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setMenuEnabled(False)
        self.plot.getAxis("left").setTicks([[(0, "OFF"), (1, "ON")]])

        # Connect own signal slots
        self.presetPicker.dataLoaded.connect(self._onLoadPreset)

        # Connect forwarded signals
        self.plot.scene().sigMouseClicked.connect(self.plotClicked)
        self.btnAddPulse.clicked.connect(self.addPulseClicked)
        self.btnRemovePulse.clicked.connect(self.removePulseClicked)

        connectMulti(self.editProperties.wavelengthChangedByUser, [float, Multivalue],
                     self.pulseWavelengthChangedByUser)
        connectMulti(self.editProperties.durationChangedByUser, [float, Multivalue],
                     self.pulseDurationChangedByUser)
        self.editProperties.moveLeftClicked.connect(self.pulseMoveLeftClicked)
        self.editProperties.moveRightClicked.connect(self.pulseMoveRightClicked)

        # Initialize presenter
        self._presenter = PulseSchemePresenter(self)

    def value(self) -> PulseScheme:
        return self._presenter.model

    def setValue(self, model: PulseScheme, emitSignal: bool = True) -> None:
        self.presetPicker.setLoadedPath(None)
        self._presenter.model = model

        if emitSignal:
            self.valueChanged.emit(model)

    def setSelectedPulse(self, pulse: Optional[Pulse]) -> None:
        """ Updates controls and pulse properties widget based on the current selection. """

        if pulse is not None:
            self.editProperties.setValue(pulse)
            self.editProperties.setEnabled(True)
            self.btnRemovePulse.setEnabled(True)
        else:
            # Clear properties
            self.editProperties.setValue(Pulse())
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

        plotEndTime = reduce(
            lambda current, pulse: 1 + current + avg_value_if_multivalue(pulse.duration),
            model._pulses.values(), 0
        )
        
        nextStartTime = 1
        for (key, pulse) in model.get_pulses_with_keys():
            wavelength = avg_value_if_multivalue(pulse.wavelength)
            duration = avg_value_if_multivalue(pulse.duration)

            curve = PulseCurveItem(key, wavelength=wavelength, startTime=nextStartTime,
                                   duration=duration, plotEndTime=plotEndTime)

            curve.sigClicked.connect(lambda clickedCurve: self.pulseClicked.emit(clickedCurve))
            self.plot.addItem(curve)
            nextStartTime += duration + 1

    # Event handling
    @pyqtSlot()
    def _onLoadPreset(self) -> None:
        self.setSelectedPulse(None)
        self.clearPlotHighlighting()
