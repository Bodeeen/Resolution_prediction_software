import functools

from PyQt5.QtCore import pyqtSignal

from frcpredict.model import PulseScheme, Pulse, PulseType, Pattern, Array2DPatternData, ValueRange
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import UserFileDirs
from frcpredict.util import avg_value_if_range
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
        
        self.editProperties.setEditWavelengthEnabled(False)

        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setMenuEnabled(False)
        self.plot.getAxis("left").setTicks([[(0, "OFF"), (1, "ON")]])

        # Connect forwarded signals
        self.plot.scene().sigMouseClicked.connect(self.plotClicked)
        self.btnAddPulse.clicked.connect(self.addPulseClicked)
        self.btnRemovePulse.clicked.connect(self.removePulseClicked)
        self.editProperties.durationChangedByUser.connect(self.pulseDurationChangedByUser)
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

    def setSelectedPulse(self, pulse: Pulse) -> None:
        """ Updates controls and pulse properties widget based on the current selection. """

        if pulse is not None:
            self.editProperties.setValue(pulse)
            self.editProperties.setEnabled(True)
            self.btnRemovePulse.setEnabled(True)
        else:
            # Clear properties
            self.editProperties.setValue(
                Pulse(
                    pulse_type=PulseType.on,
                    wavelength=0,
                    duration=0.0,
                    max_intensity=0.0,
                    illumination_pattern=Pattern(pattern_data=Array2DPatternData())
                )
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

        plotEndTime = functools.reduce(
            lambda current, pulse: 1 + current + avg_value_if_range(pulse.duration),
            model._pulses.values(), 0
        )
        
        nextStartTime = 1
        for (key, pulse) in model._pulses.items():
            curve = PulseCurveItem(
                key, wavelength=pulse.wavelength,
                startTime=nextStartTime, duration=avg_value_if_range(pulse.duration),
                plotEndTime=plotEndTime
            )

            curve.sigClicked.connect(self._onCurveClicked)
            self.plot.addItem(curve)
            nextStartTime += curve.duration + 1

    # Internal methods
    def _onCurveClicked(self, curve) -> None:
        self.pulseClicked.emit(curve)
