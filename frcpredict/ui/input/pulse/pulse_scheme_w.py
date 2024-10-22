from functools import reduce
from typing import Optional

import pyqtgraph as pg
from PyQt5.QtCore import pyqtSignal, pyqtSlot, QPointF

from frcpredict.model import PulseScheme, Pulse, Multivalue
from frcpredict.ui import BaseWidget
from frcpredict.ui.util import connectMulti, PresetFileDirs, UserFileDirs
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
    modifiedFlagSet = pyqtSignal()

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
        self.configPanel.setModelType(PulseScheme)
        self.configPanel.setPresetsDirectory(PresetFileDirs.PulseScheme)
        self.configPanel.setStartDirectory(UserFileDirs.PulseScheme)
        self.configPanel.setValueGetter(self.value)
        self.configPanel.setValueSetter(self.setValue)

        self.plot.setMouseEnabled(x=False, y=False)
        self.plot.setMenuEnabled(False)
        self.plot.getAxis("left").setTicks([[(0, "OFF"), (1, "ON")]])

        # Connect own signal slots
        self.modifiedFlagSet.connect(self._onModifiedFlagSet)
        self.editProperties.modifiedFlagSet.connect(self._onModifiedFlagSet)
        self.configPanel.dataLoaded.connect(self._onLoadConfig)

        # Connect forwarded signals
        self.plot.scene().sigMouseClicked.connect(self.plotClicked)
        self.btnAddPulse.clicked.connect(self.addPulseClicked)
        self.btnRemovePulse.clicked.connect(self.removePulseClicked)
        self.configPanel.dataLoaded.connect(self.modifiedFlagSet)

        connectMulti(self.editProperties.wavelengthChangedByUser, [float, Multivalue],
                     self.pulseWavelengthChangedByUser)
        connectMulti(self.editProperties.durationChangedByUser, [float, Multivalue],
                     self.pulseDurationChangedByUser)
        self.editProperties.moveLeftClicked.connect(self.pulseMoveLeftClicked)
        self.editProperties.moveRightClicked.connect(self.pulseMoveRightClicked)

        # Initialize presenter
        self._presenter = PulseSchemePresenter(self)

    def getPulseCurveItemAtScenePos(self, scenePosition: QPointF) -> Optional[PulseCurveItem]:
        """
        Returns the pulse curve item at the given plot scene position, or None if there is no pulse
        there.
        """

        viewBox = next(item for item in self.plot.items() if isinstance(item, pg.ViewBox))
        viewPosition = viewBox.mapSceneToView(scenePosition)

        if not(0 <= viewPosition.y() <= 1):
            return None

        pulseCurveItems = [item for item in self.plot.items() if isinstance(item, PulseCurveItem)]
        for item in pulseCurveItems:
            if item.startTime <= viewPosition.x() <= item.startTime + item.duration:
                return item

        return None

    def value(self) -> PulseScheme:
        return self._presenter.model

    def setValue(self, model: PulseScheme, emitSignal: bool = True) -> None:
        self.configPanel.setLoadedPath(None)
        self._presenter.model = model
        self.setSelectedPulse(None)
        self.clearPlotHighlighting()

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

    def setCanvasInnerRadius(self, canvasInnerRadiusNm: float) -> None:
        """
        Sets the inner radius of the canvas, in nanometres, that the pattern fields will use to
        generate pattern previews.
        """
        self.editProperties.setCanvasInnerRadius(canvasInnerRadiusNm)

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
    def _onModifiedFlagSet(self, *_args, **_kwargs) -> None:
        self.configPanel.setModifiedFlag()

    @pyqtSlot()
    def _onLoadConfig(self) -> None:
        self.setSelectedPulse(None)
        self.clearPlotHighlighting()
