from typing import Optional

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QMessageBox

from frcpredict.model import PulseScheme, Pulse
from frcpredict.ui import BasePresenter
from .add_pulse_dialog import AddPulseDialog
from .pulse_curve_item import PulseCurveItem


class PulseSchemePresenter(BasePresenter[PulseScheme]):
    """
    Presenter for the pulse scheme widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: PulseScheme) -> None:
        # Disconnect old model event handling
        try:
            self._model.pulse_added.disconnect(self._onPulseAdded)
            self._model.pulse_moved.disconnect(self._onPulseMoved)
            self._model.pulse_removed.disconnect(self._onPulseRemoved)
        except AttributeError:
            pass

        # Set model
        self._model = model

        # Trigger model change event handlers
        self.widget.updatePlot(model)

        # Prepare model events
        model.pulse_added.connect(self._onPulseAdded)
        model.pulse_moved.connect(self._onPulseMoved)
        model.pulse_removed.connect(self._onPulseRemoved)

    # Methods
    def __init__(self, widget) -> None:
        self._selectedPulseKey = None
        super().__init__(PulseScheme(), widget)

        # Prepare UI elements
        widget.setSelectedPulse(None)

        # Prepare UI events
        widget.pulseClicked.connect(self._uiPulseClicked)
        widget.plotClicked.connect(self._uiPlotClicked)
        widget.addPulseClicked.connect(self._uiClickAddPulse)
        widget.removePulseClicked.connect(self._uiClickRemovePulse)

        widget.pulseWavelengthChangedByUser.connect(self._uiChangeWavelengthByUser)
        widget.pulseDurationChangedByUser.connect(self._uiChangeDurationByUser)
        widget.pulseMoveLeftClicked.connect(self._uiMovePulseLeft)
        widget.pulseMoveRightClicked.connect(self._uiMovePulseRight)

    # Internal methods
    def _updatePlotInWidget(self, keyOfPulseToHighlight: Optional[str]) -> None:
        self.widget.updatePlot(self.model)
        self.widget.highlightPulse(keyOfPulseToHighlight)

    # Model event handling
    def _onPulseAdded(self, key: str, pulse: Pulse) -> None:
        self.widget.updatePlot(self.model)

        self._selectedPulseKey = key
        self.widget.setSelectedPulse(pulse)
        self.widget.highlightPulse(key)

    def _onPulseMoved(self, _key: str, _pulse: Pulse) -> None:
        self._updatePlotInWidget(self._selectedPulseKey)

    def _onPulseRemoved(self, _key: str, _pulse: Pulse) -> None:
        self._updatePlotInWidget(None)
        self.widget.setSelectedPulse(None)
        self.widget.clearPlotHighlighting()

    # UI event handling
    @pyqtSlot(PulseCurveItem)
    def _uiPulseClicked(self, pulseCurve: PulseCurveItem) -> None:
        """ Select and highlight the clicked pulse. """
        self._selectedPulseKey = pulseCurve.pulseKey
        self.widget.setSelectedPulse(self.model._pulses[self._selectedPulseKey])
        self.widget.clearPlotHighlighting()
        pulseCurve.highlight()

    @pyqtSlot(object)
    def _uiPlotClicked(self, mouseClickEvent) -> None:
        """
        On left click, select the pulse that that the user clicked "inside" of if wasn't clicked
        directly. When the plot is right-clicked, any selected pulses will be de-selected.
        """

        if mouseClickEvent.button() == 1:
            if not isinstance(mouseClickEvent.currentItem, PulseCurveItem):
                clickedCurve = self.widget.getPulseCurveItemAtScenePos(mouseClickEvent.scenePos())
                if clickedCurve is not None:
                    self._uiPulseClicked(clickedCurve)
        elif mouseClickEvent.button() == 2:
            self.widget.setSelectedPulse(None)
            self.widget.clearPlotHighlighting()

    @pyqtSlot()
    def _uiClickAddPulse(self) -> None:
        """
        Adds a pulse. A dialog will open for the user to enter the properties first.
        """

        pulse, okClicked = AddPulseDialog.getPulse(self.widget)
        if okClicked:
            self.model.add_pulse(pulse)

    @pyqtSlot()
    def _uiClickRemovePulse(self) -> None:
        """
        Removes the currently selected pulse. A dialog will open for the user to confirm first.
        """

        confirmation_result = QMessageBox.question(
            self.widget, "Remove Pulse", "Remove the selected pulse?",
            defaultButton=QMessageBox.No)

        if confirmation_result == QMessageBox.Yes:
            self.model.remove_pulse(self._selectedPulseKey)

    @pyqtSlot()
    def _uiChangeWavelengthByUser(self) -> None:
        self._updatePlotInWidget(self._selectedPulseKey)

    @pyqtSlot()
    def _uiChangeDurationByUser(self) -> None:
        self._updatePlotInWidget(self._selectedPulseKey)

    @pyqtSlot()
    def _uiMovePulseLeft(self) -> None:
        """ Moves the selected pulse one step to the left in the order of pulses. """
        self.model.move_pulse_left(self._selectedPulseKey)

    @pyqtSlot()
    def _uiMovePulseRight(self) -> None:
        """ Moves the selected pulse one step to the right in the order of pulses. """
        self.model.move_pulse_right(self._selectedPulseKey)
