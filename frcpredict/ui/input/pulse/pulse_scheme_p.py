import numpy as np

from PyQt5.QtCore import pyqtSlot, QObject
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
        self._model = model

        # Trigger model change event handlers
        self._onPulseAdded()

        # Prepare model events
        model.pulse_added.connect(self._onPulseAdded)
        model.pulse_moved.connect(self._onPulseMoved)
        model.pulse_removed.connect(self._onPulseRemoved)

    # Methods
    def __init__(self, widget) -> None:
        # Initialize model
        model = PulseScheme(pulses=[])

        # Initialize self
        self._selectedPulseKey = None
        super().__init__(model, widget)

        # Prepare UI elements
        widget.editProperties.setEditWavelengthEnabled(False)
        widget.setSelectedPulse(None)

        # Prepare UI events
        widget.pulseClicked.connect(self._uiPulseClicked)
        widget.plotClicked.connect(self._uiPlotClicked)
        widget.addPulseClicked.connect(self._uiClickAddPulse)
        widget.removePulseClicked.connect(self._uiClickRemovePulse)
        widget.pulseDurationChangedByUser.connect(self._uiChangeDurationByUser)
        widget.pulseMoveLeftClicked.connect(self._uiMovePulseLeft)
        widget.pulseMoveRightClicked.connect(self._uiMovePulseRight)

    # Model event handling
    def _onPulseAdded(self, *args, **kwargs) -> None:
        self.widget.updatePlot(self.model)
        if self._selectedPulseKey is not None:
            self.widget.setSelectedPulse(None)  # De-select currently selected pulse

    def _onPulseMoved(self, *args, **kwargs) -> None:
        self.widget.updatePlot(self.model)
        self.widget.highlightPulse(self._selectedPulseKey)  # Re-highlight currently selected pulse

    def _onPulseRemoved(self, *args, **kwargs) -> None:
        self.widget.updatePlot(self.model)
        self.widget.setSelectedPulse(None)

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
        """ De-select pulse when the plot is right-clicked. """
        if mouseClickEvent.button() == 2:
            self.widget.setSelectedPulse(None)
            self.widget.clearPlotHighlighting()

    @pyqtSlot()
    def _uiClickAddPulse(self) -> None:
        """
        Adds a pulse. A dialog will open for the user to enter the properties first.
        """

        pulse, ok_pressed = AddPulseDialog.getPulse(self.widget)
        if ok_pressed:
            self.model.add_pulse(pulse)

    @pyqtSlot()
    def _uiClickRemovePulse(self) -> None:
        """
        Removes the currently selected pulse. A dialog will open for the user to confirm first.
        """

        confirmation_result = QMessageBox.question(
            self.widget, "Remove Pulse", "Remove the selected pulse?")

        if confirmation_result == QMessageBox.Yes:
            self.model.remove_pulse(self._selectedPulseKey)

    @pyqtSlot()
    def _uiChangeDurationByUser(self) -> None:
        self.widget.updatePlot(self.model)
        self.widget.highlightPulse(self._selectedPulseKey)  # Re-highlight currently selected pulse

    @pyqtSlot()
    def _uiMovePulseLeft(self) -> None:
        """ Moves the selected pulse one step to the left in the order of pulses. """
        self.model.move_pulse_left(self._selectedPulseKey)

    @pyqtSlot()
    def _uiMovePulseRight(self) -> None:
        """ Moves the selected pulse one step to the right in the order of pulses. """
        self.model.move_pulse_right(self._selectedPulseKey)
