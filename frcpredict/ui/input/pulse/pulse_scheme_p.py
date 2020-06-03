import numpy as np

from PyQt5.QtCore import pyqtSlot, QObject
from PyQt5.QtWidgets import QMessageBox

from frcpredict.model import PulseScheme, Pulse
from .add_pulse_dialog import AddPulseDialog
from .pulse_curve_item import PulseCurveItem


class PulseSchemePresenter(QObject):
    """
    Presenter for the pulse scheme widget.
    """

    # Properties
    @property
    def model(self) -> PulseScheme:
        return self._model

    @model.setter
    def model(self, model: PulseScheme) -> None:
        self._model = model

        # Update data in widget
        self._onPulseAdded()

        # Prepare model events
        model.pulse_added.connect(self._onPulseAdded)
        model.pulse_moved.connect(self._onPulseMoved)
        model.pulse_removed.connect(self._onPulseRemoved)

    # Methods
    def __init__(self, widget, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self._selectedPulseKey = None

        self._widget = widget
        self._widget.editProperties.setEditWavelengthEnabled(False)
        self._widget.setSelectedPulse(None)

        # Prepare UI events
        self._widget.pulseClicked.connect(self._onPulseClicked)
        self._widget.plot.scene().sigMouseClicked.connect(self._onPlotClicked)
        self._widget.btnAddPulse.clicked.connect(self._onClickAddPulse)
        self._widget.btnRemovePulse.clicked.connect(self._onClickRemovePulse)
        
        # TODO: This is a simple way of listening to these changes, but maybe not optimal
        self._widget.editProperties.editDuration.valueChanged.connect(self._onChangeDuration)
        self._widget.editProperties.btnMoveLeft.clicked.connect(self._onMovePulseLeft)
        self._widget.editProperties.btnMoveRight.clicked.connect(self._onMovePulseRight)

        # Initialize model
        self.model = PulseScheme(
            pulses=[]
        )

    # Model event handling
    def _onPulseAdded(self, *args, **kwargs) -> None:
        self._widget.updatePlot(self.model)
        if self._selectedPulseKey is not None:
            self._widget.setSelectedPulse(None)  # De-select currently selected pulse

    def _onPulseMoved(self, *args, **kwargs) -> None:
        self._widget.updatePlot(self.model)
        self._widget.highlightPulse(self._selectedPulseKey)  # Re-highlight currently selected pulse

    def _onPulseRemoved(self, *args, **kwargs) -> None:
        self._widget.updatePlot(self.model)
        self._widget.setSelectedPulse(None)

    # UI event handling
    @pyqtSlot(PulseCurveItem)
    def _onPulseClicked(self, pulseCurve: PulseCurveItem) -> None:
        """ Select and highlight the clicked pulse. """
        self._selectedPulseKey = pulseCurve.pulseKey
        self._widget.setSelectedPulse(self.model._pulses[self._selectedPulseKey])
        self._widget.clearPlotHighlighting()
        pulseCurve.highlight()

    def _onPlotClicked(self, mouseClickEvent) -> None:
        """ De-select pulse when the plot is right-clicked. """
        if mouseClickEvent.button() == 2:
            self._widget.setSelectedPulse(None)
            self._widget.clearPlotHighlighting()

    @pyqtSlot()
    def _onClickAddPulse(self) -> None:
        """
        Adds a pulse. A dialog will open for the user to enter the properties first.
        """

        pulse, ok_pressed = AddPulseDialog.getPulse(self._widget)
        if ok_pressed:
            self.model.add_pulse(pulse)

    @pyqtSlot()
    def _onClickRemovePulse(self) -> None:
        """
        Removes the currently selected pulse. A dialog will open for the user to confirm first.
        """

        confirmation_result = QMessageBox.question(
            self._widget, "Remove Pulse", "Remove the selected pulse?")

        if confirmation_result == QMessageBox.Yes:
            self.model.remove_pulse(self._selectedPulseKey)

    @pyqtSlot()
    def _onChangeDuration(self) -> None:
        pass
        # TODO: This causes issues due to updatePlot doing things that lead to this function being
        #       called again.
        #self._widget.updatePlot(self.model)
        #self._widget.highlightPulse(self._selectedPulseKey)  # Re-highlight currently selected pulse

    @pyqtSlot()
    def _onMovePulseLeft(self) -> None:
        """ Moves the selected pulse one step to the left in the order of pulses. """
        self.model.move_pulse_left(self._selectedPulseKey)

    @pyqtSlot()
    def _onMovePulseRight(self) -> None:
        """ Moves the selected pulse one step to the right in the order of pulses. """
        self.model.move_pulse_right(self._selectedPulseKey)
