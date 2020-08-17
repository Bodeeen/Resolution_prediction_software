from typing import Union

from PyQt5.QtCore import pyqtSlot

from frcpredict.model import Multivalue, CameraProperties
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import connectMulti, connectModelToSignal, disconnectModelFromSignal


class CameraPropertiesPresenter(BasePresenter[CameraProperties]):
    """
    Presenter for the camera properties widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: CameraProperties) -> None:
        # Disconnect old model event handling
        try:
            self._model.basic_field_changed.disconnect(self._onBasicFieldChange)
            disconnectModelFromSignal(self.model, self._modifiedFlagSlotFunc)
        except AttributeError:
            pass

        # Set model
        self._model = model

        # Trigger model change event handlers
        self._onBasicFieldChange(model)

        # Prepare model events
        model.basic_field_changed.connect(self._onBasicFieldChange)
        self._modifiedFlagSlotFunc = connectModelToSignal(self.model, self.widget.modifiedFlagSet)

    # Methods
    def __init__(self, widget) -> None:
        super().__init__(CameraProperties(), widget)

        # Prepare UI events
        connectMulti(widget.readoutNoiseChanged, [float, Multivalue],
                     self._uiReadoutNoiseChange)
        connectMulti(widget.quantumEfficiencyChanged, [float, Multivalue],
                     self._uiQuantumEfficiencyChange)

    # Model event handling
    def _onBasicFieldChange(self, model: CameraProperties) -> None:
        """ Loads basic model fields (e.g. ints) into the widget. """
        self.widget.updateBasicFields(model)

    # UI event handling
    @pyqtSlot(float)
    @pyqtSlot(Multivalue)
    def _uiReadoutNoiseChange(self, value: Union[float, Multivalue[float]]) -> None:
        self.model.readout_noise = value

    @pyqtSlot(float)
    @pyqtSlot(Multivalue)
    def _uiQuantumEfficiencyChange(self, value: Union[float, Multivalue[float]]) -> None:
        self.model.quantum_efficiency = value
