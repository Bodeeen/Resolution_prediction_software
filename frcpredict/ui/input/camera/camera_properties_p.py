from typing import Union

from PyQt5.QtCore import pyqtSlot

from frcpredict.model import ValueRange, CameraProperties
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import connectMulti


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
        except AttributeError:
            pass

        # Set model
        self._model = model

        # Trigger model change event handlers
        self._onBasicFieldChange(model)

        # Prepare model events
        model.basic_field_changed.connect(self._onBasicFieldChange)

    # Methods
    def __init__(self, widget) -> None:
        # Initialize model
        model = CameraProperties(
            readout_noise=0.0,
            quantum_efficiency=0.0
        )
        
        super().__init__(model, widget)

        # Prepare UI events
        connectMulti(widget.readoutNoiseChanged, [float, ValueRange],
                     self._uiReadoutNoiseChange)
        connectMulti(widget.quantumEfficiencyChanged, [float, ValueRange],
                     self._uiQuantumEfficiencyChange)

    # Model event handling
    def _onBasicFieldChange(self, model: CameraProperties) -> None:
        """ Loads basic model fields (e.g. ints) into the widget. """
        self.widget.updateBasicFields(model)

    # UI event handling
    @pyqtSlot(float)
    @pyqtSlot(ValueRange)
    def _uiReadoutNoiseChange(self, value: Union[float, ValueRange[float]]) -> None:
        self.model.readout_noise = value

    @pyqtSlot(float)
    @pyqtSlot(ValueRange)
    def _uiQuantumEfficiencyChange(self, value: Union[float, ValueRange[float]]) -> None:
        self.model.quantum_efficiency = value
