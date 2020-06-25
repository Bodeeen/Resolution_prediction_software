from PyQt5.QtCore import pyqtSlot

from frcpredict.model import CameraProperties
from frcpredict.ui import BasePresenter


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
        widget.readoutNoiseChanged.connect(self._uiReadoutNoiseChange)
        widget.quantumEfficiencyChanged.connect(self._uiQuantumEfficiencyChange)

    # Model event handling
    def _onBasicFieldChange(self, model: CameraProperties) -> None:
        """ Loads basic model fields (e.g. ints) into the widget. """
        self.widget.updateBasicFields(model)

    # UI event handling
    @pyqtSlot(float)
    def _uiReadoutNoiseChange(self, value: float) -> None:
        self.model.readout_noise = value

    @pyqtSlot(float)
    def _uiQuantumEfficiencyChange(self, value: float) -> None:
        self.model.quantum_efficiency = value / 100  # Convert to probability fraction
