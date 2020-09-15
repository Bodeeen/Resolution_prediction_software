from typing import Union

from PyQt5.QtCore import pyqtSlot

from frcpredict.model import Multivalue, DetectorProperties
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import connectMulti, connectModelToSignal, disconnectModelFromSignal


class DetectorPropertiesPresenter(BasePresenter[DetectorProperties]):
    """
    Presenter for the detector properties widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: DetectorProperties) -> None:
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
        super().__init__(DetectorProperties(), widget)

        # Prepare UI events
        widget.typePointDetectorToggled.connect(self._uiPointDetectorToggle)
        widget.typeCameraToggled.connect(self._uiCameraToggle)
        connectMulti(widget.readoutNoiseChanged, [float, Multivalue],
                     self._uiReadoutNoiseChange)
        connectMulti(widget.quantumEfficiencyChanged, [float, Multivalue],
                     self._uiQuantumEfficiencyChange)
        connectMulti(widget.cameraPixelSizeChanged, [float, Multivalue],
                     self._uiCameraPixelSizeChange)

    # Model event handling
    def _onBasicFieldChange(self, model: DetectorProperties) -> None:
        """ Loads basic model fields (e.g. ints) into the widget. """
        self.widget.updateBasicFields(model)

    # UI event handling
    @pyqtSlot(bool)
    def _uiPointDetectorToggle(self, enabled: bool) -> None:
        if enabled:
            self.model.camera_pixel_size = None

    @pyqtSlot(bool)
    def _uiCameraToggle(self, enabled: bool) -> None:
        if enabled:
            self.model.camera_pixel_size = 20.0

    @pyqtSlot(float)
    @pyqtSlot(Multivalue)
    def _uiReadoutNoiseChange(self, value: Union[float, Multivalue[float]]) -> None:
        self.model.readout_noise = value

    @pyqtSlot(float)
    @pyqtSlot(Multivalue)
    def _uiQuantumEfficiencyChange(self, value: Union[float, Multivalue[float]]) -> None:
        self.model.quantum_efficiency = value

    @pyqtSlot(float)
    @pyqtSlot(Multivalue)
    def _uiCameraPixelSizeChange(self, value: Union[float, Multivalue[float]]) -> None:
        self.model.camera_pixel_size = value
