from PyQt5.QtCore import pyqtSlot

from frcpredict.model import SimulationSettings
from frcpredict.ui import BasePresenter


class SimulationSettingsPresenter(BasePresenter[SimulationSettings]):
    """
    Presenter for the simulation settings widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: SimulationSettings) -> None:
        # Disconnect old model event handling
        try:
            self._model.canvas_inner_radius_changed.disconnect(self._onAnyFieldChange)
            self._model.num_kernel_detection_iterations_changed.disconnect(self._onAnyFieldChange)
        except AttributeError:
            pass

        # Set model
        self._model = model

        # Trigger model change event handlers
        self._onAnyFieldChange(model)

        # Prepare model events
        model.canvas_inner_radius_changed.connect(self._onAnyFieldChange)
        model.num_kernel_detection_iterations_changed.connect(self._onAnyFieldChange)

    # Methods
    def __init__(self, widget) -> None:
        super().__init__(SimulationSettings(), widget)

        # Prepare UI events
        widget.canvasSideLengthChanged.connect(self._uiCanvasSideLengthChange)
        widget.numDetectionIterationsChanged.connect(self._uiNumDetectionIterationsChange)

    # Model event handling
    def _onAnyFieldChange(self, _) -> None:
        self.widget.updateBasicFields(self.model)

    # UI event handling
    @pyqtSlot(float)
    def _uiCanvasSideLengthChange(self, value: float) -> None:
        self.model.canvas_inner_radius = value / 2

    @pyqtSlot(int)
    def _uiNumDetectionIterationsChange(self, value: int) -> None:
        self.model.num_kernel_detection_iterations = value
