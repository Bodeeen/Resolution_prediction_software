from PyQt5.QtCore import pyqtSignal

from frcpredict.model import SimulationSettings
from frcpredict.ui import BaseWidget
from .simulation_settings_p import SimulationSettingsPresenter


class SimulationSettingsWidget(BaseWidget):
    """
    A widget where the user may edit advanced simulation settings.
    """

    # Signals
    valueChanged = pyqtSignal(SimulationSettings)

    canvasSideLengthChanged = pyqtSignal(float)
    numDetectionIterationsChanged = pyqtSignal(int)

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        # Connect forwarded signals
        self.editCanvasSideLength.valueChanged.connect(self.canvasSideLengthChanged)
        self.editNumDetectionIterations.valueChanged.connect(self.numDetectionIterationsChanged)

        # Initialize presenter
        self._presenter = SimulationSettingsPresenter(self)

    def value(self) -> SimulationSettings:
        return self._presenter.model

    def setValue(self, model: SimulationSettings, emitSignal: bool = True) -> None:
        self._presenter.model = model

        if emitSignal:
            self.valueChanged.emit(model)

    def updateBasicFields(self, model: SimulationSettings) -> None:
        self.editCanvasSideLength.setValue(model.canvas_inner_radius * 2)
        self.editNumDetectionIterations.setValue(model.num_kernel_detection_iterations)
