import numpy as np

from PyQt5.QtCore import pyqtSlot, QObject

from frcpredict.model import Pulse
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import getArrayPixmap
from frcpredict.util import patterns


class PulsePropertiesPresenter(BasePresenter[Pulse]):
    """
    Presenter for the pulse properties widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: Pulse) -> None:
        self._model = model

        # Trigger model change event handlers
        self._onIlluminationPatternChange(model.illumination_pattern)
        self._onBasicFieldChange(model)

        # Prepare model events
        model.illumination_pattern_changed.connect(self._onIlluminationPatternChange)
        model.basic_field_changed.connect(self._onBasicFieldChange)

    # Methods
    def __init__(self, widget) -> None:
        # Initialize model
        model = Pulse(
            wavelength=0.0,
            duration=0.0,
            max_intensity=0.0,
            illumination_pattern=np.zeros((80, 80))
        )

        super().__init__(model, widget)

        # Prepare UI events
        widget.wavelengthChanged.connect(self._uiWavelengthChange)
        widget.durationChanged.connect(self._uiDurationChange)
        widget.maxIntensityChanged.connect(self._uiMaxIntensityChange)
        widget.patternSelectionChanged.connect(self._uiIlluminationPatternSelectionChange)

    # Model event handling
    def _onIlluminationPatternChange(self, illuminationPattern: np.ndarray) -> None:
        """ Loads the illumination pattern into a visualization in the interface. """
        self.widget.setIlluminationPatternPixmap(getArrayPixmap(illuminationPattern))

    def _onBasicFieldChange(self, model: Pulse) -> None:
        """ Loads basic model fields (spinboxes etc.) into the interface fields. """
        self.widget.updateBasicFields(model)

    # UI event handling
    @pyqtSlot(int)
    def _uiWavelengthChange(self, value: int) -> None:
        self.model.wavelength = value

    @pyqtSlot(float)
    def _uiDurationChange(self, value: float) -> None:
        self.model.duration = value

    @pyqtSlot(float)
    def _uiMaxIntensityChange(self, value: float) -> None:
        self.model.max_intensity = value

    @pyqtSlot(int)
    def _uiIlluminationPatternSelectionChange(self, selectedIndex: int) -> None:
        selectedPattern = self.widget.listPatterns.item(selectedIndex).text()
        self.model.illumination_pattern = patterns[selectedPattern]
