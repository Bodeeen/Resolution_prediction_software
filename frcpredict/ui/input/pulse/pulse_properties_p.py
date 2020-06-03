import numpy as np

from PyQt5.QtCore import pyqtSlot, QObject

from frcpredict.model import Pulse
from frcpredict.ui.util import getArrayPixmap
from frcpredict.util import patterns


class PulsePropertiesPresenter(QObject):
    """
    Presenter for the pulse properties widget.
    """

    # Properties
    @property
    def model(self) -> Pulse:
        return self._model

    @model.setter
    def model(self, model: Pulse) -> None:
        self._model = model

        # Update data in widget
        self._onIlluminationPatternChange(model.illumination_pattern)
        self._onBasicFieldChange(model)

        # Prepare model events
        model.illumination_pattern_changed.connect(self._onIlluminationPatternChange)
        model.basic_field_changed.connect(self._onBasicFieldChange)

    # Methods
    def __init__(self, widget, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._widget = widget

        # Prepare UI events
        self._widget.editWavelength.valueChanged.connect(self._onWavelengthChange)
        self._widget.editDuration.valueChanged.connect(self._onDurationChange)
        self._widget.editMaxIntensity.valueChanged.connect(self._onMaxIntensityChange)
        self._widget.listPatterns.currentRowChanged.connect(self._onIlluminationPatternSelectionChange)

        # Initialize model
        self.model = Pulse(
            wavelength=0.0,
            duration=0.0,
            max_intensity=0.0,
            illumination_pattern=np.zeros((80, 80))
        )

    # Model event handling
    def _onIlluminationPatternChange(self, illuminationPattern: np.ndarray) -> None:
        """ Loads the illumination pattern into a visualization in the interface. """
        self._widget.setIlluminationPatternPixmap(getArrayPixmap(illuminationPattern))

    def _onBasicFieldChange(self, model: Pulse) -> None:
        """ Loads basic model fields (spinboxes etc.) into the interface fields. """
        self._widget.updateBasicFields(model)

    # UI event handling
    @pyqtSlot(int)
    def _onWavelengthChange(self, value: int) -> None:
        self.model.wavelength = value

    @pyqtSlot(float)
    def _onDurationChange(self, value: float) -> None:
        self.model.duration = value

    @pyqtSlot(float)
    def _onMaxIntensityChange(self, value: float) -> None:
        self.model.max_intensity = value

    @pyqtSlot(int)
    def _onIlluminationPatternSelectionChange(self, selectedIndex: int) -> None:
        selectedPattern = self._widget.listPatterns.item(selectedIndex).text()
        self.model.illumination_pattern = patterns[selectedPattern]
