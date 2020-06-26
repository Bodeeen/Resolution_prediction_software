from PyQt5.QtCore import pyqtSlot

from frcpredict.model import Pulse, PulseType, Pattern, Array2DPatternData
from frcpredict.ui import BasePresenter


class PulsePropertiesPresenter(BasePresenter[Pulse]):
    """
    Presenter for the pulse properties widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: Pulse) -> None:
        # Disconnect old model event handling
        try:
            self._model.basic_field_changed.disconnect(self._onBasicFieldChange)
        except AttributeError:
            pass

        # Set model
        self._model = model

        # Trigger model change event handlers
        self.widget.updateIlluminationPattern(model.illumination_pattern)
        self._onBasicFieldChange(model)

        # Prepare model events
        model.basic_field_changed.connect(self._onBasicFieldChange)

    # Methods
    def __init__(self, widget) -> None:
        # Initialize model
        model = Pulse(
            pulse_type=PulseType.on,
            wavelength=0.0,
            duration=0.0,
            max_intensity=0.0,
            illumination_pattern=Pattern(pattern_data=Array2DPatternData())
        )

        super().__init__(model, widget)

        # Prepare UI events
        widget.onTypeSelected.connect(self._uiOnTypeSelect)
        widget.offTypeSelected.connect(self._uiOffTypeSelect)
        widget.readoutTypeSelected.connect(self._uiReadoutTypeSelect)
        widget.wavelengthChanged.connect(self._uiWavelengthChange)
        widget.durationChanged.connect(self._uiDurationChange)
        widget.maxIntensityChanged.connect(self._uiMaxIntensityChange)
        widget.illuminationPatternChanged.connect(self._uiSetIlluminationPatternModel)

    # Model event handling
    def _onBasicFieldChange(self, model: Pulse) -> None:
        """ Loads basic model fields (e.g. ints) into the widget. """
        self.widget.updateBasicFields(model)

    # UI event handling
    @pyqtSlot()
    def _uiOnTypeSelect(self) -> None:
        self.model.pulse_type = PulseType.on
    
    @pyqtSlot()
    def _uiOffTypeSelect(self) -> None:
        self.model.pulse_type = PulseType.off
    
    @pyqtSlot()
    def _uiReadoutTypeSelect(self) -> None:
        self.model.pulse_type = PulseType.readout
    
    @pyqtSlot(int)
    def _uiWavelengthChange(self, value: int) -> None:
        self.model.wavelength = value

    @pyqtSlot(float)
    def _uiDurationChange(self, value: float) -> None:
        self.model.duration = value

    @pyqtSlot(float)
    def _uiMaxIntensityChange(self, value: float) -> None:
        self.model.max_intensity = value

    @pyqtSlot(Pattern)
    def _uiSetIlluminationPatternModel(self, value: Pattern) -> None:
        self.model.illumination_pattern = value
