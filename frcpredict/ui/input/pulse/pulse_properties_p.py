from typing import Union

from PyQt5.QtCore import pyqtSlot

from frcpredict.model import Pulse, Pattern, Array2DPatternData, Multivalue
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import connectMulti


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
        super().__init__(Pulse(), widget)

        # Prepare UI events
        connectMulti(widget.wavelengthChanged, [float, Multivalue], self._uiWavelengthChange)
        connectMulti(widget.durationChanged, [float, Multivalue], self._uiDurationChange)
        connectMulti(widget.maxIntensityChanged, [float, Multivalue], self._uiMaxIntensityChange)
        widget.illuminationPatternChanged.connect(self._uiSetIlluminationPatternModel)

    # Model event handling
    def _onBasicFieldChange(self, model: Pulse) -> None:
        """ Loads basic model fields (e.g. ints) into the widget. """
        self.widget.updateBasicFields(model)

    # UI event handling
    @pyqtSlot(float)
    @pyqtSlot(Multivalue)
    def _uiWavelengthChange(self, value: Union[float, Multivalue[float]]) -> None:
        self.model.wavelength = value

    @pyqtSlot(float)
    @pyqtSlot(Multivalue)
    def _uiDurationChange(self, value: Union[float, Multivalue[float]]) -> None:
        self.model.duration = value

    @pyqtSlot(float)
    @pyqtSlot(Multivalue)
    def _uiMaxIntensityChange(self, value: Union[float, Multivalue[float]]) -> None:
        self.model.max_intensity = value

    @pyqtSlot(Pattern)
    def _uiSetIlluminationPatternModel(self, value: Pattern) -> None:
        self.model.illumination_pattern = value
