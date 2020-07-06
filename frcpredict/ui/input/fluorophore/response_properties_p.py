from typing import Union

from PyQt5.QtCore import pyqtSlot

from frcpredict.model import IlluminationResponse, ValueRange
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import connectMulti


class ResponsePropertiesPresenter(BasePresenter[IlluminationResponse]):
    """
    Presenter for the response properties widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: IlluminationResponse) -> None:
        self._model = model
        self._individualStartEnd = self._model.wavelength_start != self._model.wavelength_end

        # Trigger model change event handlers
        self._onBasicFieldChange(model)

        # Connect new model events
        model.basic_field_changed.connect(self._onBasicFieldChange)

    # Methods
    def __init__(self, widget) -> None:
        # Initialize model
        model = IlluminationResponse(
            wavelength_start=0, wavelength_end=0,
            cross_section_off_to_on=0.0, cross_section_on_to_off=0.0, cross_section_emission=0.0
        )

        super().__init__(model, widget)

        # Prepare UI events
        widget.wavelengthChanged.connect(self._uiWavelengthChange)
        connectMulti(widget.offToOnChanged, [float, ValueRange], self._uiOffToOnChange)
        connectMulti(widget.onToOffChanged, [float, ValueRange], self._uiOnToOffChange)
        connectMulti(widget.emissionChanged, [float, ValueRange], self._uiEmissionChange)

    # Model event handling
    def _onBasicFieldChange(self, model: IlluminationResponse) -> None:
        """ Loads basic model fields (e.g. ints) into the widget. """
        self.widget.updateBasicFields(model)

    # UI event handling
    @pyqtSlot(int)
    def _uiWavelengthChange(self, value: int) -> None:
        self.model.wavelength_start = value
        if not self._individualStartEnd:
            self.model.wavelength_end = value

    @pyqtSlot(float)
    @pyqtSlot(ValueRange)
    def _uiOffToOnChange(self, value: Union[float, ValueRange[float]]) -> None:
        self.model.cross_section_off_to_on = value

    @pyqtSlot(float)
    @pyqtSlot(ValueRange)
    def _uiOnToOffChange(self, value: Union[float, ValueRange[float]]) -> None:
        self.model.cross_section_on_to_off = value

    @pyqtSlot(float)
    @pyqtSlot(ValueRange)
    def _uiEmissionChange(self, value: Union[float, ValueRange[float]]) -> None:
        self.model.cross_section_emission = value
