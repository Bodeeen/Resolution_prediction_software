from typing import Union

from PyQt5.QtCore import pyqtSlot

from frcpredict.model import IlluminationResponse, Multivalue
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import connectMulti, connectModelToSignal, disconnectModelFromSignal


class ResponsePropertiesPresenter(BasePresenter[IlluminationResponse]):
    """
    Presenter for the response properties widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: IlluminationResponse) -> None:
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

        # Connect new model events
        model.basic_field_changed.connect(self._onBasicFieldChange)
        self._modifiedFlagSlotFunc = connectModelToSignal(self.model, self.widget.modifiedFlagSet)

    # Methods
    def __init__(self, widget) -> None:
        super().__init__(IlluminationResponse(), widget)

        # Prepare UI events
        widget.wavelengthChanged.connect(self._uiWavelengthChange)
        connectMulti(widget.offToOnChanged, [float, Multivalue], self._uiOffToOnChange)
        connectMulti(widget.onToOffChanged, [float, Multivalue], self._uiOnToOffChange)
        connectMulti(widget.emissionChanged, [float, Multivalue], self._uiEmissionChange)

    # Model event handling
    def _onBasicFieldChange(self, model: IlluminationResponse) -> None:
        """ Loads basic model fields (e.g. float values) into the widget. """
        self.widget.updateBasicFields(model)

    # UI event handling
    @pyqtSlot(float)
    def _uiWavelengthChange(self, value: float) -> None:
        self.model.wavelength = value

    @pyqtSlot(float)
    @pyqtSlot(Multivalue)
    def _uiOffToOnChange(self, value: Union[float, Multivalue[float]]) -> None:
        self.model.cross_section_off_to_on = value

    @pyqtSlot(float)
    @pyqtSlot(Multivalue)
    def _uiOnToOffChange(self, value: Union[float, Multivalue[float]]) -> None:
        self.model.cross_section_on_to_off = value

    @pyqtSlot(float)
    @pyqtSlot(Multivalue)
    def _uiEmissionChange(self, value: Union[float, Multivalue[float]]) -> None:
        self.model.cross_section_emission = value
