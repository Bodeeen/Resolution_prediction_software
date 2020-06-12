from PyQt5.QtCore import pyqtSlot

from frcpredict.model import IlluminationResponse
from frcpredict.ui import BasePresenter


class ResponsePropertiesPresenter(BasePresenter[IlluminationResponse]):
    """
    Presenter for the response properties widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: IlluminationResponse) -> None:
        self._model = model

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
        widget.offToOnEdited.connect(self._uiOffToOnEdit)
        widget.onToOffEdited.connect(self._uiOnToOffEdit)
        widget.emissionEdited.connect(self._uiEmissionEdit)

    # Model event handling
    def _onBasicFieldChange(self, model: IlluminationResponse) -> None:
        """ Loads basic model fields (spinboxes etc.) into the widget. """
        self.widget.updateBasicFields(model)

    # UI event handling
    @pyqtSlot(float)
    def _uiOffToOnEdit(self, value: float) -> None:
        self.model.cross_section_off_to_on = value

    @pyqtSlot(float)
    def _uiOnToOffEdit(self, value: float) -> None:
        self.model.cross_section_on_to_off = value

    @pyqtSlot(float)
    def _uiEmissionEdit(self, value: float) -> None:
        self.model.cross_section_emission = value
