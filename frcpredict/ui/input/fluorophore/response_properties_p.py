from PyQt5.QtCore import pyqtSlot, QObject

from frcpredict.model import IlluminationResponse


class ResponsePropertiesPresenter(QObject):
    """
    Presenter for the response properties widget.
    """

    # Properties
    @property
    def model(self) -> IlluminationResponse:
        return self._model

    @model.setter
    def model(self, model: IlluminationResponse) -> None:
        self._model = model

        # Update data in widget
        self._onBasicFieldChange(model)

        # Prepare model events
        model.basic_field_changed.connect(self._onBasicFieldChange)

    # Methods
    def __init__(self, widget, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._widget = widget

        # Prepare UI events
        self._widget.editOffToOn.valueChanged.connect(self._onOffToOnChange)
        self._widget.editOnToOff.valueChanged.connect(self._onOnToOffChange)
        self._widget.editEmission.valueChanged.connect(self._onEmissionChange)

        # Initialize model
        self.model = IlluminationResponse(
            wavelength_start=0, wavelength_end=0,
            cross_section_off_to_on=0.0, cross_section_on_to_off=0.0, cross_section_emission=0.0
        )

    # Model event handling
    def _onBasicFieldChange(self, model: IlluminationResponse) -> None:
        """ Loads basic model fields (spinboxes etc.) into the interface fields. """
        self._widget.updateBasicFields(model)

    # UI event handling
    @pyqtSlot(float)
    def _onOffToOnChange(self, value: float) -> None:
        self.model.cross_section_off_to_on = value

    @pyqtSlot(float)
    def _onOnToOffChange(self, value: float) -> None:
        self.model.cross_section_on_to_off = value

    @pyqtSlot(float)
    def _onEmissionChange(self, value: float) -> None:
        self.model.cross_section_emission = value
