from PyQt5.QtCore import pyqtSlot, QObject

from frcpredict.model import SampleProperties
from frcpredict.ui import BasePresenter


class SamplePropertiesPresenter(BasePresenter[SampleProperties]):
    """
    Presenter for the sample properties widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: SampleProperties) -> None:
        self._model = model

        # Trigger model change event handlers
        self._onBasicFieldChange(model)

        # Prepare model events
        model.basic_field_changed.connect(self._onBasicFieldChange)

    # Methods
    def __init__(self, widget) -> None:
        # Initialize model
        model = SampleProperties(
            spectral_power=0.0,
            labelling_density=0.0
        )

        super().__init__(model, widget)

        # Prepare UI events
        widget.spectralPowerChanged.connect(self._uiSpectralPowerChange)
        widget.labellingDensityChanged.connect(self._uiLabellingDensityChange)

    # Model event handling
    def _onBasicFieldChange(self, model: SampleProperties) -> None:
        """ Loads basic model fields (spinboxes etc.) into the interface fields. """
        self._widget.updateBasicFields(model)

    # UI event handling
    @pyqtSlot(float)
    def _uiSpectralPowerChange(self, value: float) -> None:
        self.model.spectral_power = value

    @pyqtSlot(float)
    def _uiLabellingDensityChange(self, value: float) -> None:
        self.model.labelling_density = value
