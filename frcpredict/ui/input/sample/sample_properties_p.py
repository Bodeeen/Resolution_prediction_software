from typing import Union

from PyQt5.QtCore import pyqtSlot

from frcpredict.model import ValueRange, SampleProperties
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import connectMulti


class SamplePropertiesPresenter(BasePresenter[SampleProperties]):
    """
    Presenter for the sample properties widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: SampleProperties) -> None:
        # Disconnect old model event handling
        try:
            self._model.basic_field_changed.disconnect(self._onBasicFieldChange)
        except AttributeError:
            pass

        # Set model
        self._model = model

        # Trigger model change event handlers
        self._onBasicFieldChange(model)

        # Prepare model events
        model.basic_field_changed.connect(self._onBasicFieldChange)

    # Methods
    def __init__(self, widget) -> None:
        # Initialize model
        model = SampleProperties(
            spectral_power=1.0,
            labelling_density=1.0,
            K_origin=1.0
        )

        super().__init__(model, widget)

        # Prepare UI events
        connectMulti(widget.spectralPowerChanged, [float, ValueRange],
                     self._uiSpectralPowerChange)
        connectMulti(widget.labellingDensityChanged, [float, ValueRange],
                     self._uiLabellingDensityChange)
        connectMulti(widget.KOriginChanged, [float, ValueRange],
                     self._uiKOriginChange)

    # Model event handling
    def _onBasicFieldChange(self, model: SampleProperties) -> None:
        """ Loads basic model fields (e.g. ints) into the widget. """
        self.widget.updateBasicFields(model)

    # UI event handling
    @pyqtSlot(float)
    @pyqtSlot(ValueRange)
    def _uiSpectralPowerChange(self, value: Union[float, ValueRange[float]]) -> None:
        self.model.spectral_power = value

    @pyqtSlot(float)
    @pyqtSlot(ValueRange)
    def _uiLabellingDensityChange(self, value: Union[float, ValueRange[float]]) -> None:
        self.model.labelling_density = value

    @pyqtSlot(float)
    @pyqtSlot(ValueRange)
    def _uiKOriginChange(self, value: Union[float, ValueRange[float]]) -> None:
        self.model.K_origin = value
