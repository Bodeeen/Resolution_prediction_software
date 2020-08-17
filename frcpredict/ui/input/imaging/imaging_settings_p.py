from typing import Union

from PyQt5.QtCore import pyqtSlot

from frcpredict.model import ImagingSystemSettings, Pattern, Multivalue
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import connectMulti, connectModelToSignal, disconnectModelFromSignal


class ImagingSystemSettingsPresenter(BasePresenter[ImagingSystemSettings]):
    """
    Presenter for the imaging system settings widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: ImagingSystemSettings) -> None:
        # Disconnect old model event handling
        try:
            self._model.basic_field_changed.disconnect(self._onBasicFieldChange)
            disconnectModelFromSignal(self.model, self._modifiedFlagSlotFunc)
        except AttributeError:
            pass

        # Set model
        self._model = model

        # Trigger model change event handlers
        self.widget.updateOpticalPsf(model.optical_psf)
        self.widget.updatePinholeFunction(model.pinhole_function)
        self._onBasicFieldChange(model)

        # Prepare model events
        model.basic_field_changed.connect(self._onBasicFieldChange)
        self._modifiedFlagSlotFunc = connectModelToSignal(self.model, self.widget.modifiedFlagSet)

    # Methods
    def __init__(self, widget) -> None:
        super().__init__(ImagingSystemSettings(), widget)

        # Prepare UI events
        widget.opticalPsfChanged.connect(self._uiSetOpticalPsfModel)
        widget.pinholeFunctionChanged.connect(self._uiSetPinholeFunctionModel)
        connectMulti(widget.scanningStepSizeChanged, [float, Multivalue],
                     self._uiScanningStepSizeChange)
        widget.refractiveIndexChanged.connect(self._uiRefractiveIndexChange)

    # Model event handling
    def _onBasicFieldChange(self, model: ImagingSystemSettings) -> None:
        """ Loads basic model fields (e.g. ints) into the widget. """
        self.widget.updateBasicFields(model)

    # UI event handling
    @pyqtSlot(Pattern)
    def _uiSetOpticalPsfModel(self, value: Pattern) -> None:
        self.model.optical_psf = value

    @pyqtSlot(Pattern)
    def _uiSetPinholeFunctionModel(self, value: Pattern) -> None:
        self.model.pinhole_function = value

    @pyqtSlot(float)
    @pyqtSlot(Multivalue)
    def _uiScanningStepSizeChange(self, value: Union[float, Multivalue[float]]) -> None:
        self.model.scanning_step_size = value

    @pyqtSlot(float)
    def _uiRefractiveIndexChange(self, value: float) -> None:
        self.model.refractive_index = value
