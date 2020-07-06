from typing import Union

from PyQt5.QtCore import pyqtSlot

from frcpredict.model import ImagingSystemSettings, Pattern, Array2DPatternData, ValueRange
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import connectMulti


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

    # Methods
    def __init__(self, widget) -> None:
        # Initialize model
        model = ImagingSystemSettings(
            optical_psf=Pattern(pattern_data=Array2DPatternData()),
            pinhole_function=Pattern(pattern_data=Array2DPatternData()),
            scanning_step_size=1.0
        )

        super().__init__(model, widget)

        # Prepare UI events
        widget.opticalPsfChanged.connect(self._uiSetOpticalPsfModel)
        widget.pinholeFunctionChanged.connect(self._uiSetPinholeFunctionModel)
        connectMulti(widget.scanningStepSizeChanged, [float, ValueRange],
                     self._uiScanningStepSizeChange)

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
    @pyqtSlot(ValueRange)
    def _uiScanningStepSizeChange(self, value: Union[float, ValueRange[float]]) -> None:
        self.model.scanning_step_size = value
