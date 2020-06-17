import numpy as np

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog

from frcpredict.model import ImagingSystemSettings, Pattern, Array2DPatternData
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import getArrayPixmap


class ImagingSystemSettingsPresenter(BasePresenter[ImagingSystemSettings]):
    """
    Presenter for the imaging system settings widget.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: ImagingSystemSettings) -> None:
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

    # Model event handling
    def _onBasicFieldChange(self, model: ImagingSystemSettings) -> None:
        """ Loads basic model fields (e.g. ints) into the widget. """
        self.widget.updateBasicFields(model)

    # UI event handling
    @pyqtSlot(float)
    def _uiScanningStepSizeChange(self, value: float) -> None:
        self.model.scanningStepSizeChanged = value
