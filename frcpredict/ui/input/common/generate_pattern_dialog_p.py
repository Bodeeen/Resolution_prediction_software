import numpy as np
from typing import Optional, List

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog

from frcpredict.model import Pattern, Array2DPatternData, PatternType
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import getArrayPixmap


class GeneratePatternPresenter(BasePresenter[Pattern]):
    """
    Presenter for the generate pattern dialog.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: Pattern) -> None:
        self._model = model

        # Trigger model change event handlers
        self._onPatternDataChange(model)

        # Prepare model events
        model.data_loaded.connect(self._onPatternDataChange)

    # Methods
    def __init__(self, widget, normalisePreview: bool = False) -> None:
        self._normalisePreview = normalisePreview

        # Initialize model
        model = Pattern(pattern_data=Array2DPatternData())

        super().__init__(model, widget)

        # Prepare UI events
        widget.typeChanged.connect(self._uiTypeChange)
        widget.amplitudeChanged.connect(self._uiAmplitudeChange)
        widget.fwhmChanged.connect(self._uiFwhmChange)
        widget.periodicityChanged.connect(self._uiPeriodicityChange)

    # Model event handling
    def _onPatternDataChange(self, model: Pattern) -> None:
        """
        Updates the preview based on the pattern data, sets which property fields are available
        based on the pattern type, and updates the values of the property fields.
        """

        self.widget.setPreview(
            getArrayPixmap(
                model.get_numpy_array(pixels_per_nm=32),
                normalise=self._normalisePreview
            )
        )

        hasAmplitudeProperty = (model.pattern_type == PatternType.gaussian or
                                model.pattern_type == PatternType.airy)

        hasFwhmProperty = (model.pattern_type == PatternType.gaussian or
                           model.pattern_type == PatternType.airy or
                           model.pattern_type == PatternType.digital_pinhole)

        hasPeriodicityProperty = model.pattern_type == PatternType.doughnut

        if hasAmplitudeProperty:
            self.widget.updatePropertyFields(amplitude=model.pattern_data.amplitude)

        if hasFwhmProperty:
            self.widget.updatePropertyFields(fwhm=model.pattern_data.fwhm)

        if hasPeriodicityProperty:
            self.widget.updatePropertyFields(periodicity=model.pattern_data.periodicity)

        self.widget.setAvailableProperties(
            amplitude=hasAmplitudeProperty,
            fwhm=hasFwhmProperty,
            periodicity=hasPeriodicityProperty
        )

    # UI event handling
    @pyqtSlot(object)
    def _uiTypeChange(self, patternType: Optional[PatternType] = None) -> None:
        """
        Loads default data for the selected pattern type into the model. If no pattern type has
        been selected, an empty pattern is loaded.
        """

        if patternType is not None:
            self.model.load_type(patternType)
        else:
            self.model.load_data(Array2DPatternData())

    @pyqtSlot(float)
    def _uiAmplitudeChange(self, value: float) -> None:
        if hasattr(self.model.pattern_data, "amplitude") and self.model.pattern_data.amplitude != value:
            self.model.pattern_data.amplitude = value
            self._onPatternDataChange(self.model)

    @pyqtSlot(float)
    def _uiFwhmChange(self, value: float) -> None:
        if hasattr(self.model.pattern_data, "fwhm") and self.model.pattern_data.fwhm != value:
            self.model.pattern_data.fwhm = value
            self._onPatternDataChange(self.model)

    @pyqtSlot(float)
    def _uiPeriodicityChange(self, value: float) -> None:
        if hasattr(self.model.pattern_data, "periodicity") and self.model.pattern_data.periodicity != value:
            self.model.pattern_data.periodicity = value
            self._onPatternDataChange(self.model)
