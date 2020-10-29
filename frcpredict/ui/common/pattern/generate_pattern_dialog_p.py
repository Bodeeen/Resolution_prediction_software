from typing import Optional

from PyQt5.QtCore import pyqtSlot

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
        # Disconnect old model event handling
        try:
            self._model.data_loaded.disconnect(self._onPatternDataChange)
        except AttributeError:
            pass

        # Set model
        self._model = model

        # Trigger model change event handlers
        self._onPatternDataChange(model)

        # Prepare model events
        model.data_loaded.connect(self._onPatternDataChange)

    # Methods
    def __init__(self, widget, normalizePreview: bool = False) -> None:
        self._normalizePreview = normalizePreview
        super().__init__(Pattern(), widget)

        # Prepare UI events
        widget.typeChanged.connect(self._uiTypeChange)
        widget.amplitudeChanged.connect(self._uiAmplitudeChange)
        widget.radiusChanged.connect(self._uiRadiusChange)
        widget.fwhmChanged.connect(self._uiFwhmChange)
        widget.periodicityChanged.connect(self._uiPeriodicityChange)
        widget.zeroIntensityChanged.connect(self._uiZeroIntensityChange)
        widget.naChanged.connect(self._uiNaChange)
        widget.emissionWavelengthChanged.connect(self._uiEmissionWavelengthChange)

    # Internal methods
    def _updatePatternDataInModel(self, field_name: str, field_value: float) -> None:
        if (hasattr(self.model.pattern_data, field_name) and
                getattr(self.model.pattern_data, field_name) != field_value):
            setattr(self.model.pattern_data, field_name, field_value)
            self._onPatternDataChange(self.model)

    # Model event handling
    def _onPatternDataChange(self, model: Pattern) -> None:
        """
        Updates the preview based on the pattern data, sets which property fields are available
        based on the pattern type, and updates the values of the property fields.
        """

        self.widget.updatePreview(
            getArrayPixmap(
                model.get_numpy_array(canvas_inner_radius_nm=self.widget.canvasInnerRadius(),
                                      px_size_nm=self.widget.canvasInnerRadius() * 2 / 81),
                normalize=self._normalizePreview
            )
        )

        self.widget.updateType(model.pattern_type)

        hasAmplitudeProperty = (model.pattern_type == PatternType.gaussian or
                                model.pattern_type == PatternType.airy_from_FWHM)

        hasRadiusProperty = model.pattern_type == PatternType.physical_pinhole

        hasFwhmProperty = (model.pattern_type == PatternType.gaussian or
                           model.pattern_type == PatternType.airy_from_FWHM or
                           model.pattern_type == PatternType.digital_pinhole)

        hasPeriodicityProperty = model.pattern_type == PatternType.doughnut

        hasZeroIntensityProperty = model.pattern_type == PatternType.doughnut

        hasNAProperty = model.pattern_type == PatternType.airy_from_NA

        hasEmissionWavelengthProperty = model.pattern_type == PatternType.airy_from_NA

        if hasAmplitudeProperty:
            self.widget.updatePropertyFields(amplitude=model.pattern_data.amplitude)

        if hasRadiusProperty:
            self.widget.updatePropertyFields(radius=model.pattern_data.radius)

        if hasFwhmProperty:
            self.widget.updatePropertyFields(fwhm=model.pattern_data.fwhm)

        if hasPeriodicityProperty:
            self.widget.updatePropertyFields(periodicity=model.pattern_data.periodicity)

        if hasZeroIntensityProperty:
            self.widget.updatePropertyFields(zeroIntensity=model.pattern_data.zero_intensity * 100)

        if hasNAProperty:
            self.widget.updatePropertyFields(na=model.pattern_data.na)

        if hasEmissionWavelengthProperty:
            self.widget.updatePropertyFields(
                emissionWavelength=model.pattern_data.emission_wavelength
            )

        self.widget.setAvailableProperties(
            amplitude=hasAmplitudeProperty,
            radius=hasRadiusProperty,
            fwhm=hasFwhmProperty,
            periodicity=hasPeriodicityProperty,
            zeroIntensity=hasZeroIntensityProperty,
            na=hasNAProperty,
            emissionWavelength=hasEmissionWavelengthProperty
        )

    # UI event handling
    @pyqtSlot(object)
    def _uiTypeChange(self, patternType: Optional[PatternType] = None) -> None:
        """
        Loads default data for the selected pattern type into the model. If no pattern type has
        been selected, an empty pattern is loaded.
        """

        if patternType is not None:
            self.model.load_from_type(patternType)
        else:
            self.model.load_from_data(Array2DPatternData())

    @pyqtSlot(float)
    def _uiAmplitudeChange(self, value: float) -> None:
        self._updatePatternDataInModel("amplitude", value)

    @pyqtSlot(float)
    def _uiRadiusChange(self, value: float) -> None:
        self._updatePatternDataInModel("radius", value)

    @pyqtSlot(float)
    def _uiFwhmChange(self, value: float) -> None:
        self._updatePatternDataInModel("fwhm", value)

    @pyqtSlot(float)
    def _uiPeriodicityChange(self, value: float) -> None:
        self._updatePatternDataInModel("periodicity", value)

    @pyqtSlot(float)
    def _uiZeroIntensityChange(self, value: float) -> None:
        self._updatePatternDataInModel("zero_intensity", value / 100)

    @pyqtSlot(float)
    def _uiNaChange(self, value: float) -> None:
        self._updatePatternDataInModel("na", value)

    @pyqtSlot(float)
    def _uiEmissionWavelengthChange(self, value: float) -> None:
        self._updatePatternDataInModel("emission_wavelength", value)
