from typing import List

from PyQt5.QtCore import pyqtSignal, pyqtProperty, Qt
from PyQt5.QtGui import QPixmap

from frcpredict.model import Pattern, PatternType
from frcpredict.ui import BaseWidget
from .pattern_field_p import PatternFieldPresenter


class PatternFieldWidget(BaseWidget):
    """
    A widget for a pattern (2D array) input field. Here, the user may load a file that contains
    pattern data, or open a dialog to generate a pattern and load it. It also displays a preview of
    the loaded pattern. Hovering the mouse over the preview will show information about the
    loaded pattern.
    """

    # Properties
    @pyqtProperty(bool)
    def allowLoadFile(self) -> bool:
        return self._allowLoadFile

    @allowLoadFile.setter
    def allowLoadFile(self, value: bool) -> None:
        self._allowLoadFile = value
        if self.isUiLoaded():
            self.btnLoadFile.setVisible(value)

    @pyqtProperty(bool)
    def normalizeVisualisation(self) -> bool:
        return self._normalizeVisualisation

    @normalizeVisualisation.setter
    def normalizeVisualisation(self, value: bool) -> None:
        self._normalizeVisualisation = value

        # This is an ugly way to handle changes to the "normalizeVisualisation" property
        self._presenter.deleteLater()
        self._presenter = PatternFieldPresenter(self, normalizeVisualisation=value)

    # Signals
    valueChanged = pyqtSignal(Pattern)
    loadFileClicked = pyqtSignal()
    generateClicked = pyqtSignal()
    canvasInnerRadiusChanged = pyqtSignal(float)

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        self._allowEditGenerationAmplitude = True
        self._availableGenerationTypes = []
        self._fieldName = "Pattern"
        self._canvasInnerRadius = 500.0

        # Prepare UI elements
        self.setFocusPolicy(Qt.TabFocus)
        self.setFocusProxy(self.btnGenerate)

        # Connect forwarded signals
        self.btnGenerate.clicked.connect(self.generateClicked)
        self.btnLoadFile.clicked.connect(self.loadFileClicked)

        # Initialize presenter
        self._presenter = PatternFieldPresenter(self)

    def allowEditGenerationAmplitude(self) -> bool:
        """
        Returns whether the user may edit the pattern amplitude when generating a pattern, assuming
        the option is available.
        """
        return self._allowEditGenerationAmplitude

    def setAllowEditGenerationAmplitude(self, allowEditAmplitude: bool) -> None:
        """
        Sets whether the user may edit the pattern amplitude when generating a pattern, assuming
        the option is available.
        """
        self._allowEditGenerationAmplitude = allowEditAmplitude

    def availableGenerationTypes(self) -> List[PatternType]:
        """
        Returns which pattern types are available for the user to pick when generating a pattern.
        """
        return self._availableGenerationTypes

    def setAvailableGenerationTypes(self, patternTypes: List[PatternType]) -> None:
        """
        Sets which pattern types are available for the user to pick when generating a pattern.
        """
        self._availableGenerationTypes = patternTypes

    def canvasInnerRadius(self) -> float:
        """
        Returns the inner radius of the canvas, in nanometres, that is used to generate pattern
        previews.
        """
        return self._canvasInnerRadius

    def setCanvasInnerRadius(self, canvasInnerRadiusNm: float) -> None:
        """
        Sets the inner radius of the canvas, in nanometres, that will be used to generate pattern
        previews.
        """
        self._canvasInnerRadius = canvasInnerRadiusNm
        self.canvasInnerRadiusChanged.emit(canvasInnerRadiusNm)

    def fieldName(self) -> str:
        """ Returns the displayed name of the field, e.g. "Optical PSF". """
        return self._fieldName

    def setFieldName(self, fieldName: str) -> None:
        """ Sets the displayed name of the field, e.g. "Optical PSF". """
        self._fieldName = fieldName

    def value(self) -> Pattern:
        return self._presenter.model

    def setValue(self, model: Pattern, emitSignal: bool = True) -> None:
        self._presenter.model = model

        if emitSignal:
            self.valueChanged.emit(model)

    def updateVisualisation(self, pixmap: QPixmap, description: str) -> None:
        self.imgVisualisation.setPixmap(pixmap)
        self.imgVisualisation.setToolTip(description)
