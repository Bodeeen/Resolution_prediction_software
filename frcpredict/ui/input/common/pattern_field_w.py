from typing import List

from PyQt5.QtCore import pyqtSignal, pyqtProperty
from PyQt5.QtGui import QPixmap

from frcpredict.model import Pattern, PatternType
from frcpredict.ui import BaseWidget
from .pattern_field_p import PatternFieldPresenter


class PatternFieldWidget(BaseWidget):
    """
    A widget for a pattern (2D array) input field.
    """

    # Properties
    @pyqtProperty(bool)
    def normaliseVisualisation(self):
        return self._normaliseVisualisation

    @normaliseVisualisation.setter
    def normaliseVisualisation(self, value):
        self._normaliseVisualisation = value
        # This is an ugly way to listen to changes on the "normaliseVisualisation" property
        self._presenter = PatternFieldPresenter(self, normaliseVisualisation=value)

    # Signals
    loadFileClicked = pyqtSignal()
    generateClicked = pyqtSignal()

    # Methods
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(__file__, *args, **kwargs)

        self._allowEditGenerationAmplitude = True
        self._availableGenerationTypes = []
        self._fieldName = "Pattern"
        
        # Connect forwarded signals
        self.btnLoadFile.clicked.connect(self.loadFileClicked)
        self.btnGenerate.clicked.connect(self.generateClicked)

        # Initialize presenter
        self._presenter = PatternFieldPresenter(self)

    def allowEditGenerationAmplitude(self) -> bool:
        return self._allowEditGenerationAmplitude

    def setAllowEditGenerationAmplitude(self, allowEditAmplitude: bool) -> None:
        self._allowEditGenerationAmplitude = allowEditAmplitude

    def availableGenerationTypes(self) -> bool:
        return self._availableGenerationTypes

    def setAvailableGenerationTypes(self, patternTypes: List[PatternType]) -> None:
        self._availableGenerationTypes = patternTypes

    def fieldName(self) -> str:
        """ Returns the name of the field, e.g. "Optical PSF". """
        return self._fieldName

    def setFieldName(self, fieldName: str) -> None:
        """ Sets the name of the field, e.g. "Optical PSF". """
        self._fieldName = fieldName

    def value(self) -> Pattern:
        return self._presenter.model

    def setValue(self, model: Pattern) -> None:
        self._presenter.model = model

    def setVisualisation(self, pixmap: QPixmap, description: str) -> None:
        self.imgVisualisation.setPixmap(pixmap)
        self.imgVisualisation.setToolTip(description)
