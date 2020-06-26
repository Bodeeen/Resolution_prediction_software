from typing import List

from PyQt5.QtCore import pyqtSignal, pyqtProperty
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
    def normaliseVisualisation(self):
        return self._normaliseVisualisation

    @normaliseVisualisation.setter
    def normaliseVisualisation(self, value):
        self._normaliseVisualisation = value
        # This is an ugly way to listen to changes on the "normaliseVisualisation" property
        self._presenter = PatternFieldPresenter(self, normaliseVisualisation=value)

    # Signals
    valueChanged = pyqtSignal(Pattern)
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
