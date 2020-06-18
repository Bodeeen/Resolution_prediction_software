import numpy as np

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog

from frcpredict.model import Pattern, Array2DPatternData
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import getArrayPixmap
from .generate_pattern_dialog_w import GeneratePatternDialog


class PatternFieldPresenter(BasePresenter[Pattern]):
    """
    Presenter for the pattern field widget.
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
    def __init__(self, widget, normaliseVisualisation: bool = False) -> None:
        self._normaliseVisualisation = normaliseVisualisation

        # Initialize model
        model = Pattern(pattern_data=Array2DPatternData())

        super().__init__(model, widget)

        # Prepare UI events
        widget.loadFileClicked.connect(self._uiClickLoadFile)
        widget.generateClicked.connect(self._uiClickGenerate)

    # Model event handling
    def _onPatternDataChange(self, model: Pattern):
        """ Updates the visualization based on the pattern type and data. """

        self.widget.updateVisualisation(
            pixmap=getArrayPixmap(
                model.get_numpy_array(pixels_per_nm=64),
                normalise=self._normaliseVisualisation
            ),
            description=str(model)
        )

    # UI event handling
    @pyqtSlot()
    def _uiClickLoadFile(self) -> None:
        """
        Lets the user open a file that contains pattern data, and loads the contents into the field.
        """

        path, _ = QFileDialog.getOpenFileName(
            self.widget,
            caption="Open file",
            filter="Supported files (*.npy *.tif *.tiff *.png)"
        )

        if path:
            if path.endswith(".npy"):
                self.model.load_data(Array2DPatternData.from_npy_file(path))
            else:
                self.model.load_data(Array2DPatternData.from_image_file(path))
    
    @pyqtSlot()
    def _uiClickGenerate(self) -> None:
        """
        Loads a generated pattern into the field. A dialog will open for the user to pick what type
        of pattern to generate and with what parameters.
        """

        pattern_data, ok_pressed = GeneratePatternDialog.getPatternData(
            self.widget,
            title=f"Generate {self.widget.fieldName()}",
            availableTypes=self.widget.availableGenerationTypes(),
            allowEditAmplitude=self.widget.allowEditGenerationAmplitude(),
            normalisePreview=self._normaliseVisualisation
        )
        
        if ok_pressed:
            self.model.load_data(pattern_data)
