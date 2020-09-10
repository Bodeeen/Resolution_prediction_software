from typing import Optional

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog

from frcpredict.model import DisplayableSample, SampleStructure, Array2DPatternData
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import getArrayPixmap
from .gen_lines_dialog import LinesStructureDialog
from .gen_pairs_dialog import PairsStructureDialog
from .gen_random_dialog import RandomPointsStructureDialog
from .sample_from_file_dialog import SampleFromFileDialog
from .sample_structure_picker_dialog_m import SampleStructurePickerModel, SampleType


class SampleStructurePickerPresenter(BasePresenter[SampleStructurePickerModel]):
    """
    Presenter for the sample structure picker dialog.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: SampleStructurePickerModel) -> None:
        # Disconnect old model event handling
        try:
            self._model.displayableSampleChanged.disconnect(self._onDisplayableSampleChange)
        except AttributeError:
            pass

        # Set model
        self._model = model

        # Trigger model change event handlers
        self._onDisplayableSampleChange(model._displayableSample, model._displayableSampleType)

        # Prepare model events
        model.displayableSampleChanged.connect(self._onDisplayableSampleChange)

    # Methods
    def __init__(self, widget, inputSampleStructure: Optional[SampleStructure] = None) -> None:
        super().__init__(SampleStructurePickerModel(inputLoadedSample=inputSampleStructure), widget)

        if self.model.inputLoadedSample is not None:
            self.model.loadInputSample()

        # Prepare UI events
        widget.generateRandomClicked.connect(self._uiClickGenerateRandom)
        widget.generatePairsClicked.connect(self._uiClickGeneratePairs)
        widget.generateLinesClicked.connect(self._uiClickGenerateLines)
        widget.loadFromInputClicked.connect(self._uiClickLoadFromInput)
        widget.loadFromFileClicked.connect(self._uiClickLoadFromFile)

    # Model event handling
    def _onDisplayableSampleChange(self, displayableSample: DisplayableSample,
                                   displayableSampleType: SampleType) -> None:
        """ Updates the preview and property fields based on the sample. """
        image_arr = displayableSample.get_image_arr(
            1000 * displayableSample.get_area_side_um() / 250
        )
        self.widget.updatePreview(getArrayPixmap(image_arr, normalize=True))
        self.widget.updateLoadedType(displayableSampleType)

    # UI event handling
    @pyqtSlot()
    def _uiClickGenerateRandom(self) -> None:
        sampleStructure, okClicked = RandomPointsStructureDialog.getStructure(self.widget)

        if okClicked:
            self.model.loadSample(sampleStructure, SampleType.random)

    @pyqtSlot()
    def _uiClickGeneratePairs(self) -> None:
        sampleStructure, okClicked = PairsStructureDialog.getStructure(self.widget)

        if okClicked:
            self.model.loadSample(sampleStructure, SampleType.pairs)

    @pyqtSlot()
    def _uiClickGenerateLines(self) -> None:
        sampleStructure, okClicked = LinesStructureDialog.getStructure(self.widget)

        if okClicked:
            self.model.loadSample(sampleStructure, SampleType.lines)

    @pyqtSlot()
    def _uiClickLoadFromInput(self) -> None:
        self.model.loadInputSample()

    @pyqtSlot()
    def _uiClickLoadFromFile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.widget,
            caption=f"Open Sample Image File",
            filter="Supported files (*.npy;*.tif;*.tiff;*.png)"
        )

        if path:  # Check whether a file was picked
            if path.endswith(".npy"):
                image = Array2DPatternData.from_npy_file(path)
            else:
                image = Array2DPatternData.from_image_file(path, normalize_ints=False)

            sampleImage, okClicked = SampleFromFileDialog.getSampleImage(self.widget,
                                                                         image.get_numpy_array())

            if okClicked:
                self.model.loadSample(sampleImage, SampleType.file)
