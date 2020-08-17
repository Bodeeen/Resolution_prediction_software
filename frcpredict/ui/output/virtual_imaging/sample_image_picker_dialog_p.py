from typing import Optional

from PyQt5.QtCore import pyqtSlot
from PyQt5.QtWidgets import QFileDialog

from frcpredict.model import Array2DPatternData, SampleStructure
from frcpredict.ui import BasePresenter
from frcpredict.ui.util import getArrayPixmap
from frcpredict.util import get_sample_structure_data_dir_names
from .sample_image_picker_dialog_m import SampleImagePickerModel


class SampleImagePickerPresenter(BasePresenter[SampleImagePickerModel]):
    """
    Presenter for the image picker dialog.
    """

    # Properties
    @BasePresenter.model.setter
    def model(self, model: SampleImagePickerModel) -> None:
        # Disconnect old model event handling
        try:
            self._model.fluorophoresPerUnitChanged.disconnect(self._onFluorophoresPerUnitChange)
        except AttributeError:
            pass

        self._model = model

        # Trigger model change event handlers (only model event, since it in turn also triggers the
        # others)
        self._onModelSet(model)

        # Prepare model events
        model.fluorophoresPerUnitChanged.connect(self._onFluorophoresPerUnitChange)

    # Methods
    def __init__(self, widget) -> None:
        super().__init__(SampleImagePickerModel(), widget)

        # Prepare UI events
        widget.sampleStructurePicked.connect(self._uiSampleStructurePick)
        widget.fromSampleSelected.connect(self._uiFromSampleSelect)
        widget.fromFileSelected.connect(self._uiFromFileSelect)
        widget.loadFileClicked.connect(self._uiClickLoadFile)
        widget.fluorophoresPerUnitChanged.connect(self._uiFluorophoresPerUnitChange)

        # Load sample structures
        self._loadSampleStructures()

    # Internal methods
    def _loadSampleStructures(self) -> None:
        self.widget.setAvailableSampleStructures([
            SampleStructure.from_sample_data_directory_name(sampleDirName)
            for sampleDirName in get_sample_structure_data_dir_names()
        ])

    # Model event handling
    def _onModelSet(self, model: SampleImagePickerModel) -> None:
        """ Updates the preview and fields based on the image data. """

        self.widget.updateFields(not model.image.is_empty(), model.fromFile,
                                 model.sampleStructureId)

        self.widget.updatePreview(
            getArrayPixmap(model.image.get_numpy_array())
        )

        self._onFluorophoresPerUnitChange(model.fluorophoresPerUnit)

    def _onFluorophoresPerUnitChange(self, fluorophoresPerUnit: float) -> None:
        self.widget.updateFluorophoresPerUnit(fluorophoresPerUnit)

    # UI event handling
    @pyqtSlot(object)
    def _uiSampleStructurePick(self, sample: Optional[SampleStructure] = None) -> None:
        if sample is None:
            return

        self.model = SampleImagePickerModel(
            image=sample.image, sampleStructureId=sample.id,
            fluorophoresPerUnit=1.0, fromFile=False
        )

    @pyqtSlot()
    def _uiFromSampleSelect(self) -> None:
        self.model = SampleImagePickerModel(
            image=Array2DPatternData(), sampleStructureId=None,
            fluorophoresPerUnit=1.0, fromFile=False
        )

    @pyqtSlot()
    def _uiFromFileSelect(self) -> None:
        self.model = SampleImagePickerModel(
            image=Array2DPatternData(), sampleStructureId=None,
            fluorophoresPerUnit=1.0, fromFile=True
        )

    @pyqtSlot()
    def _uiClickLoadFile(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self.widget,
            caption=f"Open Sample Image File",
            filter="Supported files (*.npy;*.tif;*.tiff;*.png)"
        )

        if path:  # Check whether a file was picked
            if path.endswith(".npy"):
                image = Array2DPatternData.from_npy_file(path)
            else:
                image = Array2DPatternData.from_image_file(path)

            self.model = SampleImagePickerModel(
                image=image, sampleStructureId=None,
                fluorophoresPerUnit=self.model.fluorophoresPerUnit, fromFile=True
            )

    @pyqtSlot(float)
    def _uiFluorophoresPerUnitChange(self, fluorophoresPerUnit: float) -> None:
        self.model.fluorophoresPerUnit = fluorophoresPerUnit
